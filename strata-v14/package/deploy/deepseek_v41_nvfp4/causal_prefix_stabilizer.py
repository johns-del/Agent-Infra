"""Online Causal Session Prefix Stabilizer (Pillar 1).

Strictly causal: uses ONLY already-observed prompts within the same workflow/session.
Detects mid-prompt dynamic insertions/mutations (`B_t`) ahead of unchanged multi-turn
history (`C`), keeps `[A] + [C] + [new D_t]` as a contiguous monotone RadixCache prefix,
and relocates only the short volatile span (`B_t`) right before the final assistant
generation trigger tokens.

Every transformation preserves exact token count and exact token multiset conservation.
"""
from __future__ import annotations

from collections import defaultdict


def canonicalize_against_spine(
    cur_raw: list[int],
    spine: list[int] | None,
    *,
    window: int = 32,
    stride: int = 8,
    min_contiguous_match: int = 128,
    min_total_recovered: int = 256,
    max_volatile_tokens: int = 4096,
    max_spine_skip_tokens: int = 0,
    tail_guard_tokens: int = 4,
) -> tuple[list[int], list[int], int]:
    """Stabilize a single prompt against its session's historical spine.

    Only pure mid-prompt insertions (``p_pos == s_cur`` when ``max_spine_skip_tokens == 0``)
    are relocated to the tail, because skipping deleted/replaced tokens in ``spine`` breaks
    RadixCache prefix alignment and causes self-poisoning spine drift on subsequent turns.

    Returns:
        `(cur_canon, next_spine, relocated_tokens)`
    """
    length = len(cur_raw)
    if not spine or length < 512:
        return list(cur_raw), list(cur_raw), 0

    n_min = min(len(spine), length)
    lcp = 0
    while lcp < n_min and spine[lcp] == cur_raw[lcp]:
        lcp += 1

    if lcp >= len(spine) - 64:
        return list(cur_raw), list(cur_raw), 0

    spine_idx: dict[tuple[int, ...], int] = {}
    lim_idx = len(spine) - window + 1
    if max_spine_skip_tokens > 0:
        for pos in range(lcp, lim_idx, stride):
            key = tuple(spine[pos : pos + window])
            if key not in spine_idx:
                spine_idx[key] = pos
    elif lcp < lim_idx:
        spine_idx[tuple(spine[lcp : lcp + window])] = lcp

    if not spine_idx:
        return list(cur_raw), list(cur_raw), 0

    kept_spine_and_new = list(cur_raw[:lcp])
    volatile: list[int] = []
    i = lcp
    s_cur = lcp
    matched_after_break = 0
    exact_probe_pos = s_cur
    exact_probe_key = tuple(spine[s_cur : s_cur + window]) if s_cur + window <= len(spine) else None

    while i + window <= length:
        probe = tuple(cur_raw[i : i + window])
        # Always check exact s_cur first so stride alignment can never skip 1 token
        if exact_probe_key is not None and probe == exact_probe_key:
            p_pos = exact_probe_pos
        else:
            p_pos = spine_idx.get(probe)
        if p_pos is not None and s_cur <= p_pos <= s_cur + max_spine_skip_tokens:
            m = window
            lim = min(length - i, len(spine) - p_pos)
            while m < lim and cur_raw[i + m] == spine[p_pos + m]:
                m += 1
            if m >= min_contiguous_match:
                kept_spine_and_new.extend(cur_raw[i : i + m])
                matched_after_break += m
                i += m
                s_cur = p_pos + m
                exact_probe_pos = s_cur
                exact_probe_key = tuple(spine[s_cur : s_cur + window]) if s_cur + window <= len(spine) else None
                continue
        if s_cur < len(spine) - min_contiguous_match and len(volatile) < max_volatile_tokens:
            volatile.append(cur_raw[i])
            i += 1
        else:
            kept_spine_and_new.extend(cur_raw[i:])
            i = length
            break
    if i < length:
        kept_spine_and_new.extend(cur_raw[i:])

    # Require full spine alignment (no mid-spine deletion/replacement break that left the
    # tail of `spine` unmatched and swallowed innocent body tokens up to max_volatile_tokens).
    if not volatile or matched_after_break < min_total_recovered or s_cur < len(spine) - 64:
        return list(cur_raw), list(cur_raw), 0

    guard = min(tail_guard_tokens, len(kept_spine_and_new))
    if guard > 0:
        next_spine_head = kept_spine_and_new[:-guard]
        tail_guard = kept_spine_and_new[-guard:]
    else:
        next_spine_head = kept_spine_and_new
        tail_guard = []

    cur_canon = next_spine_head + volatile + tail_guard
    if len(cur_canon) != length:
        raise AssertionError(f"Token count mismatch after canonicalization: {len(cur_canon)} != {length}")
    return cur_canon, kept_spine_and_new, len(volatile)


class CausalPrefixStabilizer:
    """Thread/async-safe per-trial online causal prefix stabilizer keyed by (graph_id, scope_id)."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = bool(enabled)
        self.scope_spines: dict[tuple[str, str], list[int]] = {}
        self.graph_fallback_spines: dict[str, list[int]] = {}
        self.requests_seen = 0
        self.requests_stabilized = 0
        self.tokens_total = 0
        self.tokens_relocated = 0

    def reset(self):
        self.scope_spines.clear()
        self.graph_fallback_spines.clear()
        self.requests_seen = 0
        self.requests_stabilized = 0
        self.tokens_total = 0
        self.tokens_relocated = 0

    def transform(self, graph_id: str, scope_id: str, raw_input_ids: list[int]) -> tuple[list[int], int]:
        self.requests_seen += 1
        self.tokens_total += len(raw_input_ids)
        if not self.enabled:
            return raw_input_ids, 0
        key = (str(graph_id), str(scope_id))
        prior_spine = self.scope_spines.get(key)
        if prior_spine is None:
            prior_spine = self.graph_fallback_spines.get(str(graph_id))
        cur_canon, next_spine, relocated = canonicalize_against_spine(raw_input_ids, prior_spine)
        self.scope_spines[key] = next_spine
        if str(scope_id).endswith("-s0") or str(graph_id) not in self.graph_fallback_spines:
            self.graph_fallback_spines[str(graph_id)] = next_spine
        if relocated > 0:
            self.requests_stabilized += 1
            self.tokens_relocated += relocated
        return cur_canon, relocated

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "requests_seen": self.requests_seen,
            "requests_stabilized": self.requests_stabilized,
            "tokens_total": self.tokens_total,
            "tokens_relocated": self.tokens_relocated,
        }
