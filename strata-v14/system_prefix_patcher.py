"""Semantic block-level System Prefix Freeze & Trailing Override Patcher (Option 2).

When an Agent client (such as Orvix) mutates dynamic tail sections inside ``messages[0]``
(``system`` / ``developer`` prompt — e.g., step counters, ``<task>``, ``<session>``,
``<memory>`` blocks) mid-session, directly forwarding the mutated ``messages[0]`` invalidates
the entire multi-hundred-thousand-token RadixCache prefix.

This module operates at the structured message/semantic-block layer (before tokenization):
1. Freezes the initial ``messages[0]`` text for each session (bounded LRU).
2. Splits ``system`` text into complete, reversible semantic units (complete outer XML blocks,
   Markdown sections, and blank-line paragraphs) — never slicing mid-sentence or mid-tag.
3. On subsequent turns where ``messages[0]`` changes within strict safety guardrails:
   - Restores ``messages[0]`` to the frozen baseline so the 100K-260K prefix hits RadixCache.
   - Wraps the changed/added/removed semantic blocks in an explicit, authoritative update block
     and appends it to the tail of the latest ``user``/``tool`` message (never inserting a new
     ``user`` message that would trigger DeepSeek chat-template reasoning stripping).
4. Zero stale-patch accumulation + 99.5%+ steady-state hit rate:
   - If the next turn keeps the same mutated ``system`` text (steady state), the patcher
     replays the exact same patch at its original anchor message index so even the last turn
     hits KV cache (99.5%+).
   - If the next turn mutates ``system`` again (e.g., per-turn step counter), the old anchor
     patch is dropped and only the single latest state is attached to ``messages[-1]`` — zero
     stale patch buildup, while still reusing 98%-99% of the prefix up to the previous turn.
   - If ``system`` reverts to the frozen baseline, zero patch is attached at all.
   - If a mutation exceeds safety guardrails (e.g., core static prefix changed or delta too
     large), the patcher refreshes the baseline and passes through verbatim to protect quality.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import difflib
from hashlib import sha256
import json
import re
import threading
from typing import Any


_OPEN_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_:-]{0,31})(?:\s[^<>]{0,200})?>")
_MD_HEADING_SPLIT_RE = re.compile(r"(?<=\n)(?=#{1,6}\s)")
_PARA_SPLIT_RE = re.compile(r"(\n[ \t]*\n+)")

PATCH_OPEN_TAG = "<system_context_update>"
PATCH_CLOSE_TAG = "</system_context_update>"


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:20]


def _extract_system_text(msg: dict[str, Any]) -> tuple[str | None, str]:
    """Return ``(text, format_kind)`` where format_kind is ``'str'`` or ``'list'``.

    Returns ``(None, 'unsupported')`` for multimodal or non-text system messages.
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content, "str"
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text" or not isinstance(item.get("text"), str):
                return None, "unsupported"
            parts.append(item["text"])
        return "\n".join(parts), "list"
    return None, "unsupported"


def _set_system_text(msg: dict[str, Any], text: str, fmt: str) -> dict[str, Any]:
    out = dict(msg)
    if fmt == "str":
        out["content"] = text
    else:
        out["content"] = [{"type": "text", "text": text}]
    return out


def _message_text_digest(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        raw = content
    else:
        raw = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _digest(f"{msg.get('role')}|{msg.get('tool_call_id') or ''}|{raw}")


def _append_suffix_to_message(msg: dict[str, Any], suffix: str) -> dict[str, Any] | None:
    """Attach ``suffix`` to the end of a ``user``/``tool``/``function`` message without mutating input."""
    role = str(msg.get("role") or "").lower()
    if role not in {"user", "tool", "function"}:
        return None
    content = msg.get("content")
    out = dict(msg)
    if isinstance(content, str):
        out["content"] = content + suffix
        return out
    if content is None:
        out["content"] = suffix.lstrip("\n")
        return out
    if isinstance(content, list):
        new_list = [dict(x) if isinstance(x, dict) else x for x in content]
        if new_list and isinstance(new_list[-1], dict) and new_list[-1].get("type") == "text" and isinstance(new_list[-1].get("text"), str):
            new_list[-1]["text"] = new_list[-1]["text"] + suffix
        else:
            new_list.append({"type": "text", "text": suffix.lstrip("\n")})
        out["content"] = new_list
        return out
    return None


def _split_plain_paragraphs(chunk: str) -> list[str]:
    """Split non-XML text into complete paragraphs / Markdown heading blocks (reversible)."""
    if not chunk:
        return []
    out: list[str] = []
    # First split at blank-line boundaries while keeping delimiters attached to the preceding block
    parts = _PARA_SPLIT_RE.split(chunk)
    merged: list[str] = []
    idx = 0
    while idx < len(parts):
        seg = parts[idx]
        if idx + 1 < len(parts):
            seg += parts[idx + 1]
            idx += 2
        else:
            idx += 1
        if seg:
            merged.append(seg)

    for seg in merged:
        sub = _MD_HEADING_SPLIT_RE.split(seg)
        for s in sub:
            if s:
                out.append(s)
    return out


def split_semantic_blocks(text: str) -> list[str]:
    """Split system prompt text into complete, reversible semantic blocks.

    Guarantees ``"".join(split_semantic_blocks(text)) == text``.
    Top-level balanced XML blocks (e.g. ``<task>...</task>``, ``<session>...</session>``,
    ``<memory>...</memory>``) are kept intact as atomic blocks; surrounding prose is split
    on Markdown headings and blank-line paragraph boundaries.
    """
    if not text:
        return []
    blocks: list[str] = []
    n = len(text)
    pos = 0
    cursor = 0

    while cursor < n:
        m = _OPEN_TAG_RE.search(text, cursor)
        if m is None:
            break
        start = m.start()
        tag = m.group(1)
        # Skip self-closing tags like <br/>
        if m.group(0).endswith("/>"):
            cursor = m.end()
            continue
        # Only treat as a structural block if it starts at line boundary or after whitespace
        if start > 0 and text[start - 1] not in "\n\r \t":
            cursor = m.end()
            continue
        close_tok = f"</{tag}>"
        # Scan forward supporting same-tag nesting within a reasonable window
        depth = 1
        scan = m.end()
        same_open = re.compile(rf"<{re.escape(tag)}(?:\s[^<>]{{0,200}})?>")
        found_end = -1
        while depth > 0 and scan < n:
            next_close = text.find(close_tok, scan)
            if next_close == -1:
                break
            next_open_m = same_open.search(text, scan, next_close)
            if next_open_m is not None and not next_open_m.group(0).endswith("/>"):
                depth += 1
                scan = next_open_m.end()
            else:
                depth -= 1
                scan = next_close + len(close_tok)
                if depth == 0:
                    found_end = scan
        if found_end == -1:
            cursor = m.end()
            continue
        # Include trailing blank lines belonging to this block so concatenation stays exact
        end = found_end
        while end < n and text[end] in " \t":
            end += 1
        while end < n and text[end] == "\n":
            end += 1
        if start > pos:
            blocks.extend(_split_plain_paragraphs(text[pos:start]))
        blocks.append(text[start:end])
        pos = end
        cursor = end

    if pos < n:
        blocks.extend(_split_plain_paragraphs(text[pos:]))
    return blocks


@dataclass
class BlockDiff:
    static_prefix_chars: int
    common_total_chars: int
    added: list[str] = field(default_factory=list)
    modified: list[tuple[str, str]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.removed)

    @property
    def delta_chars(self) -> int:
        return (
            sum(len(x) for x in self.added)
            + sum(len(new) for _, new in self.modified)
            + sum(len(x) for x in self.removed)
        )


def _refine_large_paragraph_replace(old_blk: str, new_blk: str) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """For large non-XML paragraph replacements (e.g. a 213B reminder appended with single '\\n'
    to a 4KB static section), refine down to the changed complete lines so static prose is not
    duplicated into the trailing update."""
    # Never slice inside a balanced XML block
    if _OPEN_TAG_RE.match(old_blk.strip()) or _OPEN_TAG_RE.match(new_blk.strip()):
        return [], [(old_blk, new_blk)], []
    if max(len(old_blk), len(new_blk)) < 500:
        return [], [(old_blk, new_blk)], []

    old_lines = old_blk.splitlines(keepends=True)
    new_lines = new_blk.splitlines(keepends=True)
    if len(old_lines) < 3 and len(new_lines) < 3:
        return [], [(old_blk, new_blk)], []

    p = 0
    lim = min(len(old_lines), len(new_lines))
    while p < lim and old_lines[p] == new_lines[p]:
        p += 1
    s = 0
    rem_lim = min(len(old_lines) - p, len(new_lines) - p)
    while s < rem_lim and old_lines[len(old_lines) - 1 - s] == new_lines[len(new_lines) - 1 - s]:
        s += 1

    common_chars = sum(len(x) for x in old_lines[:p]) + sum(len(x) for x in old_lines[len(old_lines) - s :])
    if common_chars < 0.50 * max(1, len(old_blk)):
        return [], [(old_blk, new_blk)], []

    o_mid = "".join(old_lines[p : len(old_lines) - s]).strip()
    n_mid = "".join(new_lines[p : len(new_lines) - s]).strip()
    added: list[str] = []
    modified: list[tuple[str, str]] = []
    removed: list[str] = []
    if o_mid and n_mid:
        modified.append((o_mid, n_mid))
    elif n_mid:
        added.append(n_mid)
    elif o_mid:
        removed.append(o_mid)
    return added, modified, removed


def diff_semantic_blocks(frozen_blocks: list[str], curr_blocks: list[str]) -> BlockDiff:
    """Compute semantic block-level diff between frozen baseline and current system blocks."""
    # Common prefix blocks
    p = 0
    lim = min(len(frozen_blocks), len(curr_blocks))
    while p < lim and frozen_blocks[p] == curr_blocks[p]:
        p += 1
    static_prefix_chars = sum(len(b) for b in frozen_blocks[:p])

    # Common suffix blocks
    s = 0
    rem_lim = min(len(frozen_blocks) - p, len(curr_blocks) - p)
    while s < rem_lim and frozen_blocks[len(frozen_blocks) - 1 - s] == curr_blocks[len(curr_blocks) - 1 - s]:
        s += 1

    f_mid = frozen_blocks[p : len(frozen_blocks) - s]
    c_mid = curr_blocks[p : len(curr_blocks) - s]

    common_total = static_prefix_chars + sum(len(b) for b in frozen_blocks[len(frozen_blocks) - s :])
    added: list[str] = []
    modified: list[tuple[str, str]] = []
    removed: list[str] = []

    sm = difflib.SequenceMatcher(None, f_mid, c_mid, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            common_total += sum(len(b) for b in f_mid[i1:i2])
        elif tag == "insert":
            blk = "".join(c_mid[j1:j2]).strip()
            if blk:
                added.append(blk)
        elif tag == "delete":
            blk = "".join(f_mid[i1:i2]).strip()
            if blk:
                removed.append(blk)
        elif tag == "replace":
            old_blk = "".join(f_mid[i1:i2]).strip()
            new_blk = "".join(c_mid[j1:j2]).strip()
            if old_blk and new_blk:
                sub_a, sub_m, sub_r = _refine_large_paragraph_replace(old_blk, new_blk)
                # Credit refined leading equal lines toward static_prefix_chars when at prefix boundary
                if i1 == 0 and j1 == 0 and (len(sub_a) + len(sub_m) + len(sub_r) > 0):
                    old_lines = old_blk.splitlines(keepends=True)
                    new_lines = new_blk.splitlines(keepends=True)
                    lp = 0
                    while lp < min(len(old_lines), len(new_lines)) and old_lines[lp] == new_lines[lp]:
                        lp += 1
                    static_prefix_chars += sum(len(x) for x in old_lines[:lp])
                    common_total += sum(len(x) for x in old_lines[:lp])
                added.extend(sub_a)
                modified.extend(sub_m)
                removed.extend(sub_r)
            elif new_blk:
                added.append(new_blk)
            elif old_blk:
                removed.append(old_blk)

    return BlockDiff(
        static_prefix_chars=static_prefix_chars,
        common_total_chars=common_total,
        added=added,
        modified=modified,
        removed=removed,
    )


def _summarize_block_label(block: str) -> str:
    """Extract a concise label (XML tag or first non-empty line) for removed-block notices."""
    stripped = block.strip()
    m = _OPEN_TAG_RE.match(stripped)
    if m:
        return f"<{m.group(1)}> section"
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    if len(first_line) > 120:
        first_line = first_line[:117] + "..."
    return first_line or "(section)"


def format_trailing_update(diff: BlockDiff) -> str:
    """Format a clear, authoritative trailing override block for the model."""
    parts: list[str] = [
        "",
        "",
        PATCH_OPEN_TAG,
        "[Authoritative latest system context update: the sections below reflect the current state "
        "and supersede any earlier conflicting version in the system prompt or earlier updates.]",
    ]
    for old_blk, new_blk in diff.modified:
        label = _summarize_block_label(old_blk)
        parts.append(f"[Updated {label} — use this latest version instead of any earlier one]:\n{new_blk}")
    for new_blk in diff.added:
        parts.append(f"[Added system context — active now]:\n{new_blk}")
    for old_blk in diff.removed:
        label = _summarize_block_label(old_blk)
        parts.append(
            f"[Removed system context — the following earlier section ({label}) is no longer active "
            f"and must be ignored]:\n{old_blk[:240]}"
        )
    parts.append(PATCH_CLOSE_TAG)
    return "\n".join(parts)


@dataclass
class _SessionState:
    frozen_sys: str
    frozen_blocks: list[str]
    last_raw_sys: str
    active_patch: str = ""
    anchor: tuple[int, str, str] | None = None  # latest (msg_idx, orig_msg_digest, patch_text)
    anchor_steady_turns: int = 0
    committed_anchors: list[tuple[int, str, str]] = field(default_factory=list)
    patches_applied: int = 0


@dataclass
class PatchDecision:
    action: str  # disabled | skip_short | skip_unsupported | init | passthrough | reverted_to_frozen | patched_new | patched_steady_replay | guardrail_refresh
    delta_chars: int = 0
    static_prefix_ratio: float = 1.0
    anchor_index: int | None = None
    reason: str = ""


class SystemPrefixPatcher:
    """Thread-safe bounded-LRU semantic block-level system prefix patcher."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_sessions: int = 2048,
        min_system_chars: int = 512,
        min_static_prefix_ratio: float = 0.65,
        max_delta_chars: int = 8192,
        max_delta_ratio: float = 0.30,
    ):
        self.enabled = bool(enabled)
        self.max_sessions = int(max_sessions)
        self.min_system_chars = int(min_system_chars)
        self.min_static_prefix_ratio = float(min_static_prefix_ratio)
        self.max_delta_chars = int(max_delta_chars)
        self.max_delta_ratio = float(max_delta_ratio)

        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()
        self._stats: dict[str, int] = {
            "requests_seen": 0,
            "sessions_initialized": 0,
            "passthrough_unchanged": 0,
            "reverted_to_frozen": 0,
            "patched_new_delta": 0,
            "patched_steady_replay": 0,
            "guardrail_refreshes": 0,
            "skipped_other": 0,
            "frozen_prefix_chars_preserved": 0,
            "delta_chars_relocated": 0,
        }

    def stats_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "tracked_sessions": len(self._sessions),
                **self._stats,
            }

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()
            for k in self._stats:
                self._stats[k] = 0

    def _put_session(self, session_id: str, state: _SessionState) -> None:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
        elif len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)
        self._sessions[session_id] = state

    def apply(self, session_id: str, body: dict[str, Any]) -> tuple[dict[str, Any], PatchDecision]:
        """Inspect and optionally patch ``body['messages']`` for ``session_id``.

        Never mutates caller's ``body`` or nested message dicts in place; returns
        ``(effective_body, decision)``.
        """
        with self._lock:
            self._stats["requests_seen"] += 1
            if not self.enabled:
                return body, PatchDecision("disabled")

            messages = body.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                self._stats["skipped_other"] += 1
                return body, PatchDecision("skip_unsupported", reason="too_few_messages")

            m0 = messages[0]
            if not isinstance(m0, dict) or str(m0.get("role") or "").lower() not in {"system", "developer"}:
                self._stats["skipped_other"] += 1
                return body, PatchDecision("skip_unsupported", reason="no_leading_system_msg")

            curr_sys, sys_fmt = _extract_system_text(m0)
            if curr_sys is None:
                self._stats["skipped_other"] += 1
                return body, PatchDecision("skip_unsupported", reason="non_text_system")

            if len(curr_sys) < self.min_system_chars:
                self._stats["skipped_other"] += 1
                return body, PatchDecision("skip_short", reason="below_min_system_chars")

            # Do not rewrite if the last message cannot host a trailing update (e.g. assistant prefill)
            last_msg = messages[-1]
            if not isinstance(last_msg, dict) or str(last_msg.get("role") or "").lower() not in {"user", "tool", "function"}:
                self._stats["skipped_other"] += 1
                return body, PatchDecision("skip_unsupported", reason="tail_role_not_patchable")

            state = self._sessions.get(session_id)
            if state is None:
                blocks = split_semantic_blocks(curr_sys)
                state = _SessionState(frozen_sys=curr_sys, frozen_blocks=blocks, last_raw_sys=curr_sys)
                self._put_session(session_id, state)
                self._stats["sessions_initialized"] += 1
                return body, PatchDecision("init")

            self._sessions.move_to_end(session_id)

            # Helper to replay already-settled historical anchors so KV cache stays intact
            def _replay_committed(msgs_list: list[dict[str, Any]]) -> None:
                valid: list[tuple[int, str, str]] = []
                for c_idx, c_dig, c_patch in state.committed_anchors:
                    if 0 < c_idx < len(msgs_list) and isinstance(msgs_list[c_idx], dict):
                        if _message_text_digest(msgs_list[c_idx]) == c_dig:
                            pm = _append_suffix_to_message(msgs_list[c_idx], c_patch)
                            if pm is not None:
                                msgs_list[c_idx] = pm
                                valid.append((c_idx, c_dig, c_patch))
                state.committed_anchors = valid

            # Case 0: Unchanged or reverted back to the exact frozen baseline
            if curr_sys == state.frozen_sys:
                was_patched = bool(state.active_patch)
                state.last_raw_sys = curr_sys
                state.active_patch = ""
                if was_patched:
                    # If an earlier patch was settled deep in history, keep committed history intact
                    # and emit a concise revert notice at the tail; if it was only 1 turn ago (or none),
                    # clean passthrough with zero patch!
                    if state.anchor is not None and state.anchor_steady_turns >= 2 and state.anchor[0] < len(messages) - 4:
                        state.committed_anchors.append(state.anchor)
                    state.anchor = None
                    state.anchor_steady_turns = 0
                    if state.committed_anchors:
                        new_msgs = list(messages)
                        _replay_committed(new_msgs)
                        revert_patch = (
                            f"\n\n{PATCH_OPEN_TAG}\n"
                            "[Authoritative latest system context update: all temporary system overrides above "
                            "have been cleared; follow the original system prompt as written.]\n"
                            f"{PATCH_CLOSE_TAG}"
                        )
                        tail_idx = len(new_msgs) - 1
                        orig_dig = _message_text_digest(new_msgs[tail_idx])
                        pm_tail = _append_suffix_to_message(new_msgs[tail_idx], revert_patch)
                        if pm_tail is not None:
                            new_msgs[tail_idx] = pm_tail
                            state.committed_anchors.append((tail_idx, orig_dig, revert_patch))
                            new_body = dict(body)
                            new_body["messages"] = new_msgs
                            self._stats["reverted_to_frozen"] += 1
                            return new_body, PatchDecision("reverted_to_frozen", anchor_index=tail_idx)
                    self._stats["reverted_to_frozen"] += 1
                    return body, PatchDecision("reverted_to_frozen")
                if state.committed_anchors:
                    new_msgs = list(messages)
                    _replay_committed(new_msgs)
                    if state.committed_anchors:
                        new_body = dict(body)
                        new_body["messages"] = new_msgs
                        self._stats["passthrough_unchanged"] += 1
                        return new_body, PatchDecision("passthrough")
                self._stats["passthrough_unchanged"] += 1
                return body, PatchDecision("passthrough")

            # Case 1: Steady state (system prompt matches previous mutated turn -> replay at original anchor)
            if curr_sys == state.last_raw_sys and state.active_patch:
                new_msgs = list(messages)
                new_msgs[0] = _set_system_text(m0, state.frozen_sys, sys_fmt)
                _replay_committed(new_msgs)
                replayed_idx: int | None = None

                if state.anchor is not None:
                    a_idx, a_dig, a_patch = state.anchor
                    if 0 < a_idx < len(new_msgs) and isinstance(new_msgs[a_idx], dict):
                        if _message_text_digest(new_msgs[a_idx]) == a_dig:
                            patched_msg = _append_suffix_to_message(new_msgs[a_idx], a_patch)
                            if patched_msg is not None:
                                new_msgs[a_idx] = patched_msg
                                replayed_idx = a_idx
                                state.anchor_steady_turns += 1

                if replayed_idx is None:
                    # Anchor disappeared (e.g. client compacted history) -> re-anchor at current tail
                    tail_idx = len(new_msgs) - 1
                    orig_dig = _message_text_digest(new_msgs[tail_idx])
                    patched_tail = _append_suffix_to_message(new_msgs[tail_idx], state.active_patch)
                    if patched_tail is None:
                        self._stats["skipped_other"] += 1
                        return body, PatchDecision("skip_unsupported", reason="tail_append_failed")
                    new_msgs[tail_idx] = patched_tail
                    state.anchor = (tail_idx, orig_dig, state.active_patch)
                    state.anchor_steady_turns = 0
                    replayed_idx = tail_idx

                new_body = dict(body)
                new_body["messages"] = new_msgs
                self._stats["patched_steady_replay"] += 1
                self._stats["frozen_prefix_chars_preserved"] += len(state.frozen_sys)
                return new_body, PatchDecision(
                    "patched_steady_replay",
                    delta_chars=len(state.active_patch),
                    static_prefix_ratio=1.0,
                    anchor_index=replayed_idx,
                )

            # Case 2: New mutation relative to frozen baseline
            curr_blocks = split_semantic_blocks(curr_sys)
            diff = diff_semantic_blocks(state.frozen_blocks, curr_blocks)
            frozen_len = max(1, len(state.frozen_sys))
            static_ratio = diff.static_prefix_chars / frozen_len
            delta_ratio = diff.delta_chars / frozen_len

            # Safety guardrails: refresh baseline if core prefix changed, delta too large, or too many committed layers
            guardrail_fail = ""
            if not diff.has_changes:
                guardrail_fail = "no_block_diff"
            elif static_ratio < self.min_static_prefix_ratio:
                guardrail_fail = f"static_prefix_ratio={static_ratio:.2f}<{self.min_static_prefix_ratio:.2f}"
            elif diff.delta_chars > self.max_delta_chars:
                guardrail_fail = f"delta_chars={diff.delta_chars}>{self.max_delta_chars}"
            elif delta_ratio > self.max_delta_ratio:
                guardrail_fail = f"delta_ratio={delta_ratio:.2f}>{self.max_delta_ratio:.2f}"
            elif len(state.committed_anchors) >= 6:
                guardrail_fail = "max_committed_anchors_reached"

            if guardrail_fail:
                state.frozen_sys = curr_sys
                state.frozen_blocks = curr_blocks
                state.last_raw_sys = curr_sys
                state.active_patch = ""
                state.anchor = None
                state.anchor_steady_turns = 0
                state.committed_anchors.clear()
                self._stats["guardrail_refreshes"] += 1
                return body, PatchDecision(
                    "guardrail_refresh",
                    delta_chars=diff.delta_chars,
                    static_prefix_ratio=round(static_ratio, 4),
                    reason=guardrail_fail,
                )

            # Promote previous anchor into committed history ONLY if it had already settled for >=2
            # steady turns deep in history (so we never invalidate 100K-200K of downstream KV!).
            # Rapid per-turn counters (steady_turns < 2) are cleanly replaced with zero buildup.
            if state.anchor is not None and state.anchor_steady_turns >= 2 and state.anchor[0] < len(messages) - 4:
                if state.anchor not in state.committed_anchors:
                    state.committed_anchors.append(state.anchor)

            patch_text = format_trailing_update(diff)
            new_msgs = list(messages)
            new_msgs[0] = _set_system_text(m0, state.frozen_sys, sys_fmt)
            _replay_committed(new_msgs)
            tail_idx = len(new_msgs) - 1
            orig_dig = _message_text_digest(new_msgs[tail_idx])
            patched_tail = _append_suffix_to_message(new_msgs[tail_idx], patch_text)
            if patched_tail is None:
                self._stats["skipped_other"] += 1
                return body, PatchDecision("skip_unsupported", reason="tail_append_failed")
            new_msgs[tail_idx] = patched_tail

            state.last_raw_sys = curr_sys
            state.active_patch = patch_text
            state.anchor = (tail_idx, orig_dig, patch_text)
            state.anchor_steady_turns = 0
            state.patches_applied += 1

            new_body = dict(body)
            new_body["messages"] = new_msgs
            self._stats["patched_new_delta"] += 1
            self._stats["frozen_prefix_chars_preserved"] += len(state.frozen_sys)
            self._stats["delta_chars_relocated"] += diff.delta_chars
            return new_body, PatchDecision(
                "patched_new_delta",
                delta_chars=diff.delta_chars,
                static_prefix_ratio=round(static_ratio, 4),
                anchor_index=tail_idx,
            )
