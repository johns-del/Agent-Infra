"""Thread-safe bounded-LRU online CausalPrefixStabilizer hook for OpenAI chat requests."""
from __future__ import annotations

from collections import OrderedDict
import threading

from .causal_prefix_stabilizer import canonicalize_against_spine

MAX_CHAT_SPINES = 1024
_LOCK = threading.Lock()
_SPINES: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
_LAST_RAW: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
_STATS = {
    "requests_seen": 0,
    "requests_stabilized": 0,
    "tokens_relocated": 0,
    "pure_append_short_circuits": 0,
}


def _is_pure_append(prev_raw: list[int] | None, cur_raw: list[int], tolerance: int = 64) -> bool:
    if not prev_raw or len(cur_raw) < len(prev_raw) - tolerance:
        return False
    n_min = min(len(prev_raw), len(cur_raw))
    lcp = 0
    while lcp < n_min and prev_raw[lcp] == cur_raw[lcp]:
        lcp += 1
    return lcp >= len(prev_raw) - tolerance


def stabilize_chat_processed_messages(request, processed_messages, *, tail_guard_tokens: int = 12) -> int:
    """Reorder volatile mid-prompt spans on multi-turn chat prompts without changing multiset/length.

    Opt-in via ``request.custom_params["orvix_unified"]`` (injected by ``ours_node_daemon`` or
    trusted caller). Returns relocated token count. Multimodal requests are left untouched.
    """
    custom = getattr(request, "custom_params", None)
    if not isinstance(custom, dict):
        return 0
    meta = custom.get("orvix_unified")
    if not isinstance(meta, dict):
        return 0
    if (
        getattr(processed_messages, "image_data", None)
        or getattr(processed_messages, "video_data", None)
        or getattr(processed_messages, "audio_data", None)
        or getattr(processed_messages, "modalities", None)
    ):
        return 0
    ids = getattr(processed_messages, "prompt_ids", None)
    if not isinstance(ids, list) or len(ids) < 512:
        return 0
    graph_id = str(meta.get("graph_id") or meta.get("session_id") or "")
    if not graph_id:
        return 0
    scope_id = str(meta.get("scope_id") or "s0")
    key = (graph_id, scope_id)

    with _LOCK:
        _STATS["requests_seen"] += 1
        prev_raw = _LAST_RAW.get(key)
        prior_spine = _SPINES.get(key)

        # If raw prompt is already a pure append over prev_raw, avoid re-relocating
        # a stabilized prefix (unless spine already matches prior_spine cleanly).
        if prev_raw is not None and _is_pure_append(prev_raw, ids) and prior_spine is not None and not _is_pure_append(prior_spine, ids):
            if key in _SPINES:
                _SPINES.move_to_end(key)
            if key in _LAST_RAW:
                _LAST_RAW.move_to_end(key)
            _SPINES[key] = list(ids)
            _LAST_RAW[key] = list(ids)
            _STATS["pure_append_short_circuits"] += 1
            return 0

        canon, next_spine, relocated = canonicalize_against_spine(
            ids,
            prior_spine,
            tail_guard_tokens=tail_guard_tokens,
        )
        if key in _SPINES:
            _SPINES.move_to_end(key)
        elif len(_SPINES) >= MAX_CHAT_SPINES:
            _SPINES.popitem(last=False)
        _SPINES[key] = next_spine

        if key in _LAST_RAW:
            _LAST_RAW.move_to_end(key)
        elif len(_LAST_RAW) >= MAX_CHAT_SPINES:
            _LAST_RAW.popitem(last=False)
        _LAST_RAW[key] = list(ids)

        if relocated > 0 and len(canon) == len(ids):
            processed_messages.prompt_ids = canon
            _STATS["requests_stabilized"] += 1
            _STATS["tokens_relocated"] += relocated
            return relocated
    return 0


def stats_snapshot() -> dict[str, int]:
    with _LOCK:
        return dict(_STATS, tracked_sessions=len(_SPINES))
