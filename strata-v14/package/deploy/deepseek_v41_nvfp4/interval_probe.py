"""Side-effect-free tier probes, safe token-id concatenation, and prefix fences.

A host endpoint is a COMPLETE native tree edge with all component validators
accepting it. Partial host edges are not promoted into fictitious checkpoints.
Unknown component families are conservative misses. Token counts are logical,
not bytes. The native allocator remains authoritative.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from types import SimpleNamespace
from array import array


def concat_token_ids(origin, output):
    """Concatenate request input and output IDs while preserving native array('q').

    Native SGLang stores `origin_input_ids` as `array('q')` while `output_ids` may
    be `list[int]` or `array('q')`. Direct `origin + output` raises `TypeError` on
    mixed `array` + `list`, and `list(origin) + list(output)` breaks native
    `RadixKey` array-type invariants.
    """
    if isinstance(origin, array):
        out = array(origin.typecode, origin)
        if output:
            out.extend(output)
        return out
    if isinstance(output, array):
        out = array(output.typecode, origin)
        out.extend(output)
        return out
    return list(origin) + list(output)


@dataclass(frozen=True)
class TierProbe:
    gpu_ready_len: int = 0
    host_recoverable_len: int = 0
    prefetch_pending_len: int = 0
    true_missing_len: int = 0
    host_node_id: int | None = None
    swa_reprefill_tail_tokens: int = 0
    kind: str = 'unknown'


def namespace(key):
    return (getattr(key, 'extra_key', None), getattr(key, 'cache_salt', None))


def raw_ids(key):
    fn = getattr(key, 'raw_token_ids', None)
    return fn() if callable(fn) else key.token_ids[:len(key) + int(getattr(key, 'is_bigram', False))]


@dataclass
class PrefixFence:
    rid: str
    node_id: int
    tokens: array
    namespace: tuple
    safe_len: int
    total_len: int

    def intersects(self, key):
        if namespace(key) != self.namespace:
            return False
        ids = raw_ids(key)
        # At least one token past safe_len overlaps an unavailable edge.
        n = min(len(ids), len(self.tokens))
        if n <= self.safe_len:
            return False
        matched = 0
        while matched < n and ids[matched] == self.tokens[matched]:
            matched += 1
        return matched > self.safe_len


class Fences:
    def __init__(self):
        self.items = {}

    def limit(self, key):
        matches = [f.safe_len for f in self.items.values() if f.intersects(key)]
        return min(matches) if matches else None

    def blocked(self, key):
        return self.limit(key) is not None

    def install(self, cache):
        core = cache.tree_core
        original = core.match_prefix

        def guarded(params):
            limit = self.limit(params.key)
            if limit is not None:
                params = replace(params, key=params.key[:limit])
            return original(params)

        core.match_prefix = guarded
        # Other FULL-only read paths must not expose pending page IDs either.
        if callable(getattr(core, 'match_full_device_prefix', None)):
            full = core.match_full_device_prefix

            def guarded_full(key):
                limit = self.limit(key)
                return full(key if limit is None else key[:limit])

            core.match_full_device_prefix = guarded_full


def probe_tiered_prefix(cache, key, *, input_len=None, fences=None):
    total = len(key) + 1 if input_len is None else input_len
    fail = lambda reason: TierProbe(true_missing_len=total, kind=reason)
    if getattr(cache, 'disable', False):
        return fail('disabled')
    if getattr(cache, '_tree_core_backend', None) != 'python':
        return fail('unsupported_backend')
    core = cache.tree_core
    components = tuple(core.components)
    if not components or any(type(c).__name__ not in {'FullComponent', 'SWAComponent'} for c in components):
        return fail('unsupported_components')
    names = {getattr(c.component_type, 'name', '') for c in components}
    if 'FULL' not in names or not names <= {'FULL', 'SWA'} or getattr(cache, 'linker', None) is not None:
        return fail('unsupported_layout')
    if getattr(core, 'is_eagle', False):
        return fail('bigram_not_qualified')
    key = key.page_aligned(core.page_size)
    devs = [c.create_match_validator(match_device_only=True) for c in components]
    hosts = [c.create_match_validator(match_device_only=False) for c in components]
    full = next(c.component_type for c in components if c.component_type.name == 'FULL')
    node = core.root_node
    offset = 0
    gpu = host = 0
    host_node = None
    device_contiguous = True
    host_acked = True
    while offset < len(key):
        child = node.children.get(key.child_key_at(offset, core.page_size))
        if child is None:
            break
        cd = child.component_data[full]
        if cd.value is None and not child.backuped:
            break
        length = child.key.match_at(key, offset, page_size=core.page_size)
        if length <= 0:
            break
        partial = length < len(child.key)
        view = (
            SimpleNamespace(
                key=child.key[:length],
                component_data=child.component_data,
                backuped=child.backuped,
                evicted=child.evicted,
            )
            if partial
            else child
        )
        device_contiguous = device_contiguous and cd.value is not None
        dv = all([v(view) for v in devs])
        hv = all([v(view) for v in hosts])
        offset += length
        if device_contiguous and dv:
            gpu = offset
        host_acked = (
            host_acked
            and getattr(child, 'write_through_pending_id', None) is None
            and not getattr(child, 'write_back_queued', False)
        )
        if hv and not partial and host_acked:
            # Native valid endpoint may mix GPU ancestors, SWA tombstones, and host suffix.
            host = offset
            host_node = child.id
        if partial:
            break
        node = child
    limit = fences.limit(key) if fences else None
    pending = 0
    if limit is not None:
        pending = max(1, len(key) - limit)
        gpu = min(gpu, limit)
    # Original demand-load still re-matches and rebuilds request-private state.
    tail = cache.swa_reprefill_tail_tokens() if callable(getattr(cache, 'swa_reprefill_tail_tokens', None)) else 0
    tail = max(0, int(tail))
    usable = min(max(gpu, host), max(0, total - 1))
    missing = max(total - usable, min(total, tail))
    return TierProbe(gpu, host, pending, missing, host_node, tail, 'full_swa_native_validators')
