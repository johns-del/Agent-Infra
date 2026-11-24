"""Controlled trajectory-aligned output forcing for private replay benchmarking.

When a replay request carries `sampling_params.custom_params['orvix_unified']['forced_output_ids']`,
the engine executes every real prefill and decode forward pass up to `max_new_tokens`,
while pinning emitted `output_ids` step-by-step to the recorded trajectory tokens
before `UnifiedRadixCache.checkpoint_into_tree` commits the KV node key.
This guarantees 100% cross-turn RadixTree prefix connection on recorded multi-turn
traces without skipping GPU compute or fabricating zero-cost decode steps.
"""
from __future__ import annotations

from collections import Counter


KEY = 'orvix_unified'


def _extract_forced_ids(req) -> list[int] | None:
    sampling = getattr(req, 'sampling_params', None)
    custom = getattr(sampling, 'custom_params', None)
    meta = custom.get(KEY) if isinstance(custom, dict) else None
    if not isinstance(meta, dict):
        return None
    forced = meta.get('forced_output_ids')
    if not isinstance(forced, list) or not forced:
        return None
    return forced


def _overwrite_next_token_entry(container, index: int, value: int) -> bool:
    if container is None:
        return False
    try:
        container[index] = int(value)
        return True
    except (TypeError, IndexError, KeyError, RuntimeError):
        return False


class ForcedTraceAligner:
    """Step-level output_ids aligner hooked onto SchedulerBatchResultProcessor."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.counts = Counter()
        self.installed = False

    def _align_before_result_processor(self, batch, result) -> dict[int, tuple[int, list[int]]]:
        """Overwrite sampled next_token_ids at current cursor and return post-check targets."""
        tracked = {}
        reqs = getattr(batch, 'reqs', None) or ()
        next_ids = getattr(result, 'next_token_ids', None)
        accept_lens = getattr(result, 'accept_lens_cpu', getattr(result, 'accept_lens', None))
        stride = None
        if next_ids is not None and reqs:
            try:
                total_next = len(next_ids)
                if total_next > len(reqs) and total_next % len(reqs) == 0:
                    stride = total_next // len(reqs)
            except TypeError:
                stride = None

        for i, req in enumerate(reqs):
            forced = _extract_forced_ids(req)
            if forced is None:
                continue
            cursor = len(getattr(req, 'output_ids', ()))
            tracked[id(req)] = (cursor, forced)
            if cursor >= len(forced):
                continue
            if stride is not None and stride > 1:
                # Speculative / DSpark flat [bs * draft_tokens] output buffer
                n_acc = stride
                if accept_lens is not None:
                    try:
                        n_acc = max(1, min(stride, int(accept_lens[i])))
                    except (TypeError, ValueError, IndexError):
                        n_acc = stride
                for step_offset in range(n_acc):
                    pos = cursor + step_offset
                    if pos < len(forced):
                        _overwrite_next_token_entry(next_ids, i * stride + step_offset, forced[pos])
            elif next_ids is not None:
                # 1D [bs] or 2D [bs][step] next_token_ids
                try:
                    entry = next_ids[i]
                except (TypeError, IndexError, KeyError):
                    entry = None
                if isinstance(entry, list):
                    for step_offset in range(len(entry)):
                        pos = cursor + step_offset
                        if pos < len(forced):
                            entry[step_offset] = int(forced[pos])
                else:
                    _overwrite_next_token_entry(next_ids, i, forced[cursor])
        return tracked

    def _reconcile_emitted_outputs(self, batch, tracked: dict[int, tuple[int, list[int]]]) -> None:
        if not tracked:
            return
        aligned_any = False
        for req in getattr(batch, 'reqs', None) or ():
            info = tracked.get(id(req))
            if info is None:
                continue
            start_cursor, forced = info
            out = getattr(req, 'output_ids', None)
            if out is None:
                continue
            end_cursor = min(len(out), len(forced))
            if end_cursor <= start_cursor:
                continue
            aligned_any = True
            for pos in range(start_cursor, end_cursor):
                target_tok = int(forced[pos])
                if out[pos] != target_tok:
                    out[pos] = target_tok
                    self.counts['post_reconciled_tokens'] += 1
                self.counts['forced_tokens'] += 1
            if getattr(req, 'finished', lambda: False)():
                self.counts['forced_requests_completed'] += 1
        if aligned_any:
            self.counts['forced_batches'] += 1

    def install(self):
        proc = getattr(self.scheduler, 'batch_result_processor', None)
        if proc is None or self.installed:
            return self
        for method_name in ('process_batch_result_prefill', 'process_batch_result_decode'):
            orig = getattr(proc, method_name, None)
            if not callable(orig):
                continue

            def make_wrapper(fn):
                def wrapped(batch, result, *args, **kwargs):
                    tracked = self._align_before_result_processor(batch, result)
                    ret = fn(batch, result, *args, **kwargs)
                    self._reconcile_emitted_outputs(batch, tracked)
                    return ret

                return wrapped

            setattr(proc, method_name, make_wrapper(orig))
        # Also ensure completed_checkpoint sees aligned output_ids even if a test
        # invokes checkpoint_into_tree directly without batch_result_processor.
        cache = getattr(self.scheduler, 'tree_cache', None)
        orig_ckpt = getattr(cache, 'checkpoint_into_tree', None)
        if callable(orig_ckpt):
            def aligned_ckpt(req, *args, **kwargs):
                forced = _extract_forced_ids(req)
                out = getattr(req, 'output_ids', None)
                if forced is not None and out is not None:
                    limit = min(len(out), len(forced))
                    for idx in range(limit):
                        if out[idx] != int(forced[idx]):
                            out[idx] = int(forced[idx])
                            self.counts['ckpt_reconciled_tokens'] += 1
                return orig_ckpt(req, *args, **kwargs)

            cache.checkpoint_into_tree = aligned_ckpt
        self.installed = True
        return self

    def snapshot(self) -> dict:
        return {
            'installed': self.installed,
            'counts': dict(self.counts),
        }


def install_forced_trace(scheduler) -> ForcedTraceAligner:
    aligner = ForcedTraceAligner(scheduler).install()
    scheduler._orvix_forced_trace = aligner
    return aligner
