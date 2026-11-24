"""Optional per-cycle native decisions. No prefix lookups or GPU synchronization."""
from __future__ import annotations
from contextvars import ContextVar
from functools import wraps
import math
import os
import time

_current = ContextVar('orvix_admission_cycle', default=None)
_links = {}


def enabled():
    return os.environ.get('ORVIX_TRACE_ADMISSION', '0') == '1'


def link(rid, identity):
    if enabled():
        _links[str(rid)] = identity
        if len(_links) > 4096: _links.pop(next(iter(_links)))


def dispatch(obj):
    if enabled():
        identity = _links.pop(str(obj.rid), None)
        if identity is not None: obj.agent_trace_identity = {k:str(v) for k,v in identity.items() if v is not None}


def install(observer):
    """Track native group reservations without inspecting/consuming its iterator."""
    if not enabled(): return
    allocator = getattr(observer.s.req_to_token_pool, 'mamba_allocator', None)
    if allocator is None or getattr(allocator, '_orvix_admission_installed', False): return
    allocator._orvix_admission_installed = True
    allocator._orvix_group_reserved = 0
    for name in ('alloc_group_begin', 'alloc_group_end', 'alloc'):
        original = getattr(allocator, name, None)
        if not callable(original): continue
        def wrap(original=original, name=name):
            @wraps(original)
            def wrapped(*args, **kwargs):
                before = number(allocator, 'available_size')
                result = original(*args, **kwargs)
                after = number(allocator, 'available_size')
                if name == 'alloc_group_begin' and before is not None and after is not None:
                    allocator._orvix_group_reserved = max(0, before-after)
                elif name == 'alloc_group_end': allocator._orvix_group_reserved = 0
                elif result is not None and before == after:
                    count = args[0] if args else kwargs.get('need_size', 1)
                    allocator._orvix_group_reserved = max(0, allocator._orvix_group_reserved-count)
                return result
            return wrapped
        setattr(allocator, name, wrap())


def number(obj, name):
    value = getattr(obj, name, None)
    try:
        value = value() if callable(value) else value
        return value if isinstance(value, (int, float)) else None
    except Exception: return None


def pool(unit, needed=None, free=None, reserved=None, evictable=None, **kw):
    return dict(unit=unit, needed=needed, free=free, reserved=reserved, evictable=evictable, **kw)


def snapshot(s, req, adder=None, values=None):
    v = values or {}; alloc = s.token_to_kv_pool_allocator; cache = s.tree_cache
    page = int(getattr(s, 'page_size', 64)); kv = getattr(req, 'kv', req)
    needed = v.get('total_tokens', v.get('paged_input'))
    if needed is None and 'cand_extend_input_len' in v:
        needed = v['cand_extend_input_len'] + v.get('max_new', 0) + page
    free = number(alloc, 'full_available_size')
    if free is None: free = number(alloc, 'available_size')
    evict = number(cache, 'full_evictable_size')
    if evict is None: evict = number(cache, 'evictable_size')
    beam = getattr(req, 'beam_group', None)
    pools = {'request_rows': pool('request_row', getattr(beam, 'beam_width', 1) if beam else 1,
                number(s.req_to_token_pool, 'available_size'), len(getattr(adder, 'can_run_list', [])), 0),
             'full_tokens': pool('token', needed, free, number(adder, 'rem_total_token_offset'), evict)}
    pools['full_pages'] = pool('page', math.ceil(needed/page) if needed is not None else None,
        free//page if free is not None else None,
        math.ceil(pools['full_tokens']['reserved']/page) if pools['full_tokens']['reserved'] is not None else None,
        evict//page if evict is not None else None,
        page_size_tokens=page, alias_of='full_tokens_not_an_independent_pool')
    if needed is None:
        for key in ('full_tokens','full_pages'):
            pools[key]['needed_missing_reason']='native_guard_did_not_evaluate_token_budget'
    running=getattr(getattr(s,'running_batch',None),'reqs',[])
    limit=number(s,'max_running_requests')
    pools['running_window']=pool('request',1,max(0,limit-len(running)) if limit is not None else None,
        len(getattr(adder,'can_run_list',[])),0)
    mamba = getattr(s.req_to_token_pool, 'mamba_allocator', None)
    if mamba is not None:
        pools['mamba_slots'] = pool('state_slot', int(getattr(kv, 'mamba_pool_idx', None) is None),
            number(mamba, 'schedulable_available_size'), number(mamba, '_orvix_group_reserved'), number(cache, 'mamba_evictable_size'),
            candidate_already_holds_slot=getattr(kv, 'mamba_pool_idx', None) is not None)
    if getattr(adder, 'is_hybrid_swa', False):
        pools['swa'] = pool('request_ring_capacity' if getattr(adder, '_swa_req_ring', False) else 'token',
            v.get('swa_needed'), number(alloc, 'swa_available_size'), number(adder, 'rem_swa_token_offset'),
            number(cache, 'swa_evictable_size'))
    return pools


def timed_call(state, fn):
    start = time.perf_counter_ns()
    try: return fn()
    except Exception: state['errors'] += 1
    finally: state['hook_ns'] += time.perf_counter_ns() - start


def consider(s, req, adder, running_batch):
    state = _current.get()
    if state is None: return
    def capture():
        state['last'] = str(req.rid)
        state['rows'][str(req.rid)] = dict(req=req, considered=True, admitted=False, failed_guard=None,
            pools=snapshot(s, req, adder), guard_expression=None)
    timed_call(state, capture)


def native_guard(adder, req, guard, expression, values):
    state = _current.get()
    if state is None: return
    def capture():
        row = state['rows'].setdefault(str(req.rid), dict(req=req, considered=True, admitted=False))
        row.update(failed_guard=guard, guard_expression=expression, pools=snapshot(state['s'], req, adder, values))
    timed_call(state, capture)


def scheduler_guard(s, guard, expression, values):
    state = _current.get()
    if state is None: return
    def capture():
        # batch_is_full is also latched by a previous NO_TOKEN result. The
        # latch is a native gate, not proof that request rows are exhausted.
        if guard == 'slot_full' and 'batch_is_full' in expression:
            batch=values.get('running_batch')
            if batch is not None and s.get_num_allocatable_reqs(len(batch.reqs),running_batch=batch)>0:
                actual_guard='batch_full_latch'
            else:actual_guard=guard
        else:actual_guard=guard
        state['stop'] = actual_guard
        state['stop_expression'] = expression
        req = values.get('req')
        if req is not None and str(req.rid) in state['rows']:
            row = state['rows'][str(req.rid)]
            if not row.get('admitted') and not row.get('failed_guard'):
                row.update(failed_guard=actual_guard, guard_expression=expression)
    timed_call(state, capture)


def candidate(fn):
    @wraps(fn)
    def wrapped(adder, req, *args, **kwargs):
        state = _current.get()
        if state is None: return fn(adder, req, *args, **kwargs)
        result = fn(adder, req, *args, **kwargs)
        def capture():
            row = state['rows'][str(req.rid)]
            row['admitted'] = any(x is req for x in adder.can_run_list)
            if row['admitted']: row['failed_guard'] = None
            row['native_result'] = getattr(result, 'name', str(result))
        timed_call(state, capture)
        return result
    return wrapped


def cycle(fn):
    @wraps(fn)
    def wrapped(s, *args, **kwargs):
        observer = getattr(s, '_orvix_observer', None)
        if not enabled() or observer is None or not s.waiting_queue: return fn(s, *args, **kwargs)
        start = time.perf_counter_ns()
        seq = getattr(s, '_orvix_admission_seq', 0) + 1; s._orvix_admission_seq = seq
        state = dict(s=s, rows={}, initial=list(s.waiting_queue), last=None, stop=None,
                     hook_ns=0, errors=0, cycle_id=seq)
        token = _current.set(state); state['hook_ns'] += time.perf_counter_ns() - start
        try: return fn(s, *args, **kwargs)
        finally:
            def finish():
                for req in state['initial']:
                    rid = str(req.rid); row = state['rows'].get(rid)
                    if row is None:
                        row = dict(req=req, admitted=False, considered=False,
                            failed_guard=('break_truncated' if state['last'] else state['stop'] or 'sorting_not_reached'),
                            pools=snapshot(s, req), guard_expression=state.get('stop_expression') if not state['last'] else None,
                            global_guard=not bool(state['last']) and bool(state['stop']))
                    identity = getattr(observer, 'admission_identities', {}).get(rid, {})
                    observer.emit('engine.admission.candidate', dict(
                        admission_schema_version=1, schema_version=1, cycle_id=seq, rid=rid,
                        ts_mono_ns=time.monotonic_ns(), **{k:v for k,v in row.items() if k != 'req'}, **identity,
                        cutoff_after_rid=state['last'] if not row['considered'] else None,
                        cycle_hook_ns_before_emit=state['hook_ns'], units_cross_pool_additive=False), rid)
            timed_call(state, finish)
            try:
                observer.emit('engine.admission.cycle', dict(admission_schema_version=1, cycle_id=seq,
                    candidate_count=len(state['initial']), hook_ns=state['hook_ns'], callback_errors=state['errors'],
                    timing_scope='probe_collection_and_candidate_serialization_enqueue_excludes_summary_emit'))
            except Exception: pass
            _current.reset(token)
    return wrapped
