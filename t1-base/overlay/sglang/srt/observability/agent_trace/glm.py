"""GLM hybrid DSA/mamba index ledger. Byte geometry remains explicitly unknown.

The adaptive draft worker has private scratch pools; only scheduler-owned
(target model) pools are covered. No kernel, cache policy or MTP setting changes.
"""
from functools import wraps


def install(observer):
    from .hooks import observe_method, argument
    from .lifecycle import Lifecycle, IndexCopy, install_request_slot_observer
    layouts = observer.sampler.layouts
    if type(layouts.pool).__name__ != 'HybridLinearKVPool':
        raise ValueError('unsupported GLM pool')
    allocator = layouts.alloc
    if type(allocator).__name__ != 'PagedTokenToKVPoolAllocator':
        raise ValueError('unsupported GLM allocator')
    life = Lifecycle(observer, layouts)
    observer.lifecycle = life
    allocator._trace_lifecycle_callback = lambda action,pages,method,**kw: observer.w.safe(
        life.allocator, 'GPU:full', action, pages, method, **kw)
    observe_method(allocator, '_release_page_ids', observer.w, None,
        lambda a,k,r,e,c: life.capture('release','GPU:full',list(a),method='_release_page_ids') if not e else None)
    observe_method(allocator,'clear',observer.w,None,
        lambda a,k,r,e,c: life.capture('reset','GPU:full',[],method='clear') if not e else None)
    req_pool = observer.s.req_to_token_pool
    install_request_slot_observer(observer, life, req_pool)
    mamba = getattr(req_pool,'mamba_allocator',None)
    if mamba is not None:
        observe_method(mamba,'_do_alloc',observer.w,None,
            lambda a,k,r,e,c: life.capture('allocate','GPU:mamba',[r],method='MambaSlotAllocator.alloc') if r is not None and not e else None)
        observe_method(mamba,'free',observer.w,
            lambda a,k: IndexCopy([argument(a,k,'free_index')]),
            lambda a,k,r,e,c: life.capture('release','GPU:mamba',[],staged=c,method='MambaSlotAllocator.free') if not e else None)
        observe_method(mamba,'clear',observer.w,None,
            lambda a,k,r,e,c: life.capture('reset','GPU:mamba',[],method='MambaSlotAllocator.clear') if not e else None)
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    for name in ('prepare_for_extend','prepare_for_decode'):
        original = getattr(ScheduleBatch,name)
        @wraps(original)
        def prepare(batch,*a,_original=original,**k):
            life.contexts.append([str(r.rid) for r in batch.reqs])
            try:return _original(batch,*a,**k)
            finally:life.contexts.pop()
        setattr(ScheduleBatch,name,prepare)
    observer.emit('engine.trace.coverage', {'adapter':'glm_hybrid_target',
        'allocator_units':['full_pages','mamba_slots','request_slots'],
        'byte_geometry_complete':False, 'draft_worker_pools_observed':False,
        'missing':['DSA_compressed_index_byte_geometry','MTP_private_draft_scratch_and_rollback_storage'],
        'host_migration':'disabled_by_qualified_baseline'})
    return life


def native_unit_counts(layouts):
    """Native reusable capacity, excluding padding; units are never added as bytes."""
    allocator=layouts.alloc
    request_pool=layouts.s.req_to_token_pool
    mamba=getattr(request_pool,'mamba_allocator',None)
    page=int(allocator.page_size)
    rows={}
    for group,pool,divisor in (('GPU:full',allocator,page),
                              ('GPU:request_slot',request_pool,1),('GPU:mamba',mamba,1)):
        if pool is None:continue
        capacity=int(pool.size);available=int(pool.available_size())
        rows[group]={'capacity':capacity//divisor,'free':available//divisor,
            'occupied':(capacity-available)//divisor,
            'unit':'page' if divisor>1 else 'slot',
            'valid':0<=available<=capacity and capacity%divisor==0 and available%divisor==0}
    return rows
