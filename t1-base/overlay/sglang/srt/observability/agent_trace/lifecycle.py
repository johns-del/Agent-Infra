"""Ordered allocator and copy-descriptor ledger for the private DSV4 engine.

Only index metadata is staged. CUDA events are polled; no synchronize or KV
tensor read is added. Native call order, rather than asynchronous ready time,
defines page generations. A generation ends at allocator return, which means
reusable capacity, not a CUDA backing-storage deallocation.
"""
from collections import defaultdict, deque
from functools import wraps
import uuid

from .runtime import point


class IndexCopy:
    def __init__(self, values):
        import torch
        self.events, self.holds, self.values = [], [], []
        devices = set()
        for value in values:
            if value is None:
                self.values.append(None)
            elif not hasattr(value, 'device'):
                self.values.append(list(value))
            elif value.device.type == 'cpu':
                self.values.append(value.detach().clone())
            elif value.device.type == 'cuda':
                host = torch.empty_like(value, device='cpu', pin_memory=True)
                host.copy_(value.detach(), non_blocking=True)
                self.values.append(host)
                self.holds.append(value)
                devices.add(value.device)
            else:
                raise ValueError('unsupported index device')
        for device in devices:
            with torch.cuda.device(device):
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                self.events.append(event)

    def ready(self):
        return all(e.query() for e in self.events)

    def finish(self):
        return [v.tolist() if hasattr(v, 'tolist') else v for v in self.values]


class Lifecycle:
    def __init__(self, observer, layouts):
        self.o, self.layouts = observer, layouts
        self.sequence, self.pending = 0, deque()
        self.generations, self.live = {}, {}
        self.contexts, self.transfers, self.copies = [], [], []
        self.copy_details = {}
        self.unit_specs = {}
        self.gaps = 0
        for tier, component in layouts.targets:
            group = tier + ':' + component
            self.unit_specs[group] = [layouts.specs[(tier, n)] for n in layouts.targets[(tier, component)]]
        observer.emit('kv.lifecycle.begin', {
            'groups': self.unit_specs, 'baseline': 'observer_attached_before_first_request',
            'generation_domain': observer.w.source,
            'release_semantics': 'native_allocator_capacity_return_not_backing_storage_free',
            'copy_bytes_semantics': 'native_copy_descriptor_payload_not_PCIe_bus_measurement',
        })

    def context(self):
        return {'candidate_rids': list(self.contexts[-1]) if self.contexts else
                [self.o.request_stack[-1]] if self.o.request_stack and self.o.request_stack[-1] else [],
                'migration_id': self.transfers[-1] if self.transfers else None,
                'object_id': self.o.victim_stack[-1] if self.o.victim_stack else None,
                'operation_id': self.o.operation_stack[-1] if self.o.operation_stack else None}

    def capture(self, action, group, values, *, method=None, divisor=1,
                owner_counts=None, owners=None, staged=None, **extra):
        self.sequence += 1
        context = self.context()
        context.update(extra)
        context.update(ledger_seq=self.sequence, observed_at=point(), action=action,
                       group=group, method=method, divisor=divisor,
                       owner_counts=owner_counts, owners=owners)
        # A gap is explicit; never resume with a silently continuous ledger.
        if len(self.pending) >= 4096:
            self.gaps += 1
            self.o.emit('kv.lifecycle.gap', {'ledger_seq': self.sequence, 'reason': 'pending_index_budget'})
            return
        self.pending.append((context, staged if staged is not None else IndexCopy(values)))

    def allocator(self, group, action, pages, method, **lengths):
        if pages is not None and int(pages.numel())==0:return
        candidates = self.context()['candidate_rids']
        counts = None
        if method in ('alloc_extend', 'alloc_decode') and candidates:
            seq = lengths['seq_lens_cpu'].tolist()
            ps = lengths['page_size']
            if method == 'alloc_extend':
                prefix = lengths['prefix_lens_cpu'].tolist()
                counts = [(s + ps - 1)//ps - (p + ps - 1)//ps for s, p in zip(seq, prefix)]
            else:
                counts = [int(s % ps == 1) for s in seq]
            if len(counts) != len(candidates):
                counts = None
        self.capture(action, group, [] if pages is None else [pages], method=method,
                     owner_counts=counts, owners=candidates)

    def drain(self):
        while self.pending and self.pending[0][1].ready():
            context, staged = self.pending.popleft()
            self.consume(context, staged.finish())

    def consume(self, context, vectors):
        action, group = context['action'], context['group']
        divisor = context['divisor']
        if action == 'copy_descriptor':
            self._copy_descriptor(context, vectors)
            return
        raw = [int(v)//divisor for vector in vectors if vector is not None for v in vector]
        # Preserve native allocation order for per-request CPU page counts.
        pages = list(dict.fromkeys(raw))
        if action == 'reset':
            pages = [p for (g, p) in self.live if g == group]
        owners = context.get('owners') or []
        per_page = []
        counts = context.get('owner_counts')
        if counts is not None and sum(counts) == len(pages):
            per_page = [rid for rid, count in zip(owners, counts) for _ in range(count)]
        elif len(owners) == 1:
            per_page = owners * len(pages)
        units, violations = [], []
        for i, page in enumerate(pages):
            key = (group, page)
            previous = self.live.get(key)
            if action == 'allocate':
                if previous is not None:
                    violations.append({'page_index': page, 'reason': 'allocated_while_live'})
                generation = self.generations.get(key, 0) + 1
                self.generations[key] = generation
                previous = {'page_index': page, 'generation': generation,
                            'created_by_rids': [per_page[i]] if per_page else [],
                            'content_origin_rids': [per_page[i]] if per_page else [],
                            'allocation_seq': context['ledger_seq'],
                            'migration_id': context.get('migration_id')}
                self.live[key] = previous
            if previous is None:
                previous = {'page_index': page, 'generation': None, 'created_by_rids': [],
                            'missing_reason': 'allocation_not_observed'}
                violations.append({'page_index': page, 'reason': 'allocation_not_observed'})
            units.append(dict(previous))
            if action in ('release', 'reset'):
                self.live.pop(key, None)
        specs = self.unit_specs.get(group, [])
        charges = [{'pool': spec['pool'], 'bytes': len(pages)*spec['page_bytes']}
                   for spec in specs if spec['page_bytes'] is not None]
        self.o.emit('kv.lifecycle.pages', {
            **{k: v for k, v in context.items() if k not in ('owners', 'owner_counts', 'divisor')},
            'units': units, 'pool_charges': charges, 'unit_count': len(pages),
            'bytes': sum(v['bytes'] for v in charges) if len(charges)==len(specs) and specs else None,
            'byte_geometry_complete': bool(specs) and len(charges)==len(specs), 'violations': violations,
            'owner_assignment': 'native_batch_CPU_page_counts' if per_page else 'unknown_or_not_allocation',
            'release_semantics': 'allocator_reusable_capacity' if action in ('release','reset') else None,
        }, owners[0] if len(owners) == 1 else None)

    def _copy_descriptor(self, context, vectors):
        source, destination = vectors
        pool, direction = context['pool'], context['direction']
        ps = context['divisor']
        src = list(dict.fromkeys(int(v)//ps for v in source))
        dst = list(dict.fromkeys(int(v)//ps for v in destination))
        def aligned(values):
            return len(values)%ps==0 and all(values[i]%ps==0 and
                   values[i:i+ps]==list(range(values[i],values[i]+ps)) for i in range(0,len(values),ps))
        complete = len(source) == len(destination) and aligned(source) and aligned(destination)
        # Descriptor payload uses the native pool page layout and local layer count.
        payload = len(source)//ps * context['bytes_per_page_per_layer']*context['layers'] if complete else None
        detail = {**{k:v for k,v in context.items() if k not in ('divisor','owners','owner_counts')},
                  'source_pages': src, 'destination_pages': dst, 'payload_bytes': payload,
                  'byte_basis': 'native_aligned_copy_descriptor' if complete else 'unaligned_descriptor_unknown',
                  'hardware_bus_bytes': None}
        provenance=[]
        if complete and len(src)==len(dst):
            for source_page,destination_page in zip(src,dst):
                source_state=self.live.get((context['source_group'],source_page))
                target_state=self.live.get((context['destination_group'],destination_page))
                origin=(source_state or {}).get('content_origin_rids',[])
                if target_state is not None and origin:target_state['content_origin_rids']=list(origin)
                provenance.append({'source_page':source_page,'source_generation':(source_state or {}).get('generation'),
                    'destination_page':destination_page,'destination_generation':(target_state or {}).get('generation'),
                    'content_origin_rids':list(origin)})
        detail['provenance']=provenance
        self.copy_details.setdefault(context['copy_id'], []).append(detail)
        self.o.emit('kv.transfer.copy_descriptor', detail)

    def copy_begin(self, direction):
        cc = self.o.s.tree_cache.cache_controller
        queue = cc.write_queue if direction == 'D2H' else cc.load_queue
        mids = []
        for op in queue:
            for node in getattr(op, 'node_ids', [getattr(op, 'node_id', None)]):
                mids.extend(self.o.pending_transfers.get((direction, str(node)), []))
        self.copies.append({'copy_id': uuid.uuid4().hex, 'direction': direction,
                            'migration_ids': sorted(set(mids)), 'layers_seen': set()})

    def pool_copy(self, pool, direction, args, kwargs):
        if not self.copies:
            return None
        context = self.copies[-1]
        layer = kwargs.get('layer_id', args[3] if len(args)>3 else None) if direction=='H2D' else None
        # Record each actual local-layer call for H2D; D2H copies every local layer.
        key = (str(pool.pool_name), layer)
        if key in context['layers_seen']:
            return None
        context['layers_seen'].add(key)
        host = kwargs.get('host_indices', args[1] if len(args)>1 else None)
        device = kwargs.get('device_indices', args[2] if len(args)>2 else None)
        if host is None or device is None:
            return None
        if int(host.numel())==0 and int(device.numel())==0:return None
        size = getattr(pool, 'item_bytes', getattr(pool, 'state_page_bytes', None))
        if size is None:
            return None
        src, dst = (device, host) if direction=='D2H' else (host, device)
        component=self.layouts.specs[('Host',str(pool.pool_name))]['component']
        source_group=('GPU:' if direction=='D2H' else 'Host:')+component
        destination_group=('Host:' if direction=='D2H' else 'GPU:')+component
        self.capture('copy_descriptor', 'GPU:'+str(pool.pool_name), [src,dst],
                     divisor=pool.page_size, copy_id=context['copy_id'], direction=direction,
                     migration_ids=context['migration_ids'], pool=str(pool.pool_name),
                     layers=pool.layer_num if direction=='D2H' else 1,
                     local_layer=layer, bytes_per_page_per_layer=int(size),
                     source_group=source_group,destination_group=destination_group)
        return context['copy_id']

    def copy_end(self):
        return self.copies.pop() if self.copies else None


def install_request_slot_observer(observer, lifecycle, req_pool):
    """Track native allocated owners independently of scheduler queue transitions."""
    from .hooks import observe_method, argument
    def slot(req):
        return getattr(req,'kv',req).req_pool_idx
    def slots_before(args,kwargs):
        reqs = argument(args,kwargs,'reqs')
        return [(r, slot(r)) for r in reqs]
    def slots_after(args,kwargs,result,error,context):
        if result is None or error:return
        for req,old in context:
            if old is None:
                observer.allocated_requests[str(req.rid)]=req
                lifecycle.capture('allocate','GPU:request_slot',[[int(slot(req))]],
                                  method='ReqToTokenPool.alloc',owners=[str(req.rid)])
    def slots_freed(args,kwargs,result,error,context):
        if error:return
        idx,rid=context
        observer.allocated_requests.pop(rid,None)
        lifecycle.capture('release','GPU:request_slot',[[int(idx)]],
                          method='ReqToTokenPool.free',owners=[rid])
    def slots_cleared(args,kwargs,result,error,context):
        if error:return
        observer.allocated_requests.clear()
        lifecycle.capture('reset','GPU:request_slot',[],method='ReqToTokenPool.clear')
    observe_method(req_pool,'alloc',observer.w,slots_before,slots_after)
    observe_method(req_pool,'free',observer.w,
        lambda args,kw:(slot(argument(args,kw,'req')), str(argument(args,kw,'req').rid)),
        slots_freed)
    observe_method(req_pool,'clear',observer.w,None,slots_cleared)


def install(observer):
    """Attach only supported private native allocator/copy contracts."""
    from .hooks import observe_method, argument
    layouts=observer.sampler.layouts
    if type(layouts.pool).__name__ == "HybridLinearKVPool":
        from .glm import install as install_glm
        return install_glm(observer)
    if not layouts.dsv4 or getattr(layouts.pool,'_unified_kv',True):
        raise ValueError('lifecycle adapter requires native DSV4 separate full/SWA allocators')
    lifecycle = Lifecycle(observer, observer.sampler.layouts)
    observer.lifecycle = lifecycle
    allocator = observer.s.token_to_kv_pool_allocator
    for component, attr in (('full','full_attn_allocator'),('swa','swa_attn_allocator')):
        target = getattr(allocator, attr, None)
        if target is not None and type(target).__name__ == 'PagedTokenToKVPoolAllocator':
            target._trace_lifecycle_callback = lambda action,pages,method,c=component,**kw: observer.w.safe(
                lifecycle.allocator, 'GPU:'+c, action, pages, method, **kw)
            observe_method(target, '_release_page_ids', observer.w, None,
                lambda args,kw,result,error,ctx,c=component: lifecycle.capture(
                    'release','GPU:'+c,list(args),method='_release_page_ids') if not error else None)
    # Scope per-request allocation to the actual native batch, before run_batch.
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    for method in ('prepare_for_extend','prepare_for_decode'):
        original = getattr(ScheduleBatch, method)
        @wraps(original)
        def prepare(batch,*args,_original=original,**kwargs):
            lifecycle.contexts.append([str(r.rid) for r in batch.reqs])
            try:
                return _original(batch,*args,**kwargs)
            finally:
                lifecycle.contexts.pop()
        setattr(ScheduleBatch, method, prepare)
    req_pool = observer.s.req_to_token_pool
    install_request_slot_observer(observer,lifecycle,req_pool)
    group = observer.s.tree_cache.host_pool_group
    sidecars = {str(s.pool_name):str(s.indices_from_pool) for s in observer.s.tree_cache.sidecar_pool_specs}
    for entry in group.entries:
        pool, name = entry.host_pool, str(entry.name)
        if name not in sidecars:
            component = {'kv':'full','swa':'swa'}.get(name)
            if component:
                observe_method(pool,'alloc',observer.w,None,
                    lambda a,k,r,e,ctx,p=pool,c=component:lifecycle.capture('allocate','Host:'+c,[r],
                      divisor=p.page_size,method=type(p).__name__+'.alloc') if r is not None and not e else None)
                observe_method(pool,'free',observer.w,
                    lambda a,k,p=pool:IndexCopy([argument(a,k,'indices')]),
                    lambda a,k,r,e,stage,p=pool,c=component:lifecycle.capture('release','Host:'+c,[],staged=stage,
                      divisor=p.page_size,method=type(p).__name__+'.free') if not e else None)
                observe_method(pool,'clear',observer.w,None,
                    lambda a,k,r,e,ctx,p=pool,c=component:lifecycle.capture('reset','Host:'+c,[],
                      method=type(p).__name__+'.clear') if not e else None)
        if type(pool).__name__ in ('DeepSeekV4PagedHostPool','DeepSeekV4StateHostPool'):
            for method,direction in (('backup_from_device_all_layer','D2H'),('load_to_device_per_layer','H2D')):
                observe_method(pool,method,observer.w,
                    lambda a,k,p=pool,d=direction:lifecycle.pool_copy(p,d,a,k),None)
    return lifecycle
