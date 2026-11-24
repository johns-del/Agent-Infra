"""Native metadata snapshots and optional asynchronous INDEX-only staging.

No KV values are copied. A physical snapshot is not a lifetime ledger: slot
identity is scoped to this snapshot unless an allocation generation is known.
Unsupported layouts stay unknown; logical FULL anchors never become fake bytes.
"""
from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import math
import time
import uuid

from .runtime import flag, integer, point, primitive
from .accounting import allocation_totals, reserved_audit


def count(x):
    if x is None: return 0
    try: return int(x.numel()) if hasattr(x,"numel") else len(x)
    except (TypeError,ValueError): return 0


def node_handle(x):
    if x is None:return None
    if isinstance(x,(int,str)):return str(x)
    n=getattr(x,"id",None)
    return str(n) if isinstance(n,(int,str)) else None


def buffers(x):
    if x is None:return []
    if isinstance(x,(list,tuple)):
        return [t for t in x if hasattr(t,"untyped_storage")]
    return [x] if hasattr(x,"untyped_storage") else []


def native_pool_layout(pool, page_size=None):
    """Return allocation charge from verified native buffer geometry, not model math."""
    name=type(pool).__name__
    ps=int(page_size or getattr(pool,"page_size",1))
    base={"native_type":name,"page_size":ps,"page_bytes":None,
          "byte_basis":"unsupported_layout","zero_is_padding":True}
    if pool is None:return base
    if name=="LogicalHostPool":
        return {**base,"page_bytes":0,"byte_basis":"logical_anchor_no_storage","zero_is_padding":False}
    if name=="DeepSeekV4PagedHostPool":
        return {**base,"page_bytes":int(pool.layer_num*pool.item_bytes),
                "byte_basis":"native_host_page_geometry","zero_is_padding":False}
    if name=="DeepSeekV4StateHostPool":
        return {**base,"page_bytes":int(pool.layer_num*pool.state_page_bytes),
                "byte_basis":"native_host_state_page_geometry","zero_is_padding":False}
    if name=="DeepSeekV4SingleKVPool":
        return {**base,"page_bytes":int(pool.bytes_per_page_padded*pool.layer_num),
                "byte_basis":"native_padded_device_page_geometry"}
    if name=="DeepSeekV4IndexerPool":
        ts=buffers(pool.index_k_with_scale_buffer)
        return {**base,"page_bytes":sum(int(t[0].numel()*t.element_size()) for t in ts),
                "byte_basis":"native_indexer_page_geometry"}
    # Restrict names: subclasses with scales/compaction/quantized sidecars need adapters.
    if name in ("MHATokenToKVPool","MLATokenToKVPool"):
        if getattr(pool,"post_capture_active",False):
            return {**base,"byte_basis":"post_capture_backing_requires_allocator_adapter"}
        ts=buffers(getattr(pool,"k_buffer",None))+buffers(getattr(pool,"v_buffer",None))+buffers(getattr(pool,"kv_buffer",None))
        size=int(getattr(pool,"size",-1))
        if ts and all(t.ndim==3 and t.shape[0]==size+ps for t in ts):
            return {**base,"page_bytes":sum(int(t[0].numel()*t.element_size()*ps) for t in ts),
                    "byte_basis":"native_NHD_or_MLA_buffer_geometry"}
        if ts and getattr(pool,"kv_cache_layout",None)=="vectorized_5d" and all(t.ndim==5 for t in ts):
            return {**base,"page_bytes":sum(int(t[0].numel()*t.element_size()) for t in ts),
                    "byte_basis":"native_vectorized_5d_page_geometry"}
    return base


class Layouts:
    def __init__(self, scheduler):
        self.s=scheduler
        self.cache=scheduler.tree_cache
        self.alloc=scheduler.token_to_kv_pool_allocator
        self.pool=self.alloc.get_kvcache()
        self.dsv4=type(self.pool).__name__=="DeepSeekV4TokenToKVPool"
        self.full_page=int(getattr(self.cache,"page_size",1))
        self.specs={}
        self.targets=defaultdict(list)
        self.unassigned=[]
        self._device()
        self._host()

    def add(self,name,tier,component,pool,logical_page=None,layout=None):
        spec=layout or native_pool_layout(pool,logical_page)
        spec={**spec,"pool":name,"tier":tier,"component":component}
        self.specs[(tier,name)]=spec
        self.targets[(tier,component)].append(name)
        if spec["page_bytes"] is None:self.unassigned.append(tier+":"+name)

    def _device(self):
        p=self.pool
        if self.dsv4:
            # Full values are LOGICAL indices. Map a logical full page to native
            # compressed pool pages, using the shipped V4 HiCache assembler contract.
            if getattr(p,"_unified_kv",False):
                self.unassigned.extend(["GPU:unified_swa_ring","GPU:unified_compressed_region"])
            else:
                for n in ("c4","c128"):
                    sub=getattr(p,n+"_kv_pool",None)
                    if sub is not None:self.add(n,"GPU","full",sub,self.full_page)
                self.add("swa","GPU","swa",getattr(p,"swa_kv_pool",None),getattr(p,"swa_page_size",self.full_page))
            idx=getattr(p,"c4_indexer_kv_pool",None)
            if idx is not None:self.add("c4_indexer","GPU","full",idx,self.full_page)
            # Native C4 state rings are addressed by SWA page; C128 rings by
            # request slot. Charge those allocation units, never context length.
            if not getattr(p,"_unified_kv",False):
                for key in ("compress_state_pools","indexer_compress_state_pools"):
                    grouped=defaultdict(int)
                    for pool in getattr(p,key,[]):
                        if pool is not None:
                            grouped[int(pool.ratio)]+=int(pool.ring_size)*int(pool.kv_score_buffer.kv_score[0].nbytes)
                    for ratio,n in grouped.items():
                        name=key+":c"+str(ratio)
                        if ratio not in (4,128):
                            self.unassigned.append("GPU:"+name);continue
                        self.add(name,"GPU","swa" if ratio==4 else "request_slot",None,
                                 layout={"native_type":"DeepSeekV4CompressionStateRing","page_size":int(p.swa_page_size) if ratio==4 else 1,
                                         "page_bytes":n,"byte_basis":"native_state_ring_geometry","zero_is_padding":True})
        elif type(p).__name__=="HybridLinearKVPool":
            self.add("full","GPU","full",p.full_kv_pool,self.full_page)
            self.add("mamba","GPU","mamba",getattr(p,"mamba_pool",None),1)
            self.add("request_slot","GPU","request_slot",None,1)
        elif type(p).__name__=="SWAKVPool":
            self.add("full","GPU","full",p.full_kv_pool)
            self.add("swa","GPU","swa",p.swa_kv_pool)
        else:
            self.add("full","GPU","full",p)
            if getattr(self.cache,"supports_mamba",lambda:False)():
                self.unassigned.append("GPU:mamba_state")

    def _host(self):
        group=getattr(self.cache,"host_pool_group",None)
        entries=getattr(group,"entries",[]) if group is not None else []
        sidecars={str(x.pool_name):str(x.indices_from_pool) for x in getattr(self.cache,"sidecar_pool_specs",[])}
        for e in entries:
            n=str(e.name); source=sidecars.get(n,n)
            component={"kv":"full","swa":"swa","mamba":"mamba"}.get(source)
            if component:self.add(n,"Host",component,e.host_pool)
            else:self.unassigned.append("Host:"+n)

    def inventory(self):
        """Backing tensors only: shape/dtype/bytes, never data_ptr or KV contents."""
        seen_obj=set();seen_storage={};out=[]
        keys=("k_buffer","v_buffer","kv_buffer","index_k_with_scale_buffer",
              "kv_score_buffer","kv_score","swa_kv_pool","full_kv_pool","c4_kv_pool",
              "c128_kv_pool","c4_indexer_kv_pool","unified_kv_pool","compress_state_pools",
              "indexer_compress_state_pools","host_pool","entries","buffer","_buffer",
              "mamba_pool","mamba_cache","conv","temporal","state")
        def walk(x,path,depth=0):
            if x is None or depth>9 or id(x) in seen_obj or len(out)>1024:return
            seen_obj.add(id(x))
            if hasattr(x,"untyped_storage"):
                st=x.untyped_storage(); key=(str(x.device),st.data_ptr(),st.nbytes())
                # address is only used in-memory for alias detection, never exported.
                sid=seen_storage.setdefault(key,"backing-"+str(len(seen_storage)))
                out.append({"path":path,"backing_id":sid,"device":str(x.device),
                            "dtype":str(x.dtype),"shape":list(x.shape),
                            "view_bytes":int(x.numel()*x.element_size()),
                            "storage_bytes":int(st.nbytes()),"storage_offset_elements":int(x.storage_offset())})
            elif isinstance(x,(list,tuple)):
                for i,v in enumerate(x):walk(v,f"{path}[{i}]",depth+1)
            else:
                for k in keys:
                    if k in getattr(x,"__dict__",{}):walk(vars(x)[k],path+"."+k,depth+1)
        walk(self.pool,"device_pool")
        walk(getattr(self.cache,"host_pool_group",None),"host_group")
        unique={x["backing_id"]:x for x in out}
        by_device=defaultdict(int)
        for x in unique.values():by_device[x["device"]]+=x["storage_bytes"]
        return {"buffers":out,"unique_storage_bytes_by_device":dict(by_device),
                "meaning":"reserved_backing_storage_not_valid_or_active_KV_bytes",
                "complete":False,"enumeration_budget_exhausted":len(out)>1024,"unattributed_pools":sorted(set(self.unassigned)),
                "post_capture_backing":"not_certified"}

    def counts(self):
        result={}
        for n in ("available_size","full_available_size","swa_available_size"):
            fn=getattr(self.alloc,n,None)
            if callable(fn):result[n]=int(fn())
        for n in ("full_evictable_size","full_protected_size","swa_evictable_size",
                  "swa_protected_size","mamba_evictable_size","mamba_protected_size"):
            fn=getattr(self.cache,n,None)
            if callable(fn):result[n]=int(fn())
        if type(self.pool).__name__=="HybridLinearKVPool":
            from .glm import native_unit_counts
            result["allocation_units"]=native_unit_counts(self)
        result["units"]="native_index_counts_per_named_pool_not_additive_tokens_or_bytes"
        return result


def snapshot_metadata(cache,max_nodes=4096):
    core=getattr(cache,"tree_core",None)
    arena=getattr(core,"_node_arena",None)
    if not isinstance(arena,dict):
        return {"complete":False,"reason":"python_unified_arena_unavailable","objects":[]},[]
    rows=[];tensors=[]
    root=getattr(core,"root_node",None)
    lengths={getattr(root,"id",None):0}
    def total(n):
        path=[];seen=set()
        while n is not None and n.id not in lengths:
            if n.id in seen:raise ValueError("cycle in native tree")
            seen.add(n.id);path.append(n);n=n.parent
        end=lengths.get(getattr(n,"id",None),0)
        for v in reversed(path):end+=len(v.key) if v.key is not None else 0;lengths[v.id]=end
        return end
    for n in arena.values():
        if n is root:continue
        if max_nodes is not None and len(rows)>=max_nodes:break
        end=total(n);size=len(n.key) if n.key is not None else 0
        row={"cache_object_id":str(n.id),"parent_id":node_handle(n.parent),
             "token_range":[end-size,end],"components":{},"native_priority":getattr(n,"priority",None),
             "last_access_counter":getattr(n,"last_access_time",None)}
        for ct in getattr(n,"component_types",getattr(core,"component_types",[])):
            cd=n.component_data[ct];name=str(ct)
            comp={"device_present":cd.value is not None,"host_present":cd.host_value is not None,
                  "device_index_count":count(cd.value),"host_index_count":count(cd.host_value),
                  "device_lock_ref":int(cd.lock_ref),"host_lock_ref":int(cd.host_lock_ref),
                  "native_session_ref":int(cd.session_ref),
                  "native_session_ids":sorted(cd.session_ids or []),
                  "count_unit":"native_index_entries_may_include_sentinel"}
            row["components"][name]=comp
            for tier,value in (("GPU",cd.value),("Host",cd.host_value)):
                if hasattr(value,"numel") and value.numel():
                    tensors.append(({"object_id":str(n.id),"component":name,"tier":tier,
                                     "lock_ref":comp["device_lock_ref" if tier=="GPU" else "host_lock_ref"],
                                     "native_session_ids":comp["native_session_ids"],
                                     "active_rids":[]},value))
        rows.append(row)
    return {"scope":"unified_tree_host_metadata","complete":len(rows)==max(0,len(arena)-1),
            "objects":rows,"total_native_nodes":max(0,len(arena)-1),
            "consistency":"single_scheduler_thread_metadata_cut",
            "continuous_mutation_coverage":False},tensors


class IndexStage:
    """Copy only small index tensors. CUDA completion is polled; never synchronize."""
    def __init__(self,entries,limit):
        self.entries=[];self.holds=[];self.events=[];self.skipped=[]
        self.started=point();budget=limit;devices={}
        import torch
        # Retained Radix entries can exhaust the entire staging budget before
        # active request slots are reached. Preserve active request/page links
        # first, with full indices across requests before auxiliary SWA indices.
        # The same total budget still applies; omitted entries stay in skipped.
        ordered=sorted(entries,key=lambda item:
                       0 if item[0].get("active_rids") and item[0].get("component")=="full"
                       else 1 if item[0].get("active_rids") else 2)
        for meta,t in ordered:
            n=int(t.numel())
            if budget is not None and n>budget:
                self.skipped.append({**meta,"reason":"index_budget","count":n});continue
            if t.ndim!=1 or t.dtype not in (torch.int32,torch.int64):
                self.skipped.append({**meta,"reason":"unsupported_index_tensor"});continue
            if t.device.type=="cpu":
                h=t.detach().clone()
            elif t.device.type=="cuda":
                # Staging is issued on the scheduler's CURRENT stream after
                # metadata updates; later writes on that stream follow the copy.
                h=torch.empty(t.shape,dtype=t.dtype,device="cpu",pin_memory=True)
                h.copy_(t.detach(),non_blocking=True)
                self.holds.append(t)
                devices[str(t.device)]=t.device
            else:
                self.skipped.append({**meta,"reason":"unsupported_device"});continue
            self.entries.append((meta,h))
            if budget is not None:budget-=n
        for device in devices.values():
            with torch.cuda.device(device):
                ev=torch.cuda.Event();ev.record(torch.cuda.current_stream(device));self.events.append(ev)
        self.index_count=sum(int(h.numel()) for _,h in self.entries)

    def ready(self):return all(e.query() for e in self.events)

    def finish(self,layouts,max_pages):
        if not self.ready():return None
        import torch
        pages={};unknown=[];invalid=[]
        for meta,h in self.entries:
            names=layouts.targets.get((meta["tier"],meta["component"]),[])
            if not names:
                unknown.append({**meta,"reason":"pool_adapter_unavailable"});continue
            for name in names:
                spec=layouts.specs[(meta["tier"],name)];ps=spec["page_size"]
                # Native GPU slot 0 is padding; host page zero is valid.
                # Token-level indices stay in the staged CPU tensor; only the
                # distinct page numbers become Python ints. Converting ~1M
                # indices per checkpoint stalled the scheduler thread.
                valid=h[h>0] if spec["zero_is_padding"] else h[h>=0]
                skipped=int(h.numel())-int(valid.numel())
                if skipped:invalid.append({**meta,"sentinel_or_negative_entries":skipped})
                for page in torch.unique(torch.div(valid,ps,rounding_mode="floor")).tolist():
                    key=(meta["tier"],name,page)
                    if key not in pages and max_pages is not None and len(pages)>=max_pages:
                        unknown.append({**meta,"reason":"page_budget"});break
                    row=pages.setdefault(key,{"tier":meta["tier"],"pool":name,"page_index":page,
                       "bytes_allocated":spec["page_bytes"],"byte_basis":spec["byte_basis"],
                       "allocation_generation":None,"object_ids":set(),"active_rids":set(),
                       "native_session_ids":set(),"locked":False})
                    if meta["component"]=="request_slot":
                        row["allocation_generation"]=meta.get("request_slot_generation")
                    if meta.get("object_id") is not None:row["object_ids"].add(meta["object_id"])
                    row["active_rids"].update(meta.get("active_rids",[]))
                    row["native_session_ids"].update(meta.get("native_session_ids",[]))
                    row["locked"]|=meta.get("lock_ref",0)>0
        rows=[]
        for row in pages.values():
            rows.append({k:sorted(v) if isinstance(v,set) else v for k,v in row.items()})
        known=defaultdict(int)
        for row in rows:
            if row["bytes_allocated"] is not None:known[row["tier"]]+=row["bytes_allocated"]
        return {"pages":rows,"index_count":self.index_count,
                "covered_page_bytes_by_tier":dict(known),"skipped":self.skipped+unknown,
                "sentinel_exclusions":invalid,
                "unattributed_pools":sorted(set(layouts.unassigned)),
                "all_engine_allocations_covered":False,
                "page_set_complete_for_supplied_indices":not self.skipped and not unknown,
                "physical_generation_scope":"this_checkpoint_only",
                "precision":"sampled_native_index_mapping",
                "capture_submitted_at":self.started,"capture_ready_observed_at":point(),
                "device_completion_exact_wall_time":None,
                "independent_stream_consistency":"index_metadata_only_not_KV_tensor_readiness"}


class Sampler:
    def __init__(self,observer):
        self.o=observer;self.s=observer.s;self.w=observer.w
        self.layouts=Layouts(self.s);self.pending=None;self.last=0.0
        self.interval=float(__import__('os').environ.get("ORVIX_TRACE_CHECKPOINT_SECONDS","5"))
        if self.interval<0.1:raise ValueError("checkpoint interval must be >=0.1s")
        self.max_nodes=integer("ORVIX_TRACE_MAX_NODES",4096)
        self.max_indices=integer("ORVIX_TRACE_MAX_INDICES",262144)
        self.max_pages=integer("ORVIX_TRACE_MAX_PAGES",20000)
        # Private audit runs can collect the whole native index mapping. This
        # removes observation caps, never changes allocator or cache policy.
        # CUDA copies still use the scheduler stream and nonblocking polling.
        self.complete_purpose=flag("ORVIX_TRACE_COMPLETE_PURPOSE")
        inventory=self.layouts.inventory()
        inventory["reserved_storage_audit"]=reserved_audit(self.layouts,inventory)
        self.w.emit("engine.pool.layout",{"specs":list(self.layouts.specs.values()),
                                          "inventory":inventory})

    def tick(self,force=False):
        # A checkpoint builds many short-lived, acyclic objects (page rows,
        # event copies). In the scheduler's large heap they triggered repeated
        # full cyclic-GC passes on the scheduler thread. Refcounting still frees
        # them; the previous GC state is always restored.
        paused=gc.isenabled()
        if paused:gc.disable()
        try:
            return self._tick(force)
        finally:
            if paused:gc.enable()

    def _tick(self,force=False):
        if self.pending is not None:
            cp,epoch,stage,context=self.pending
            ready=stage.finish(self.layouts,None if self.complete_purpose else self.max_pages)
            if ready is not None:
                self.w.emit("kv.physical.checkpoint",{"checkpoint_id":cp,"cache_epoch":epoch,**context,**ready})
                self.pending=None
        if not flag('ORVIX_TRACE_CACHE_CHECKPOINTS', True):
            return
        now=time.monotonic()
        if not force and now-self.last<self.interval:return
        self.last=now
        capture=point();cp=uuid.uuid4().hex
        data,tensors=snapshot_metadata(self.s.tree_cache,None if self.complete_purpose else self.max_nodes)
        data.update(checkpoint_id=cp,capture_at=capture,cache_epoch=self.o.cache_epoch,
                    watermark_source_seq=self.w.seq,queues=self.o.queue_snapshot(),
                    pool_counts=self.layouts.counts(),allocation_totals=allocation_totals(self.layouts))
        data['lifecycle_watermark']=self.o.lifecycle.sequence if getattr(self.o,'lifecycle',None) is not None else None
        self.w.emit("cache.checkpoint",data)
        if not flag("ORVIX_TRACE_PHYSICAL") or self.pending is not None:return
        # Include active request slots, including states not inserted into Radix.
        pool=getattr(self.s,"req_to_token_pool",None)
        table=getattr(pool,"req_to_token",None)
        gather_budget=self.max_indices
        active=[]
        physical_requests=getattr(self.o,'physical_requests',self.o.active_requests)
        prepared={str(r.rid) for r in getattr(getattr(self.o,'prepared_batch',None),'reqs',[]) or []}
        running={str(r.rid) for r in self.o.active_requests()}
        for req in physical_requests():
            request_kv=getattr(req,"kv",req)
            idx=getattr(request_kv,"req_pool_idx",None);n=int(getattr(request_kv,"kv_committed_len",0))
            generation=None
            if idx is not None and getattr(pool,"req_generation",None) is not None:
                generation=int(pool.req_generation[idx]) # Native CPU tensor, no device sync.
            active.append({"rid":str(req.rid),"req_pool_idx":idx,"request_slot_generation":generation,
                           "request_index_state":("scheduler_active" if str(req.rid) in running else
                               "prepared_before_forward" if str(req.rid) in prepared else "allocator_reserved"),
                           "kv_committed_len":n,"index_mapping_available":idx is not None and n>0 and table is not None})
            if idx is not None and ("GPU","request_slot") in self.layouts.targets:
                import torch
                tensors.append(({"object_id":None,"component":"request_slot","tier":"GPU","lock_ref":1,
                                 "active_rids":[str(req.rid)],"request_slot_generation":generation},torch.tensor([idx],dtype=torch.int64)))
            if idx is None or n<=0 or table is None:continue
            value=table[idx,:n]
            tensors.append(({"object_id":None,"component":"full","tier":"GPU","lock_ref":1,
                             "active_rids":[str(req.rid)],"native_session_ids":[]},value))
            mamba_idx=getattr(req,"mamba_pool_idx",getattr(getattr(req,"kv",None),"mamba_pool_idx",None))
            if mamba_idx is not None and ("GPU","mamba") in self.layouts.targets:
                import torch
                value_mamba = mamba_idx.reshape(-1) if hasattr(mamba_idx,"reshape") else torch.tensor([mamba_idx],dtype=torch.int64)
                tensors.append(({"object_id":None,"component":"mamba","tier":"GPU","lock_ref":1,
                                 "active_rids":[str(req.rid)],"native_session_ids":[]},value_mamba))
            # Read-only lookup of the existing full->SWA map; no new allocation.
            mapping=getattr(self.layouts.alloc,"full_to_swa_index_mapping",None)
            if mapping is None:mapping=getattr(self.layouts.pool,"full_to_swa_index_mapping",None)
            if mapping is not None and (self.complete_purpose or count(value)<=gather_budget):
                if not self.complete_purpose:gather_budget-=count(value)
                swa=mapping[value.long()]
                tensors.append(({"object_id":None,"component":"swa","tier":"GPU","lock_ref":1,
                                 "active_rids":[str(req.rid)],"native_session_ids":[]},swa))
        stage=IndexStage(tensors,None if self.complete_purpose else self.max_indices)
        context={"active_requests":active,"allocation_totals":data["allocation_totals"],
                 "pool_counts":data["pool_counts"],"capture_at":capture,
                 "queues":data["queues"],"lifecycle_watermark":data['lifecycle_watermark'],
                 "purpose_capture":{"mode":"complete_native_indices" if self.complete_purpose else "bounded_native_indices",
                    "cache_nodes_complete":data.get("complete",False),
                    "cache_nodes_captured":len(data.get("objects",[])),
                    "cache_nodes_total":data.get("total_native_nodes"),
                    "index_budget":None if self.complete_purpose else self.max_indices,
                    "page_budget":None if self.complete_purpose else self.max_pages}}
        self.pending=(cp,self.o.cache_epoch,stage,context)
        if stage.ready():
            ready=stage.finish(self.layouts,None if self.complete_purpose else self.max_pages)
            self.w.emit("kv.physical.checkpoint",{"checkpoint_id":cp,"cache_epoch":self.o.cache_epoch,**context,**ready})
            self.pending=None
