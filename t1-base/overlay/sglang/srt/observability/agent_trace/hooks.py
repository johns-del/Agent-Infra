"""Instance-local wrappers around native observation boundaries.

Original methods run exactly once; observation failures never swallow business
exceptions. No locks, waits, cache queries, eviction decisions or scheduling
parameters are added to the inference path.
"""
from __future__ import annotations
from array import array
from collections import defaultdict
from functools import wraps
import hashlib
import hmac
import os
import time
import uuid

from .runtime import get_writer, integer, number, point, primitive
from .sampler import Sampler, count, node_handle, snapshot_metadata


def argument(args,kwargs,name,index=0,default=None):
    return kwargs.get(name,args[index] if len(args)>index else default)


def observe_method(obj,name,w,before=None,after=None):
    original=getattr(obj,name,None)
    if not callable(original):return False
    label=type(obj).__name__+"."+name
    timed=getattr(w,"safe_labeled",None)
    def call(suffix,fn,*a):
        return timed(label+suffix,fn,*a) if timed else w.safe(fn,*a)
    @wraps(original)
    def wrapped(*args,**kwargs):
        context=call(":before",before,args,kwargs) if before else None
        result=None;error=None
        try:
            result=original(*args,**kwargs)
            return result
        except BaseException as exc:
            error=type(exc).__name__
            raise
        finally:
            if after:call(":after",after,args,kwargs,result,error,context)
    setattr(obj,name,wrapped)
    return True


class EngineObserver:
    def __init__(self,scheduler,*,rank=None):
        self.s=scheduler
        if rank is None:
            ps=getattr(scheduler,"ps",scheduler)
            rank={k:getattr(ps,k,None) for k in ("tp_rank","tp_size","pp_rank","pp_size","dp_rank","gpu_id")}
        self.w=get_writer("scheduler",rank)
        if self.w is None:raise RuntimeError("trace writer unavailable")
        self.cache_epoch=0;self.states={};self.requests={};self.steps=0
        self.prepared_batch=None
        self.allocated_requests={}
        self.last_sample=0.;self.sample_interval=number("ORVIX_TRACE_SAMPLE_SECONDS",1.)
        self.batch_details=os.environ.get("ORVIX_TRACE_BATCH_EVENTS","0")=="1"
        self.batch_context={};self.request_stack=[];self.operation_stack=[];self.victim_stack=[];self.pending_transfers=defaultdict(list)
        self.migrations={};self.acks={};self.restores_by_node=defaultdict(set);self.last_sample_counter={};self.sampler=None
        self.max_active=integer("ORVIX_TRACE_MAX_REQUESTS",4096)
        self.lifecycle=None
        self.admission_identities={}

    def emit(self,kind,data,rid=None,span=None):
        return self.w.emit(kind,{"cache_epoch":self.cache_epoch,**data},rid,span)

    def attach(self):
        s=self.s;w=self.w;c=s.tree_cache
        attached=[]
        def wrap(obj,name,b=None,a=None):
            if observe_method(obj,name,w,b,a):attached.append(type(obj).__name__+"."+name)
        wrap(s,"handle_generate_request",self.received,None)
        wrap(s,"_add_request_to_queue",None,self.queued)
        wrap(s,"run_batch",self.run_before,self.run_after)
        wrap(s,"process_batch_result",None,self.result_after)
        wrap(s,"get_next_batch_to_run",None,self.prepared_after)
        wrap(s,"abort_request",self.abort_before,None)
        wrap(c,"match_prefix",None,self.match_after)
        for n in ("cache_finished_req","cache_unfinished_req"):
            wrap(c,n,self.cache_req_before,self.cache_req_after)
        wrap(c,"insert",None,self.insert_after)
        for n in ("evict","evict_host"):
            wrap(c,n,lambda a,k,n=n:self.operation_before(n,a,k),
                 lambda a,k,r,e,ctx,n=n:self.operation_after(n,a,k,r,e,ctx))
        for n in ("_evict_device_leaf","_demote","_drop_subtree_no_host"):
            wrap(c,n,lambda a,k,n=n:self.victim_before(n,a,k),self.victim_after)
        wrap(c,"_free_values",self.free_before,self.free_after)
        wrap(c,"_apply_cache_action",self.action_before,self.action_after)
        core=getattr(c,"tree_core",None)
        if core is not None:
            wrap(core,"_unregister_node",self.unregister_before,None)
            wrap(core,"_split_node",None,self.split_after)
            wrap(core,"reset",None,self.reset_after)
            for n in ("inc_lock_ref","dec_lock_ref","inc_host_lock_ref","dec_host_lock_ref",
                      "set_component_device_value","commit_backup","commit_load_back","finish_write_through"):
                wrap(core,n,None,lambda a,k,r,e,ctx,n=n:self.node_update_after(n,a,k,r,e))
        cc=getattr(c,"cache_controller",None)
        if cc is not None:
            for n,d in (("write","D2H"),("load","H2D")):
                wrap(cc,n,lambda a,k,d=d:self.transfer_before(d,a,k),
                     lambda a,k,r,e,ctx,d=d:self.transfer_after(d,a,k,r,e,ctx))
            for n,d in (("start_writing","D2H"),("start_loading","H2D")):
                wrap(cc,n,lambda a,k,d=d:self.ack_before(d,copy=True),
                     lambda a,k,r,e,ctx,d=d:self.ack_after(d,ctx,e))
            for n,d in (("writing_check","D2H"),("loading_check","H2D")):
                wrap(c,n,lambda a,k,d=d:self.ack_before(d),
                     lambda a,k,r,e,ctx,d=d:self.completed_acks(d,ctx,e))
        self.w.emit("engine.trace.attached",{"methods":attached,
             "cache_class":type(c).__name__,"tree_core_class":type(core).__name__,
             "allocator_class":type(s.token_to_kv_pool_allocator).__name__,
             "raw_model_config":{k:primitive(getattr(getattr(s,"server_args",None),k,None)) for k in
                 ("model_path","served_model_name","tp_size","pp_size","dp_size","page_size",
                  "kv_cache_dtype","enable_hierarchical_cache","hicache_write_policy","hicache_mem_layout")},
             "mutations_exhaustive":False,"physical_allocation_generations":False,
             "exact_arrival_time_lookup":False,
             "note":"Observe native lookups only; no extra match_prefix that can touch LRU/split nodes."})
        # Unsupported cache layouts must not disable request lifecycle capture.
        self.sampler=w.safe(Sampler,self)
        if self.sampler is not None and os.environ.get('ORVIX_TRACE_LIFECYCLE','0')=='1':
            from .lifecycle import install
            w.safe(install,self)
        self.tick(force=True)
        from .admission import install as install_admission
        w.safe(install_admission,self)

    def received(self,args,kwargs):
        req=argument(args,kwargs,"recv_req")
        rid=getattr(req,"rid",None)
        from .admission import enabled as admission_enabled
        if admission_enabled():
            self.admission_identities[str(rid)]=getattr(req,"agent_trace_identity",None) or {}
            if len(self.admission_identities)>self.max_active:
                self.admission_identities.pop(next(iter(self.admission_identities)))
        n=len(getattr(req,"input_ids",[]) or [])
        data={"input_tokens":n,"boundary":"scheduler_handle_generate_request",
              "native_session_id":getattr(req,"session_id",None),
              "priority":getattr(req,"priority",None),"input_fingerprint":None}
        sampling=getattr(req,"sampling_params",None)
        data["output_budget"]={name:(sampling.get(name) if isinstance(sampling,dict) else getattr(sampling,name,None))
                               for name in ("max_new_tokens","max_thinking_tokens")}
        key=os.environ.get("ORVIX_TRACE_FINGERPRINT_KEY")
        max_tokens=integer("ORVIX_TRACE_FINGERPRINT_TOKENS",131072)
        ids=getattr(req,"input_ids",None)
        if key and ids is not None and n<=max_tokens:
            page=int(getattr(self.s.tree_cache,"page_size",64))
            domain=os.environ.get("ORVIX_TRACE_CACHE_DOMAIN",self.w.instance+":"+self.w.deployment)
            chain=hmac.new(key.encode(),(domain+":"+str(getattr(req,"extra_key",None))).encode(),hashlib.sha256).digest()
            compatibility_tag=chain.hex()
            hashes=[]
            for i in range(0,n,page):
                a=array('q',ids[i:i+page])
                if __import__('sys').byteorder!='little':a.byteswap()
                chain=hmac.new(key.encode(),chain+a.tobytes(),hashlib.sha256).digest()
                hashes.append(chain.hex())
            data["input_fingerprint"]={"algorithm":"HMAC-SHA256-chain-int64le-v1","page_size":page,"cache_domain":domain,
                 "page_hashes":hashes,"last_page_tokens":n%page or page,"compatibility_tag":compatibility_tag,
                 "exact_token_values_stored":False,"compatibility":"same_key_and_engine_configuration_only",
                 "not_valid_for_external_embeddings":getattr(req,"input_embeds",None) is not None}
        else:data["fingerprint_missing_reason"]="disabled_or_input_budget"
        self.emit("engine.request.received",data,rid)

    def queued(self,args,kwargs,result,error,context):
        req=argument(args,kwargs,"req")
        if req is None:return
        self.requests[str(req.rid)]=req
        if len(self.requests)>self.max_active:
            removed=next(iter(self.requests));self.requests.pop(removed,None);self.states.pop(removed,None)
            self.emit("engine.trace.coverage",{"kind":"tracked_request_budget","request_id":removed})
        waiting=any(x is req for x in getattr(self.s,"waiting_queue",[]))
        state="waiting" if waiting else "admission_result_not_in_normal_waiting_queue"
        retracted=argument(args,kwargs,"is_retracted",1,False)
        self.state(req,state,{"is_retracted":bool(retracted),"native_error_type":error})

    def state(self,req,state,extra=None):
        rid=str(req.rid);prev=self.states.get(rid)
        if prev!=state:
            self.emit("engine.request.state",{"state":state,"previous_state":prev,
                        "reason":"native_observation","priority":getattr(req,"priority",None),**(extra or {})},rid)
            self.states[rid]=state

    def active_requests(self):
        result={}
        for attr in ("running_batch","cur_batch","last_batch"):
            for r in getattr(getattr(self.s,attr,None),"reqs",[]) or []:
                if not r.finished():result[str(r.rid)]=r
        for rid,r in list(self.requests.items()):
            if self.states.get(rid) in ("prefill","decode","target_verify","running") and not r.finished():result[rid]=r
        return list(result.values())[:self.max_active]

    def physical_requests(self):
        # get_next_batch_to_run has allocated the returned batch before the
        # caller assigns cur_batch or invokes run_batch. Include these native
        # request mappings in physical ownership, without counting them as
        # running in queue_snapshot.
        result={str(r.rid):r for r in self.active_requests()}
        # Native slot allocation can happen inside batch preparation, before
        # get_next_batch_to_run returns. Keep its owner until native free,
        # including a finished request whose allocation is still held.
        result.update(self.allocated_requests)
        for req in getattr(self.prepared_batch,'reqs',[]) or []:
            if not req.finished():result[str(req.rid)]=req
        return list(result.values())[:self.max_active]

    def prepared_after(self,args,kwargs,result,error,context):
        self.prepared_batch=result if error is None else None
        self.tick()

    def queue_snapshot(self):
        waiting=getattr(self.s,"waiting_queue",[]) or []
        active=self.active_requests()
        return {"running_rids":[str(r.rid) for r in active],
                "waiting_rids":[str(r.rid) for r in waiting[:self.max_active]],
                "waiting_total":len(waiting),"lists_truncated":len(waiting)>self.max_active,
                "coverage":"normal_scheduler_queue_not_all_disaggregation_queues"}

    def run_before(self,args,kwargs):
        batch=argument(args,kwargs,"batch")
        if batch is None:return None
        self.prepared_batch=None
        self.steps+=1;bid=self.w.source+":batch:"+str(self.steps)
        mode=str(getattr(getattr(batch,"forward_mode",None),"name",getattr(batch,"forward_mode","unknown"))).lower()
        ctx={"batch_id":bid,"mode":mode,"submitted_at":point(),"rids":[],"prefixes":{},'prefix_use_ids':{}}
        for req in batch.reqs:
            rid=str(req.rid);self.requests[rid]=req;ctx["rids"].append(rid)
            phase="decode" if "decode" in mode else "target_verify" if "verify" in mode else "prefill" if "extend" in mode else "running"
            changed=self.states.get(rid)!=phase
            self.state(req,phase,{"batch_id":bid})
            ctx["prefixes"][rid]=self.path(getattr(req,"last_node",None))
            if changed and self.lifecycle is not None:
                prefix=getattr(req,'prefix_indices',None)
                if prefix is not None and count(prefix):
                    uid=uuid.uuid4().hex;ctx['prefix_use_ids'][rid]=uid
                    self.lifecycle.capture('scheduled_reference','GPU:full',[prefix],
                        divisor=self.sampler.layouts.full_page,owners=[rid],request_id=rid,
                        use_id=uid,batch_id=bid,method='prefix_indices_before_native_forward')
            if changed:
                self.emit("engine.request.scheduled",{
                    "batch_id":bid,"mode":mode,"prefix_tokens_device":count(getattr(req,"prefix_indices",None)),
                    "host_hit_length":getattr(req,"host_hit_length",None),
                    "swa_host_hit_length":getattr(req,"swa_host_hit_length",None),
                    "committed_tokens":getattr(req,"kv_committed_len",None),
                    "extend_input_len":getattr(req,"extend_input_len",None),
                    "object_ids":ctx["prefixes"][rid],
                    "phase_time_basis":"host_submission_not_GPU_kernel_start"},rid)
        self.batch_context[id(batch)]=ctx
        if self.batch_details:self.emit("engine.batch.start",ctx,span=bid)
        return ctx

    def run_after(self,args,kwargs,result,error,ctx):
        batch=argument(args,kwargs,"batch")
        if ctx is None:return
        # Overlap execution passes shallow copies at result processing. The native
        # forward_iter is assigned inside run_batch, so bind it after the call.
        self.batch_context.pop(id(batch),None)
        seq=getattr(batch,"forward_iter",None)
        if not error and seq is not None:
            self.batch_context[("forward_iter",int(seq))]=ctx
        elif not error:
            self.batch_context[id(batch)]=ctx
        if error:
            self.emit("engine.batch.error",{"batch_id":ctx.get("batch_id"),"error_type":error})
        if len(self.batch_context)>128:
            lost=list(self.batch_context)[:-128]
            for key in lost:self.batch_context.pop(key,None)
            self.emit("engine.trace.coverage",{"kind":"unmatched_batch_context_budget","count":len(lost)})

    def result_after(self,args,kwargs,result,error,context):
        batch=argument(args,kwargs,"batch");native_result=argument(args,kwargs,"result",1)
        if batch is None:return
        seq=getattr(batch,"forward_iter",None)
        key=("forward_iter",int(seq)) if seq is not None else id(batch)
        ctx=self.batch_context.pop(key,{})
        # Never pair an unmatched shallow copy by timestamp.
        bid=ctx.get("batch_id")
        for req in batch.reqs:
            rid=str(req.rid)
            uid=ctx.get('prefix_use_ids',{}).get(rid)
            if uid and not error:
                self.emit('kv.lifecycle.use_completed',{'use_id':uid,'batch_id':bid,
                    'meaning':'scheduled_prefix_used_by_completed_native_forward_not_bytewise_GPU_read'},rid)
            if len(getattr(req,"output_ids",[]))>0 and not getattr(req,"_orvix_first_observed",False):
                req._orvix_first_observed=True
                self.emit("engine.request.first_output_observed",{"boundary":"native_batch_result_processed",
                       "exact_device_first_token_time":None},rid)
            path=ctx.get("prefixes",{}).get(rid,self.path(getattr(req,"last_node",None)))
            # A completed forward establishes logical use, not reads of every KV byte.
            if path and not error:
                signature=(rid,tuple(path),self.cache_epoch)
                if getattr(req,"_orvix_used_path",None)!=signature:
                    req._orvix_used_path=signature
                    self.emit("cache.request.use",{"object_ids":path,"batch_id":bid,
                           "use_semantics":"logical_prefix_used_by_completed_forward",
                           "native_return_counts":{k:getattr(req,k,None) for k in
                             ("cached_tokens","cached_tokens_device","cached_tokens_host","cached_tokens_storage")}},rid)
                for node in path:
                    mids=self.restores_by_node.pop((self.cache_epoch,node),set())
                    for mid in mids:
                        m=self.migrations.get(mid)
                        if m is None or m.get("failed") or m.get("first_use"):continue
                        m["first_use"]=True
                        self.emit("kv.transfer.first_use_observed",{"migration_id":mid,"object_ids":path,
                                  "boundary":"batch_result_processed_upper_bound",
                                  "direct_tensor_read_measured":False},rid)
            if req.finished() and not getattr(req,"_orvix_terminal_observed",False):
                req._orvix_terminal_observed=True
                self.state(req,"finished",{"finish_reason":self.finish_reason(req),"batch_id":bid})
                self.emit("engine.request.completed",{"output_tokens":len(getattr(req,"output_ids",[])),
                    "finish_reason":self.finish_reason(req),
                    "cache_counts":{k:getattr(req,k,None) for k in ("cached_tokens","cached_tokens_device","cached_tokens_host","cached_tokens_storage")},
                    "speculation":{"accepted_output_only":True,"verify_count":getattr(req,"spec_verify_ct",None)},
                    "exact_prefill_gpu_seconds":None},rid)
                self.requests.pop(rid,None);self.states.pop(rid,None)
        if self.batch_details and bid:
            self.emit("engine.batch.end",{"batch_id":bid,"submitted_at":ctx["submitted_at"],
                       "result_observed_at":point(),"mode":ctx.get("mode"),
                       "extend_num_tokens":getattr(batch,"extend_num_tokens",None),
                       "num_correct_drafts":getattr(native_result,"num_correct_drafts",None),
                       "duration_semantics":"host_busy_interval_includes_overlap_not_exclusive_GPU_service"},span=bid)
        self.tick()

    def finish_reason(self,req):
        x=getattr(req,"finished_reason",None)
        if hasattr(x,"to_json"):return primitive(x.to_json())
        return type(x).__name__ if x is not None else None

    def abort_before(self,args,kwargs):
        r=argument(args,kwargs,"recv_req")
        self.emit("engine.request.cancel_requested",{"scope":"scheduler_native_abort",
                  "termination_confirmed":False},getattr(r,"rid",None))

    def path(self,handle):
        core=getattr(self.s.tree_cache,"tree_core",None);arena=getattr(core,"_node_arena",{})
        key=handle if isinstance(handle,int) else getattr(handle,"id",None)
        n=arena.get(key);out=[];seen=set()
        while n is not None and getattr(n,"parent",None) is not None and len(out)<4096:
            if n.id in seen:break
            seen.add(n.id);out.append(str(n.id));n=n.parent
        return out

    def match_after(self,args,kwargs,result,error,ctx):
        p=argument(args,kwargs,"params");req=getattr(p,"req",None)
        if error or result is None:return
        device=count(getattr(result,"device_indices",None));host=int(getattr(result,"host_hit_length",0))
        full=getattr(result,"full_kv_hit_length",None)
        self.emit("cache.lookup",{
            "lookup_stage":"native_match_call_not_forced_arrival_probe",
            "key_tokens":len(p.key) if p is not None and getattr(p,"key",None) is not None else None,
            "device_match_tokens":device,"host_hit_tokens":host,
            "swa_host_hit_length":getattr(result,"swa_host_hit_length",None),
            "mamba_host_hit_length":getattr(result,"mamba_host_hit_length",None),
            "full_kv_hit_length":full,"effective_prefix_tokens_before_restore":device+host,
            "component_restriction_observed":bool(full is not None and full>device+host),
            "best_match_node":node_handle(getattr(result,"best_match_node",None)),
            "last_device_node":node_handle(getattr(result,"last_device_node",None)),
            "object_ids":self.path(getattr(result,"best_match_node",None)),
            "host_count_is_not_additional_SWA_tokens":True,
            "miss_cause":"component_constraint" if full is not None and full>device+host else "not_determined"},
            getattr(req,"rid",None))
        if self.lifecycle is not None and req is not None:
            indices=getattr(result,'device_indices',None)
            if indices is not None and count(indices):
                self.lifecycle.capture('reference','GPU:full',[indices],
                    divisor=self.sampler.layouts.full_page,method='native_match_prefix',
                    owners=[str(req.rid)],request_id=str(req.rid))

    def cache_req_before(self,args,kwargs):
        req=argument(args,kwargs,"req")
        rid=str(req.rid) if req is not None else None
        self.request_stack.append(rid)
        return rid

    def cache_req_after(self,args,kwargs,result,error,ctx):
        if self.request_stack:self.request_stack.pop()
        self.emit("cache.request.commit_observed",{"object_ids":self.path(getattr(argument(args,kwargs,"req"),"last_node",None)),
                    "error_type":error},ctx)

    def insert_after(self,args,kwargs,result,error,ctx):
        if result is None:return
        p=argument(args,kwargs,"params")
        self.emit("cache.insert",{"key_tokens":len(p.key) if getattr(p,"key",None) is not None else None,
                  "prefix_len":getattr(result,"prefix_len",None),
                  "object_ids":self.path(getattr(result,"last_device_node",None)),
                  "relation":"inserted_or_reused_not_exclusive_ownership","invalidates_last_inventory":True},
                  self.request_stack[-1] if self.request_stack else None)

    def operation_before(self,name,args,kwargs):
        ctx={"operation_id":uuid.uuid4().hex,"operation":name,
             "pressure_before":self.sampler.layouts.counts() if self.sampler else None}
        self.operation_stack.append(ctx["operation_id"])
        self.emit("cache.operation.begin",ctx)
        return ctx

    def operation_after(self,name,args,kwargs,result,error,ctx):
        data={**(ctx or {}),"operation":name,"error_type":error,"invalidates_last_inventory":True,
              "pressure_after":self.sampler.layouts.counts() if self.sampler else None,
              "native_return":primitive(vars(result)) if hasattr(result,"__dict__") else primitive(result),
              "trigger_request_id":None,"causality":"native_call_only_no_inferred_victim_beneficiary"}
        self.emit("cache.operation.finish",data)
        if ctx and self.operation_stack and self.operation_stack[-1]==ctx["operation_id"]:
            self.operation_stack.pop()

    def node_metadata(self,node_id):
        core=getattr(self.s.tree_cache,"tree_core",None)
        node=getattr(core,"_node_arena",{}).get(node_id)
        if node is None:return None
        return {str(ct):{"device_index_count":count(node.component_data[ct].value),
                        "host_index_count":count(node.component_data[ct].host_value),
                        "device_present":node.component_data[ct].value is not None,
                        "host_present":node.component_data[ct].host_value is not None,
                        "lock_ref":int(node.component_data[ct].lock_ref)} for ct in node.component_types}

    def victim_before(self,name,args,kwargs):
        nid=argument(args,kwargs,"node_id")
        ctx={"operation":name,"object_id":node_handle(nid),"native_node_id":nid,
             "operation_id":self.operation_stack[-1] if self.operation_stack else None,
             "before_components":self.node_metadata(nid)}
        self.victim_stack.append(ctx["object_id"])
        return ctx

    def victim_after(self,args,kwargs,result,error,ctx):
        if self.victim_stack:self.victim_stack.pop()
        if ctx is None:return
        nid=ctx.pop("native_node_id")
        self.emit("cache.victim.transition",{**ctx,"after_components":self.node_metadata(nid),
                    "native_error_type":error,"invalidates_last_inventory":True,
                    "meaning":"native_eviction_or_demote_step_not_inferred_cross_agent_cause"})

    def free_before(self,args,kwargs):
        device=argument(args,kwargs,"device_frees",0,{}) or {}
        host=argument(args,kwargs,"host_frees",1,{}) or {}
        return {"device_index_counts":{str(k):sum(count(x) for x in v) for k,v in device.items()},
                "host_index_counts":{str(k):sum(count(x) for x in v) for k,v in host.items()},
                "operation_id":self.operation_stack[-1] if self.operation_stack else None,
                "object_id":self.victim_stack[-1] if self.victim_stack else None,
                "pressure_before":self.sampler.layouts.counts() if self.sampler else None}

    def free_after(self,args,kwargs,result,error,ctx):
        if ctx and (any(ctx["device_index_counts"].values()) or any(ctx["host_index_counts"].values())):
            self.emit("kv.pool.release_observed",{**ctx,"error_type":error,
                      "pressure_after":self.sampler.layouts.counts() if self.sampler else None,
                      "bytes_reclaimed":None,"count_unit":"native_component_indices_not_bytes",
                      "success":error is None,"invalidates_last_inventory":True,
                      "boundary":"native_free_values_return_after_allocator_actions"})

    def action_before(self,args,kwargs):
        action=argument(args,kwargs,"action")
        names=("FreeDeviceKV","FreeComponentDeviceSlot","FreeComponentHostSlot",
               "RecoverSWAWithLockedFull","SWARebuild","RebuildFullToSWAMapping")
        name=type(action).__name__
        if name not in names:return None
        fields={}
        for field in ("indices","host_indices","full_indices","swa_indices","kept_full","incoming_full","source_value"):
            value=getattr(action,field,None)
            if value is not None:
                fields[field]=sum(count(x) for x in value) if isinstance(value,(list,tuple)) else count(value)
        return {"native_action":name,"component":str(getattr(action,"component_type","full")),
                "native_index_counts":fields,
                "object_id":node_handle(getattr(action,"node_id",None)),
                "operation_id":self.operation_stack[-1] if self.operation_stack else None}

    def action_after(self,args,kwargs,result,error,ctx):
        if ctx:self.emit("cache.native_action.observed",{**ctx,"error_type":error,
                        "invalidates_last_inventory":True,"bytes_reclaimed":None})

    def unregister_before(self,args,kwargs):
        n=argument(args,kwargs,"node")
        self.emit("cache.object.removed",{"object_id":node_handle(n),"meaning":"native_tree_removal",
                   "physical_free_not_implied":True,"invalidates_last_inventory":True})

    def split_after(self,args,kwargs,result,error,ctx):
        child=argument(args,kwargs,"child",1)
        new_parent=result[0] if isinstance(result,tuple) and result else None
        self.emit("cache.object.split",{"suffix_id":node_handle(child),"prefix_id":node_handle(new_parent),
                    "split_len":argument(args,kwargs,"split_len",2),"error_type":error,
                    "invalidates_last_inventory":True})

    def reset_after(self,*args):
        self.cache_epoch+=1
        self.emit("cache.reset",{"meaning":"logical_tree_reset_not_pool_deallocation"})

    def node_update_after(self,name,args,kwargs,result,error):
        h=argument(args,kwargs,"node_id")
        self.emit("cache.component.change",{"operation":name,"object_id":node_handle(h),
                   "error_type":error,"invalidates_last_inventory":True,
                   "presence_does_not_prove_async_copy_ready":True})

    def transfer_before(self,direction,args,kwargs):
        node=argument(args,kwargs,"node_id",2,-1)
        vec=argument(args,kwargs,"device_indices" if direction=="D2H" else "host_indices")
        extras=argument(args,kwargs,"extra_pools",3,[]) or []
        mid=uuid.uuid4().hex
        info={"migration_id":mid,"direction":direction,"node_id":node_handle(node),
              "native_index_count":count(vec),"actual_transfer_bytes":None,
              "extra_pools":[{"pool":str(x.name),"host_indices":count(x.host_indices),
                               "device_indices":count(x.device_indices)} for x in extras],
              "source_reclaimed_at":None,"destination_usable_at":None,
              "byte_missing_reason":"per_pool_copy_bytes_require_completed_native_transfer_descriptor"}
        info["cache_epoch"]=self.cache_epoch
        self.migrations[mid]=info
        self.pending_transfers[(direction,node_handle(node))].append(mid)
        self.emit("kv.transfer.requested",info)
        if self.lifecycle is not None:self.lifecycle.transfers.append(mid)
        return mid

    def transfer_after(self,direction,args,kwargs,result,error,mid):
        if mid is None:return
        if self.lifecycle is not None and self.lifecycle.transfers:self.lifecycle.transfers.pop()
        self.emit("kv.transfer.accepted",{"migration_id":mid,"direction":direction,
                 "accepted":result is not None and error is None,"error_type":error,
                 "boundary":"controller_return_after_allocation_not_copy_completion"})
        if result is not None and not error and direction=="H2D":
            m=self.migrations[mid]
            self.restores_by_node[(m["cache_epoch"],m["node_id"])].add(mid)
        if result is None or error:
            self.migrations[mid]["failed"]=True
            k=(direction,self.migrations[mid]["node_id"])
            if mid in self.pending_transfers[k]:self.pending_transfers[k].remove(mid)

    def ack_queue(self,direction):
        cc=getattr(self.s.tree_cache,"cache_controller",None)
        return getattr(cc,"ack_write_queue" if direction=="D2H" else "ack_load_queue",[]) or []

    def ack_before(self,direction,copy=False):
        if copy and self.lifecycle is not None:self.lifecycle.copy_begin(direction)
        return {id(a):a for a in self.ack_queue(direction)}

    def ack_after(self,direction,before,error):
        copy=self.lifecycle.copy_end() if self.lifecycle is not None else None
        for ack in self.ack_queue(direction):
            if id(ack) in (before or {}):continue
            mids=[]
            for n in ack.node_ids:
                k=(direction,node_handle(n));mids+=self.pending_transfers.pop(k,[])
            group=uuid.uuid4().hex;self.acks[id(ack)]={"transfer_group_id":group,"migration_ids":mids,"direction":direction}
            if copy:self.acks[id(ack)].update(copy_id=copy['copy_id'],copy_descriptor_count=len(copy['layers_seen']))
            self.emit("kv.transfer.copy_submitted",{**self.acks[id(ack)],
                 "node_ids":[node_handle(n) for n in ack.node_ids],
                 "actual_device_start_at":None,"timing_enabled":bool(getattr(ack,"timing_enabled",False))})

    def completed_acks(self,direction,before,error):
        if error:return
        if self.lifecycle is not None:self.lifecycle.drain()
        remaining={id(a) for a in self.ack_queue(direction)}
        for ident,ack in (before or {}).items():
            if ident in remaining:continue
            # Native method consumed the ack AFTER its own completion check/wait.
            info=self.acks.pop(ident,{"direction":direction,"migration_ids":[],"transfer_group_id":None})
            ms=None
            if getattr(ack,"timing_enabled",False):
                ms=float(ack.start_event.elapsed_time(ack.finish_event))
            descriptors=self.lifecycle.copy_details.pop(info.get('copy_id'),[]) if self.lifecycle is not None else []
            payload=sum(d['payload_bytes'] for d in descriptors) if descriptors and len(descriptors)==info.get('copy_descriptor_count') and all(d['payload_bytes'] is not None for d in descriptors) else None
            self.emit("kv.transfer.copy_completed_observed",{**info,"device_duration_ms":ms,
                      'completed_payload_bytes':payload,
                      'byte_basis':'completed_native_copy_descriptors_not_hardware_bus_bytes' if payload is not None else None,
                      "completion_observation":"native_ack_consumed_upper_bound",
                      "exact_wall_completion_at":None,"source_reclaimed_at":None,
                      "all_layers_copy_complete":True,
                      "first_layer_usable_at":None})
            for mid in info["migration_ids"]:
                if mid in self.migrations:self.migrations[mid]["complete"]=True
        # Bound completed history. Ongoing or never-used histories can be discarded
        # from RAM, but their events remain on disk and analysis reports censoring.
        if len(self.migrations)>10000:
            for mid in list(self.migrations)[:2000]:
                old=self.migrations.pop(mid,None)
                if old:
                    key=(old["cache_epoch"],old["node_id"])
                    self.restores_by_node[key].discard(mid)
                    if not self.restores_by_node[key]:self.restores_by_node.pop(key,None)
            self.emit("engine.trace.coverage",{"kind":"transfer_consumer_tracking_pruned","count":2000})

    def tick(self,force=False):
        if self.lifecycle is not None:self.w.safe(self.lifecycle.drain)
        if self.sampler:self.w.safe(self.sampler.tick,force)
        now=time.monotonic()
        if force or now-self.last_sample>=self.sample_interval:
            self.last_sample=now
            self.emit("engine.sample",{"queues":self.queue_snapshot(),
                       "pool_counts":self.sampler.layouts.counts() if self.sampler else None,
                       "observer_health":self.w.health(),
                       'lifecycle_health':({'pending_index_events':len(self.lifecycle.pending),
                           'last_sequence':self.lifecycle.sequence,'gaps':self.lifecycle.gaps,
                           'live_allocation_units':len(self.lifecycle.live)} if self.lifecycle is not None else None),
                       "sample_semantics":"scheduler_metadata_not_gpu_utilization",
                       "steps_observed":self.steps})
