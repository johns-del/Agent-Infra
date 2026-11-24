"""v6 native multi-partial adapter for the qualified SGLang commit.

Native memory admission/checkpoint/retraction own all KV mutations. This module
owns compute proposals, request-list transitions and user service accounting.
No tensor allocator, no homemade CPU KV swap, no future workload oracle.
"""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
import math
import time
try:
    from .multi_chunk_core import MultiConfig, CreditLedger, Candidate, plan_batch
    from .multi_chunk_compat import partials, annotate
except ImportError:
    from multi_chunk_core import MultiConfig, CreditLedger, Candidate, plan_batch
    from multi_chunk_compat import partials, annotate

VERSION='orvix-multi-prefill-v6'


@dataclass
class Work:
    req:object
    user:str
    inputs:int
    useful:int
    generated:bool
    start:int
    end:int


class MultiController:
    def __init__(self,scheduler,readonly_match,parallel,consensus=None):
        self.s=scheduler; self.probe=readonly_match; self.parallel=parallel
        self.consensus=consensus or (lambda value:None)
        self.has_consensus = consensus is not None
        self.config=MultiConfig(); self.ledger=CreditLedger(self.config)
        self.counts=Counter(); self.receipts=deque(maxlen=128); self.steps=deque(maxlen=128)
        self.tickets={}; self.abort_pending={}; self.backoff={}
        self.epoch=0; self.pass_id=0; self.prefill_streak=0
        self.last_reclaim=-1000000; self.cpu_seconds=0.; self._pass=None; self.stalled_passes=0; self.stalled_since=None
        self._install_guards()

    @property
    def active(self):return self.config.active
    @property
    def now(self):return self.s.orvix_rank_policy.now

    def partial_requests(self):return tuple(getattr(self.s, 'chunked_reqs', ()))

    def _check(self,value):
        if self.config.consensus:self.consensus(value)

    def _idle(self):
        if self.tickets or self.abort_pending:
            return False
        native=getattr(self.s,'is_fully_idle',None)
        if callable(native):return native()
        return not (self.partial_requests() or self.tickets or self.s.waiting_queue
                    or getattr(self.s.running_batch,'reqs',()) or getattr(self.s,'result_queue',()))

    def unavailable(self):
        if getattr(self.s,'_orvix_multi_patch_level',0)!=6:return 'v6 native lifecycle patch missing'
        # v5's qualified envelope remains useful even though its actuator is OFF.
        base=getattr(self.s,'_orvix_chunk_control',None)
        if base is not None and (reason:=base._unsupported()):return reason
        if not getattr(self.s,'is_generation',True):return 'only autoregressive generation is qualified'
        if getattr(self.s,'uocr_runtime',None) is not None:return 'UOCR is not qualified'
        if getattr(self.s, 'enable_overlap_mlx', False):return 'MLX overlap is not qualified'
        if getattr(self.s, 'enable_unified_cache_external_linker', False):return 'external cache linker is not qualified'
        if getattr(self.s, 'enable_priority_scheduling', False):return 'native priority scheduling conflicts with multi_chunk'
        if getattr(self.s, 'is_mixed_chunk', False):return 'v6 first release uses native alternating execution; mixed is not qualified'
        algo=getattr(self.s,'spec_algorithm',None)
        if algo is not None:
            none=getattr(algo,'is_none',lambda:False)()
            dspark=getattr(algo,'is_dspark',lambda:False)()
            if not (none or dspark):return 'only NONE and DSPARK speculative paths are qualified'
        if not hasattr(self.s,'chunked_reqs'):return 'native CHUNKED request list is required'
        return None

    def validate_control(self,value):
        c=self.config if value is None else MultiConfig(**value)
        if c.observes and (reason:=self.unavailable()):raise ValueError(reason)
        if c.observes and getattr(self.parallel,'tp_size',1)>1 and (not c.consensus or not self.has_consensus):
            raise ValueError('multi-rank execution requires a real plan-consensus transport')
        if c.observes and c.max_partial_prefills>self.s.max_running_requests:
            raise ValueError('partial limit exceeds resident request capacity')
        if c.latency_profile is not None:
            if getattr(self.s,'is_mixed_chunk',False):raise ValueError('pure-prefill time profile cannot govern mixed execution')
            identity=getattr(self.s,'_orvix_runtime_identity',{})
            if c.latency_profile['runtime_fingerprint']!=identity.get('python_fingerprint'):
                raise ValueError('latency profile runtime fingerprint mismatch')
        if c!=self.config and (c.observes or self.config.observes) and not self._idle():raise ValueError('drain engine before changing multi_chunk')
        return c

    def configure(self,config,reset=False):
        if (reset or config!=self.config) and (config.observes or self.config.observes) and not self._idle():raise ValueError('drain before reset')
        if reset or config!=self.config:
            self.config=config; self.ledger=CreditLedger(config); self.epoch+=1
            self.backoff.clear(); self.abort_pending.clear(); self.receipts.clear(); self.steps.clear()
            self.prefill_streak=0; self.last_reclaim=-1000000; self.stalled_passes=0; self.stalled_since=None
        self.counts['control_updates']+=1

    def metadata(self,r):
        m=self.s.orvix_rank_policy._metadata(r)
        if getattr(r,'_orvix_multi_epoch',None)!=self.epoch:
            r._orvix_multi_epoch=self.epoch; r._orvix_multi_ready=self.now
            r._orvix_multi_progress=self.now; r._orvix_multi_step=self.s.forward_ct
            r._orvix_multi_output_seen=len(r.output_ids)
            r._orvix_multi_highwater=0; r._orvix_multi_launches=0
            r._orvix_multi_replay_upto=0
        return m['family_id']

    def supported(self,r):
        return all(getattr(r,k,None) is None for k in
                   ('beam_group','session','input_embeds','multimodal_inputs','mm_inputs',
                    'token_indices_to_pool','positional_embed_overrides','grammar')) and not getattr(r, 'is_prefill_only', False)

    def residents(self,running):
        rows=[*getattr(running, 'reqs', ()),*self.partial_requests(),*getattr(self.s,'prefill_launched_reqs',())]
        return list({id(r):r for r in rows if not r.finished()}.values())

    def _install_guards(self):
        old=getattr(self.s, 'should_process_last_batch_early', lambda batch: False)
        def early(batch):
            if self.active:
                reclaim = self._reclaim_candidate(self.s.running_batch) if batch is not None else None
                need = batch is not None and (bool(partials(batch)) or reclaim is not None)
                self._check({'phase':'early_result_fence', 'step':self.s.forward_ct,
                             'needed':need, 'partial':[str(r.rid) for r in partials(batch)],
                             'reclaim':str(reclaim[0].rid) if reclaim else None})
                if need:
                    self.counts['early_result_fences']+=1
                    return True
            return old(batch)
        self.s.should_process_last_batch_early=early
        old_defer=getattr(self.s, '_should_defer_prefill', lambda batch: False)
        def defer(batch):
            native=old_defer(batch)
            if self.active and getattr(batch, 'reqs', ()) and not getattr(batch,'is_prefill_only',False) and self.prefill_streak>=self.config.max_prefill_streak:
                self.counts['decode_progress_guards']+=1; return True
            return native
        self.s._should_defer_prefill=defer
        old_gate=getattr(self.s,'_orvix_min_slots_should_delay',None)
        def gate(**kw):
            base=old_gate(**kw) if old_gate else self.s.min_free_slots_delayer.should_delay(**kw)
            if not self.active or not base or kw['num_allocatable_reqs']<=0:return base
            if any(self.now-self._ready(r)>=self.config.slot_gate_max_wait_s for r in self.s.waiting_queue):
                self.counts['coalescing_gate_overrides']+=1; return False
            return base
        self.s._orvix_min_slots_should_delay=gate
        arrive = getattr(self.s, '_add_request_to_queue', None)
        if callable(arrive):
            def arrival(req, *args, **kwargs):
                if self.config.observes:
                    self.metadata(req)
                return arrive(req, *args, **kwargs)
            self.s._add_request_to_queue = arrival

    def _ready(self,r):
        self.metadata(r);return r._orvix_multi_ready

    def before_admission(self,running):
        if not self.active:return
        self.process_aborts()
        candidate=self._reclaim_candidate(running)
        self._check({'phase':'reclaim_eligibility', 'step':self.s.forward_ct,
                     'pair':[str(r.rid) for r in candidate] if candidate else None})
        if candidate is not None:
            victim,beneficiary=candidate
            if getattr(victim,'inflight_steps',0):
                self.counts['reclaim_waiting_safe_boundary']+=1;return
            self._retract(victim,running,'borrowed_state_reclaim',beneficiary)

    def _reclaim_candidate(self,running):
        if not self.active or self.config.reclaim!='native_recompute':return None
        step=self.s.forward_ct
        if step-self.last_reclaim<self.config.reclaim_cooldown_steps:return None
        waiting=[r for r in self.s.waiting_queue if not r.finished() and self.supported(r)]
        if not waiting:return None
        residents=self.residents(running)
        slots_full=len(residents)>=self.s.max_running_requests or self.s.req_to_token_pool.available_size()<=0
        partial_full=len(self.partial_requests())>=self.config.max_partial_prefills
        if not (slots_full or partial_full):return None
        counts=Counter(self.metadata(r) for r in residents)
        all_users=set(counts)|{self.metadata(r) for r in waiting}
        shares={u:max(1,math.ceil(self.s.max_running_requests*self.config.user_weights.get(u,1.)/
                 sum(self.config.user_weights.get(v,1.) for v in all_users))) for u in all_users}
        waiting.sort(key=lambda r:(r._orvix_multi_ready,str(r.rid)))
        for b in waiting:
            bu=self.metadata(b)
            if self.now-b._orvix_multi_ready<self.config.reclaim_wait_s or counts[bu]>=shares[bu]:continue
            donors=[]
            for r in residents:
                u=self.metadata(r)
                is_partial=any(p is r for p in self.partial_requests())
                if u==bu or not self.supported(r) or r.finished():continue
                if not slots_full and not is_partial:continue
                if counts[u]<=shares[u] and not (partial_full and is_partial and counts[bu]==0):continue
                if r._orvix_multi_launches<self.config.reclaim_min_service_steps:continue
                if len(r.output_ids)+1>=r.sampling_params.max_new_tokens:continue
                donors.append(r)
            if donors:
                # Favor lower replay footprint; no prediction of remaining output.
                victim=min(donors,key=lambda r:(len(r.origin_input_ids)+len(r.output_ids),str(r.rid)))
                return victim,b
        return None

    def _retract(self,r,running,reason,beneficiary=None):
        if getattr(r,'inflight_steps',0):raise RuntimeError('cannot retract in-flight state')
        self._check({'phase':'retract','rid':str(r.rid),'beneficiary':str(beneficiary.rid) if beneficiary else None,'step':self.s.forward_ct})
        from sglang.srt.managers.schedule_batch import retract_all
        is_partial = any(p is r for p in self.partial_requests())
        # Only work ALREADY executed can be exempt from a scheduler-induced
        # replay charge. A partial's unprocessed prompt is not free future work.
        committed = getattr(r.kv, 'kv_committed_len', None)
        if type(committed) is not int or committed < 0:
            committed = (r.prefix_len if is_partial else
                         max(r.prefix_len, len(r.origin_input_ids) + max(0, len(r.output_ids)-1)))
        r._orvix_multi_replay_upto = max(r._orvix_multi_replay_upto,
                                        r._orvix_multi_highwater, committed)
        if getattr(self.s,'decode_offload_manager',None) is not None:
            self.s.decode_offload_manager.drain_before_retraction()
        unresumable=retract_all(reqs=[r],tree_cache=self.s.tree_cache,request_retirer=self.s.request_retirer,offload_kv=False)
        if unresumable:raise RuntimeError('native retraction reported an unresumable request')
        running.filter_batch(reqs_to_exclude=[r])
        self.s.running_reqs.leave([r]);self.s.prefill_launched_reqs.leave([r]);self.s.chunked_reqs.leave([r])
        self.s._add_request_to_queue(r,is_retracted=True)
        self.backoff[str(r.rid)]=self.pass_id+self.config.rejected_backoff_steps+1
        running.batch_is_full=False;self.last_reclaim=self.s.forward_ct
        self.counts[reason]+=1
        self.receipts.append({'action':reason,'victim':str(r.rid),'beneficiary':str(beneficiary.rid) if beneficiary else None,
                              'recovery':'native cache/recompute; no CPU swap promise'})

    def admit_pass(self,adder,running):
        started=time.thread_time();self.pass_id+=1
        self._pass=(self.pass_id,adder);adder._orvix_multi_plan=None;adder._orvix_multi_actual=[]
        adder._orvix_multi_partials=[]
        native_select = adder._select_prefill_admission
        adder._orvix_allow_new_partial = True
        def select_with_limit(*args, **kwargs):
            decision = native_select(*args, **kwargs)
            if getattr(decision, 'is_chunked', False) and not adder._orvix_allow_new_partial:
                from sglang.srt.managers.schedule_policy import AddReqResult
                return AddReqResult.OTHER
            return decision
        adder._select_prefill_admission = select_with_limit
        residents=self.residents(running)
        queue=list(self.s.waiting_queue)
        # All ranks reach the same agreement even if a local capability changed.
        bad=[str(r.rid) for r in [*residents,*queue] if not self.supported(r)]
        self._check({'phase':'request_contract','bad':bad,'pass':self.pass_id})
        if bad:raise RuntimeError('multi-prefill unsupported request type; drain and use native mode: '+','.join(bad[:4]))
        users=[self.metadata(r) for r in [*residents,*queue]];self.ledger.sync(users)
        occupancy=Counter(self.metadata(r) for r in residents)
        held=set(id(r) for r in self.partial_requests())
        candidates=[];objects={}
        step=self.s.forward_ct
        for r in [*self.partial_requests(),*sorted(queue,key=lambda r:(self._ready(r),str(r.rid)))[:self.config.scan_limit]]:
            rid=str(r.rid);u=self.metadata(r)
            if r.finished() or rid in self.abort_pending or getattr(r,'inflight_steps',0):continue
            if self.backoff.get(rid,-1)>self.pass_id:continue
            continuing=id(r) in held
            if continuing:
                r.init_next_round_input();prefix=int(r.prefix_len)
                remaining=len(r.full_untruncated_fill_ids)-prefix
            else:
                ids=r.origin_input_ids+r.output_ids
                prefix=int(self.probe(r,ids[:-1]).device_prefix_len) if len(ids)>1 else 0
                prefix=min(max(0,prefix),max(0,len(ids)-1));remaining=max(1,len(ids)-prefix)
            if remaining<=0:raise RuntimeError('nonpositive partial input progress')
            objects[rid]=r
            candidates.append(Candidate(rid,u,remaining,prefix,r._orvix_multi_ready,r._orvix_multi_progress,r._orvix_multi_step,continuing))
        budget=min(adder.rem_chunk_tokens or 0,adder.rem_input_tokens)
        alignment=math.lcm(adder.page_size,getattr(self.s,'truncation_align_size',None) or 1)
        row_slots=max(0,self.s.req_to_token_pool.available_size())
        new_slots=min(row_slots,max(0,self.s.max_running_requests-len(residents)))
        native_max = getattr(adder, 'prefill_max_requests', None)
        pass_config = self.config
        if native_max is not None:
            from dataclasses import replace
            pass_config = replace(self.config, max_prefill_reqs=min(self.config.max_prefill_reqs, native_max))
        plan=plan_batch(candidates,budget=max(0,budget),alignment=alignment,new_slots=new_slots,
                        partial_count=len(self.partial_requests()),resident_by_user=occupancy,
                        state_capacity=self.s.max_running_requests,credit={u:self.ledger.available(u) for u in users},
                        config=pass_config,now=self.now,step=step)
        adder._orvix_multi_plan=plan
        self._check({'phase':'plan','pass':self.pass_id,'plan':plan.frozen(),'resident':[str(r.rid) for r in residents]})
        new_partial_count=len(self.partial_requests())
        for share in plan.shares:
            r=objects[share.rid]
            global_budget=adder.rem_chunk_tokens
            cap=min(global_budget,share.cap)
            if cap<=0:break
            adder.rem_chunk_tokens=cap
            try:
                if share.continuing:
                    adder.chunked_req_limit=cap
                    adder.add_chunked_req(r)
                    result=adder.budget_state()
                else:
                    adder._orvix_allow_new_partial = new_partial_count < self.config.max_partial_prefills
                    r.init_next_round_input(self.s.tree_cache)
                    result=adder.add_one_req(r,has_chunked_req=False,truncation_align_size=getattr(self.s,'truncation_align_size',None))
            finally:
                consumed=cap-adder.rem_chunk_tokens
                if consumed<0 or consumed>cap:raise RuntimeError('native local-cap accounting changed')
                adder.rem_chunk_tokens=global_budget-consumed
                adder.chunked_req_limit=None
            accepted=any(x is r for x in adder.can_run_list)
            outcome={'rid':share.rid,'cap':cap,'accepted':accepted,'compute_charge':consumed,'result':getattr(result,'name',str(result))}
            # Native return enum can describe the NEXT request; membership wins.
            if accepted:
                partial=r.extend_end<len(r.full_untruncated_fill_ids)
                outcome['partial']=partial
                if partial:adder._orvix_multi_partials.append(r)
                if share.continuing and not partial:new_partial_count-=1
                if not share.continuing and partial:new_partial_count+=1
                if new_partial_count>self.config.max_partial_prefills:raise RuntimeError('native shortened into too many partials')
                self.counts['accepted_fragments']+=1
            else:
                self.backoff[share.rid]=self.pass_id+self.config.rejected_backoff_steps
                self.counts['native_rejections']+=1
            adder.new_chunked_req=None
            adder._orvix_multi_actual.append(outcome)
            self._check({'phase':'admission_result','pass':self.pass_id,'outcome':outcome})
        self.cpu_seconds+=time.thread_time()-started
        # Keep rejection history bounded and remove IDs that are no longer live.
        live_ids={str(r.rid) for r in [*queue,*residents]}
        self.backoff={k:v for k,v in self.backoff.items() if k in live_ids}
        if not adder.can_run_list and not running.reqs and self.partial_requests():
            self.stalled_passes += 1
            if self.stalled_since is None:
                self.stalled_since = self.now
            self.counts['no_progress_partial_passes'] += 1
            if (self.stalled_passes >= self.config.max_stalled_passes
                    and self.now - self.stalled_since >= self.config.stall_timeout_s):
                stalled = [r for r in self.partial_requests() if not getattr(r, 'inflight_steps', 0)]
                if stalled and self.config.reclaim == 'native_recompute':
                    victim = max(stalled, key=lambda r: (r.prefix_len, str(r.rid)))
                    self._retract(victim, running, 'capacity_progress_reclaim')
                    self.stalled_passes = 0
                    self.stalled_since = None
                else:
                    raise RuntimeError('multi-prefill made no progress after bounded retries; '
                                       'capacity recovery is disabled; inspect native capacity and reduce active partials')
        else:
            self.stalled_passes = 0
            self.stalled_since = None

    def commit_admissions(self,adder):
        middle=set(id(r) for r in adder._orvix_multi_partials)
        self.s.chunked_reqs.leave([r for r in adder.can_run_list if id(r) not in middle])
        self.s.chunked_reqs.join(adder._orvix_multi_partials)
        if adder.can_run_list:
            self.ledger.grant(self.pass_id,adder._orvix_multi_plan.grants)
        self.receipts.append({'pass':self.pass_id,'proposal':adder._orvix_multi_plan.frozen(),
                              'native':adder._orvix_multi_actual,'partial_ids':[str(r.rid) for r in self.partial_requests()]})

    def annotate_batch(self,batch):
        annotate(batch,batch.reqs,self.s.model_config.vocab_size)
        self.counts['multi_partial_batches']+=int(len(partials(batch))>1)

    def mark_abort(self,message):
        for r in self.partial_requests():
            if message.matches(r.rid):self.abort_pending[str(r.rid)]=(r,message)

    def process_aborts(self):
        for rid, (r, message) in list(self.abort_pending.items()):
            if getattr(r, 'inflight_steps', 0):
                continue
            if not any(p is r for p in self.partial_requests()):
                # Last chunk may have moved to PREFILL_LAUNCHED before abort.
                del self.abort_pending[rid]
                if not r.finished() and getattr(r.kv, 'holds_kv', False):
                    self.s.abort_request(message)
                continue
            self._check({'phase': 'abort', 'rid': rid, 'step': self.s.forward_ct})
            if not r.finished():
                from sglang.srt.managers.req_types import FINISH_ABORT
                from sglang.srt.disaggregation.utils import prepare_abort
                reason = r.discard_output_reason or FINISH_ABORT("Aborted")
                prepare_abort(r, reason.message, reason.status_code)
                r.time_stats.trace_ctx.abort(abort_info={'reason': reason.message})
                r.clear_requested_finish()
                self.s.request_retirer.abort(
                    r, finished_reason=r.discard_output_reason and reason.to_json())
            self.s.chunked_reqs.leave([r])
            del self.abort_pending[rid]
            self.counts['partial_aborts'] += 1

    def on_launch(self,batch):
        if not self.config.observes:return
        mode=batch.forward_mode
        if getattr(mode,'is_idle',lambda:False)() or getattr(mode,'is_prebuilt',lambda:False)():return
        prefill=getattr(mode,'is_extend_without_speculative',mode.is_extend)()
        decode={id(r) for r in (getattr(batch,'decoding_reqs',None) or ())}
        lens=list(batch.extend_lens) if prefill else [0]*len(batch.reqs)
        starts=list(batch.prefix_lens) if prefill else [0]*len(batch.reqs)
        works=[]
        for r,n,start in zip(batch.reqs,lens,starts,strict=True):
            u=self.metadata(r);inputs=n if prefill and id(r) not in decode else 0
            end=start+inputs
            useful=max(0,end-max(start,r._orvix_multi_replay_upto)) if inputs else 0
            r._orvix_multi_highwater=max(r._orvix_multi_highwater,end)
            generated=not prefill or id(r) in decode or not any(p is r for p in partials(batch))
            works.append(Work(r,u,inputs,useful,generated,start,end))
            r._orvix_multi_launches+=1
        ticket=batch.forward_iter
        self.ledger.reserve(ticket,[(w.user,w.useful,int(w.generated)) for w in works])
        self.tickets[ticket]=tuple(works)
        self.prefill_streak=self.prefill_streak+1 if prefill and not decode else 0
        self.counts['launched_batches']+=1
        self.counts['launched_mixed_batches']+=int(bool(decode))

    def on_result(self,batch,result):
        if not self.config.observes:return
        ticket=batch.forward_iter
        if ticket not in self.tickets:
            if getattr(batch.forward_mode,'is_idle',lambda:False)() or getattr(batch.forward_mode,'is_prebuilt',lambda:False)():return
            raise RuntimeError('result without launch ticket')
        works=self.tickets.pop(ticket);charges=[];out=0
        for w in works:
            r=w.req;delta=len(r.output_ids)-r._orvix_multi_output_seen
            if delta<0:raise RuntimeError('output counter regressed')
            if not w.generated and delta:
                raise RuntimeError('middle prefill chunk committed a pseudo output token')
            r._orvix_multi_output_seen=len(r.output_ids)
            charges.append((w.user,w.useful,delta));out+=delta
            self.ledger.recomputed+=w.inputs-w.useful
            if w.inputs or delta:
                r._orvix_multi_progress=self.now;r._orvix_multi_step=self.s.forward_ct
        self.ledger.settle(ticket,charges)
        if self.active:
            from sglang.srt.mem_cache.common import checkpoint_kv_cache
            for r in partials(batch):
                if getattr(r,'inflight_steps',0):raise RuntimeError('checkpoint before partial execution drained')
                if getattr(r, 'discard_output_reason', None) is not None:
                    from sglang.srt.managers.io_struct import AbortReq
                    self.abort_pending[str(r.rid)] = (r, AbortReq(rid=r.rid))
                if str(r.rid) in self.abort_pending or r.finished():continue
                checkpoint_kv_cache(r,self.s.tree_cache)
            self.process_aborts()
        self.counts['processed_batches']+=1
        self.counts['physical_prefill_tokens']+=sum(w.inputs for w in works)
        self.counts['accepted_output_tokens']+=out
        self.steps.append({'step':ticket,'prefill':sum(w.inputs for w in works),'output':out,
                           'users':sorted({w.user for w in works}),'partial_ids':[str(r.rid) for r in partials(batch)]})

    def observe_shadow(self, running):
        if self.config.mode != 'shadow':
            return
        rows = list(getattr(self.s, 'waiting_queue', ()))
        rows += list(self.partial_requests())
        self.ledger.sync([self.metadata(r) for r in [*rows, *getattr(running, 'reqs', ())]])
        self.counts['shadow_observation_passes'] += 1

    def snapshot(self):
        return {'version':VERSION,'config':self.config.frozen(),'unavailable_reason':self.unavailable(),
                'counts':dict(self.counts),'ledger':self.ledger.snapshot(),'partial_ids':[str(r.rid) for r in self.partial_requests()],
                'recent_plans':list(self.receipts),'recent_steps':list(self.steps),'cpu_seconds':self.cpu_seconds,
                'gpu_validated':False,'state_recovery':'native cache/recompute; no custom CPU offload',
                'fairness_scope':'bounded service at feasible fragment choices; state reclaim optional; no hard SLO theorem',
                'partial_result_fence':True, 'mixed_supported':False,
                'shadow_semantics':'native scheduling plus service observation, no hypothetical KV allocation'}
