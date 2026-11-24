"""Native execution-share adapter for the supplied DeepSeek/SGLang snapshot.

Does not allocate/free KV, invent multiple partial prefills, touch CUDA streams,
change model outputs, or interrupt kernels. Native admission can reject any
proposal. Opt-in, version-qualified and single attention-DP/CP/PP only.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import math
import time

try:
    from .chunk_fair_core import Candidate, ChunkConfig, ServiceLedger, ceil_to, plan_shares
except ImportError:
    from chunk_fair_core import Candidate, ChunkConfig, ServiceLedger, ceil_to, plan_shares

CHUNK_VERSION = "orvix-fair-chunk-v5"
UNKNOWN_USER = "__unattributed__"


@dataclass
class Work:
    req: object
    user: str
    inputs: int
    provisional_outputs: int


class ChunkController:
    def __init__(self, scheduler, readonly_match, parallel):
        self.scheduler, self.readonly_match, self.parallel = scheduler, readonly_match, parallel
        self.config = ChunkConfig()
        self.ledger = ServiceLedger(self.config)
        self.counts = Counter()
        self.receipts = deque(maxlen=128)
        self.steps = deque(maxlen=128)
        self.tickets: dict[int, tuple[Work, ...]] = {}
        self.blocked: OrderedDict[str, int] = OrderedDict()
        self.epoch = 0
        self.prefill_streak = 0
        self.current_adder = None
        self.cpu_seconds = 0.0
        self._install_adder()
        self._install_decode_guard()
        self._install_slot_gate()
        self._install_arrivals()

    @property
    def now(self):
        # Only the IPC-broadcast clock participates in decisions.
        return self.scheduler.orvix_rank_policy.now

    def _unsupported(self):
        s = self.scheduler
        if getattr(s, "_orvix_chunk_patch_level", None) != 5:
            return "native chunk patch missing; derive a v5 private runtime"
        if any(getattr(self.parallel, k, 1) != 1 for k in ("pp_size", "attn_dp_size", "attn_cp_size")):
            return "requires single PP, attention DP and attention CP"
        if getattr(s, "require_mlp_sync", False):
            return "MLP synchronization path is not qualified"
        for attr in ("enable_lora", "enable_priority_preemption", "enable_hisparse", "enable_lmcache",
                     "enable_hicache_storage", "enable_unified_memory"):
            if getattr(s, attr, False):
                return attr + " is outside this adapter's qualified scope"
        if getattr(s, "dllm_config", None) is not None or getattr(s, "dynamic_chunk_sizer", None) is not None:
            return "another scheduler owns prefill shape"
        mode = getattr(s, "disaggregation_mode", None)
        if getattr(mode, "value", mode) not in (None, "none", "null"):
            return "disaggregation is not qualified"
        cache = s.tree_cache
        if getattr(cache, "supports_mamba", lambda: True)():
            return "SSM/Mamba state is not qualified"
        if getattr(cache, "linker", None) is not None or getattr(cache, "buffer_pipeline", None) is not None:
            return "external linker / staged-buffer path is not qualified"
        if getattr(cache, "disable", False) or getattr(cache, "enable_session_radix_cache", False):
            return "requires native radix cache, no session borrowing"
        if getattr(s, "orvix_encoder_replay_enabled", False):
            return "encoder SWA replay needs separate continuation accounting"
        chunk, page = getattr(s, "chunked_prefill_size", None), getattr(s, "page_size", None)
        truncation = getattr(s, "truncation_align_size", None)
        if type(chunk) is not int or chunk <= 0 or type(page) is not int or page <= 0:
            return "positive native chunk and page sizes required"
        if truncation is not None and (type(truncation) is not int or truncation <= 0):
            return "invalid native truncation alignment"
        if chunk < math.lcm(page, truncation or 1):
            return "native chunk cannot contain one aligned progress quantum"
        if not callable(getattr(s, "get_num_allocatable_reqs", None)):
            return "native request-slot API unavailable"
        if not callable(getattr(s, "get_prefill_adder_class", None)):
            return "native prefill-adder API unavailable"
        if not callable(getattr(s, "_should_defer_prefill", None)):
            return "native decode protection hook unavailable"
        return None

    def validate_control(self, value):
        config = ChunkConfig(**value) if isinstance(value, dict) else (self.config if value is None else value)
        if not isinstance(config, ChunkConfig):
            raise ValueError("chunk_fair must be an object")
        if config.observes:
            reason = self._unsupported()
            if reason:
                raise ValueError("chunk_fair unavailable: " + reason)
            cap = config.max_profiled_prefill_tokens
            q = math.lcm(self.scheduler.page_size, getattr(self.scheduler, "truncation_align_size", None) or 1)
            if cap and cap < q:
                raise ValueError("profiled cap below native alignment")
        if config != self.config and (self.tickets or self._has_live_work()):
            raise ValueError("drain live work before changing chunk configuration")
        return config

    def configure(self, config, reset=False):
        if reset and (self.tickets or (self.config.observes and self._has_live_work())):
            raise ValueError("drain live work before resetting service ledger")
        if reset or config != self.config:
            self.config = config
            self.ledger = ServiceLedger(config)
            self.blocked.clear()
            self.receipts.clear()
            self.steps.clear()
            self.epoch += 1
            self.prefill_streak = 0
        self.counts["control_updates"] += 1

    def _has_live_work(self):
        # A final prefill can be in flight but not yet merged into running_batch.
        # Use the engine's own flush/drain predicate, not only queue lengths.
        native_idle = getattr(self.scheduler, "is_fully_idle", None)
        if callable(native_idle):
            return not native_idle()
        return (bool(getattr(self.scheduler, "waiting_queue", ()))
                or getattr(self.scheduler, "chunked_req", None) is not None
                or bool(getattr(self.scheduler, "result_queue", ()))
                or bool(getattr(getattr(self.scheduler, "running_batch", None), "reqs", ()))
                or bool(getattr(getattr(self.scheduler, "last_batch", None), "reqs", ())))

    def metadata(self, req):
        m = self.scheduler.orvix_rank_policy._metadata(req)
        if getattr(req, "_orvix_chunk_ready_epoch", None) != self.epoch:
            req._orvix_chunk_ready_epoch = self.epoch
            req._orvix_chunk_ready = self.now
        # Ignore caller-supplied ready_at/lane_weight for this actuator. User
        # identity still has to be supplied by the authenticated gateway.
        return m["family_id"], req._orvix_chunk_ready, m["arrival"]

    def _install_arrivals(self):
        original = getattr(self.scheduler, "_add_request_to_queue", None)
        if not callable(original):
            return
        def arrive(req, *args, **kwargs):
            if self.config.observes:
                self.metadata(req)
            return original(req, *args, **kwargs)
        self.scheduler._add_request_to_queue = arrive

    def _request_supported(self, req):
        return (getattr(req, "beam_group", None) is None and getattr(req, "session", None) is None
                and getattr(req, "input_embeds", None) is None
                and getattr(req, "mm_inputs", None) is None
                and getattr(req, "token_indices_to_pool", None) is None)

    def _sync(self, running):
        users = set()
        for req in [*self.scheduler.waiting_queue, *getattr(running, "reqs", ())]:
            users.add(self.metadata(req)[0])
        cont = getattr(self.scheduler, "chunked_req", None)
        if cont is not None:
            users.add(self.metadata(cont)[0])
        self.ledger.sync(users)

    def prepare_pass(self, adder, running, continuation):
        """Called after native construction and continuation input initialization."""
        self.current_adder = adder
        adder._orvix_share_plan = None
        adder._orvix_share_map = {}
        adder._orvix_share_actual = []
        adder._orvix_share_candidates = {}
        if not self.config.observes:
            return
        self._sync(running)
        if self.config.mode == "guard":
            return
        if self._unsupported():
            raise RuntimeError("runtime capabilities changed while chunk policy active")
        started = time.thread_time()
        try:
            native = min(adder.rem_chunk_tokens or 0, adder.rem_input_tokens)
            if self.config.max_profiled_prefill_tokens:
                native = min(native, self.config.max_profiled_prefill_tokens)
            alignment = math.lcm(adder.page_size, getattr(self.scheduler, "truncation_align_size", None) or 1)
            if native < alignment:
                self.counts["below_alignment_native_fallback"] += 1
                return
            if continuation is not None and not self._request_supported(continuation):
                self.counts["unsupported_continuation_native_fallback"] += 1
                return
            held = int(continuation is not None and getattr(getattr(continuation, "kv", None), "holds_kv", False))
            slots = self.scheduler.get_num_allocatable_reqs(
                len(running.reqs), running_batch=running, num_held_rows=held)
            slots = min(slots, self.scheduler.max_running_requests - len(running.reqs))
            slots = max(0, slots - int(continuation is not None))
            if getattr(adder, "prefill_max_requests", None) is not None:
                slots = min(slots, max(0, adder.prefill_max_requests - int(continuation is not None)))
            waiting = list(self.scheduler.waiting_queue)
            # Old reorders do not redefine arrival. IDs are only deterministic tie breaks.
            waiting.sort(key=lambda r: (*self.metadata(r)[1:], str(r.rid)))
            candidates = []
            if continuation is not None:
                user, ready, arrival = self.metadata(continuation)
                remaining = len(continuation.full_untruncated_fill_ids) - continuation.prefix_len
                if remaining <= 0:
                    raise RuntimeError("nonpositive native continuation length")
                candidates.append(Candidate(str(continuation.rid), user, remaining, ready, arrival, True))
            step = getattr(self.scheduler, "forward_ct", 0)
            for req in waiting[:self.config.scan_limit]:
                rid = str(req.rid)
                if self.blocked.get(rid, -1) > step:
                    self.counts["rejected_candidate_backoffs"] += 1
                    continue
                if not self._request_supported(req):
                    # Native fallback for the WHOLE pass: do not quietly starve a
                    # newly introduced request type behind an adapter filter.
                    self.counts["unsupported_request_native_fallback"] += 1
                    return
                tokens = list(req.origin_input_ids) + list(req.output_ids)
                hit = int(self.readonly_match(req, tokens[:-1]).device_prefix_len) if len(tokens) > 1 else 0
                work = max(1, len(tokens) - min(max(0, hit), max(0, len(tokens)-1)))
                user, ready, arrival = self.metadata(req)
                candidates.append(Candidate(rid, user, work, ready, arrival))
            plan = plan_shares(candidates, native, alignment, slots, self.config,
                               {c.user: self.ledger.effective(c.user) for c in candidates}, self.now)
            self.counts["planned_passes"] += 1
            adder._orvix_share_plan = plan
            adder._orvix_share_candidates = {c.rid: c for c in candidates}
            adder._orvix_share_map = {p.rid: p for p in plan.shares}
            if self.config.mode == "shadow":
                self.counts["shadow_plans"] += 1
                return
            if not plan.shares:
                return
            adder.rem_chunk_tokens = min(adder.rem_chunk_tokens, plan.budget)
            if continuation is not None:
                # Native fit_chunk/state checks STILL apply to this cap.
                adder.chunked_req_limit = adder._orvix_share_map[str(continuation.rid)].cap
            positions = {share.rid: i for i, share in enumerate(plan.shares)}
            waiting.sort(key=lambda r: (positions.get(str(r.rid), len(positions)),
                                        *self.metadata(r)[1:], str(r.rid)))
            self.scheduler.waiting_queue.replace(waiting)
        finally:
            self.cpu_seconds += time.thread_time() - started

    def before_candidate(self, adder, req):
        if not self.config.allocates or getattr(adder, "_orvix_share_plan", None) is None:
            return True
        plan = adder._orvix_share_plan
        if not plan.shares:
            return True  # let native allocator decide; no invented denial
        if str(req.rid) not in adder._orvix_share_map:
            self.counts["outside_current_share_plan"] += 1
            return False
        return True

    def _install_adder(self):
        native_get = getattr(self.scheduler, "get_prefill_adder_class", None)
        if not callable(native_get):
            return
        owner, classes = self, {}
        def get_class():
            base = native_get()
            if base not in classes:
                class ShareAdder(base):
                    def _select_prefill_admission(self, req, *args, **kwargs):
                        result = super()._select_prefill_admission(req, *args, **kwargs)
                        if getattr(self, "_orvix_require_complete", False) and getattr(result, "is_chunked", False):
                            # Before native init_load_back or any admission commit.
                            from sglang.srt.managers.schedule_policy import AddReqResult
                            owner.counts["second_partial_prevented"] += 1
                            return AddReqResult.OTHER
                        return result
                    def add_one_req(self, req, *args, **kwargs):
                        share = getattr(self, "_orvix_share_map", {}).get(str(req.rid))
                        if not owner.config.allocates or share is None:
                            return super().add_one_req(req, *args, **kwargs)
                        global_budget = self.rem_chunk_tokens
                        cap = min(global_budget, share.cap)
                        self.rem_chunk_tokens = cap
                        self._orvix_require_complete = share.completes
                        try:
                            result = super().add_one_req(req, *args, **kwargs)
                        finally:
                            consumed = cap - self.rem_chunk_tokens
                            self.rem_chunk_tokens = global_budget - consumed
                            self._orvix_require_complete = False
                        accepted = any(r is req for r in self.can_run_list)
                        self._orvix_share_actual.append({"request_id": str(req.rid), "accepted": accepted,
                            "cap": cap, "charged_budget": consumed, "result": getattr(result, "name", str(result))})
                        if accepted:
                            if self.new_chunked_req is req and share.completes:
                                raise RuntimeError("native admitted an untracked second partial")
                            owner.counts["share_admissions"] += 1
                            # OTHER may mean our LOCAL cap is spent. Restore the
                            # GLOBAL budget before deciding whether to add more.
                            return self.budget_state()
                        owner.counts["share_native_rejections"] += 1
                        owner.blocked[str(req.rid)] = getattr(owner.scheduler, "forward_ct", 0) + owner.config.rejected_backoff_steps
                        owner.blocked.move_to_end(str(req.rid))
                        while len(owner.blocked) > owner.config.scan_limit * 4:
                            owner.blocked.popitem(last=False)
                        return result
                ShareAdder.__name__ = "OrvixShare" + base.__name__
                classes[base] = ShareAdder
            return classes[base]
        self.scheduler.get_prefill_adder_class = get_class

    def finish_pass(self, native_plan):
        adder, self.current_adder = self.current_adder, None
        if adder is None or not self.config.observes:
            return
        plan = getattr(adder, "_orvix_share_plan", None)
        if plan is None:
            return
        batch = native_plan.batch_to_run
        decode = {id(r) for r in (getattr(batch, "decoding_reqs", None) or ())} if batch else set()
        selected = [r for r in getattr(batch, "reqs", ()) if id(r) not in decode]
        actual = [{"request_id": str(r.rid), "prefill_tokens": int(r.extend_len)} for r in selected]
        proposed = [s.rid for s in plan.shares]
        used = sum(row["prefill_tokens"] for row in actual)
        self.counts["selected_prefill_tokens"] += used
        self.counts["selected_prefill_batches"] += int(bool(selected))
        if self.config.allocates and plan.shares:
            unused = max(0, plan.budget - used)
            self.counts["unspent_plan_budget_tokens"] += unused
            # Do not append another chunk of an already-admitted continuation:
            # that would double-charge native memory reservations. Observe
            # underfill and back off failed beneficiaries on following steps.
            self.counts["multi_request_prefill_batches"] += int(len(selected) > 1)
            self.counts["continuation_shared_batches"] += int(any(s.continuing for s in plan.shares) and len(selected) > 1)
        self.receipts.append({"epoch": self.epoch, "clock": self.now,
            "prefill_pass": getattr(getattr(self.scheduler, "_orvix_admission_feedback", None), "pass_id", None),
            "mode": self.config.mode, "proposal": plan.frozen(), "selected": actual,
            "missing_proposals": [r for r in proposed if r not in {str(q.rid) for q in selected}],
            "native_attempts": list(adder._orvix_share_actual),
            "semantics": "selection is not a completed GPU execution; service charged from launch/result hooks"})

    def _install_decode_guard(self):
        original = getattr(self.scheduler, "_should_defer_prefill", None)
        if not callable(original):
            return
        def defer(batch):
            native = original(batch)
            if self.config.mode not in {"guard", "share", "fair"} or not self.config.max_prefill_streak:
                return native
            if not getattr(batch, "reqs", ()) or getattr(batch, "is_prefill_only", False):
                self.prefill_streak = 0
                return native
            if self.prefill_streak >= self.config.max_prefill_streak:
                self.counts["decode_progress_guard_hits"] += 1
                return True
            return native
        self.scheduler._should_defer_prefill = defer

    def _install_slot_gate(self):
        def decision(**kwargs):
            gate = self.scheduler.min_free_slots_delayer
            native = gate.should_delay(**kwargs)
            if (not native or self.config.mode not in {"guard", "share", "fair"}
                    or self.config.slot_gate_max_wait_s <= 0 or kwargs["num_allocatable_reqs"] <= 0):
                return native
            waiting = list(self.scheduler.waiting_queue)
            oldest = max((max(0., self.now-self.metadata(r)[1]) for r in waiting), default=0.)
            if oldest >= self.config.slot_gate_max_wait_s:
                self.counts["slot_coalescing_wait_overrides"] += 1
                return False  # only coalescing gate; all capacity checks remain
            return native
        self.scheduler._orvix_min_slots_should_delay = decision

    def on_launch(self, batch):
        if not self.config.observes:
            return
        mode = batch.forward_mode
        if getattr(mode, "is_idle", lambda: False)() or getattr(mode, "is_prebuilt", lambda: False)():
            return
        step = batch.forward_iter
        pure_prefill = getattr(mode, "is_extend_without_speculative", mode.is_extend)()
        decode_ids = {id(r) for r in (getattr(batch, "decoding_reqs", None) or ())}
        requests = list(batch.reqs)
        lens = list(batch.extend_lens) if pure_prefill else [0] * len(requests)
        if len(lens) != len(requests) or any(type(v) is not int or v < 0 for v in lens):
            raise RuntimeError("native CPU extend-lens contract changed")
        works = []
        for req, length in zip(requests, lens):
            user = self.metadata(req)[0]
            if getattr(req, "_orvix_chunk_output_epoch", None) != self.epoch:
                req._orvix_chunk_output_epoch = self.epoch
                req._orvix_chunk_output_seen = len(req.output_ids)
            inputs = length if pure_prefill and id(req) not in decode_ids else 0
            generate = (not pure_prefill or id(req) in decode_ids or
                        req.extend_end >= len(req.full_untruncated_fill_ids))
            works.append(Work(req, user, inputs, int(generate)))
        if step in self.tickets:
            raise RuntimeError("duplicate native batch launch")
        self.ledger.reserve(step, [(w.user, w.inputs, w.provisional_outputs) for w in works])
        self.tickets[step] = tuple(works)
        if pure_prefill and not decode_ids:
            self.prefill_streak += 1
        else:
            self.prefill_streak = 0
        self.counts["launched_batches"] += 1
        self.counts["launched_mixed_batches"] += int(bool(decode_ids))
        self.counts["launched_prefill_batches"] += int(pure_prefill)
        self.counts["launched_generation_batches"] += int(not pure_prefill or bool(decode_ids))

    def on_result(self, batch, result):
        if not self.config.observes:
            return
        step = batch.forward_iter
        if step not in self.tickets:
            if getattr(batch.forward_mode, "is_idle", lambda: False)() or getattr(batch.forward_mode, "is_prebuilt", lambda: False)():
                return
            raise RuntimeError("missing native launch ticket; do not report fictional fair service")
        works = self.tickets.pop(step)
        rows, out = [], 0
        for w in works:
            current = len(w.req.output_ids)
            before = w.req._orvix_chunk_output_seen
            if current < before:
                self.counts["output_counter_regressions"] += 1
                raise RuntimeError("committed output counter regressed")
            delta = current - before
            w.req._orvix_chunk_output_seen = current
            rows.append((w.user, w.inputs, delta))
            out += delta
        self.ledger.settle(step, rows)
        self.counts["processed_batches"] += 1
        self.counts["executed_prefill_tokens"] += sum(w.inputs for w in works)
        self.counts["accepted_output_tokens"] += out
        self.steps.append({"forward_iter": step, "inputs": sum(w.inputs for w in works), "outputs": out,
                           "users": sorted(set(w.user for w in works))})

    def snapshot(self):
        return {"version": CHUNK_VERSION, "config": self.config.frozen(),
            "unavailable_reason": self._unsupported(), "counts": dict(self.counts),
            "cpu_seconds": self.cpu_seconds, "ledger": self.ledger.snapshot(),
            "recent_plans": list(self.receipts), "recent_steps": list(self.steps),
            "fairness_scope": "weighted input/output service at feasible prefill choices; one native continuation; no decode preemption",
            "gpu_validated": False, "kv_policy": "native HiCache/allocator only; custom proactive demotion must be observe"}
