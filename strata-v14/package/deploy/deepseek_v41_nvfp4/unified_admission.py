"""Observe actual native admission without bypassing a single allocator check.

NO_TOKEN can include state-slot and reserve constraints, not just free KV. These
receipts are necessary evidence for opt-in demotion, never sufficient evidence
that demotion will help. All timestamps use the broadcast policy clock.
"""
from collections import Counter, OrderedDict, deque


class AdmissionFeedback:
    def __init__(self, scheduler, capacity=4096):
        self.scheduler = scheduler
        self.capacity = capacity
        self.counts = Counter()
        self.rejections = OrderedDict()
        self.recent = deque(maxlen=64)
        self.pass_id = 0
        self.available = False
        self.gates = deque(maxlen=128)
        self.admissions = deque(maxlen=2048)
        self._last_gate_sample = {}
        self.epoch = 0

    @property
    def now(self):
        return self.scheduler.orvix_rank_policy.now

    def begin_pass(self):
        self.pass_id += 1
        self.counts['prefill_attempt_passes'] += 1

    def reset(self):
        self.rejections.clear()
        self.recent.clear()
        self.gates.clear()
        self.admissions.clear()
        self._last_gate_sample.clear()
        self.epoch += 1
        # pass_id/counters remain cumulative; trial reports use counter deltas.

    def record_gate(self, reason, running_batch, **values):
        """Called only by an actually taken native early-return/break branch.

        Time here is the broadcast logical clock. A repeated busy loop counts
        more observations, NOT more milliseconds blocked. No allocator method
        or prefix lookup is invoked for instrumentation.
        """
        self.counts['gate_' + reason] += 1
        now = self.now
        if self._last_gate_sample.get(reason) == now:
            return
        self._last_gate_sample[reason] = now
        waiting = self.scheduler.waiting_queue
        # Only identities, never prompt contents or token arrays.
        from itertools import islice
        self.gates.append({'reason': reason, 'at_broadcast_clock': now,
            'prefill_pass': self.pass_id,
            'running_requests': len(getattr(running_batch, 'reqs', ())),
            'waiting_requests': len(waiting),
            'has_continuation': getattr(self.scheduler, 'chunked_req', None) is not None,
            'waiting_head_ids': [str(getattr(r, 'rid', '')) for r in islice(iter(waiting), 8)],
            **values})

    def record(self, req, result, admitted):
        rid = getattr(req, 'rid', None)
        name = getattr(result, 'name', type(result).__name__)
        self.counts['candidate_attempts'] += 1
        self.counts['candidate_' + ('accepted' if admitted else str(name))] += 1
        if not isinstance(rid, str):
            self.counts['missing_request_identity'] += 1
            return
        row = {'request_id': rid, 'result': str(name), 'accepted': bool(admitted),
               'at_broadcast_clock': self.now, 'prefill_pass': self.pass_id}
        self.recent.append(row)
        if not admitted and name == 'NO_TOKEN':
            self.rejections[rid] = row
            self.rejections.move_to_end(rid)
            while len(self.rejections) > self.capacity:
                self.rejections.popitem(last=False)
        else:
            self.rejections.pop(rid, None)

    def recently_capacity_rejected(self, req, max_age_s=2.0):
        row = self.rejections.get(str(getattr(req, 'rid', '')))
        return self.available and row is not None and 0 <= self.now - row['at_broadcast_clock'] <= max_age_s

    def on_admitted(self, requests):
        for req in requests:
            rid = str(getattr(req, 'rid', ''))
            self.rejections.pop(rid, None)
            if getattr(req, '_orvix_admission_receipt_epoch', None) == self.epoch:
                continue
            custom = getattr(getattr(req, 'sampling_params', None), 'custom_params', None)
            meta = custom.get('orvix_unified', {}) if isinstance(custom, dict) else {}
            if not isinstance(meta, dict):
                meta = {}
            if len(self.admissions) == self.admissions.maxlen:
                self.counts['admission_receipts_dropped'] += 1
            self.admissions.append({'request_id': rid, 'family_id': meta.get('family_id'),
                'at_broadcast_clock': self.now, 'ready_at_broadcast_clock': meta.get('ready_at'),
                'prefill_pass': self.pass_id,
                'input_tokens': len(getattr(req, 'origin_input_ids', ())),
                'native_prefix_len': getattr(req, 'prefix_len', None)})
            req._orvix_admission_receipt_epoch = self.epoch

    def install(self):
        get_class = getattr(self.scheduler, 'get_prefill_adder_class', None)
        if not callable(get_class):
            return
        observer = self
        classes = {}

        def observed_class():
            base = get_class()
            if base not in classes:
                class ObservedPrefillAdder(base):
                    def add_one_req(self, req, *args, **kwargs):
                        # Invoke native code exactly once, including its locks,
                        # host loads, state reserves, and rejection cleanup.
                        result = super().add_one_req(req, *args, **kwargs)
                        accepted = any(item is req for item in self.can_run_list)
                        observer.record(req, result, accepted)
                        return result
                ObservedPrefillAdder.__name__ = 'Observed' + base.__name__
                classes[base] = ObservedPrefillAdder
            return classes[base]
        self.scheduler.get_prefill_adder_class = observed_class
        self.available = True

    def snapshot(self):
        return {'available': self.available, 'counts': dict(self.counts),
                'tracked_rejections': len(self.rejections), 'recent': list(self.recent),
                'gate_samples': list(self.gates), 'first_admission_receipts': list(self.admissions),
                'semantics': 'actual native branch observations and add_one_req outcomes; '
                             'counts are not blocked-time shares; NO_TOKEN is not a pool-specific cause'}
