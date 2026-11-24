"""UNIFIED request ordering for the isolated DeepSeek V4.1 comparison.

Legacy ordering remains the default. ``fairness_mode=bounded`` adds an
estimated-admission service ledger and a read-only Full/SWA prefix probe.
Only broadcast logical time participates in scheduling, on every TP rank.
"""

import json
import math
import os
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace

from .unified_core import Parameters, rank_candidates
from .unified_prediction import OutputHistory


KEY = "orvix_unified"
DEFAULT_KNOBS = {"native": "fcfs", "min_candidates": 2, "interleave": 0}
UNKNOWN_FAMILY = "__orvix_unattributed__"


def _concat_ids(origin, output):
    from array import array as _array
    if isinstance(origin, _array):
        out = _array(origin.typecode, origin)
        if output:
            out.extend(output)
        return out
    if isinstance(output, _array):
        out = _array(output.typecode, origin)
        out.extend(output)
        return out
    return list(origin) + list(output)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def read_only_device_prefix(cache, key):
    """Read a device-prefix length without splitting, touching LRU, or copying KV.

    Qualified only for the Python Full/SWA family (including a second SWA slot).
    A temporary view of a partially matched edge exposes its prefix length to
    the native component validators; unlike ``match_prefix`` it never creates
    a tree node. Missing FULL pages terminate the contiguous device prefix.
    Host-only hits are intentionally not counted as resident device hits.

    Unknown component families yield a conservative miss. This probe is for
    ordering only: native admission still rematches, locks, and restores pages.
    """
    miss = lambda reason: SimpleNamespace(device_prefix_len=0, probe_kind=reason)
    if getattr(cache, "disable", False):
        return miss("disabled")
    if getattr(cache, "_tree_core_backend", None) != "python":
        return miss("unsupported_backend")
    core = cache.tree_core
    components = tuple(core.components)
    names = {getattr(c.component_type, "name", "") for c in components}
    if ("FULL" not in names or not names <= {"FULL", "SWA", "AUXILIARY_SWA"}
            or any(type(c).__name__ not in {"FullComponent", "SWAComponent"}
                   for c in components)):
        return miss("unsupported_components")
    if getattr(cache, "linker", None) is not None:
        return miss("external_linker")
    full = next(c.component_type for c in components if c.component_type.name == "FULL")
    validators = tuple(c.create_match_validator(match_device_only=True) for c in components)
    key, _ = key.maybe_to_bigram_view(core.is_eagle)
    key = key.page_aligned(core.page_size)
    node, offset, best = core.root_node, 0, 0
    while offset < len(key):
        child = node.children.get(key.child_key_at(offset, core.page_size))
        if child is None or child.component_data[full].value is None:
            break
        length = child.key.match_at(key, offset, page_size=core.page_size)
        if length <= 0:
            break
        partial = length < len(child.key)
        candidate = (SimpleNamespace(key=child.key[:length],
                                     component_data=child.component_data,
                                     backuped=child.backuped, evicted=child.evicted)
                     if partial else child)
        # Evaluate every validator: SWA validators carry path-local state.
        valid = all([validator(candidate) for validator in validators])
        offset += length
        if valid:
            best = offset
        if partial:
            break
        node = child
    return SimpleNamespace(device_prefix_len=best, probe_kind="readonly_full_swa")


class RankPolicy:
    def __init__(self, match, readonly_match=None, monotonic=time.monotonic):
        self.match = match
        self.readonly_match = readonly_match or match
        self.monotonic = monotonic
        self.now = 0.0
        self.enabled = False
        self.knobs = DEFAULT_KNOBS.copy()
        self.params = Parameters()
        self.counts = Counter()
        self.cpu_seconds = 0.0
        self.last_control_received_at = None
        self.family_service = OrderedDict()
        self.virtual_time = 0.0
        self.epoch = 0
        self.arrivals = 0
        self.last_decision = {}
        self.decision_trace = deque(maxlen=128)
        self.ledger_qualified = True
        self.output_history = OutputHistory()
        self.last_prediction = {}

    def _forecast(self, meta, length):
        prior = self.params.uniform_output_tokens
        raw, samples, source = self.output_history.predict(meta, length, prior)
        used = raw if self.params.cost_model == "online" else prior
        if self.params.cost_model == "online_guarded":
            used, decision = self.output_history.guard(raw, prior)
            source = source + ":" + decision
        return used, samples, source, raw, prior

    def validate_control(self, value):
        if not isinstance(value, dict):
            raise ValueError("scheduler control must be an object")
        now = value["now"]
        enabled = value["enabled"]
        reset = value.get("reset", False)
        if (not _finite(now) or now < self.now or type(enabled) is not bool
                or type(reset) is not bool):
            raise ValueError("invalid scheduler clock or mode")
        params = Parameters(**value["parameters"])
        knobs = DEFAULT_KNOBS | value.get("knobs", {})
        if (
            knobs.keys() != DEFAULT_KNOBS.keys()
            or knobs["native"] not in {"fcfs", "shortest-prefill-first"}
            or type(knobs["min_candidates"]) is not int
            or knobs["min_candidates"] < 2
            or type(knobs["interleave"]) is not int
            or knobs["interleave"] not in (0, 1)
            or (enabled and knobs["native"] != "fcfs")
        ):
            raise ValueError("unqualified scheduler knobs")
        return {"now": float(now), "enabled": enabled, "params": params,
                "knobs": knobs, "reset": reset}

    def apply_control(self, prepared):
        params = prepared["params"]
        changed_epoch = prepared["enabled"] != self.enabled or params != self.params
        if prepared["reset"] or changed_epoch:
            self.reset_ledger()
        self.now, self.enabled = prepared["now"], prepared["enabled"]
        self.params, self.knobs = params, prepared["knobs"]
        self.last_control_received_at = self.monotonic()
        self.counts["control_updates"] += 1

    def control(self, value):
        self.apply_control(self.validate_control(value))

    def reset_ledger(self):
        self.family_service.clear()
        self.output_history.clear()
        self.last_prediction = {}
        self.virtual_time = 0.0
        self.epoch += 1
        self.ledger_qualified = True
        self.last_decision = {}
        self.decision_trace.clear()
        self.counts["ledger_resets"] += 1

    def _metadata(self, req):
        custom = getattr(getattr(req, "sampling_params", None), "custom_params", None)
        value = custom.get(KEY) if isinstance(custom, dict) else None
        if not isinstance(value, dict):
            value = {}
        family = value.get("family_id")
        valid = isinstance(family, str) and 0 < len(family) <= 256
        if not valid:
            family = UNKNOWN_FAMILY
        lane = value.get("lane", "unknown")
        if not isinstance(lane, str) or lane not in self.params.lane_weights:
            lane = "unknown"
        if not hasattr(req, "_orvix_rank_arrival"):
            self.arrivals += 1
            req._orvix_rank_arrival = self.arrivals
            req._orvix_rank_ready = self.now
        ready = value.get("ready_at")
        # A heartbeat may be 0.25 s behind an arriving request. Preserve future
        # ready_at and clamp only the derived wait, not the event timestamp.
        if not _finite(ready) or ready < 0:
            ready = req._orvix_rank_ready
            valid = False
        weight = value.get("lane_weight")
        if not _finite(weight) or weight <= 0:
            weight = self.params.lane_weights[lane]
        meta = {"family_id": family, "lane": lane, "ready_at": float(ready),
                "arrival": req._orvix_rank_arrival, "lane_weight": weight,
                "metadata_valid": valid,
                "session_id": value.get("session_id", ""),
                "scope_id": value.get("scope_id", "")}
        if self.params.fairness_mode == "wait_inheritance":
            # The private replay driver supplies verified same-round linkage.
            for key in ("graph_id", "round_id", "predecessor_request_id"):
                item = value.get(key)
                meta[key] = item if isinstance(item, str) and item else None
            if all(meta[k] is not None for k in ("graph_id", "round_id", "predecessor_request_id")):
                meta["predecessor_queue_s"] = value.get("predecessor_queue_s")
        return meta

    def _sync_ledger(self, families):
        families = list(dict.fromkeys(families))
        capacity = self.params.fairness_max_families
        if len(families) > capacity:
            # Keep memory bounded. Age/cost ranking continues; report that this
            # cycle cannot enforce the estimated-service fairness budget.
            self.counts["ledger_capacity_overflow_cycles"] += 1
            self.ledger_qualified = False
            return False
        active = set(families)
        current = [self.family_service[f] for f in families if f in self.family_service]
        if current:
            self.virtual_time = max(self.virtual_time, min(current))
        for family in families:
            if family not in self.family_service:
                while len(self.family_service) >= capacity:
                    victim = next(f for f in self.family_service if f not in active)
                    del self.family_service[victim]
                    self.counts["ledger_inactive_evictions"] += 1
            self.family_service[family] = max(self.family_service.get(family, self.virtual_time),
                                              self.virtual_time)
            self.family_service.move_to_end(family)
        self.ledger_qualified = True
        return True

    def _order_inherited_wait(self, requests, metadata, diagnostics):
        if any(not row["metadata_valid"] for row in metadata):
            self.counts["cycles_without_metadata"] += 1
            if diagnostics is not None:
                diagnostics["reason"] = "native_order_missing_metadata"
            return requests
        order, details = rank_candidates(metadata, self.now, self.params)
        self.counts["inheritance_ranked_cycles"] += 1
        self.counts["inheritance_credited_candidates"] += sum(row["inherited_wait_s"] > 0 for row in details)
        if diagnostics is not None:
            without = sorted(range(len(requests)), key=lambda i: (
                not details[i]["protected"], metadata[i]["ready_at"], metadata[i]["arrival"]))
            diagnostics.update(
                reason="inherited_wait_order",
                without_inheritance=[str(requests[i].rid) for i in without],
                scored_candidates=[
                    {"request_id": str(req.rid),
                     **{k: row.get(k) for k in ("graph_id", "round_id", "predecessor_request_id",
                                               "predecessor_queue_s", "ready_at", "lane")},
                     **detail}
                    for req, row, detail in zip(requests, metadata, details)],
            )
        return [requests[i] for i in order]

    def order(self, requests, running=(), *, continuing_chunk=False, diagnostics=None):
        requests = list(requests)
        self.counts["cycles"] += 1
        self.counts["peak_candidates"] = max(self.counts["peak_candidates"], len(requests))
        if not self.enabled:
            return requests
        started = time.thread_time()
        if self.params.fairness_mode == "wait_inheritance":
            # The native scheduler still admits a live prefill continuation first.
            # This order is only for requests in the waiting queue.
            ordered = self._order_inherited_wait(requests, [self._metadata(req) for req in requests], diagnostics)
            self.cpu_seconds += time.thread_time() - started
            return ordered
        oldest = None
        native_head = requests[0] if requests else None
        if self.params.fairness_mode == "guarded" and len(requests) >= 2:
            # SGLang retains the last proposed order in its waiting queue.
            # Reconstruct the oldest waiter from its native enqueue timestamp.
            index = min(range(len(requests)),
                        key=lambda i: requests[i].time_stats.wait_queue_entry_time)
            oldest = requests[index]
            if self.params.preserve_fcfs_head and index:
                requests.insert(0, requests.pop(index))
                self.counts["restored_fcfs_head_cycles"] += 1
        bounded = self.params.fairness_mode in ("bounded", "guarded")
        metadata = [self._metadata(req) for req in requests]
        if not bounded:
            for meta in metadata:
                meta["arrival"] = meta["ready_at"]
        if not bounded and any(not meta["metadata_valid"] for meta in metadata):
            self.counts["cycles_without_metadata"] += 1
            return requests
        if bounded:
            families = [meta["family_id"] for meta in metadata]
            families.extend(self._metadata(req)["family_id"] for req in running)
            self._sync_ledger(families)
        if len(requests) < self.knobs["min_candidates"]:
            self.counts["oldest_waiter_bypassed_cycles"] += int(oldest is not None and requests[0] is not oldest)
            return requests
        if self.params.fairness_mode == "guarded" and continuing_chunk:
            # A continuation is selected ahead of the waiting queue by native
            # SGLang. Do not rescan tens of long prefixes for every 1k/2k chunk.
            # Native admission may still use the retained order for leftovers.
            self.counts["continuation_rerank_skips"] += 1
            self.counts["oldest_waiter_bypassed_cycles"] += int(oldest is not None and requests[0] is not oldest)
            return requests
        if self.params.fairness_mode == "guarded":
            pairs = list(zip(requests, metadata))
            sort_from = int(self.params.preserve_fcfs_head)
            pairs[sort_from:] = sorted(pairs[sort_from:],
                                       key=lambda p: (p[1]["ready_at"], p[1]["arrival"]))
            k = self.params.candidate_window_k
            k = len(pairs) if k is None else max(1, k)
            head, tail = pairs[:k], [p[0] for p in pairs[k:]]
            requests, metadata = map(list, zip(*head))
        else:
            tail = []
        rows = []
        probe = self.readonly_match if bounded else self.match
        for req, meta in zip(requests, metadata):
            tokens = _concat_ids(req.origin_input_ids, req.output_ids)
            match = probe(req, tokens)
            matched = max(0, min(len(tokens), int(match.device_prefix_len)))
            row = {**meta, "input_tokens": len(tokens), "matched_tokens": matched}
            if self.params.cost_model in ("online", "online_guarded"):
                prediction, count, source, _, _ = self._forecast(meta, len(tokens))
                row.update(predicted_output_tokens=prediction, prediction_count=count)
                self.counts["prediction_" + source] += 1
            if bounded:
                row["family_service_s"] = (self.family_service.get(meta["family_id"], self.virtual_time)
                                           if self.ledger_qualified else self.virtual_time)
            rows.append(row)
            self.counts["prefix_probe_" + getattr(match, "probe_kind", "native")] += 1
            self.counts["prefix_probe_matched_tokens"] += matched
            self.counts["metadata_fallback_candidates"] += int(not meta["metadata_valid"])
        order, details = rank_candidates(rows, self.now, self.params)
        proposed_head = requests[order[0]]
        self.counts["oldest_waiter_bypassed_cycles"] += int(oldest is not None and proposed_head is not oldest)
        if self.params.preserve_fcfs_head:
            self.counts["preserved_fcfs_head_cycles"] += 1
        for req, row, detail in zip(requests, rows, details):
            req._orvix_rank_reservation = (self.epoch, row["family_id"], detail["service_s"])
        self.counts["ranked_cycles"] += 1
        self.counts["reordered_cycles"] += int(order != list(range(len(requests))))
        self.counts["protected_candidates"] += sum(row["protected"] for row in details)
        self.last_decision = {"candidates": len(rows),
                              "families": len({r["family_id"] for r in rows}),
                              "protected": sum(d["protected"] for d in details),
                              "head_service_s": details[order[0]]["service_s"] if order else None,
                              "head_wait_s": details[order[0]]["wait"] if order else None,
                              "ledger_qualified": self.ledger_qualified}
        if order != list(range(len(requests))):
            if len(self.decision_trace) == self.decision_trace.maxlen:
                self.counts["decision_receipts_dropped"] += 1
            self.decision_trace.append({"at_broadcast_clock": self.now,
                "semantics": "proposed_order_not_native_admission",
                "native_head_request_id": str(getattr(native_head, "rid", "")),
                "oldest_waiting_request_id": str(getattr(oldest, "rid", "")),
                "proposed_head_request_id": str(getattr(proposed_head, "rid", "")),
                "candidates": [{"request_id": str(getattr(requests[i], "rid", "")),
                    "fifo_window_position": i, "selected_position": pos,
                    "family_id": rows[i]["family_id"],
                    "input_tokens": rows[i]["input_tokens"],
                    "matched_device_tokens": rows[i]["matched_tokens"],
                    **details[i]} for pos, i in enumerate(order)]})
        self.cpu_seconds += time.thread_time() - started
        return [requests[index] for index in order] + tail

    def on_admitted(self, requests):
        """Reserve estimated service once per request, after native admission.

        Chunk continuation and a retracted request do not reserve a whole second
        request. This deliberately measures admitted demand, not actual GPU time;
        inaccurate output priors and recomputation are experiment limitations.
        """
        if not self.enabled or self.params.fairness_mode not in ("bounded", "guarded"):
            return
        for req in requests:
            if getattr(req, "_orvix_rank_admitted_epoch", None) == self.epoch:
                continue
            meta = self._metadata(req)
            family = meta["family_id"]
            reservation = getattr(req, "_orvix_rank_reservation", None)
            if (self.params.fairness_mode != "guarded" and reservation is not None
                    and reservation[:2] == (self.epoch, family)):
                service = reservation[2]
            else:
                # The native allocator has already matched the request. Use that
                # result for a singleton that skipped the ranking fast path.
                length = len(req.origin_input_ids) + len(req.output_ids)
                matched = max(0, min(length, getattr(req, "prefix_len", 0)))
                output = self.params.uniform_output_tokens if self.params.cost_model in ("prefill_uniform", "online", "online_guarded") else 0
                if self.params.fairness_mode == "guarded" and self.params.cost_model in ("online", "online_guarded", "prefill_uniform"):
                    output, samples, source, raw, prior = self._forecast(meta, length)
                    req._orvix_output_prediction = (self.epoch, output, samples, source)
                    req._orvix_shadow_prediction = (self.epoch, raw, prior)
                service = self.params.service(length - matched, output)
            if family not in self.family_service:
                while len(self.family_service) >= self.params.fairness_max_families:
                    self.family_service.popitem(last=False)
                    self.counts["ledger_admission_evictions"] += 1
                self.family_service[family] = self.virtual_time
            self.family_service[family] = max(self.family_service[family], self.virtual_time)
            self.family_service[family] += service / self.params.family_weights.get(family, 1.)
            self.family_service.move_to_end(family)
            req._orvix_rank_admitted_epoch = self.epoch
            if self.params.fairness_mode == "guarded":
                # Save the admission-time feature. Later chunk prefix_len is not
                # an estimate of the original uncached prompt any more.
                length = len(req.origin_input_ids)
                matched = max(0, min(length, getattr(req, "prefix_len", 0)))
                req._orvix_rank_accounting = (self.epoch, family, service, length - matched)
            self.counts["admitted_requests"] += 1
            self.counts["estimated_admitted_service_s"] += service

    def completed(self, req):
        """Reconcile token-based estimated service and learn after completion.

        This remains a cost model, not elapsed GPU time. Cancelled/errored work
        is not learned. The native completed request supplies output_ids; the
        replay driver's frozen output_budget is never read here.
        """
        if not self.enabled or self.params.fairness_mode != "guarded":
            return
        if not callable(getattr(req, "finished", None)) or not req.finished():
            return
        if getattr(req, "_orvix_rank_learned_epoch", None) == self.epoch:
            return
        # Ignore native health checks / unannotated probes for learned priors.
        meta = self._metadata(req)
        if not meta["metadata_valid"]:
            return
        reason = getattr(req, "finished_reason", None)
        error_flag = getattr(reason, "is_error", False)
        if callable(error_flag):
            error_flag = error_flag()
        reason_name = (str(reason.get("type", "")) if isinstance(reason, dict)
                       else type(reason).__name__).lower()
        if error_flag is True or any(word in reason_name for word in ("abort", "cancel", "error")):
            return
        output = len(req.output_ids)
        account = getattr(req, "_orvix_rank_accounting", None)
        if account is not None and account[0] == self.epoch:
            _, family, reserved, uncached = account
            realized = self.params.service(uncached, output)
            adjustment = (realized - reserved) / self.params.family_weights.get(family, 1.)
            if family in self.family_service:
                self.family_service[family] = max(self.virtual_time, self.family_service[family] + adjustment)
            self.counts["reconciled_requests"] += 1
            self.counts["service_reconciliation_s"] += realized - reserved
        if self.params.cost_model in ("online", "online_guarded", "prefill_uniform"):
            forecast = getattr(req, "_orvix_output_prediction", None)
            if forecast is not None and forecast[0] == self.epoch:
                _, prediction, samples, source = forecast
                self.counts["prediction_absolute_error_tokens"] += abs(prediction - output)
                self.counts["prediction_observations"] += 1
                self.last_prediction = {"predicted_tokens": prediction, "observed_tokens": output,
                                        "source": source, "samples_at_admission": samples,
                                        "measurement": "admission_forecast_not_completion_retrodiction"}
            shadow = getattr(req, "_orvix_shadow_prediction", None)
            if shadow is not None and shadow[0] == self.epoch:
                _, raw, prior = shadow
                self.output_history.record_forecast_error(raw, prior, output)
                self.counts["shadow_online_absolute_error_tokens"] += abs(raw - output)
                self.counts["fixed_prior_absolute_error_tokens"] += abs(prior - output)
                self.counts["shadow_prediction_observations"] += 1
            self.output_history.observe(meta, len(req.origin_input_ids), output)
        req._orvix_rank_learned_epoch = self.epoch

    def snapshot(self):
        values = list(self.family_service.values())
        heartbeat_age = (None if self.last_control_received_at is None else
                         max(0., self.monotonic() - self.last_control_received_at))
        return {
            "enabled": self.enabled,
            "now": self.now,
            "parameters": self.params.frozen(),
            "knobs": self.knobs,
            "counts": dict(self.counts),
            "adapter_cpu_s": self.cpu_seconds,
            "control_heartbeat_age_s": heartbeat_age,
            "clock_semantics": "broadcast_logical_time; local_time_is_diagnostic_only",
            "fairness": {"mode": self.params.fairness_mode,
                         "accounting": ("token_cost_reserved_then_completion_reconciled"
                                        if self.params.fairness_mode == "guarded" else
                                        "estimated_service_reserved_on_first_admission"),
                         "epoch": self.epoch, "tracked_families": len(values),
                         "ledger_capacity": self.params.fairness_max_families,
                         "virtual_time_s": self.virtual_time,
                         "service_spread_s": max(values) - min(values) if values else 0.,
                         "qualified": self.ledger_qualified},
            "last_decision": self.last_decision,
            "decision_receipts": list(self.decision_trace),
            "last_prediction": self.last_prediction,
            "prediction_groups": len(self.output_history.groups),
            "prediction_backstop": self.output_history.error_snapshot(),
        }


def install(scheduler):
    if os.environ.get("ORVIX_DSV41_UNIFIED") != "1":
        return
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    from sglang.srt.managers.schedule_policy import CacheAwarePolicy
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.common import _req_radix_key
    from sglang.srt.runtime_context import get_parallel
    from .unified_cache_control import install_cache_control
    from .unified_prefill import PrefillGovernor
    from .unified_admission import AdmissionFeedback
    from .unified_provenance import runtime_identity

    cache = scheduler.tree_cache
    if (
        type(cache).__name__ != "UnifiedRadixCache"
        or cache._tree_core_backend != "python"
        or cache.supports_mamba()
        or scheduler.policy.policy.value != "fcfs"
        or scheduler.policy.enable_priority_scheduling
    ):
        raise RuntimeError("UNIFIED rank requires the qualified Python SWA/Full FCFS setup")

    def key_for(req, tokens):
        # Preserve native per-request max-prefix, SWA-tail, extra-key and salt.
        return _req_radix_key(cache, req, tokens, req._compute_max_prefix_len(len(tokens)))

    def match(req, tokens):
        return cache.match_prefix(MatchPrefixParams(key=key_for(req, tokens), cow_mamba=False, req=req))

    def readonly_match(req, tokens):
        if getattr(req, "session", None) is not None:
            return SimpleNamespace(device_prefix_len=0, probe_kind="session_conservative")
        return read_only_device_prefix(cache, key_for(req, tokens))

    policy = RankPolicy(match, readonly_match)
    scheduler.orvix_rank_policy = policy
    admission_feedback = AdmissionFeedback(scheduler)
    admission_feedback.install()
    scheduler._orvix_admission_feedback = admission_feedback
    provenance = runtime_identity()
    try:
        from .forced_trace_runtime import install_forced_trace
    except ImportError:
        from forced_trace_runtime import install_forced_trace
    forced_trace_policy = install_forced_trace(scheduler)
    cache_policy = install_cache_control(scheduler)
    cache_policy.prefix_probe = readonly_match
    # CacheControl already wraps the native completed checkpoint. Chain onto
    # that wrapper; never replace the native insertion or its acknowledgement.
    original_checkpoint_with_cache = cache.checkpoint_into_tree
    def completed_checkpoint(req, *args, **kwargs):
        result = original_checkpoint_with_cache(req, *args, **kwargs)
        policy.completed(req)
        if interval_policy is not None:
            interval_policy.completed(req)
        return result
    cache.checkpoint_into_tree = completed_checkpoint
    parallel = get_parallel()
    prefill_policy = PrefillGovernor(
        scheduler, pp_size=parallel.pp_size,
        attn_dp_size=parallel.attn_dp_size, attn_cp_size=parallel.attn_cp_size)
    scheduler._orvix_prefill_governor = prefill_policy
    # v5 owns only native execution shares; old F/C/P must be off when enabled.
    try:
        from .unified_chunk import ChunkController
    except ImportError:
        from unified_chunk import ChunkController
    chunk_policy = ChunkController(scheduler, readonly_match, parallel)
    scheduler._orvix_chunk_control = chunk_policy
    from .multi_chunk_runtime import MultiController
    scheduler._orvix_runtime_identity = provenance
    transport = cache_policy.consensus if callable(getattr(cache, "_all_reduce", None)) else None
    multi_policy = MultiController(scheduler, readonly_match, parallel, consensus=transport)
    scheduler._orvix_multi_control = multi_policy
    interval_policy = None
    if getattr(scheduler, "_orvix_interval_patch_level", 0) == 7:
        from .interval_runtime import install_interval
        def interval_key(req, tokens):
            total = len(req.origin_input_ids) + len(req.output_ids)
            return _req_radix_key(cache, req, tokens, req._compute_max_prefix_len(total))
        interval_policy = install_interval(scheduler, cache_policy, interval_key, parallel)
    native_choices = {"fcfs": scheduler.policy.policy, "shortest-prefill-first":
                      CacheAwarePolicy.SHORTEST_PREFILL_FIRST}
    original_order = scheduler.policy.calc_priority
    original_get = scheduler.get_internal_state
    original_set = scheduler.set_internal_state
    original_prefill = scheduler.get_new_batch_prefill
    trace_root = Path(__file__).resolve().parents[4]
    trace_path = trace_root / "decision-trace.jsonl"
    tracing = (trace_root / "decision-trace-enabled").is_file() and parallel.tp_rank == 0
    pending_trace = []
    orders = []
    first_seen = set()
    if tracing:
        trace_path.touch(mode=0o600, exist_ok=False)

    def flush_trace():
        if pending_trace:
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row, separators=(",", ":")) + "\n" for row in pending_trace)
            pending_trace.clear()

    def order(*args, **kwargs):
        running_batch = args[1] if len(args) > 1 else kwargs.get("running_batch")
        running = getattr(running_batch, "reqs", ())
        native = original_order(*args, **kwargs)
        chunked = getattr(scheduler, "chunked_req", None) is not None
        if not tracing:
            ordered = policy.order(native, running=running, continuing_chunk=chunked)
            return interval_policy.order(ordered) if interval_policy is not None else ordered
        native = list(native)
        details = {"reason": "native_order"}
        ordered = policy.order(native, running=running, continuing_chunk=chunked, diagnostics=details)
        if interval_policy is not None:
            ordered = interval_policy.order(ordered)
        if native:
            orders.append({"before": [r.rid for r in native], "after": [r.rid for r in ordered], **details})
        return ordered

    def prefill(running_batch, *args, **kwargs):
        admission_feedback.begin_pass()
        if tracing:
            orders.clear()
            continuing_rid = getattr(getattr(scheduler, "chunked_req", None), "rid", None)
            running_before = len(running_batch.reqs)
        admitted = []
        if interval_policy is not None:
            interval_policy.before_admission()
        multi_policy.observe_shadow(running_batch)
        plan = prefill_policy.call(original_prefill, running_batch, *args, **kwargs)
        chunk_policy.finish_pass(plan)
        batch = plan.batch_to_run
        if batch is not None:
            # A native mixed batch can contain existing decode requests.
            decoding = {id(req) for req in (getattr(batch, "decoding_reqs", None) or ())}
            admitted = [req for req in batch.reqs if id(req) not in decoding]
            admission_feedback.on_admitted(admitted)
            policy.on_admitted(admitted)
            cache_policy.on_admitted(admitted)
            if interval_policy is not None:
                interval_policy.on_admitted(admitted)
        if tracing and (orders or admitted):
            attempts = []
            for row in reversed(admission_feedback.recent):
                if row["prefill_pass"] != admission_feedback.pass_id:
                    break
                attempts.append(row)
            selected = [r.rid for r in admitted]
            first = [rid for rid in selected if rid not in first_seen]
            first_seen.update(first)
            pending_trace.append({
                "epoch": policy.epoch, "prefill_pass": admission_feedback.pass_id,
                "monotonic_s": time.monotonic(), "broadcast_clock_s": policy.now,
                "inheritance_cap_s": policy.params.inheritance_cap_s,
                "running_before_count": running_before, "continuing_rid": continuing_rid,
                "orders": orders.copy(), "native_attempts": attempts[::-1],
                "selected_prefill_rids": selected, "first_admitted_rids": first,
            })
        return plan

    def get_state(req):
        result = original_get(req)
        if tracing:
            flush_trace()
        result.internal_state[KEY] = policy.snapshot() | {
            "decision_trace_active": tracing,
            "cache": cache_policy.snapshot(), "prefill": prefill_policy.snapshot(),
            "adapter_version": ("orvix-dsv41-multi-prefill-v6" if
                                getattr(scheduler, "_orvix_multi_patch_level", 0) == 6 else
                                "orvix-dsv41-fair-chunk-v5"),
            "multi_chunk": multi_policy.snapshot(),
            "chunk_fair": chunk_policy.snapshot(),
            "native_admission": admission_feedback.snapshot(),
            "runtime_identity": provenance,
        }
        result.internal_state[KEY]["forced_trace"] = forced_trace_policy.snapshot()
        if interval_policy is not None:
            result.internal_state[KEY]["interval_cache"] = interval_policy.snapshot()
            result.internal_state[KEY]["adapter_version"] = "orvix-dsv41-interval-cache-v7"
        return result

    def set_state(req):
        if req.orvix_unified is None:
            return original_set(req)
        if req.server_args:
            return SetInternalStateReqOutput(updated=False)
        validation_error = None
        try:
            previous_knobs = policy.knobs
            prepared = policy.validate_control(req.orvix_unified)
            cache_prepared = cache_policy.validate_control(req.orvix_unified)
            prefill_prepared = prefill_policy.validate_control(req.orvix_unified.get("prefill"))
            chunk_prepared = chunk_policy.validate_control(req.orvix_unified.get("chunk_fair"))
            multi_prepared = multi_policy.validate_control(req.orvix_unified.get("multi_chunk"))
            interval_prepared = (interval_policy.validate_control(req.orvix_unified)
                                 if interval_policy is not None else None)
            if interval_policy is None and ("interval_cache" in req.orvix_unified or "interval_events" in req.orvix_unified):
                raise ValueError("v7 runtime required for interval cache control")
            if interval_prepared is not None and interval_prepared[0].mode != "off" and (
                    multi_prepared.observes or chunk_prepared.observes or prepared["enabled"]
                    or prefill_prepared.mode not in {"off", "decode_cadence", "balanced"}
                    or cache_prepared["config"].mode != "observe"
                    or prepared["knobs"]["native"] != "fcfs" or prepared["knobs"]["interleave"] != 0):
                raise ValueError("interval v11 requires native FCFS, old F/C/decode_guard/v5/v6 actions disabled")
            if multi_prepared.observes and (
                    chunk_prepared.observes or prepared["enabled"]
                    or prefill_prepared.mode != "off"
                    or cache_prepared["config"].mode != "observe"
                    or prepared["knobs"]["native"] != "fcfs"
                    or prepared["knobs"]["interleave"] != 0):
                raise ValueError("multi_chunk requires v5/old F/P off, C observe, native FCFS")
            if prepared["reset"] and (multi_policy.config.observes or multi_prepared.observes) and not multi_policy._idle():
                raise ValueError("drain engine before resetting multi-chunk ledger")
            if chunk_prepared.observes and (
                    prepared["enabled"] or prefill_prepared.mode != "off"
                    or cache_prepared["config"].mode != "observe"
                    or prepared["knobs"]["native"] != "fcfs"
                    or prepared["knobs"]["interleave"] != 0):
                raise ValueError("chunk_fair requires old F/P off, C observe and native fcfs")
            if prepared["reset"] and (chunk_policy.tickets or (chunk_policy.config.observes and chunk_policy._has_live_work())):
                raise ValueError("drain native inflight work before resetting chunk service")
        except (ValueError, TypeError, KeyError) as exc:
            validation_error = str(exc)
        if (getattr(scheduler, "_orvix_multi_patch_level", 0) == 6 or interval_policy is not None) and transport is not None:
            # Called also for rejected proposals, so ranks cannot split between
            # apply and reject. Divergence fails consistently before mutation.
            transport({"phase": "multi_control_validation", "valid": validation_error is None,
                       "payload": req.orvix_unified})
        if validation_error is not None:
            return SetInternalStateReqOutput(updated=False)
        # Validate every subsystem before applying any setting or lifecycle
        # event. A rejected request must not leave a partially updated policy.
        multi_policy.configure(multi_prepared, reset=prepared["reset"])
        chunk_policy.configure(chunk_prepared, reset=prepared["reset"])
        policy.apply_control(prepared)
        cache_policy.apply_control(cache_prepared)
        if interval_policy is not None:
            interval_policy.apply_control(interval_prepared)
        if policy.knobs != previous_knobs:
            scheduler.policy.policy = native_choices[policy.knobs["native"]]
            scheduler.prefill_decode_interval = policy.knobs["interleave"]
            scheduler._prefill_decode_interval_remaining = 0
        # Apply last: cadence owns native interval while enabled. On transition
        # back to off it releases that ownership before honoring baseline knobs.
        prefill_policy.configure(prefill_prepared)
        if prepared["reset"]:
            prefill_policy.reset_state()
            admission_feedback.reset()
            if tracing:
                flush_trace()
                first_seen.clear()
        if prefill_prepared.mode not in {"decode_cadence", "balanced"}:
            scheduler.prefill_decode_interval = policy.knobs["interleave"]
        return SetInternalStateReqOutput(updated=True)

    scheduler.policy.calc_priority = order
    scheduler.get_new_batch_prefill = prefill
    scheduler.get_internal_state = get_state
    scheduler.set_internal_state = set_state
