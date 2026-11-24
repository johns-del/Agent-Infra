"""Optional bounded prefill/decode interleaving and legacy prefill budgets.

This changes admission granularity, not CUDA kernels.  The engine still owns
allocation, alignment, prefix restoration, speculative decoding and streams.
Only the qualified single-PP, non-DP-attention path is supported.  No rank-local
wall clock or runtime measurement participates in a scheduling decision.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import math


V11_PREFILL_DEFAULTS = {
    "mode": "balanced",
    "decode_chunk_tokens": 2048,
    "decode_min_batch": 24,
    "decode_interval": 3,
    "prefill_burst": 4,
    "max_prefill_burst": 8,
    "max_prefill_streak_with_running": 8,
    "strict_burst_above_min_batch": False,
    "fill_ratio": 0.45,
    "queue_age_s": 3600.0,
}

V12_PREFILL_DEFAULTS = {
    "mode": "balanced",
    "decode_chunk_tokens": 2048,
    "idle_chunk_tokens": 4096,
    "decode_min_batch": 24,
    "decode_interval": 3,
    "high_decode_interval": 5,
    "high_fill_ratio": 0.70,
    "prefill_burst": 4,
    "max_prefill_burst": 10,
    "max_prefill_streak_with_running": 12,
    "max_prefill_tokens_between_decode": 16384,
    "drain_prefill_tokens_cap": 10240,
    "strict_burst_above_min_batch": False,
    "fill_ratio": 0.45,
    "queue_age_s": 3600.0,
}

V13_PREFILL_DEFAULTS = {
    "mode": "balanced",
    "decode_chunk_tokens": 2048,
    "idle_chunk_tokens": 4096,
    "decode_min_batch": 24,
    "decode_interval": 3,
    "high_decode_interval": 5,
    "high_fill_ratio": 0.80,
    "unpressured_fill_ratio": 0.95,
    "min_slots_for_chunk_boost": 64,
    "min_slots_for_high_decode": 64,
    "endgame_waiting_cap": 4,
    "prefill_burst": 4,
    "max_prefill_burst": 10,
    "max_prefill_streak_with_running": 10,
    "max_prefill_tokens_between_decode": 16384,
    "drain_prefill_tokens_cap": 10240,
    "strict_burst_above_min_batch": False,
    "fill_ratio": 0.45,
    "queue_age_s": 3600.0,
}


@dataclass(frozen=True)
class PrefillConfig:
    mode: str = "off"
    decode_chunk_tokens: int = 2048
    decode_min_batch: int = 1
    decode_interval: int = 1
    # balanced is opt-in. Counts are actual launched prefill batches, not polls.
    prefill_burst: int = 4
    max_prefill_burst: int = 16
    max_prefill_streak_with_running: int = 0
    strict_burst_above_min_batch: bool = False
    fill_ratio: float = 0.75
    queue_age_s: float = 5.0
    # V12 First-Principles extensions (all default 0/1.0 inert so V11 configs stay bit-identical)
    idle_chunk_tokens: int = 0
    high_decode_interval: int = 0
    high_fill_ratio: float = 1.0
    max_prefill_tokens_between_decode: int = 0
    drain_prefill_tokens_cap: int = 0
    # V13 Phase-Adaptive Roofline-Little extensions (all default 0/0.0 inert so V11/V12 stay bit-identical)
    unpressured_fill_ratio: float = 0.0
    min_slots_for_chunk_boost: int = 0
    min_slots_for_high_decode: int = 0
    endgame_waiting_cap: int = 0

    def __post_init__(self):
        if self.mode not in {"off", "decode_guard", "decode_cadence", "balanced"}:
            raise ValueError("prefill mode must be off, decode_guard, decode_cadence or balanced")
        for name in ("decode_chunk_tokens", "decode_min_batch", "decode_interval", "prefill_burst", "max_prefill_burst"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")
        for name in (
            "idle_chunk_tokens",
            "max_prefill_tokens_between_decode",
            "drain_prefill_tokens_cap",
            "min_slots_for_chunk_boost",
            "min_slots_for_high_decode",
            "endgame_waiting_cap",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a non-negative integer")
        if type(self.high_decode_interval) is not int or not 0 <= self.high_decode_interval <= 16:
            raise ValueError("high_decode_interval must be an int in [0, 16]")
        if type(self.max_prefill_streak_with_running) is not int or not 0 <= self.max_prefill_streak_with_running <= 64:
            raise ValueError("max_prefill_streak_with_running must be an int in [0, 64]")
        if type(self.strict_burst_above_min_batch) is not bool:
            raise ValueError("strict_burst_above_min_batch must be a bool")
        if not self.prefill_burst <= self.max_prefill_burst <= 64:
            raise ValueError("require prefill_burst <= max_prefill_burst <= 64")
        for name, lower, upper in (
            ("fill_ratio", 0.01, 1.0),
            ("high_fill_ratio", 0.01, 1.0),
            ("unpressured_fill_ratio", 0.0, 1.0),
            ("queue_age_s", 0., 3600.),
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError("invalid " + name)

    @classmethod
    def v11(cls, **overrides):
        return cls(**(V11_PREFILL_DEFAULTS | overrides))

    @classmethod
    def v12(cls, **overrides):
        return cls(**(V12_PREFILL_DEFAULTS | overrides))

    @classmethod
    def v13(cls, **overrides):
        return cls(**(V13_PREFILL_DEFAULTS | overrides))


class PrefillGovernor:
    """Keep native allocation, optionally interleave already runnable decode.

    Default off; balanced counts selected prefill batches and bounds each burst.
    Legacy decode_guard changes chunk size; decode_cadence uses a fixed interval.
    None of these modes is a calibrated latency guarantee or a GPU kernel change.
    """

    def __init__(self, scheduler, *, pp_size=1, attn_dp_size=1, attn_cp_size=1):
        self.scheduler = scheduler
        self.pp_size = pp_size
        self.attn_dp_size = attn_dp_size
        self.attn_cp_size = attn_cp_size
        self.config = PrefillConfig()
        self.counts = Counter()
        self.last_native_tokens = None
        self.last_effective_tokens = None
        self._saved_cadence = None
        self.prefill_streak = 0
        self.prefill_tokens_since_decode = 0
        self.balanced_decode_remaining = 0
        self.last_decision = {}
        native_arm = getattr(scheduler, "_arm_prefill_decode_interval", None)
        if callable(native_arm):
            def observe_arm(batch):
                result = native_arm(batch)
                if self.config.mode == "balanced":
                    if batch is None:
                        self.prefill_tokens_since_decode = 0
                        if self.balanced_decode_remaining > 0:
                            self.balanced_decode_remaining = 0
                            self.prefill_streak = 0
                    else:
                        mode = getattr(batch, "forward_mode", None)
                        is_extend = bool(mode is not None and mode.is_extend())
                        if is_extend:
                            self.prefill_streak += 1
                            ext_lens = getattr(batch, "extend_lens", None)
                            launched_toks = int(sum(ext_lens)) if ext_lens else int(getattr(scheduler, "chunked_prefill_size", 0) or 0)
                            self.prefill_tokens_since_decode += max(0, launched_toks)
                            self.balanced_decode_remaining = 0
                            self.counts["launched_prefill_batches"] += 1
                        elif not getattr(batch, "is_prefill_only", False):
                            self.prefill_streak = 0
                            self.prefill_tokens_since_decode = 0
                            self.balanced_decode_remaining = max(0, self.balanced_decode_remaining - 1)
                            self.counts["launched_decode_batches"] += 1
                return result
            scheduler._arm_prefill_decode_interval = observe_arm
        native_defer = getattr(scheduler, "_should_defer_prefill", None)
        if callable(native_defer):
            def observe_defer(batch):
                if self.config.mode == "balanced":
                    return self._balanced_defer(batch)
                result = native_defer(batch)
                if self.config.mode == "decode_cadence":
                    self.counts["cadence_decisions"] += 1
                    self.counts["native_prefill_deferrals"] += int(result)
                return result
            scheduler._should_defer_prefill = observe_defer

    def _unsupported_reason(self):
        scheduler = self.scheduler
        if self.pp_size != 1 or self.attn_dp_size != 1 or self.attn_cp_size != 1:
            return "requires single PP, attention DP and attention CP"
        if getattr(scheduler, "require_mlp_sync", False):
            return "DP-attention synchronized scheduling is not qualified"
        if getattr(scheduler, "dynamic_chunk_sizer", None) is not None:
            return "native dynamic chunking already owns this budget"
        chunk = getattr(scheduler, "chunked_prefill_size", None)
        if type(chunk) is not int or chunk <= 0:
            return "native chunked prefill is disabled"
        page = getattr(scheduler, "page_size", None)
        if type(page) is not int or page <= 0:
            return "native page alignment is unavailable"
        truncation = getattr(scheduler, "truncation_align_size", None)
        if truncation is not None and (type(truncation) is not int or truncation <= 0):
            return "native truncation alignment is invalid"
        alignment = math.lcm(page, truncation or 1)
        if chunk < alignment:
            return "native chunk is shorter than its alignment"
        mode = getattr(scheduler, "disaggregation_mode", None)
        mode = getattr(mode, "value", mode)
        if mode not in (None, "null", "none"):
            return "prefill/decode disaggregation is not qualified"
        if getattr(scheduler, "is_mixed_chunk", False):
            return "native mixed-chunk budgeting is not qualified"
        return None

    def validate_control(self, value=None):
        if value is None:
            config = self.config
        elif isinstance(value, PrefillConfig):
            config = value
        elif isinstance(value, dict):
            config = PrefillConfig(**value)
        else:
            raise ValueError("prefill control must be an object")
        if config.mode != "off":
            reason = self._unsupported_reason()
            if config.mode in {"decode_cadence", "balanced"} and (not callable(getattr(self.scheduler, "_should_defer_prefill", None))
                    or not callable(getattr(self.scheduler, "_arm_prefill_decode_interval", None))):
                reason = "native prefill/decode cadence hooks are missing"
            if config.mode == "balanced" and (
                    type(getattr(self.scheduler, "max_running_requests", None)) is not int
                    or self.scheduler.max_running_requests <= 0
                    or not callable(getattr(self.scheduler, "get_num_allocatable_reqs", None))
                    or not hasattr(self.scheduler, "orvix_rank_policy")):
                reason = "balanced cadence requires request slots and broadcast rank clock"
            if config.decode_interval > 16:
                reason = "decode cadence interval is outside the qualified sweep range"
            if reason is not None:
                raise ValueError("decode_guard unavailable: " + reason)
        return config

    def configure(self, config):
        config = self.validate_control(config)
        scheduler = self.scheduler
        if config.mode in {"decode_cadence", "balanced"}:
            if self._saved_cadence is None:
                self._saved_cadence = (getattr(scheduler, "prefill_decode_interval", 0),
                                      getattr(scheduler, "prefill_decode_interval_min_bs", 0))
            changed = config != self.config
            scheduler.prefill_decode_interval = config.decode_interval if config.mode == "decode_cadence" else 0
            scheduler.prefill_decode_interval_min_bs = config.decode_min_batch
            if changed:
                scheduler._prefill_decode_interval_remaining = 0
        elif self._saved_cadence is not None:
            scheduler.prefill_decode_interval, scheduler.prefill_decode_interval_min_bs = self._saved_cadence
            scheduler._prefill_decode_interval_remaining = 0
            self._saved_cadence = None
        if config != self.config:
            self.reset_state()
        self.config = config

    def reset_state(self):
        """Reset per-trial control state, preserving cumulative instrumentation."""
        self.prefill_streak = 0
        self.prefill_tokens_since_decode = 0
        self.balanced_decode_remaining = 0
        self.last_decision = {}
        self.scheduler._prefill_decode_interval_remaining = 0

    def _balanced_defer(self, batch):
        # All decisions use identical TP scheduler state and a broadcast clock.
        # Never use a rank-local timer or measured per-rank kernel duration here.
        self.counts["balanced_decisions"] += 1
        n = len(getattr(batch, "reqs", ()))
        waiting = list(self.scheduler.waiting_queue)
        chunk = getattr(self.scheduler, "chunked_req", None)
        reason, defer = "no_pending_prefill", False
        age = None
        max_reqs = max(1, int(getattr(self.scheduler, "max_running_requests", self.config.decode_min_batch) or self.config.decode_min_batch))
        hbm_pressured = bool(getattr(self.scheduler, "_orvix_hbm_pressure_active", False))
        active_chain = bool(getattr(self.scheduler, "_orvix_active_chain_in_running", False))
        eff_fill_ratio = (
            max(self.config.fill_ratio, self.config.unpressured_fill_ratio)
            if (self.config.unpressured_fill_ratio > self.config.fill_ratio and not hbm_pressured and not active_chain)
            else self.config.fill_ratio
        )
        target = math.ceil(max_reqs * eff_fill_ratio)
        high_target = math.ceil(max_reqs * self.config.high_fill_ratio)
        eff_min_batch = (
            self.config.decode_min_batch
            if self.config.strict_burst_above_min_batch
            else min(self.config.decode_min_batch, max(2, math.ceil(max_reqs * 0.20)))
        )
        raw_crit_running = bool(getattr(self.scheduler, "_orvix_critical_in_running", False))
        crit_running = raw_crit_running and (
            self.config.endgame_waiting_cap <= 0 or len(waiting) <= self.config.endgame_waiting_cap
        )
        tok_cap = self.config.max_prefill_tokens_between_decode
        drain_tok_cap = (
            self.config.drain_prefill_tokens_cap
            if (self.config.min_slots_for_chunk_boost <= 0 or max_reqs >= self.config.min_slots_for_chunk_boost)
            else 0
        )
        if n == 0 or getattr(batch, "is_prefill_only", False):
            self.balanced_decode_remaining = 0
            reason = "small_decode_batch"
        elif n < eff_min_batch:
            streak_cap = self.config.max_prefill_streak_with_running
            streak_hit = streak_cap > 0 and self.prefill_streak >= streak_cap
            tok_hit = tok_cap > 0 and self.prefill_tokens_since_decode >= tok_cap
            if (waiting or chunk is not None) and (
                self.balanced_decode_remaining > 0 or streak_hit or tok_hit
            ):
                if self.balanced_decode_remaining == 0:
                    self.balanced_decode_remaining = 1
                reason, defer = "small_batch_streak_guard", True
            else:
                self.balanced_decode_remaining = 0
                reason = "small_decode_batch"
        elif waiting or chunk is not None:
            clock = self.scheduler.orvix_rank_policy.now
            ages = []
            for req in waiting:
                custom = getattr(getattr(req, "sampling_params", None), "custom_params", None)
                meta = custom.get("orvix_unified", {}) if isinstance(custom, dict) else {}
                ready = meta.get("ready_at") if isinstance(meta, dict) else None
                if type(ready) in (int, float) and math.isfinite(ready) and ready >= 0:
                    ages.append(max(0., clock - ready))
            age = max(ages, default=None)
            if self.config.strict_burst_above_min_batch:
                prefer_admission = False
                limit = self.config.prefill_burst
            else:
                slots = self.scheduler.get_num_allocatable_reqs(n, running_batch=batch)
                admit_demand = bool(waiting) and slots > 0 and n < self.scheduler.max_running_requests
                prefer_admission = admit_demand and (not hbm_pressured) and (
                    n < target or age is None or age >= self.config.queue_age_s
                )
                limit = self.config.max_prefill_burst if prefer_admission else self.config.prefill_burst
            if prefer_admission:
                tok_hit = tok_cap > 0 and self.prefill_tokens_since_decode >= tok_cap
                if self.balanced_decode_remaining > 0 or self.prefill_streak >= limit or tok_hit:
                    if self.balanced_decode_remaining == 0:
                        self.balanced_decode_remaining = 1
                    reason, defer = "hard_burst_guard", True
                else:
                    reason = "admission_first"
            else:
                eff_limit = max(1, limit - 1) if (crit_running and not hbm_pressured and limit > 1) else limit
                allow_high_decode = (
                    self.config.high_decode_interval > 0
                    and (self.config.min_slots_for_high_decode <= 0 or max_reqs >= self.config.min_slots_for_high_decode)
                )
                if allow_high_decode and (
                    n >= high_target or (hbm_pressured and n >= target)
                ):
                    dec_int = self.config.high_decode_interval
                elif crit_running:
                    dec_int = min(max(5, self.config.high_decode_interval or 5), self.config.decode_interval + 1)
                else:
                    dec_int = self.config.decode_interval
                tok_hit = drain_tok_cap > 0 and self.prefill_tokens_since_decode >= drain_tok_cap
                if self.balanced_decode_remaining > 0 or self.prefill_streak >= eff_limit or tok_hit:
                    if self.balanced_decode_remaining == 0:
                        self.balanced_decode_remaining = dec_int
                    reason, defer = "decode_burst_guard", True
                else:
                    reason = "prefill_burst"
        else:
            self.balanced_decode_remaining = 0
        self.counts["balanced_" + reason] += 1
        self.counts["balanced_prefill_deferrals"] += int(defer)
        self.last_decision = {
            "reason": reason,
            "defer_prefill": defer,
            "decode_batch": n,
            "waiting_requests": len(waiting),
            "oldest_wait_s": age,
            "prefill_streak": self.prefill_streak,
            "prefill_tokens_since_decode": self.prefill_tokens_since_decode,
            "fill_target": target,
            "hbm_pressured": hbm_pressured,
            "critical_in_running": crit_running,
        }
        return defer

    def planned_budget(self, running_batch):
        """Return the next admission cap without updating metrics or state."""
        native = self.scheduler.chunked_prefill_size
        if self.config.mode in {"off", "decode_cadence"}:
            return native
        if self.config.mode == "balanced":
            if self.config.idle_chunk_tokens <= 0 or self._unsupported_reason() is not None:
                return native
            requests = getattr(running_batch, "reqs", ())
            n = len(requests)
            max_reqs = max(1, int(getattr(self.scheduler, "max_running_requests", self.config.decode_min_batch) or self.config.decode_min_batch))
            eff_min_batch = (
                self.config.decode_min_batch
                if self.config.strict_burst_above_min_batch
                else min(self.config.decode_min_batch, max(2, math.ceil(max_reqs * 0.20)))
            )
            hbm_pressured = bool(getattr(self.scheduler, "_orvix_hbm_pressure_active", False))
            active_chain = bool(getattr(self.scheduler, "_orvix_active_chain_in_running", False))
            eff_fill_ratio = (
                max(self.config.fill_ratio, self.config.unpressured_fill_ratio)
                if (self.config.unpressured_fill_ratio > self.config.fill_ratio and not hbm_pressured and not active_chain)
                else self.config.fill_ratio
            )
            target = math.ceil(max_reqs * eff_fill_ratio)
            unpressured_phase1 = (
                self.config.unpressured_fill_ratio > self.config.fill_ratio
                and not hbm_pressured
                and not active_chain
                and n < target
            )
            alignment = math.lcm(self.scheduler.page_size, getattr(self.scheduler, "truncation_align_size", None) or 1)
            idle_cap = max(alignment, (self.config.idle_chunk_tokens // alignment) * alignment)
            max_pref = getattr(self.scheduler, "max_prefill_tokens", None)
            if type(max_pref) is int and max_pref > 0:
                idle_cap = min(idle_cap, max_pref)
            tok_cap = self.config.max_prefill_tokens_between_decode
            within_tok_cap = (tok_cap <= 0) or (self.prefill_tokens_since_decode + idle_cap <= tok_cap)
            allow_ramp_boost = (
                self.config.min_slots_for_chunk_boost <= 0
                or max_reqs >= self.config.min_slots_for_chunk_boost
            )
            if n == 0 or (
                allow_ramp_boost
                and not getattr(running_batch, "is_prefill_only", False)
                and (n < eff_min_batch or unpressured_phase1)
                and not hbm_pressured
                and within_tok_cap
            ):
                return max(native, idle_cap)
            return native
        reason = self._unsupported_reason()
        if reason is not None:
            return native
        requests = getattr(running_batch, "reqs", ())
        decoding = len(requests)
        if (getattr(running_batch, "is_prefill_only", False)
                or decoding < self.config.decode_min_batch):
            return native
        alignment = math.lcm(self.scheduler.page_size,
                             getattr(self.scheduler, "truncation_align_size", None) or 1)
        # Never make the budget zero: native truncation would otherwise prevent
        # a long request from making any progress.
        cap = max(alignment, (self.config.decode_chunk_tokens // alignment) * alignment)
        return min(native, cap)

    def budget(self, running_batch):
        native = self.scheduler.chunked_prefill_size
        effective = self.planned_budget(running_batch)
        self.last_native_tokens = native
        self.last_effective_tokens = effective
        if self.config.mode in {"off", "decode_cadence"}:
            return effective
        if self.config.mode == "balanced":
            if self.config.idle_chunk_tokens > 0:
                if self._unsupported_reason() is not None:
                    self.counts["unsupported_fallbacks"] += 1
                elif effective > native:
                    self.counts["idle_boosted_cycles"] += 1
            return effective
        if self._unsupported_reason() is not None:
            # Another native control plane may have changed the configuration.
            self.counts["unsupported_fallbacks"] += 1
            return effective
        if (getattr(running_batch, "is_prefill_only", False)
                or len(getattr(running_batch, "reqs", ())) < self.config.decode_min_batch):
            self.counts["unrestricted_cycles"] += 1
            return effective
        if effective < native:
            self.counts["guarded_cycles"] += 1
            self.counts["budget_tokens_removed"] += native - effective
        else:
            self.counts["already_within_cap_cycles"] += 1
        return effective

    def call(self, original, running_batch, *args, **kwargs):
        """Wrap one synchronous get_new_batch_prefill call; restore on errors."""
        self.counts["cycles"] += 1
        native = self.scheduler.chunked_prefill_size
        if self.config.mode in {"off", "decode_cadence"} or (
            self.config.mode == "balanced" and self.config.idle_chunk_tokens <= 0
        ):
            self.last_native_tokens = self.last_effective_tokens = native
            result = original(running_batch, *args, **kwargs)
            if getattr(result, "batch_to_run", None) is not None:
                self.counts["prefill_batches"] += 1
            return result
        prev_base = getattr(self.scheduler, "_orvix_base_chunk_size", None)
        self.scheduler._orvix_base_chunk_size = native
        self.scheduler.chunked_prefill_size = self.budget(running_batch)
        try:
            result = original(running_batch, *args, **kwargs)
            batch = getattr(result, "batch_to_run", None)
            if batch is not None:
                self.counts["prefill_batches"] += 1
            return result
        finally:
            self.scheduler.chunked_prefill_size = native
            if prev_base is None:
                if hasattr(self.scheduler, "_orvix_base_chunk_size"):
                    delattr(self.scheduler, "_orvix_base_chunk_size")
            else:
                self.scheduler._orvix_base_chunk_size = prev_base

    def snapshot(self):
        return {"config": asdict(self.config), "counts": dict(self.counts),
                "native_chunk_tokens": self.last_native_tokens,
                "effective_chunk_tokens": self.last_effective_tokens,
                "unavailable_reason": self._unsupported_reason(),
                "native_decode_interval": getattr(self.scheduler, "prefill_decode_interval", 0),
                "native_decode_min_batch": getattr(self.scheduler, "prefill_decode_interval_min_bs", 0),
                "last_decision": self.last_decision,
                "scope": "native_budget_and_bounded_prefill_decode_interleaving"}
