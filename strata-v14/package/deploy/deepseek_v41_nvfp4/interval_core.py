"""Causal, bounded policy for tool-interval cache management (no torch imports).

Only completed observations train the history. No ETA, recording duration,
future output, or offline single-turn label is a scheduling input.
"""
from __future__ import annotations
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, asdict, field
import math

VERSION = 'orvix-interval-cache-v7'
IMPLEMENTATION = 'orvix-interval-cache-v13-phase-adaptive'
KINDS = {'tool_start', 'tool_end', 'request_ready', 'workflow_end', 'session_close'}
STATUSES = {'ok', 'error', 'cancelled', 'timeout'}

V11_INTERVAL_DEFAULTS = {
    'mode': 'active',
    'soft_keep': True,
    'eager_backup': True,
    'demand_release': False,
    'queue_prefetch': False,
    'tool_end_prefetch': False,
    'countdown_prefetch': False,
    'streaming_prefetch': False,
    'ready_order': True,
    'tier_signal_only_order': True,
    'prefetch_only_reorder': False,
    'bimodal_critical_path_order': True,
    'critical_turn_threshold': 1,
    'critical_max_missing_chunks': 4,
    'lane1_short_max_chunks': 2,
    'max_queue_age_s': 300.0,
    'block_unrelated_during_prefetch': False,
    'pressure_gated_backup': True,
    'backup_free_fraction': 0.25,
    'proactive_release': True,
    'release_free_fraction': 0.10,
    'hbm_pressure_free_fraction': 0.35,
    'swa_tail_watermark': 0.65,
    'reserve_fraction': 0.02,
    'growth_reserve_tokens': 2048,
    'streaming_backup': True,
    'backup_chunk_tokens': 16384,
    'host_swa_reclaim': True,
    'keep_s': 1.0,
    'keep_max_s': 3.0,
    'tool_expire_s': 120.0,
    'backup_after_s': 0.25,
    'release_min_remaining_s': 3.0,
    'tick_s': 0.1,
    'age_guard_s': 10.0,
    'ready_lease_s': 6.0,
    'anchor_lease_s': 8.0,
    'prefetch_lead_s': 2.5,
    'min_host_tokens': 256,
    'max_transfer_tokens': 262144,
    'max_keep_tokens': 2097152,
    'per_user_keep_tokens': 262144,
    'max_keep_fraction': 0.35,
    'prefetch_window': 8,
    'max_inflight': 4,
    'scan_limit': 256,
    'max_actions': 32,
    'max_records': 16384,
    'history_size': 256,
    'history_min_samples': 4,
    'history_max_types': 128,
    'max_events': 131072,
    'max_tools': 128,
    'transfer_burst_bytes': 4294967296,
    'transfer_rate_bytes_s': 4294967296,
}

V12_INTERVAL_DEFAULTS = V11_INTERVAL_DEFAULTS | {
    'dead_leaf_bypass': True,
    'allow_host_kv_reclaim': True,
    'hrrn_lane1_order': True,
    'hrrn_tau_s': 4.0,
    'hrrn_age_cap_s': 90.0,
    'critical_in_batch_fold': True,
    'critical_fold_turn_threshold': 8,
}

V13_INTERVAL_DEFAULTS = V12_INTERVAL_DEFAULTS | {
    'hrrn_lane1_order': False,
    'hrrn_age_cap_s': 30.0,
    'superseded_drop_requires_full_pressure': True,
    'closed_immediate_drop': True,
    'backed_only_cold_demote': True,
    'superseded_keep_s': 30.0,
    'lane0_log_aging': True,
    'lane0_depth_cap': 24,
    'lane0_age_quantum_s': 8.0,
    'lane0_max_age_bonus': 10.0,
    'phase_chain_turn_threshold': 2,
}

OURS_INTERVAL_DEFAULTS = V13_INTERVAL_DEFAULTS | {
    'crowd_aware_order': True,
    'turn0_ttft_guard_s': 45.0,
}


@dataclass(frozen=True)
class IntervalConfig:
    mode: str = 'off'                 # off / observe / active
    soft_keep: bool = True
    eager_backup: bool = True
    demand_release: bool = True
    queue_prefetch: bool = True
    ready_order: bool = True
    tier_signal_only_order: bool = False
    # V9/V10 active closed-loop extensions (safe defaults preserve V7/V8 unit-test contracts)
    streaming_backup: bool = True
    backup_chunk_tokens: int = 16384
    host_swa_reclaim: bool = True
    proactive_release: bool = False
    release_min_remaining_s: float = 3.0
    swa_tail_watermark: float = 0.70
    release_free_fraction: float = 0.20
    hbm_pressure_free_fraction: float = 0.0
    pressure_gated_backup: bool = False
    backup_free_fraction: float = 0.25
    block_unrelated_during_prefetch: bool = True
    prefetch_only_reorder: bool = False
    bimodal_critical_path_order: bool = False
    critical_turn_threshold: int = 2
    critical_max_missing_chunks: int = 256
    lane1_short_max_chunks: int = 2
    max_queue_age_s: float = 90.0
    streaming_prefetch: bool = False
    tool_end_prefetch: bool = True
    countdown_prefetch: bool = True
    prefetch_lead_s: float = 2.0
    anchor_lease_s: float = 6.0
    prefetch_during_long_continuation: bool = False
    allow_full_slot_prefetch: bool = False
    keep_s: float = 1.0
    keep_max_s: float = 3.0
    tool_expire_s: float = 60.0
    backup_after_s: float = 0.5
    tick_s: float = 0.1
    age_guard_s: float = 10.0
    ready_lease_s: float = 1.0
    min_host_tokens: int = 1024
    max_transfer_tokens: int = 65536
    max_keep_tokens: int = 65536
    per_user_keep_tokens: int = 32768
    max_keep_fraction: float = 0.15
    reserve_fraction: float = 0.10
    growth_reserve_tokens: int = 2048
    prefetch_window: int = 2
    max_inflight: int = 1
    max_actions: int = 2
    scan_limit: int = 32
    filler_max_chunks: int = 1
    prefetch_max_remaining_chunks: int = 1
    max_bypass: int = 4
    service_slack: float = 16384.0
    output_weight: float = 32.0
    restore_cost_weight: float = 0.25  # uncalibrated ordering proxy, NOT transfer bandwidth
    max_records: int = 4096
    history_size: int = 128
    history_min_samples: int = 8
    history_max_types: int = 128
    max_events: int = 32768
    max_tools: int = 128
    # V12 First-Principles extensions (all default False / inert to preserve V7..V11 backward compatibility)
    dead_leaf_bypass: bool = False
    allow_host_kv_reclaim: bool = False
    hrrn_lane1_order: bool = False
    hrrn_tau_s: float = 4.0
    hrrn_age_cap_s: float = 90.0
    critical_in_batch_fold: bool = False
    critical_fold_turn_threshold: int = 8
    # V13 Phase-Adaptive extensions (all default False / 0 inert to preserve V7..V12 backward compatibility)
    superseded_drop_requires_full_pressure: bool = False
    closed_immediate_drop: bool = False
    backed_only_cold_demote: bool = False
    superseded_keep_s: float = 0.0
    lane0_log_aging: bool = False
    lane0_depth_cap: int = 0
    lane0_age_quantum_s: float = 8.0
    lane0_max_age_bonus: float = 0.0
    phase_chain_turn_threshold: int = 0
    # 3-Pillar Unified extensions (default False / 45.0 inert to preserve V7..V13 backward compatibility)
    crowd_aware_order: bool = False
    turn0_ttft_guard_s: float = 45.0
    # Published external baseline ordering modes for same-engine hardware/CPU evaluation
    plas_program_order: bool = False
    continuum_program_fcfs: bool = False
    # Limits, NOT a measured transfer speed. Includes sidecar upper bounds.
    transfer_burst_bytes: int = 268435456
    transfer_rate_bytes_s: int = 268435456

    def validate(self):
        if self.mode not in {'off', 'observe', 'active'}:
            raise ValueError('invalid interval mode')
        for key in (
            'soft_keep',
            'eager_backup',
            'demand_release',
            'queue_prefetch',
            'ready_order',
            'tier_signal_only_order',
            'streaming_backup',
            'host_swa_reclaim',
            'proactive_release',
            'pressure_gated_backup',
            'block_unrelated_during_prefetch',
            'prefetch_only_reorder',
            'bimodal_critical_path_order',
            'streaming_prefetch',
            'tool_end_prefetch',
            'countdown_prefetch',
            'prefetch_during_long_continuation',
            'allow_full_slot_prefetch',
            'dead_leaf_bypass',
            'allow_host_kv_reclaim',
            'hrrn_lane1_order',
            'critical_in_batch_fold',
            'superseded_drop_requires_full_pressure',
            'closed_immediate_drop',
            'backed_only_cold_demote',
            'lane0_log_aging',
            'crowd_aware_order',
            'plas_program_order',
            'continuum_program_fcfs',
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(key + ' must be bool')
        ints = {
            'min_host_tokens': (1, 1048576),
            'max_transfer_tokens': (1, 4194304),
            'backup_chunk_tokens': (1, 4194304),
            'max_keep_tokens': (0, 16777216),
            'per_user_keep_tokens': (0, 4194304),
            'growth_reserve_tokens': (1, 1048576),
            'prefetch_window': (1, 32),
            'max_inflight': (1, 16),
            'max_actions': (1, 64),
            'scan_limit': (1, 256),
            'filler_max_chunks': (1, 16),
            'prefetch_max_remaining_chunks': (1, 256),
            'critical_turn_threshold': (1, 64),
            'critical_max_missing_chunks': (1, 1024),
            'lane1_short_max_chunks': (0, 1024),
            'critical_fold_turn_threshold': (1, 128),
            'lane0_depth_cap': (0, 512),
            'phase_chain_turn_threshold': (0, 128),
            'max_bypass': (1, 100),
            'max_records': (1, 32768),
            'history_size': (1, 4096),
            'history_min_samples': (1, 512),
            'history_max_types': (1, 1024),
            'max_events': (1, 262144),
            'max_tools': (1, 1024),
            'transfer_burst_bytes': (1, 1 << 36),
            'transfer_rate_bytes_s': (1, 1 << 36),
        }
        for key, (lo, hi) in ints.items():
            v = getattr(self, key)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError('invalid ' + key)
        for key in (
            'keep_s',
            'keep_max_s',
            'tool_expire_s',
            'backup_after_s',
            'tick_s',
            'age_guard_s',
            'max_queue_age_s',
            'hrrn_tau_s',
            'hrrn_age_cap_s',
            'superseded_keep_s',
            'lane0_age_quantum_s',
            'lane0_max_age_bonus',
            'turn0_ttft_guard_s',
            'ready_lease_s',
            'anchor_lease_s',
            'prefetch_lead_s',
            'release_min_remaining_s',
            'swa_tail_watermark',
            'release_free_fraction',
            'hbm_pressure_free_fraction',
            'backup_free_fraction',
            'max_keep_fraction',
            'reserve_fraction',
            'service_slack',
            'output_weight',
            'restore_cost_weight',
        ):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError('invalid ' + key)
        if not 0 <= self.keep_s <= self.keep_max_s <= self.tool_expire_s <= 3600:
            raise ValueError('invalid lease range')
        if (
            not 0.01 <= self.tick_s <= 10
            or not 0 < self.age_guard_s <= 3600
            or not 0 < self.max_queue_age_s <= 3600
            or not 0.0 <= self.hrrn_tau_s <= 600.0
            or not 0.0 < self.hrrn_age_cap_s <= 3600.0
            or not 0.1 <= self.lane0_age_quantum_s <= 3600.0
            or not 0.0 <= self.turn0_ttft_guard_s <= 3600.0
        ):
            raise ValueError('invalid clock bounds')
        if not 0 < self.ready_lease_s <= 60 or not 0 < self.anchor_lease_s <= 60 or not 0 < self.output_weight <= 4096:
            raise ValueError('invalid cost/lease')
        if not 0 <= self.max_keep_fraction <= 0.65 or not 0.02 <= self.reserve_fraction <= 0.5:
            raise ValueError('invalid watermarks')
        if (
            not 0.05 <= self.swa_tail_watermark <= 1.0
            or not 0.0 <= self.release_free_fraction <= 0.9
            or not 0.0 <= self.hbm_pressure_free_fraction <= 0.9
            or not 0.0 <= self.backup_free_fraction <= 0.9
        ):
            raise ValueError('invalid release watermarks')
        if not 0.0 <= self.prefetch_lead_s <= 60.0 or not 0.0 <= self.release_min_remaining_s <= 3600.0:
            raise ValueError('invalid predictor thresholds')
        return self

    @classmethod
    def v11(cls, **overrides):
        return cls(**(V11_INTERVAL_DEFAULTS | overrides)).validate()

    @classmethod
    def v12(cls, **overrides):
        return cls(**(V12_INTERVAL_DEFAULTS | overrides)).validate()

    @classmethod
    def v13(cls, **overrides):
        return cls(**(V13_INTERVAL_DEFAULTS | overrides)).validate()

    @classmethod
    def ours(cls, **overrides):
        return cls(**(OURS_INTERVAL_DEFAULTS | overrides)).validate()


def validate_events(events, now):
    if not isinstance(events, list) or len(events) > 4096:
        raise ValueError('bounded interval_events list required')
    out = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('invalid event')
        if set(event) - {'kind', 'event_id', 'request_id', 'tool_id', 'tool_type', 'status', 'observed_at'}:
            raise ValueError('unknown/noncausal lifecycle fields')
        kind = event.get('kind')
        if kind not in KINDS:
            raise ValueError('unknown lifecycle kind')
        for k in ('event_id', 'request_id'):
            if not isinstance(event.get(k), str) or not 0 < len(event[k]) <= 512:
                raise ValueError('invalid ' + k)
        if kind in {'tool_start', 'tool_end'}:
            if not isinstance(event.get('tool_id'), str) or not 0 < len(event['tool_id']) <= 512:
                raise ValueError('tool_id required')
        if kind == 'tool_start':
            if not isinstance(event.get('tool_type'), str) or not 0 < len(event['tool_type']) <= 128:
                raise ValueError('causal tool_type required')
        if kind == 'tool_end' and event.get('status') not in STATUSES:
            raise ValueError('terminal tool status required')
        t = event.get('observed_at', now)
        if isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or not 0 <= t <= now:
            raise ValueError('invalid observation clock')
        out.append(dict(event, observed_at=float(t)))
    return out


@dataclass
class Tool:
    tool_type: str
    start: float


@dataclass
class Interval:
    rid: str
    user: str
    tools: dict[str, Tool] = field(default_factory=dict)
    ended: set[str] = field(default_factory=set)
    ready: bool = False
    closed: bool = False
    superseded: bool = False
    last_event: float = 0.0
    tools_all_ended_at: float | None = None
    anchor_ready_at: float | None = None


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, q)) * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


class Lifecycle:
    def __init__(self, config: IntervalConfig):
        self.config = config
        self.records = OrderedDict()
        self.seen = OrderedDict()
        self.history = OrderedDict()
        self.counts = Counter()
        self.overflow = False

    def register(self, rid, user):
        if rid in self.records:
            return self.records[rid]
        if len(self.records) >= self.config.max_records:
            self.overflow = True
            self.counts['record_overflow'] += 1
            return None
        self.records[rid] = Interval(rid, user)
        return self.records[rid]

    def event(self, event, known_user=None):
        eid = event['event_id']
        rid = event['request_id']
        t = event['observed_at']
        kind = event['kind']
        if eid in self.seen:
            self.counts['duplicates'] += 1
            return
        self.seen[eid] = None
        if len(self.seen) > self.config.max_events:
            self.seen.popitem(last=False)
        record = self.records.get(rid)
        if record is None:
            # Tombstone unknown end/ready/start. A missing completed checkpoint
            # must never become permission to act retroactively.
            record = self.register(rid, known_user or '__unknown__')
            if record is not None:
                record.ready = True
            self.counts['unknown_request_events'] += 1
            return
        record.last_event = max(t, record.last_event)
        self.counts[kind] += 1
        if kind in {'workflow_end', 'session_close', 'request_ready'}:
            record.ready = True
            record.closed = kind != 'request_ready'
            record.tools.clear()
            record.tools_all_ended_at = None
            if kind == 'request_ready':
                record.anchor_ready_at = float(t)
            else:
                record.anchor_ready_at = None
            return
        tid = event['tool_id']
        if kind == 'tool_end':
            tool = record.tools.pop(tid, None)
            if tool is not None and t >= tool.start:
                self.observe(tool.tool_type, event['status'], t - tool.start)
            if len(record.ended) < self.config.max_tools:
                record.ended.add(tid)
                if not record.tools and not record.ready and not record.closed:
                    record.tools_all_ended_at = float(t)
                    record.anchor_ready_at = float(t)
            else:
                record.ready = True
                record.tools.clear()
                record.tools_all_ended_at = None
                record.anchor_ready_at = None
            return
        if record.ready or record.closed or tid in record.ended:
            self.counts['late_start'] += 1
            return
        if len(record.tools) >= self.config.max_tools:
            record.ready = True
            record.tools.clear()
            record.tools_all_ended_at = None
            record.anchor_ready_at = None
            self.counts['tool_overflow'] += 1
            return
        record.tools.setdefault(tid, Tool(event['tool_type'], t))
        record.tools_all_ended_at = None
        record.anchor_ready_at = None

    def observe(self, tool_type, status, duration):
        # Include all observed terminal outcomes for start-time decisions;
        # outcome-separated buckets remain inspectable. Never select "ok"
        # at start based on the future outcome of this call.
        if tool_type not in self.history:
            if len(self.history) >= self.config.history_max_types:
                self.history.popitem(last=False)
            self.history[tool_type] = deque(maxlen=self.config.history_size)
        self.history[tool_type].append((status, float(duration)))

    def keep_until(self, record, now):
        if record.ready or not record.tools:
            return 0.0
        # Parallel tools: all must finish, so protect until the latest estimate.
        ends = []
        for tool in record.tools.values():
            history = [d for _, d in self.history.get(tool.tool_type, ())]
            keep = self.config.keep_s
            if len(history) >= self.config.history_min_samples:
                ordered = sorted(history)
                keep = ordered[int((len(ordered) - 1) * 0.8)]
            keep = min(self.config.keep_max_s, max(0.0, keep))
            ends.append(tool.start + keep)
        return min(max(ends), min(t.start for t in record.tools.values()) + self.config.tool_expire_s)

    def residual_survival(self, tool_type, elapsed, horizon):
        xs = [d for _, d in self.history.get(tool_type, ()) if d > elapsed]
        # Missing tail samples are unknown, not zero remaining time.
        if len(xs) < self.config.history_min_samples:
            return None
        return sum(d > elapsed + horizon for d in xs) / len(xs)

    def conditional_remaining_quantiles(self, tool_type: str, elapsed: float):
        """Online causal quantiles of (T - elapsed | T > elapsed) for tool_type.

        Uses only completed historical tool executions; never peeks at future
        trace durations. Returns None when history is insufficient or elapsed
        exceeds all observed samples (tail is censored, not zero remaining).
        """
        elapsed = max(0.0, float(elapsed))
        all_d = [d for _, d in self.history.get(tool_type, ())]
        min_total = self.config.history_min_samples
        if len(all_d) < min_total:
            return None
        rem = sorted(d - elapsed for d in all_d if d > elapsed)
        if not rem:
            return None
        min_surv = max(2, min_total // 2)
        return {
            'p20': _quantile(rem, 0.20),
            'p50': _quantile(rem, 0.50),
            'p80': _quantile(rem, 0.80),
            'survival_prob': len(rem) / len(all_d),
            'n_survivors': len(rem),
            'n_total': len(all_d),
            'confident': len(rem) >= min_surv,
        }

    def predict_remaining(self, record: Interval, now: float):
        """Predict barrier conditional remaining wait across active parallel tools."""
        if record.ready or record.closed or not record.tools:
            return None
        p20s, p50s, p80s = [], [], []
        max_elapsed = 0.0
        confident = True
        for tool in record.tools.values():
            elapsed = max(0.0, float(now) - tool.start)
            max_elapsed = max(max_elapsed, elapsed)
            q = self.conditional_remaining_quantiles(tool.tool_type, elapsed)
            if q is None:
                return None
            p20s.append(q['p20'])
            p50s.append(q['p50'])
            p80s.append(q['p80'])
            confident = confident and q['confident']
        return {
            'remaining_p20_s': max(p20s),
            'remaining_p50_s': max(p50s),
            'remaining_p80_s': max(p80s),
            'max_elapsed_s': max_elapsed,
            'confident': confident,
        }

    def expire(self, now):
        for r in self.records.values():
            if r.tools and now - min(t.start for t in r.tools.values()) >= self.config.tool_expire_s:
                # Expiration is censored; do not train a fictitious completion.
                r.tools.clear()
                r.tools_all_ended_at = None
                r.anchor_ready_at = None
                r.ready = True
                self.counts['expired'] += 1


class ByteBucket:
    def __init__(self, capacity, rate):
        self.capacity = capacity
        self.rate = rate
        self.tokens = float(capacity)
        self.now = 0.0

    def allow(self, n, now):
        if now < self.now or n < 0:
            raise ValueError('invalid token-bucket clock/size')
        self.tokens = min(self.capacity, self.tokens + (now - self.now) * self.rate)
        self.now = now
        return n <= self.tokens

    def consume(self, n, now):
        if not self.allow(n, now):
            return False
        self.tokens -= n
        return True


@dataclass(frozen=True)
class QueueItem:
    rid: str
    user: str
    ready_at: float
    gpu: int
    host: int
    missing: int
    pending: bool = False
    prefetched: bool = False
    completed_turns: int = 0
    arrival_seq: int = 0
    program_service_tokens: float = 0.0
    program_arrival_at: float = -1.0


def order_window(items, now, config, service, bypass, chunk_tokens, *, running_count=0, max_slots=64):
    """Bounded reordering, not whole-request length prediction or CPU=miss.

    Only called on a small FIFO window. Pending requests stay represented in
    the plan, but cannot be launched; their age is never reset.
    """
    items = list(items)
    debt = dict(service)
    pending = any(x.pending for x in items)
    protected = next(
        (x for x in items if now - x.ready_at >= config.age_guard_s or bypass.get(x.rid, 0) >= config.max_bypass),
        None,
    )
    eligible = [x for x in items if not x.pending]
    if pending and config.block_unrelated_during_prefetch:
        eligible = [x for x in eligible if x.missing <= chunk_tokens * config.filler_max_chunks and x.host <= x.gpu]

    if config.crowd_aware_order:
        step_quantum = max(1, int(chunk_tokens))
        deep_threshold = max(int(config.critical_turn_threshold) + 1, 3)
        q0_tok = max(1.0, float(chunk_tokens) * 2.0)
        rate_tok_s = max(1.0, float(chunk_tokens) * 4.0)
        congested = (int(running_count) + len(eligible)) >= max(8, int(0.70 * max(1, int(max_slots))))
        lane1_age_cap = min(float(config.max_queue_age_s), float(config.hrrn_age_cap_s))

        def _compute_lane0_rank_ca(x: QueueItem, age: float) -> float:
            if not config.lane0_log_aging:
                return float(-x.completed_turns)
            capped_age = min(max(0.0, age), float(config.max_queue_age_s))
            eff_turns = (
                min(float(x.completed_turns), float(config.lane0_depth_cap))
                if config.lane0_depth_cap > 0
                else math.log2(1.0 + float(x.completed_turns))
            )
            age_bonus = capped_age / float(config.lane0_age_quantum_s)
            if config.lane0_max_age_bonus > 0.0:
                age_bonus = min(age_bonus, float(config.lane0_max_age_bonus))
            return -round(eff_turns + age_bonus, 6)

        def _crowd_aware_key(x: QueueItem):
            age = max(0.0, now - x.ready_at)
            missing_chunks = (max(0, x.missing) + step_quantum - 1) // step_quantum
            if age >= config.max_queue_age_s:
                return (0, 0, 0.0, 0, x.ready_at, x.arrival_seq, x.rid)
            if x.completed_turns == 0 and age >= config.turn0_ttft_guard_s:
                return (1, 0, 0.0, 0, x.ready_at, x.arrival_seq, x.rid)
            if congested:
                s_tok = max(0.0, float(x.program_service_tokens))
                s_sec_est = max(0.5, s_tok / rate_tok_s)
                if s_tok > 0.0 and (age / s_sec_est) >= 5.0:
                    lvl = 0
                else:
                    lvl = int(math.floor(math.log2(s_tok / q0_tok + 1.0)))
                return (2, lvl, float(missing_chunks), s_tok, x.ready_at, x.arrival_seq, x.rid)
            if (
                x.completed_turns >= config.critical_turn_threshold
                and missing_chunks <= config.critical_max_missing_chunks
            ) or x.completed_turns >= deep_threshold:
                return (2, 0, _compute_lane0_rank_ca(x, age), missing_chunks, x.ready_at, x.arrival_seq, x.rid)
            if age >= lane1_age_cap:
                return (2, 1, 0.0, 0, x.ready_at, x.arrival_seq, x.rid)
            if missing_chunks <= config.lane1_short_max_chunks:
                return (2, 2, float(missing_chunks), 0, x.ready_at, x.arrival_seq, x.rid)
            return (2, 3, 0.0, 0, x.ready_at, x.arrival_seq, x.rid)

        return sorted(eligible, key=_crowd_aware_key)

    if config.plas_program_order:
        # Agentix / Autellix PLAS (Luo et al., USENIX NSDI '26):
        # Program-level cumulative GPU service time + exponential MLFQ + beta=5.0 starvation promotion
        q0_tok = max(1.0, float(chunk_tokens) * 2.0)
        rate_tok_s = max(1.0, float(chunk_tokens) * 4.0)

        def _plas_key(x: QueueItem):
            wait_s = max(0.0, now - x.ready_at)
            if wait_s >= config.max_queue_age_s:
                return (0, 0, 0.0, x.ready_at, x.arrival_seq, x.rid)
            s_tok = max(0.0, float(x.program_service_tokens))
            s_sec_est = max(0.5, s_tok / rate_tok_s)
            if s_tok > 0.0 and (wait_s / s_sec_est) >= 5.0:
                lvl = 0
            else:
                lvl = int(math.floor(math.log2(s_tok / q0_tok + 1.0)))
            return (1, lvl, s_tok, x.ready_at, x.arrival_seq, x.rid)

        return sorted(eligible, key=_plas_key)

    if config.continuum_program_fcfs:
        # Continuum (Li et al., ICLR '26): Program-Arrival Continuity FCFS + anti-starvation ceiling
        def _continuum_key(x: QueueItem):
            wait_s = max(0.0, now - x.ready_at)
            if wait_s >= config.max_queue_age_s:
                return (0, 0.0, x.ready_at, x.arrival_seq, x.rid)
            prog_t0 = x.program_arrival_at if x.program_arrival_at >= 0.0 else x.ready_at
            return (1, prog_t0, x.ready_at, x.arrival_seq, x.rid)

        return sorted(eligible, key=_continuum_key)

    if config.bimodal_critical_path_order:
        # PACER / V11..V13 Online Bimodal Critical-Path & Warm-Guarded Ordering:
        #   Tier 0 (starvation_tier = 0): queue wait >= max_queue_age_s -> unconditional FIFO anti-starvation ceiling
        #   Tier 1, Lane 0: active multi-turn serial chains (`completed_turns >= critical_turn_threshold` and warm,
        #     OR `completed_turns >= deep_threshold`), ordered by D*-capped bounded-horizon critical-path index:
        #       I_L0(i, t) = min(d_i, D*) + min(w_i(t) / tau_L0, Delta_age*)
        #   Tier 1, Lane 1:
        #     - When `hrrn_lane1_order=False`: `hrrn_age_cap_s`-guarded FIFO Sublane 1 + 2-chunk warm-filler Sublane 2 + FIFO Sublane 3
        #     - When `hrrn_lane1_order=True`: `hrrn_age_cap_s`-guarded FIFO Sublane 1 + sqrt(missing_chunks) HRRN Sublane 2
        step_quantum = max(1, int(chunk_tokens))
        deep_threshold = max(int(config.critical_turn_threshold) + 1, 3)

        def _compute_lane0_rank(x: QueueItem, age: float) -> float:
            if not config.lane0_log_aging:
                return float(-x.completed_turns)
            capped_age = min(max(0.0, age), float(config.max_queue_age_s))
            eff_turns = (
                min(float(x.completed_turns), float(config.lane0_depth_cap))
                if config.lane0_depth_cap > 0
                else math.log2(1.0 + float(x.completed_turns))
            )
            age_bonus = capped_age / float(config.lane0_age_quantum_s)
            if config.lane0_max_age_bonus > 0.0:
                age_bonus = min(age_bonus, float(config.lane0_max_age_bonus))
            return -round(eff_turns + age_bonus, 6)

        if config.hrrn_lane1_order:
            lane1_age_cap = min(float(config.max_queue_age_s), float(config.hrrn_age_cap_s))
            tau = float(config.hrrn_tau_s)

            def _v12_bimodal_hrrn_key(x: QueueItem):
                age = max(0.0, now - x.ready_at)
                missing_chunks = (max(0, x.missing) + step_quantum - 1) // step_quantum
                if age >= config.max_queue_age_s:
                    return (0, 0.0, 0.0, x.ready_at, x.arrival_seq, x.rid)
                if (
                    x.completed_turns >= config.critical_turn_threshold
                    and missing_chunks <= config.critical_max_missing_chunks
                ) or x.completed_turns >= deep_threshold:
                    return (1, _compute_lane0_rank(x, age), float(missing_chunks), x.ready_at, x.arrival_seq, x.rid)
                if age >= lane1_age_cap:
                    return (2, 0.0, 0.0, x.ready_at, x.arrival_seq, x.rid)
                eff_c = math.sqrt(max(0.5, float(missing_chunks)))
                hrrn = (age + tau) / eff_c
                return (3, -round(hrrn, 6), float(missing_chunks), x.ready_at, x.arrival_seq, x.rid)

            return sorted(eligible, key=_v12_bimodal_hrrn_key)

        lane1_age_cap = min(float(config.max_queue_age_s), float(config.hrrn_age_cap_s))

        def _bimodal_key(x: QueueItem):
            age = now - x.ready_at
            missing_chunks = (max(0, x.missing) + step_quantum - 1) // step_quantum
            if age >= config.max_queue_age_s:
                return (0, 0, 0.0, 0, x.ready_at, x.arrival_seq, x.rid)
            if (
                x.completed_turns >= config.critical_turn_threshold
                and missing_chunks <= config.critical_max_missing_chunks
            ) or x.completed_turns >= deep_threshold:
                return (1, 0, _compute_lane0_rank(x, age), missing_chunks, x.ready_at, x.arrival_seq, x.rid)
            if age >= lane1_age_cap:
                return (1, 1, 0.0, 0, x.ready_at, x.arrival_seq, x.rid)
            if missing_chunks <= config.lane1_short_max_chunks:
                return (1, 2, float(missing_chunks), 0, x.ready_at, x.arrival_seq, x.rid)
            return (1, 3, 0.0, 0, x.ready_at, x.arrival_seq, x.rid)

        return sorted(eligible, key=_bimodal_key)

    if config.prefetch_only_reorder:
        # Preserve strict FIFO arrival order unless a candidate holds a freshly
        # prefetched GPU lease (consume before lease expiry without SJF-starving
        # long critical-path chains).
        if protected is None and not any(x.prefetched for x in eligible):
            return eligible
        result = []
        while eligible:
            low = min(debt.get(x.user, 0.0) for x in eligible)
            feasible = [x for x in eligible if debt.get(x.user, 0.0) <= low + config.service_slack]
            if protected in eligible:
                selected = protected
                protected = None
            else:
                pref_ready = [x for x in feasible if x.prefetched]
                if pref_ready:
                    selected = min(pref_ready, key=lambda x: (x.ready_at, x.rid))
                else:
                    # Strictly preserve FIFO order among non-prefetched requests
                    selected = eligible[0]
            result.append(selected)
            debt[selected.user] = debt.get(selected.user, 0.0) + max(1, selected.missing)
            eligible.remove(selected)
        return result

    if config.tier_signal_only_order and (not pending or not config.block_unrelated_during_prefetch) and protected is None:
        has_tier_signal = any(x.host > x.gpu or x.gpu >= config.min_host_tokens for x in eligible)
        if not has_tier_signal:
            return eligible
    result = []
    while eligible:
        low = min(debt.get(x.user, 0.0) for x in eligible)
        feasible = [x for x in eligible if debt.get(x.user, 0.0) <= low + config.service_slack]
        if protected in eligible:
            selected = protected
            protected = None
        else:
            # Host availability is not counted as READY GPU state. It can
            # reduce true missing work, while its transfer cost stays visible.
            selected = min(
                feasible,
                key=lambda x: (
                    x.missing + config.restore_cost_weight * max(0, x.host - x.gpu),
                    -x.gpu,
                    x.ready_at,
                    x.rid,
                ),
            )
        result.append(selected)
        debt[selected.user] = debt.get(selected.user, 0.0) + max(1, selected.missing)
        eligible.remove(selected)
    return result
