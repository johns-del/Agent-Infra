"""Dependency-free UNIFIED policy math, shared by gateway, adapter and simulator.

Inputs are online features only. Workload execution data never enters this API.
"""
from __future__ import annotations

import math
import heapq
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

LABEL = 'v1 开发负载，tool_scale 1.0，仅用于开发'
LEVER_COUNTERS = (
    'interleave_armed', 'interleave_deferred_iterations', 'short_reserve_cycles',
    'short_reserved_tokens', 'short_admitted_with_reserve', 'latch_resets',
    'latch_reset_followed_by_admission', 'short_reserve_skipped_no_row',
)


@dataclass(frozen=True)
class Parameters:
    theta_fg: float = 5.
    theta_bg: float = 60.
    tau: float = 10.
    horizon: float = 30.
    reuse_default: float = .9
    lane_weights: dict = field(default_factory=lambda: {
        'foreground': 1., 'subagent': 1., 'background': .25, 'unknown': .25})
    family_weights: dict = field(default_factory=dict)
    # Opt-in estimated-admission fairness. This is not measured GPU service or
    # token-level VTC fairness; the adapter updates the ledger on admission.
    fairness_mode: str = 'legacy'
    age_guard_quota: int = 1
    fairness_slack_s: float = 2.
    fairness_max_families: int = 4096
    candidate_window_k: int | None = None
    preserve_fcfs_head: bool = False
    inheritance_cap_s: float = 0.
    prefill_tokens_per_s: float = 2000.
    decode_tokens_per_s: float = 50.
    intercept: float = 0.
    cost_model: str = 'prefill_uniform'
    # Frozen v1 median of all 125 graph/content-disjoint prior observations.
    uniform_output_tokens: float = 303.
    interleave_steps: int = 1
    short_threshold_tokens: int = 2048
    short_reserve_cap_tokens: int = 2048
    levers: dict = field(default_factory=lambda: {
        'interleave': True, 'short_first': True, 'latch_reset': True})

    def __post_init__(self):
        if self.fairness_mode not in ('legacy', 'bounded', 'guarded', 'wait_inheritance'):
            raise ValueError('unknown fairness mode')
        if (type(self.inheritance_cap_s) not in (int, float)
                or not 0 <= self.inheritance_cap_s <= 5
                or (self.inheritance_cap_s and self.fairness_mode != 'wait_inheritance')):
            raise ValueError('wait inheritance requires a cap between zero and five seconds')
        if type(self.preserve_fcfs_head) is not bool or (
                self.preserve_fcfs_head and self.fairness_mode != 'guarded'):
            raise ValueError('preserving the native FCFS head requires guarded mode')
        if (isinstance(self.fairness_slack_s, bool)
                or not isinstance(self.fairness_slack_s, (int, float))
                or not math.isfinite(self.fairness_slack_s) or self.fairness_slack_s < 0):
            raise ValueError('finite nonnegative fairness slack required')
        if (isinstance(self.fairness_max_families, bool)
                or not isinstance(self.fairness_max_families, int)
                or self.fairness_max_families < 1):
            raise ValueError('positive fairness ledger capacity required')
        for key in ('age_guard_quota', 'interleave_steps', 'short_threshold_tokens', 'short_reserve_cap_tokens'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('nonnegative integer required: ' + key)
        if (set(self.levers) != {'interleave', 'short_first', 'latch_reset'}
                or any(type(v) is not bool for v in self.levers.values())):
            raise ValueError('three explicit boolean lever switches required')
        for key in ('theta_fg', 'theta_bg', 'tau', 'horizon', 'prefill_tokens_per_s', 'decode_tokens_per_s'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('finite positive parameter required: ' + key)
        for key in ('reuse_default', 'intercept', 'uniform_output_tokens'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('finite nonnegative parameter required: ' + key)
        if self.reuse_default > 1:
            raise ValueError('probabilities must be in [0, 1]')
        k = self.candidate_window_k
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 0):
            raise ValueError('candidate_window_k must be None or nonnegative integer')
        if self.cost_model not in ('prefill_uniform', 'prefill', 'group_median', 'online', 'online_guarded', 'fcfs'):
            raise ValueError('unknown cost model')
        if set(self.lane_weights) != {'foreground', 'subagent', 'background', 'unknown'}:
            raise ValueError('all four lane weights required')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in [*self.lane_weights.values(), *self.family_weights.values()]):
            raise ValueError('finite positive weights required')

    def frozen(self):
        return asdict(self)

    def service(self, uncached, predicted_output=0):
        return max(1e-9, self.intercept + max(0, uncached) / self.prefill_tokens_per_s
                   + max(0, predicted_output) / self.decode_tokens_per_s)


def group_key(lane, purpose=None, previous_tool='none', channel='native'):
    return '|'.join(str(v or 'unknown') for v in (lane, purpose, previous_tool, channel))


def rank_candidates(rows, now, params):
    """Return row indices and diagnostics. Stable FCFS ties, hard protection.

    Only group_median falls back to weighted fairness when group evidence is
    missing. prefill_uniform uses one frozen prior length for every candidate.
    """
    if params.fairness_mode == 'wait_inheritance':
        return rank_inherited_wait(rows, now, params)
    if params.fairness_mode in ('bounded', 'guarded'):
        return _rank_bounded(rows, now, params)
    counts = Counter(r['family_id'] for r in rows)
    protected = defaultdict(list)
    normal = []
    details = []
    fallback = params.cost_model == 'group_median' and any(
        not r.get('prediction_count', 0) for r in rows)
    for i, row in enumerate(rows):
        lane = row.get('lane', 'unknown')
        wait = max(0., now - row['ready_at'])
        threshold = params.theta_fg if lane in ('foreground', 'subagent') else params.theta_bg
        q = (row.get('lane_weight', params.lane_weights[lane])
             * params.family_weights.get(row['family_id'], 1.)
             * (1 + wait / params.tau) / counts[row['family_id']])
        output = (params.uniform_output_tokens if params.cost_model == 'prefill_uniform' else
                  row.get('predicted_output_tokens', 0) if params.cost_model in ('group_median', 'online', 'online_guarded') else 0)
        service = params.service(row['input_tokens'] - row['matched_tokens'], output)
        score = q if fallback else q / service
        details.append({'wait': wait, 'protected': wait >= threshold, 'q': q, 'service_s': service,
                        'score': score, 'fallback': 'fairness' if fallback else params.cost_model})
        if wait >= threshold:
            protected[row['family_id']].append((-(wait - threshold), row['arrival'], i))
        else:
            normal.append((0 if params.cost_model == 'fcfs' else -score, row['arrival'], i))
    # Take the most overdue request of each family, then their seconds, etc.
    queues = [sorted(v) for v in protected.values()]
    order = []
    for depth in range(max(map(len, queues), default=0)):
        order.extend(item[2] for item in sorted(q[depth] for q in queues if depth < len(q)))
    order.extend(item[2] for item in sorted(normal))
    return order, details


def rank_inherited_wait(rows, now, params):
    """Promote a confirmed successor within its actual-wait protection tier."""
    details = []
    for row in rows:
        ready = row['ready_at']
        waited = max(0., now - ready)
        threshold = params.theta_fg if row['lane'] in ('foreground', 'subagent') else params.theta_bg
        previous = row.get('predecessor_queue_s')
        credit = (min(previous, params.inheritance_cap_s)
                  if type(previous) in (int, float) and math.isfinite(previous) and previous > 0
                  else 0.)
        details.append({'actual_wait_s': waited, 'protected': waited >= threshold,
                        'inherited_wait_s': credit, 'effective_ready_at': ready - credit})
    order = sorted(range(len(rows)), key=lambda i: (
        not details[i]['protected'], details[i]['effective_ready_at'], rows[i]['ready_at'], rows[i]['arrival']))
    return order, details


def _rank_bounded(rows, now, params):
    """Order every candidate using a soft service budget and an age guard.

    ``family_service_s`` is already divided by the family's weight. Eligibility
    is within ``fairness_slack_s`` of the least-served *ready* family. Within
    that set, prefer useful work per estimated service second. The local ledger
    below only simulates this ordering; the caller commits costs on admission.

    An overdue request bypasses the soft service budget. Thus the age guard is
    a priority guarantee, not a wall-clock latency bound under overload or an
    allocator that cannot fit that request. No candidate is removed or gated.
    """
    details, virtual = [], {}
    fallback = params.cost_model == 'group_median' and any(
        not r.get('prediction_count', 0) for r in rows)
    for i, row in enumerate(rows):
        lane = row.get('lane', 'unknown')
        if lane not in params.lane_weights:
            lane = 'unknown'
        family = row['family_id']
        virtual[family] = max(virtual.get(family, 0.),
                              max(0., row.get('family_service_s', 0.)))
        wait = max(0., now - row['ready_at'])
        threshold = params.theta_fg if lane in ('foreground', 'subagent') else params.theta_bg
        output = (params.uniform_output_tokens if params.cost_model == 'prefill_uniform' else
                  row.get('predicted_output_tokens', 0) if params.cost_model in ('group_median', 'online', 'online_guarded') else 0)
        service = params.service(row['input_tokens'] - row['matched_tokens'], output)
        # The virtual ledger accounts for fan-out and family weights. Dividing
        # this utility by queue length again would penalize fan-out twice.
        q = row.get('lane_weight', params.lane_weights[lane]) * (1 + wait / params.tau)
        score = q if fallback else q / service
        details.append({'wait': wait, 'protected': wait >= threshold, 'q': q,
                        'service_s': service, 'score': score,
                        'fallback': 'fairness' if fallback else params.cost_model,
                        'family_service_s': virtual[family],
                        'overdue_s': max(0., wait - threshold)})
    guard_set = None
    if params.fairness_mode == 'guarded':
        overdue = sorted((i for i, d in enumerate(details) if d['protected']),
                         key=lambda i: (-details[i]['overdue_s'], rows[i]['arrival'], i))
        guard_set = set(overdue[:params.age_guard_quota])
        for i, detail in enumerate(details):
            detail['overdue'] = detail['protected']
            detail['protected'] = i in guard_set
    protected, normal = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        if i == 0 and params.preserve_fcfs_head:
            continue
        (protected if details[i]['protected'] else normal)[row['family_id']].append(i)
    order = []

    def take(selected, reason):
        family = rows[selected]['family_id']
        details[selected]['selection_reason'] = reason
        details[selected]['virtual_service_before_s'] = virtual[family]
        virtual[family] += details[selected]['service_s'] / params.family_weights.get(family, 1.)
        order.append(selected)

    if rows and params.preserve_fcfs_head:
        # Debit the preserved request first so the remaining candidates see its
        # service reservation when the rest of the batch is ordered.
        take(0, 'fcfs_head')

    # One overdue request per family before its second, with the most overdue
    # candidate first within each round. Heap entries have a unique row index
    # before any family key, so arbitrary family identifiers never compare.
    overdue_heap = []
    for family, indices in protected.items():
        indices.sort(key=lambda i: (-details[i]['overdue_s'], rows[i]['arrival'], i))
        head = indices[0]
        heapq.heappush(overdue_heap, (0, -details[head]['overdue_s'], rows[head]['arrival'], head))
    while overdue_heap:
        depth, _, _, selected = heapq.heappop(overdue_heap)
        take(selected, 'age_guard')
        indices = protected[rows[selected]['family_id']]
        if depth + 1 < len(indices):
            head = indices[depth + 1]
            heapq.heappush(overdue_heap, (depth + 1, -details[head]['overdue_s'],
                                         rows[head]['arrival'], head))

    # Each family's best remaining candidate is its frontier. The minimum
    # virtual service and eligibility threshold only advance during this pass;
    # move frontiers from the service heap to the utility heap once eligible.
    # Lazy versions invalidate old service entries after each simulated debit.
    families = list(normal)
    queues = [sorted(normal[f], key=lambda i: (
        0 if params.cost_model == 'fcfs' else -details[i]['score'], rows[i]['arrival'], i))
        for f in families]
    positions, versions = [0] * len(families), [0] * len(families)
    active = set(range(len(families)))
    service_heap = [(virtual[f], j, 0) for j, f in enumerate(families)]
    waiting_heap = list(service_heap)
    heapq.heapify(service_heap)
    heapq.heapify(waiting_heap)
    utility_heap = []

    def live(j, version):
        return j in active and versions[j] == version

    while active:
        while service_heap and not live(service_heap[0][1], service_heap[0][2]):
            heapq.heappop(service_heap)
        threshold = service_heap[0][0] + params.fairness_slack_s
        while waiting_heap and waiting_heap[0][0] <= threshold:
            _, j, version = heapq.heappop(waiting_heap)
            if not live(j, version):
                continue
            i = queues[j][positions[j]]
            heapq.heappush(utility_heap, (
                0 if params.cost_model == 'fcfs' else -details[i]['score'],
                rows[i]['arrival'], i, j, version))
        while utility_heap and not live(utility_heap[0][3], utility_heap[0][4]):
            heapq.heappop(utility_heap)
        _, _, selected, j, _ = heapq.heappop(utility_heap)
        take(selected, 'service_budget')
        positions[j] += 1
        versions[j] += 1
        if positions[j] == len(queues[j]):
            active.remove(j)
        else:
            entry = (virtual[families[j]], j, versions[j])
            heapq.heappush(service_heap, entry)
            heapq.heappush(waiting_heap, entry)
    return order, details


def shared_probability(probabilities):
    return 1 - math.prod(1 - p for p in probabilities)


def mean(values):
    values=list(values)
    if not values:raise ValueError('mean requires observations')
    return sum(values)/len(values)


def median(values):
    values=sorted(values)
    if not values:raise ValueError('median requires observations')
    mid=len(values)//2
    return values[mid] if len(values)%2 else (values[mid-1]+values[mid])/2
