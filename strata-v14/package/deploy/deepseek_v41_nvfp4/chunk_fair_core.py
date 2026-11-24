"""Pure policy for the qualified ONE-continuation native prefill scheduler.

No tensors, cache mutations, full-output predictions or local clocks. A plan is
an execution-share proposal, NOT an allocator reservation. The native engine
may reject or shorten it. Keeping one partial prefill is deliberate: inventing
multiple partials would violate this version's scheduler/checkpoint contract.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Mapping


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def ceil_to(value: int, quantum: int) -> int:
    return (value + quantum - 1) // quantum * quantum


@dataclass(frozen=True)
class ChunkConfig:
    mode: str = "off"  # off / shadow / guard / share / fair
    min_progress_tokens: int = 512
    short_max_tokens: int = 1024
    short_fraction: float = 0.5
    max_new_prefills: int = 4
    scan_limit: int = 64
    fairness_slack_units: float = 256.0
    age_guard_s: float = 20.0
    input_weight: float = 1.0
    output_weight: float = 32.0
    user_weights: dict[str, float] = field(default_factory=dict)
    max_users: int = 4096
    max_inflight_steps: int = 32
    max_prefill_streak: int = 4
    slot_gate_max_wait_s: float = 0.0  # 0: do NOT override the native slots gate
    rejected_backoff_steps: int = 2
    max_profiled_prefill_tokens: int = 0  # 0: retain native total token budget

    def __post_init__(self):
        if self.mode not in {"off", "shadow", "guard", "share", "fair"}:
            raise ValueError("unknown chunk_fair mode")
        for key, lo, hi in (
            ("min_progress_tokens", 1, 1 << 20), ("short_max_tokens", 1, 1 << 20),
            ("max_new_prefills", 1, 64), ("scan_limit", 1, 4096),
            ("max_users", 1, 65536), ("max_inflight_steps", 1, 256),
            ("max_prefill_streak", 0, 64), ("rejected_backoff_steps", 0, 128),
            ("max_profiled_prefill_tokens", 0, 1 << 20),
        ):
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"invalid {key}")
        for key, lo, hi in (("short_fraction", 0.0, 0.9),
                            ("fairness_slack_units", 0.0, 1e9),
                            ("age_guard_s", 0.0, 3600.0),
                            ("input_weight", 1e-9, 1e9),
                            ("output_weight", 1e-9, 1e9),
                            ("slot_gate_max_wait_s", 0.0, 3600.0)):
            value = getattr(self, key)
            if not finite(value) or not lo <= value <= hi:
                raise ValueError(f"invalid {key}")
        if not isinstance(self.user_weights, dict) or len(self.user_weights) > self.max_users:
            raise ValueError("invalid user_weights")
        for user, weight in self.user_weights.items():
            if not isinstance(user, str) or not 0 < len(user) <= 256:
                raise ValueError("invalid user identity in weights")
            if not finite(weight) or not 1e-3 <= weight <= 1e3:
                raise ValueError("invalid user weight")

    @property
    def allocates(self) -> bool:
        return self.mode in {"share", "fair"}

    @property
    def observes(self) -> bool:
        return self.mode != "off"

    def frozen(self) -> dict:
        return asdict(self)


class ServiceLedger:
    """Weighted *token service proxy*, NOT GPU seconds or a VTC theorem.

    Charge input rows that actually execute and accepted output increments.
    In-flight input reservations prevent overlapping launches from repeatedly
    seeing the same uncharged user. A decode reserves one output token and is
    reconciled when its result commits; speculative overshoot is recorded, not
    falsely claimed to be bounded by one token. Idle users get no banked credit.
    """
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.served: OrderedDict[str, float] = OrderedDict()
        self.pending: dict[int, dict[str, float]] = {}
        self.active: set[str] = set()
        self.floor = 0.0
        self.total_input = 0
        self.total_output = 0
        self.evicted_users = 0

    def _weight(self, user: str) -> float:
        return float(self.config.user_weights.get(user, 1.0))

    def _ensure(self, user: str):
        if user in self.served:
            return
        protected = self.active | {u for p in self.pending.values() for u in p}
        while len(self.served) >= self.config.max_users:
            victim = next((u for u in self.served if u not in protected), None)
            if victim is None:
                raise RuntimeError("too many live users for qualified fairness ledger")
            del self.served[victim]
            self.evicted_users += 1
        self.served[user] = self.floor

    def sync(self, users: Iterable[str]):
        users = set(users) | {u for p in self.pending.values() for u in p}
        if len(users) > self.config.max_users:
            raise RuntimeError("live user count exceeds max_users; fairness not silently disabled")
        # Advance only with users that were already backlogged. New arrivals
        # join at this floor instead of receiving an ever-growing idle credit.
        continuing = self.active & users
        if continuing:
            self.floor = max(self.floor, min(self.served[u] for u in continuing))
        elif self.active:
            self.floor = max(self.floor, min(self.served[u] for u in self.active))
        previous = self.active
        self.active = users
        for user in sorted(users):
            self._ensure(user)
            if user not in previous:
                self.served[user] = max(self.served[user], self.floor)
            self.served.move_to_end(user)

    def effective(self, user: str) -> float:
        return self.served.get(user, self.floor) + sum(p.get(user, 0.) for p in self.pending.values())

    def units(self, user: str, inputs: int, outputs: int) -> float:
        if type(inputs) is not int or type(outputs) is not int or inputs < 0 or outputs < 0:
            raise ValueError("token increments must be nonnegative integers")
        return (inputs * self.config.input_weight + outputs * self.config.output_weight) / self._weight(user)

    def reserve(self, step: int, rows: Iterable[tuple[str, int, int]]):
        if step in self.pending:
            raise RuntimeError("duplicate launch ticket")
        if len(self.pending) >= self.config.max_inflight_steps:
            raise RuntimeError("inflight accounting overflow")
        charges: dict[str, float] = {}
        for user, inputs, outputs in rows:
            self._ensure(user)
            charges[user] = charges.get(user, 0.) + self.units(user, inputs, outputs)
        self.pending[step] = charges

    def settle(self, step: int, rows: Iterable[tuple[str, int, int]]):
        if step not in self.pending:
            raise RuntimeError("missing or already settled launch ticket")
        del self.pending[step]
        for user, inputs, outputs in rows:
            self._ensure(user)
            self.served[user] += self.units(user, inputs, outputs)
            self.total_input += inputs
            self.total_output += outputs

    def snapshot(self):
        return {"units": "weighted executed input / accepted output token proxy, not GPU seconds",
                "users": {u: {"served": v, "including_inflight": self.effective(u)}
                          for u, v in self.served.items()},
                "virtual_floor": self.floor, "inflight_steps": len(self.pending),
                "executed_input_tokens": self.total_input,
                "accepted_output_tokens": self.total_output,
                "evicted_inactive_users": self.evicted_users}


@dataclass(frozen=True)
class Candidate:
    rid: str
    user: str
    remaining: int
    ready: float
    arrival: int
    continuing: bool = False

    def __post_init__(self):
        if not self.rid or not self.user or type(self.remaining) is not int or self.remaining <= 0:
            raise ValueError("invalid chunk candidate")
        if not finite(self.ready) or self.ready < 0 or type(self.arrival) is not int:
            raise ValueError("invalid candidate arrival")


@dataclass(frozen=True)
class Share:
    rid: str
    user: str
    cap: int  # native compute-budget units, including page rounding
    estimated_input: int
    completes: bool
    continuing: bool
    reason: str


@dataclass(frozen=True)
class Plan:
    budget: int
    alignment: int
    shares: tuple[Share, ...]
    partial_rid: str | None
    reason: str

    def frozen(self):
        return asdict(self)


def plan_shares(candidates: list[Candidate], budget: int, alignment: int,
                new_slots: int, config: ChunkConfig, service: Mapping[str, float],
                now: float) -> Plan:
    """Plan finite progress plus complete short prefills; at most ONE partial.

    ``new_slots`` excludes the held continuation row. No plan can evict, load
    host cache or allocate a slot. Fair mode gates a short promotion by user
    service; completed input frees room for subsequent native generation, NOT
    a request row. Cost here is known prefill work, never predicted output.
    """
    if type(budget) is not int or budget < 0 or type(alignment) is not int or alignment <= 0:
        raise ValueError("invalid native budget/alignment")
    if type(new_slots) is not int or new_slots < 0 or not finite(now):
        raise ValueError("invalid slots/clock")
    if len({r.rid for r in candidates}) != len(candidates):
        raise ValueError("duplicate candidate ID")
    continuing = [r for r in candidates if r.continuing]
    if len(continuing) > 1:
        raise ValueError("native scheduler supports exactly one partial prefill")
    budget = budget // alignment * alignment
    if budget <= 0 or not candidates:
        return Plan(budget, alignment, (), None, "empty_budget_or_candidates")
    waiting = sorted((r for r in candidates if not r.continuing), key=lambda r: (r.ready, r.arrival, r.rid))
    waiting = waiting[:config.scan_limit]
    slots = min(new_slots, config.max_new_prefills)
    if not continuing and (not waiting or slots == 0):
        return Plan(budget, alignment, (), None, "no_state_slot")
    virtual = dict(service)
    fair = config.mode in {"fair", "shadow"}
    shares: list[Share] = []

    def debt(r):
        return virtual.get(r.user, 0.) if fair else 0.

    def choose(rows, *, allow_age=True, prefer_finish=False):
        overdue = [r for r in rows if config.age_guard_s > 0 and now - r.ready >= config.age_guard_s]
        if allow_age and overdue:
            return min(overdue, key=lambda r: (r.ready, r.arrival, r.rid)), "age_guard"
        lower = min(debt(r) for r in rows)
        eligible = [r for r in rows if debt(r) <= lower + (config.fairness_slack_units if fair else 0.)]
        if prefer_finish:
            return min(eligible, key=lambda r: (ceil_to(r.remaining, alignment), r.ready, r.arrival, r.rid)), "short_finish_in_fair_band"
        return min(eligible, key=lambda r: (r.ready, r.arrival, r.rid)), "user_service_band"

    primary, why = (continuing[0], "continuation_progress") if continuing else choose(waiting)
    if not continuing:
        waiting.remove(primary)
        slots -= 1
    minimum = min(ceil_to(primary.remaining, alignment), budget,
                  max(alignment, ceil_to(config.min_progress_tokens, alignment)))
    primary_finishes = ceil_to(primary.remaining, alignment) <= minimum
    # A long primary keeps a useful floor; a finishing primary need not hold
    # half a batch merely for being first.
    reserve_limit = budget - minimum
    if not primary_finishes:
        reserve_limit = min(reserve_limit, int(budget * config.short_fraction) // alignment * alignment)
    chosen: list[tuple[Candidate, str]] = []
    used = 0
    used_users = {primary.user}
    virtual[primary.user] = debt(primary) + minimum * config.input_weight / config.user_weights.get(primary.user, 1.)
    while slots > 0:
        fits = [r for r in waiting
                if r.remaining <= config.short_max_tokens
                and ceil_to(r.remaining, alignment) <= reserve_limit - used]
        if fair and fits:
            minimum_service = min([debt(primary), *(debt(r) for r in fits)])
            fits = [r for r in fits if debt(r) <= minimum_service + config.fairness_slack_units
                    or (config.age_guard_s > 0 and now - r.ready >= config.age_guard_s)]
        if not fits:
            break
        # Round across users before taking a second short from the same user.
        distinct = [r for r in fits if r.user not in used_users]
        selected, reason = choose(distinct or fits, allow_age=not any(x[1] == "age_guard" for x in chosen),
                                  prefer_finish=True)
        amount = ceil_to(selected.remaining, alignment)
        used += amount
        chosen.append((selected, reason))
        waiting.remove(selected)
        used_users.add(selected.user)
        slots -= 1
        virtual[selected.user] = debt(selected) + amount * config.input_weight / config.user_weights.get(selected.user, 1.)
    primary_cap = min(ceil_to(primary.remaining, alignment), budget - used)
    shares.append(Share(primary.rid, primary.user, primary_cap,
                        min(primary.remaining, primary_cap), primary.remaining <= primary_cap,
                        primary.continuing, why))
    for row, reason in chosen:
        shares.append(Share(row.rid, row.user, ceil_to(row.remaining, alignment), row.remaining,
                            True, False, reason))
    left = budget - sum(s.cap for s in shares)
    partial = primary.rid if primary.remaining > primary_cap else None
    # A short primary can leave room to START one long request. This is not
    # legal when another unfinished prefill already exists.
    if partial is None and slots > 0 and left >= alignment and waiting:
        next_row, reason = choose(waiting, allow_age=why != "age_guard")
        cap = min(left, ceil_to(next_row.remaining, alignment))
        shares.append(Share(next_row.rid, next_row.user, cap, min(cap, next_row.remaining),
                            next_row.remaining <= cap, False, reason))
        if next_row.remaining > cap:
            partial = next_row.rid
    if sum(s.cap for s in shares) > budget or sum(not s.completes for s in shares) > 1:
        raise AssertionError("invalid execution-share plan")
    return Plan(budget, alignment, tuple(shares), partial, "one_partial_plus_finishers")
