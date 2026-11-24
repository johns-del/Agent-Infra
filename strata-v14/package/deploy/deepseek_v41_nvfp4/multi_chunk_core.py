"""v6: multiple partial-prefill proposals with bounded user credit/locality.

Pure CPU policy. A proposal neither allocates KV nor bills a user. No future
output lengths or tool durations are read. Native allocation is authoritative.
SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import math


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def ceil_to(n, q):
    return ((n + q - 1) // q) * q


@dataclass(frozen=True)
class MultiConfig:
    mode: str = "off"                 # off / shadow / locality / fair
    max_partial_prefills: int = 4
    max_prefill_reqs: int = 8
    max_new_prefills: int = 4
    scan_limit: int = 128
    min_chunk_tokens: int = 512
    max_chunk_tokens: int = 2048
    quantum: float = 1024.0             # normalized service proxy, NOT seconds
    credit_cap_quanta: int = 2
    input_weight: float = 1.0
    output_weight: float = 32.0
    user_weights: dict[str, float] = field(default_factory=dict)
    max_users: int = 4096
    max_inflight_steps: int = 32
    stall_timeout_s: float = 30.0
    max_stalled_passes: int = 16
    age_guard_s: float = 20.0
    max_unserved_steps: int = 32
    max_prefill_streak: int = 4
    slot_gate_max_wait_s: float = 2.0
    rejected_backoff_steps: int = 2
    soft_state_limits: bool = True
    state_borrow_slack: int = 1
    reclaim: str = "off"                # off / native_recompute; NOT CPU offload
    reclaim_wait_s: float = 5.0
    reclaim_cooldown_steps: int = 64
    reclaim_min_service_steps: int = 8
    consensus: bool = True
    latency_profile: dict | None = None
    batch_budget_ms: float = 0.0
    time_margin: float = 1.2

    def __post_init__(self):
        if self.mode not in {"off", "shadow", "locality", "fair"}:
            raise ValueError("unknown multi_chunk mode")
        if self.reclaim not in {"off", "native_recompute"}:
            raise ValueError("unknown state reclaim backend")
        ints = {"max_partial_prefills": (1,64), "max_prefill_reqs": (1,128),
                "max_new_prefills": (1,64), "scan_limit": (1,4096),
                "min_chunk_tokens": (1,1<<20), "max_chunk_tokens": (1,1<<20),
                "credit_cap_quanta": (1,16), "max_users": (1,65536),
                "max_inflight_steps": (1,256), "max_stalled_passes": (1,1024), "max_unserved_steps": (0,1000000),
                "max_prefill_streak": (1,64), "state_borrow_slack": (0,128),
                "rejected_backoff_steps": (0,128), "reclaim_cooldown_steps": (1,100000),
                "reclaim_min_service_steps": (1,100000)}
        for k,(lo,hi) in ints.items():
            if type(getattr(self,k)) is not int or not lo <= getattr(self,k) <= hi:
                raise ValueError("invalid " + k)
        for k,lo,hi in (("quantum",1e-6,1e12),("input_weight",1e-9,1e9),
                        ("output_weight",1e-9,1e9),("age_guard_s",0,3600),
                        ("slot_gate_max_wait_s",0,3600),("stall_timeout_s",1,3600),("reclaim_wait_s",0,3600),
                        ("batch_budget_ms",0,1e6),("time_margin",1,10)):
            if not finite(getattr(self,k)) or not lo <= getattr(self,k) <= hi:
                raise ValueError("invalid " + k)
        if type(self.soft_state_limits) is not bool or type(self.consensus) is not bool:
            raise ValueError("boolean settings must be booleans")
        if self.min_chunk_tokens > self.max_chunk_tokens:
            raise ValueError("min chunk exceeds max chunk")
        if not isinstance(self.user_weights,dict) or len(self.user_weights)>self.max_users:
            raise ValueError("invalid user weights")
        for u,w in self.user_weights.items():
            if not isinstance(u,str) or not 0<len(u)<=256 or not finite(w) or not 1e-3<=w<=1e3:
                raise ValueError("invalid weighted principal")
        if self.latency_profile is not None:
            BatchCost(self.latency_profile)
        if self.batch_budget_ms and self.latency_profile is None:
            raise ValueError("a measured latency profile is required for a time budget")

    @property
    def active(self): return self.mode in {"locality","fair"}
    @property
    def observes(self): return self.mode != "off"
    def frozen(self): return asdict(self)


class BatchCost:
    """Measured pure-prefill model; reject unsupported shapes, never extrapolate.

    Coefficients are in milliseconds for features (1, tokens, requests,
    sum(tokens*(2*prefix+tokens))/1e6). NOT a latency guarantee. No GPU timing
    is inferred from asynchronous Python wall time.
    """
    def __init__(self, p):
        if not isinstance(p,dict) or set(p)!={"schema","runtime_fingerprint","samples","coefficients","envelope"}:
            raise ValueError("invalid measured profile schema")
        if p['schema']!='orvix-prefill-ms-v1' or not isinstance(p['runtime_fingerprint'],str) or not p['runtime_fingerprint']:
            raise ValueError("profile needs runtime identity")
        if type(p['samples']) is not int or p['samples']<12:
            raise ValueError("profile needs at least 12 measured samples")
        if not isinstance(p['coefficients'],list) or len(p['coefficients'])!=4 or any(not finite(v) or v<0 for v in p['coefficients']):
            raise ValueError("invalid nonnegative coefficients")
        if not isinstance(p['envelope'],dict) or set(p['envelope'])!={'tokens','requests','context'} or any(type(v) is not int or v<1 for v in p['envelope'].values()):
            raise ValueError("invalid profile envelope")
        self.p=p

    def estimate(self, rows):
        # rows=(executed prefill tokens, already computed prefix)
        n=sum(x for x,_ in rows); e=self.p['envelope']
        if n>e['tokens'] or len(rows)>e['requests'] or max((x+y for x,y in rows),default=0)>e['context']:
            return None
        features=(1,n,len(rows),sum(x*(2*y+x) for x,y in rows)/1e6)
        return sum(a*b for a,b in zip(self.p['coefficients'],features))


class CreditLedger:
    """Bounded normalized credit, finite history, in-flight reservations.

    Negative debt is not erased on idle/retraction. Returning users cannot
    accumulate wall-time credit. A ledger overflow fails rather than silently
    dropping debt. This is NOT a VTC/DLPM fairness proof for this runtime.
    """
    def __init__(self, config):
        self.config=config
        self.credit=OrderedDict(); self.served={}; self.active=set(); self.pending={}
        self.last_ticket=-1; self.granted_plans=set()
        self.inputs=0; self.outputs=0; self.recomputed=0

    def ensure(self,u):
        if u in self.credit: return
        if len(self.credit)>=self.config.max_users:
            protected=self.active|{u for t in self.pending.values() for u in t}
            victim=next((u for u,c in self.credit.items() if u not in protected and c>=0),None)
            if victim is None: raise RuntimeError("user ledger full; cannot erase live or negative debt")
            del self.credit[victim],self.served[victim]
        self.credit[u]=self.config.quantum; self.served[u]=0.

    def sync(self, users):
        users=set(users)|{u for t in self.pending.values() for u in t}
        if len(users)>self.config.max_users: raise RuntimeError("too many active principals")
        previous=self.active; self.active=users
        for u in sorted(users):
            self.ensure(u)
            if u not in previous: self.credit[u]=min(self.credit[u],self.config.quantum)
            self.credit.move_to_end(u)

    def units(self,u,inputs,outputs):
        if type(inputs) is not int or type(outputs) is not int or min(inputs,outputs)<0:
            raise ValueError("work increments must be nonnegative integers")
        return (inputs*self.config.input_weight+outputs*self.config.output_weight)/self.config.user_weights.get(u,1.)

    def available(self,u):
        return self.credit.get(u,self.config.quantum)-sum(t.get(u,0.) for t in self.pending.values())

    def grant(self,plan_id,grants):
        if plan_id in self.granted_plans: raise RuntimeError("duplicate grant")
        if any(not finite(v) or v<0 for v in grants.values()): raise ValueError("invalid grant")
        for u,n in grants.items():
            self.ensure(u)
            self.credit[u]=min(self.config.quantum*self.config.credit_cap_quanta,self.credit[u]+n)
        self.granted_plans.add(plan_id)
        if len(self.granted_plans)>1024: self.granted_plans=set(sorted(self.granted_plans)[-512:])

    def reserve(self,ticket,rows):
        if ticket<=self.last_ticket or ticket in self.pending: raise RuntimeError("duplicate/out-of-order launch")
        if len(self.pending)>=self.config.max_inflight_steps: raise RuntimeError("in-flight ledger overflow")
        charges={}
        for u,i,o in rows:
            self.ensure(u); charges[u]=charges.get(u,0.)+self.units(u,i,o)
        self.pending[ticket]=charges; self.last_ticket=ticket

    def settle(self,ticket,rows):
        if ticket not in self.pending: raise RuntimeError("missing/duplicate settlement")
        charges=[(u,i,o,self.units(u,i,o)) for u,i,o in rows]
        del self.pending[ticket]
        for u,i,o,c in charges:
            self.credit[u]-=c; self.served[u]+=c; self.inputs+=i; self.outputs+=o

    def snapshot(self):
        return {"units":"normalized useful executed input + accepted output proxy; not GPU seconds",
                "users":{u:{"credit":v,"available":self.available(u),"served":self.served[u]} for u,v in self.credit.items()},
                "inflight":len(self.pending),"inputs":self.inputs,"outputs":self.outputs,"recomputed_not_rebilled":self.recomputed}


@dataclass(frozen=True)
class Candidate:
    rid:str
    user:str
    remaining:int
    prefix:int
    ready:float
    progress:float
    step:int
    continuing:bool=False

    def __post_init__(self):
        if not self.rid or not self.user or type(self.remaining) is not int or self.remaining<1 or type(self.prefix) is not int or self.prefix<0:
            raise ValueError("invalid candidate")
        if type(self.step) is not int or not finite(self.ready) or not finite(self.progress) or min(self.ready,self.progress)<0:
            raise ValueError("invalid progress")


@dataclass(frozen=True)
class Share:
    rid:str
    user:str
    cap:int
    remaining:int
    prefix:int
    continuing:bool
    reason:str


@dataclass(frozen=True)
class Plan:
    shares:tuple[Share,...]
    budget:int
    grants:dict
    skipped:dict
    predicted_ms:float|None=None
    profile_fallback:bool=False
    def frozen(self): return asdict(self)


def plan_batch(rows, *, budget, alignment, new_slots, partial_count,
               resident_by_user, state_capacity, credit, config, now, step):
    for k,n in (("budget",budget),("alignment",alignment),("new_slots",new_slots),
                ("partial_count",partial_count),("state_capacity",state_capacity),("step",step)):
        if type(n) is not int or n<(1 if k in {'alignment','state_capacity'} else 0):
            raise ValueError("invalid "+k)
    if len({r.rid for r in rows})!=len(rows) or not finite(now): raise ValueError("duplicate IDs or invalid clock")
    budget=budget//alignment*alignment
    if not budget or not rows: return Plan((),budget,{},{})
    pending=sorted(rows,key=lambda r:(r.progress,r.ready,r.rid))[:config.scan_limit]
    balances={r.user:float(credit.get(r.user,config.quantum)) for r in pending}
    if any(not finite(v) for v in balances.values()): raise ValueError("invalid credit")
    occupancy=dict(resident_by_user); users=set(occupancy)|set(balances)
    weights={u:config.user_weights.get(u,1.) for u in users}; total=sum(weights.values())
    soft={u:max(1,math.ceil(state_capacity*weights[u]/total))+config.state_borrow_slack for u in users}
    partial_users={u:sum(r.continuing and r.user==u for r in rows) for u in users}
    psoft={u:max(1,math.ceil(config.max_partial_prefills*weights[u]/total)) for u in users}
    remaining=budget; slots=new_slots; partials=partial_count; nnew=0
    shares=[]; grants={}; skipped={}; age_used=False
    fair=config.mode in {'fair','shadow'}
    model=BatchCost(config.latency_profile) if config.latency_profile else None
    fallback=False
    while pending and remaining>=alignment and len(shares)<config.max_prefill_reqs:
        feasible=[]
        for r in pending:
            finish=ceil_to(r.remaining,alignment)
            cap=min(finish,remaining,max(alignment,config.max_chunk_tokens//alignment*alignment))
            floor=min(finish,max(alignment,ceil_to(config.min_chunk_tokens,alignment)))
            if not r.continuing:
                if slots<=0 or nnew>=config.max_new_prefills:
                    skipped[r.rid]='state_slots'; continue
                if partials>=config.max_partial_prefills:
                    if r.remaining>cap:
                        skipped[r.rid]='partial_limit'; continue
                    # A new request at the partial limit is admissible only if
                    # it finishes. Credit, age and time-budget adjustments must
                    # not silently turn it into one more retained partial.
                    floor=max(floor,finish)
                if fair and config.soft_state_limits:
                    other=any(q.user!=r.user and occupancy.get(q.user,0)<soft[q.user] for q in pending)
                    partial_other=any(q.user!=r.user and partial_users.get(q.user,0)<psoft[q.user] for q in pending)
                    if occupancy.get(r.user,0)>=soft[r.user] and other:
                        skipped[r.rid]='borrowed_state_limit'; continue
                    if partial_users.get(r.user,0)>=psoft[r.user] and partial_other:
                        if r.remaining>cap:
                            skipped[r.rid]='borrowed_state_limit'; continue
                        floor=max(floor,finish)
            if cap<floor and finish>cap:
                skipped[r.rid]='useful_quantum'; continue
            feasible.append((r,cap,floor))
        if not feasible: break
        overdue=[x for x in feasible if (config.age_guard_s and now-x[0].progress>=config.age_guard_s)
                 or (config.max_unserved_steps and step-x[0].step>=config.max_unserved_steps)]
        if overdue and not age_used:
            r,cap,floor=min(overdue,key=lambda x:(x[0].progress,x[0].step,x[0].ready,x[0].rid))
            cap=min(cap,floor); reason='progress_guard'; age_used=True
        else:
            eligible=[x for x in feasible if not fair or balances[x[0].user]>0]
            if not eligible:
                best=max(balances[r.user] for r,_,_ in feasible)
                amount=(math.floor(-best/config.quantum)+1)*config.quantum
                for u in balances:
                    old=balances[u]; balances[u]=min(config.credit_cap_quanta*config.quantum,old+amount)
                    grants[u]=grants.get(u,0.)+balances[u]-old
                eligible=feasible
            r,cap,floor=min(eligible,key=lambda x:(-x[0].prefix,x[0].remaining,x[0].progress,x[0].rid))
            if fair:
                affordable=int(max(0.,balances[r.user])*config.user_weights.get(r.user,1.)/config.input_weight)//alignment*alignment
                cap=min(cap,max(floor,affordable))
            reason='credit_locality' if fair else 'locality'
        if model and config.batch_budget_ms:
            while cap>=floor:
                estimate=model.estimate([(min(s.cap,s.remaining),s.prefix) for s in shares]+[(min(cap,r.remaining),r.prefix)])
                if estimate is None:
                    fallback=True
                    break
                if estimate*config.time_margin<=config.batch_budget_ms: break
                cap-=alignment
            if cap<floor:
                skipped[r.rid]='measured_time_budget'; pending.remove(r); continue
        shares.append(Share(r.rid,r.user,cap,r.remaining,r.prefix,r.continuing,reason))
        balances[r.user]-=min(cap,r.remaining)*config.input_weight/config.user_weights.get(r.user,1.)
        remaining-=cap; pending.remove(r)
        if r.continuing and r.remaining<=cap:
            partials-=1; partial_users[r.user]-=1
        elif not r.continuing:
            slots-=1; nnew+=1; occupancy[r.user]=occupancy.get(r.user,0)+1
            if r.remaining>cap: partials+=1; partial_users[r.user]=partial_users.get(r.user,0)+1
    estimate=model.estimate([(min(s.cap,s.remaining),s.prefix) for s in shares]) if model and shares else None
    if sum(s.cap for s in shares)>budget: raise AssertionError("budget overrun")
    if partial_count<=config.max_partial_prefills and partials>config.max_partial_prefills: raise AssertionError("partial overrun")
    return Plan(tuple(shares),budget,grants,skipped,estimate,fallback)
