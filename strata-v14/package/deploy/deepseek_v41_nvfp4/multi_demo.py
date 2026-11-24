"""Deterministic CPU policy example, NOT a model/performance simulation."""
import argparse
import json
from multi_chunk_core import MultiConfig, CreditLedger, Candidate, plan_batch


def demonstrate():
    config=MultiConfig(mode='fair',age_guard_s=0,max_unserved_steps=0)
    ledger=CreditLedger(config)
    ledger.sync({'user_A','user_B'})
    done={'A':0,'B':0}; lengths={'A':8192,'B':4096}; rows=[]
    log=[]
    for step in range(1,5):
        rows=[Candidate(k,'user_'+k,lengths[k]-v,v,0.,float(step-1),step-1,v>0)
              for k,v in done.items() if v<lengths[k]]
        plan=plan_batch(rows,budget=2048,alignment=256,new_slots=16-len([v for v in done.values() if v]),
                        partial_count=sum(0<v<lengths[k] for k,v in done.items()),
                        resident_by_user={'user_'+k:1 for k,v in done.items() if v},state_capacity=16,
                        credit={u:ledger.available(u) for u in ledger.active},config=config,now=float(step),step=step)
        ledger.grant(step,plan.grants)
        work=[(s.user,min(s.cap,s.remaining),0) for s in plan.shares]
        ledger.reserve(step,work);ledger.settle(step,work)
        for s in plan.shares:done[s.rid]+=min(s.cap,s.remaining)
        log.append({'step':step,'selected':[(s.rid,s.cap,s.reason) for s in plan.shares],
                    'processed':dict(done),'partial_requests':[k for k,v in done.items() if 0<v<lengths[k]]})
    return {'kind':'CPU_POLICY_EXAMPLE_NOT_GPU_MEASUREMENT','rounds':log,'ledger':ledger.snapshot()}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(demonstrate(),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
