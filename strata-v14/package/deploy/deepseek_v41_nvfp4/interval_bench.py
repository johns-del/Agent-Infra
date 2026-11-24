"""Controlled-continuity MECHANISM benchmark on one private v7 engine.

This does not relabel a modified production trace as the same workload. Every
next prompt contains the prior ACTUAL input+output and explicit tool suffix.
Cross-policy input hashes are checked: divergent generated trajectories are
reported as non-identical work, not as a strict same-input speedup result.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, math, os, statistics, time
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4
import httpx
from interval_core import IntervalConfig
from interval_control import IntervalClient
from chunk_control import private_endpoint
from compare_dev_replay import engine_state, engine_configuration, cache_flush_acknowledged, payload
from replay_stream import generate_with_measurements

METHODS=('B0','AA','O','K','E','H','KEH')

def config_for(name):
    base=IntervalConfig(mode='observe')
    if name in {'B0','AA'}:return base
    if name not in METHODS:raise ValueError('unknown method')
    return replace(base,mode='active',soft_keep=name in {'K','KEH'},eager_backup=name in {'E','KEH'},
                   demand_release=name in {'E','KEH'},queue_prefetch=name in {'H','KEH'},ready_order=name in {'O','H','KEH'})

def ids_hash(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()

def validate_fixture(f):
    if set(f)-{'schema','vocab_size','context_length','workflows','label'} or f.get('schema')!='orvix-causal-fixture-v1':raise ValueError('invalid mechanism fixture schema')
    vocab,context=f.get('vocab_size'),f.get('context_length')
    if type(vocab) is not int or vocab<2 or type(context) is not int or context<8:raise ValueError('explicit vocab/context required')
    seen=set()
    for w in f.get('workflows',[]):
        if set(w)-{'id','user','input_ids','calls','start_delay_s'}:raise ValueError('unknown workflow field')
        if not isinstance(w.get('id'),str) or w['id'] in seen or not isinstance(w.get('user'),str):raise ValueError('unique workflow/user required')
        seen.add(w['id']);length=0
        arrays=[w.get('input_ids')]
        calls=w.get('calls')
        if not isinstance(calls,list) or not calls:raise ValueError('calls required')
        for call in calls:
            if set(call)-{'max_new_tokens','tool'}:raise ValueError('unknown call field')
            n=call.get('max_new_tokens')
            if type(n) is not int or not 1<=n<=16384:raise ValueError('bounded output count required')
            length+=n;tool=call.get('tool')
            if tool is not None:
                if set(tool)-{'type','delay_s','result_input_ids','status'}:raise ValueError('unknown tool field')
                if not isinstance(tool.get('type'),str) or not tool['type']:raise ValueError('tool type required')
                if tool.get('status','ok') not in {'ok','error','timeout','cancelled'}:raise ValueError('invalid terminal status')
                delay=tool.get('delay_s',0)
                if isinstance(delay,bool) or not isinstance(delay,(int,float)) or not math.isfinite(delay) or not 0<=delay<=120:raise ValueError('invalid synthetic tool delay')
                arrays.append(tool.get('result_input_ids'))
        for ids in arrays:
            if not isinstance(ids,list) or not ids or any(type(x) is not int or not 0<=x<vocab for x in ids):raise ValueError('valid token-id segments required')
            length+=len(ids)
        if length>=context:raise ValueError('entire causal chain exceeds context')
        delay=w.get('start_delay_s',0)
        if isinstance(delay,bool) or not isinstance(delay,(int,float)) or not math.isfinite(delay) or not 0<=delay<=120:raise ValueError('invalid arrival delay')
    if not seen or len(seen)>256:raise ValueError('1..256 workflows required')
    return f

async def workload(h,control,fixture,*,stream_metrics=False,concurrency=32):
    rows=[];sem=asyncio.Semaphore(concurrency);start=time.monotonic()
    async def workflow(w):
        await asyncio.sleep(w.get('start_delay_s',0));offered=time.monotonic()
        async with sem:
            current=list(w['input_ids']);prior=None
            for i,call in enumerate(w['calls']):
                if prior:await control.event('request_ready',prior)
                rid=str(uuid4());expected=list(current);ready=time.monotonic()
                body=payload({'input_ids':current,'output_budget':call['max_new_tokens']},w['user'],'foreground',control.now(),request_id=rid,
                             identity={'session_id':w['id'],'scope_id':'main','graph_id':w['id']})
                if stream_metrics:
                    result,measure=await generate_with_measurements(h,body,ready)
                else:
                    response=await h.post('/generate',json=body)
                    response.raise_for_status(); result=response.json(); measure={}
                output=result.get('output_ids')
                if not isinstance(output,list) or len(output)!=call['max_new_tokens'] or any(type(x) is not int or not 0<=x<fixture['vocab_size'] for x in output):
                    raise ValueError('missing/malformed output IDs: cannot assert causal continuity')
                end=time.monotonic();meta=result.get('meta_info',{})
                if current!=expected:raise RuntimeError('input mutated while in flight')
                rows.append({'workflow':w['id'],'user':w['user'],'call':i,'input_tokens':len(current),'output_tokens':len(output),
                             'input_sha256':ids_hash(current),'output_sha256':ids_hash(output),'http_s':end-ready,
                             'sent_relative_s':ready-start,'done_relative_s':end-start,'meta_info':meta,'measurement':measure,
                             'causal_prefix_verified':prior is not None})
                current=current+output;prior=rid
                tool=call.get('tool')
                if tool is not None:
                    tid=str(uuid4());await control.event('tool_start',rid,tool_id=tid,tool_type=tool['type'])
                    # Known delay/result belong to the DRIVER only, never to policy metadata.
                    await asyncio.sleep(tool.get('delay_s',0))
                    await control.event('tool_end',rid,tool_id=tid,status=tool.get('status','ok'))
                    current=current+tool['result_input_ids']
            await control.event('workflow_end',prior)
            return {'workflow':w['id'],'user':w['user'],'offered_to_done_s':time.monotonic()-offered}
    tasks=[asyncio.create_task(workflow(w)) for w in fixture['workflows']]
    try:flows=await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True);raise
    return {'wall_s':time.monotonic()-start,'requests':rows,'workflows':flows,'fixture_kind':'controlled_continuity_synthetic_tools',
            'not_production_replay':True,'max_active_workflows':concurrency}

async def trial(h,control,fixture,method,*,stream_metrics=False,concurrency=32):
    await control.wait_drained()
    flush=await h.post('/flush_cache');cache_flush_acknowledged(flush)
    cfg=config_for(method);await control.update(config=cfg,reset=True)
    engine_before,state=await engine_state(h)
    observed=state['interval_cache']
    if observed['config']!=asdict(cfg) or observed['effective_mode']!=cfg.mode:raise RuntimeError('requested interval mode not actually active')
    before_cfg=engine_configuration(engine_before)
    control_before=control.control_seconds
    heart=asyncio.create_task(control.heartbeat())
    job=asyncio.create_task(workload(h,control,fixture,stream_metrics=stream_metrics,concurrency=concurrency))
    try:
        done,_=await asyncio.wait({job,heart},return_when=asyncio.FIRST_COMPLETED)
        if heart in done:
            job.cancel();await asyncio.gather(job,return_exceptions=True);heart.result();raise RuntimeError('heartbeat stopped')
        result=job.result();await control.wait_drained()
    finally:
        heart.cancel();await asyncio.gather(heart,return_exceptions=True)
    engine_after,state_after=await engine_state(h)
    if engine_configuration(engine_after)!=before_cfg:raise RuntimeError('engine configuration changed inside trial')
    if state_after['interval_cache']['effective_mode']!=cfg.mode or state_after['interval_cache']['config']!=asdict(cfg):
        raise RuntimeError('interval configuration degraded/changed inside trial; result invalid')
    result.update(method=method,config=asdict(cfg),before=observed,after=state_after['interval_cache'],engine=before_cfg,
                  runtime_identity=state['runtime_identity'],stream_metrics=stream_metrics,
                  execution_source='remote_engine_unverified_hardware',automatic_promotion=False,
                  counter_delta={k:v-observed.get('counts',{}).get(k,0) for k,v in state_after['interval_cache'].get('counts',{}).items()},
                  native_writeback_calls_delta=state_after['interval_cache']['native_writeback_calls']-observed['native_writeback_calls'],
                  native_writeback_wall_s_delta=state_after['interval_cache']['native_writeback_wall_s']-observed['native_writeback_wall_s'],
                  control_messages_seconds=control.control_seconds-control_before)
    result['summary']=describe(result)
    return result

def describe(result):
    def q(xs,p):
        xs=sorted(xs);return xs[max(0,math.ceil(p*len(xs))-1)] if xs else None
    flows=[w['offered_to_done_s'] for w in result['workflows']]
    req=result['requests'];cache=[];queues=[];ttft=[];gaps=[]
    for r in req:
        m=r.get('meta_info',{});v=m.get('cached_tokens')
        if type(v) in (float,int) and 0<=v<=r['input_tokens']:cache.append((r['input_tokens'],v))
        for k in ('scheduler_queue_time','queue_time_s','queue_time'):
            if type(m.get(k)) in (float,int):queues.append(m[k]);break
        obs=r.get('measurement',{})
        if obs.get('client_ttft_s') is not None:ttft.append(obs['client_ttft_s'])
        if obs.get('client_max_delivery_gap_s') is not None:gaps.append(obs['client_max_delivery_gap_s'])
    return {'workflow_p50_s':q(flows,.5),'workflow_p95_s':q(flows,.95),
            'request_queue_p95_s':q(queues,.95),'queue_observed_calls':len(queues),
            'client_ttft_p95_s':q(ttft,.95),'client_max_delivery_gap_p95_s':q(gaps,.95),
            'cache_observed_calls':len(cache),'cache_missing_input_proxy':sum(a-b for a,b in cache) if cache else None,
            'cache_proxy_is_not_retraction_recompute_or_gpu_time':True,
            'completed_workflows_per_second':len(flows)/result['wall_s']}

def paired(a,b):
    key=lambda r:(r['workflow'],r['call'])
    aa={key(r):r for r in a['requests']};bb={key(r):r for r in b['requests']}
    same=aa.keys()==bb.keys() and all(aa[k]['input_sha256']==bb[k]['input_sha256'] and aa[k]['output_tokens']==bb[k]['output_tokens'] for k in aa)
    protocol_same=(a.get('stream_metrics')==b.get('stream_metrics') and
                   a.get('max_active_workflows')==b.get('max_active_workflows') and
                   a.get('fixture_kind')==b.get('fixture_kind'))
    users={}
    for name in {w['user'] for w in a['workflows']}:
        av=[w['offered_to_done_s'] for w in a['workflows'] if w['user']==name];bv=[w['offered_to_done_s'] for w in b['workflows'] if w['user']==name]
        users[name]=100*(statistics.mean(bv)/statistics.mean(av)-1)
    return {'same_actual_inputs':same,'baseline':a['method'],'candidate':b['method'],
            'descriptive_wall_delta_pct':100*(b['wall_s']/a['wall_s']-1),
            'max_paired_user_mean_delta_pct':max(users.values()),'per_user_mean_delta_pct':users,
            'same_protocol':protocol_same,
            'same_work_eligible':same and protocol_same and a['engine']==b['engine'] and a['runtime_identity']==b['runtime_identity'],
            'performance_claim_allowed':False,'reason':'mechanism experiment only; inspect repeat noise, GPU correctness and compile stability; divergent prompts are not same-input comparison'}

async def run(args):
    fixture=validate_fixture(json.loads(args.fixture.read_text()))
    args.out.mkdir(exist_ok=False,parents=True)
    (args.out/'fixture.json').write_text(json.dumps(fixture,ensure_ascii=False,indent=2)+'\n')
    headers={'Authorization':'Bearer '+os.environ['ORVIX_ENGINE_API_KEY']} if os.getenv('ORVIX_ENGINE_API_KEY') else {}
    async with httpx.AsyncClient(base_url=args.endpoint,headers=headers,timeout=1200,trust_env=False) as h:
        _,state=await engine_state(h);control=IntervalClient(h,state);summary=[]
        # Identical warmup in each config before its measured run, then flush/reset.
        warm={'schema':fixture['schema'],'vocab_size':fixture['vocab_size'],'context_length':fixture['context_length'],'workflows':fixture['workflows'][:1]}
        for method in args.methods.split(','):
            if method not in METHODS or method=='B0':raise ValueError('methods must be AA,O,K,E,H,KEH')
            for rep in range(args.repeats):
                order=['B0',method] if rep%2==0 else [method,'B0'];results={}
                for name in order:
                    if not args.skip_warmup:await trial(h,control,warm,name,stream_metrics=args.stream_metrics,concurrency=1)
                    try:r=await trial(h,control,fixture,name,stream_metrics=args.stream_metrics,concurrency=args.concurrency)
                    except BaseException as exc:
                        (args.out/f'{method}-{rep}-{name}-FAILED.json').write_text(json.dumps({'valid_performance_result':False,'error':type(exc).__name__+':'+str(exc)},indent=2));raise
                    r['warmup_performed']=not args.skip_warmup
                    results[name]=r;(args.out/f'{method}-{rep}-{name}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
                summary.append(paired(results['B0'],results[method])|{'repeat':rep,'order':order})
                (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        # Retain native strong baseline on exit, never automatically promote.
        await control.wait_drained();await control.update(config=IntervalConfig(mode='off'),reset=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--endpoint',type=private_endpoint,default='http://127.0.0.1:32781');p.add_argument('--methods',default='K,E,H,KEH')
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--concurrency',type=int,default=32);p.add_argument('--stream-metrics',action='store_true')
    p.add_argument('--skip-warmup',action='store_true',help='debug only; mark results unqualified')
    a=p.parse_args()
    if not 1<=a.concurrency<=256 or not 1<=a.repeats<=20:p.error('invalid bounds')
    asyncio.run(run(a))
if __name__=='__main__':main()
