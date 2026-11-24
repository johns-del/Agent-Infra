"""One authenticated, private control-stream owner for v7 (not a public API)."""
from __future__ import annotations
import argparse, asyncio, json, os, time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import httpx
from interval_core import IntervalConfig, validate_events
from chunk_control import private_endpoint
from compare_dev_replay import engine_state, KEY
from run_unified_campaign import DEFAULT

class IntervalClient:
    def __init__(self,client,state):
        if state.get('adapter_version')!='orvix-dsv41-interval-cache-v7': raise ValueError('v7 private runtime required')
        if not state.get('runtime_identity',{}).get('verified'): raise ValueError('unverified derived Python runtime')
        self.client=client;self.state=state;self.lock=asyncio.Lock();self.origin=time.monotonic()
        self.base=max(self.origin,float(state.get('clock',state.get('now',0))),float(state.get('interval_cache',{}).get('clock',0)))
        self.events=0;self.control_seconds=0.
    def now(self):return self.base+time.monotonic()-self.origin
    async def update(self,*,config=None,events=(),reset=False):
        async with self.lock:
            now=self.now()
            normalized=[dict(e,observed_at=now) if 'observed_at' not in e else dict(e) for e in events]
            validate_events(normalized,now)
            payload={'enabled':False,'now':now,'parameters':self.state['parameters'],'knobs':DEFAULT,
                     'cache':{'mode':'observe'},'prefill':{'mode':'off'},'chunk_fair':{'mode':'off'},
                     'multi_chunk':{'mode':'off'},'interval_events':normalized,'reset':reset}
            if config is not None:payload['interval_cache']=asdict(config.validate())
            t=time.perf_counter();r=await self.client.post('/set_internal_state',json={'server_args':{},KEY:payload})
            r.raise_for_status()
            if r.json()!=[True]:raise RuntimeError('v7 control rejected; drain engine, inspect capabilities and clock owner')
            self.control_seconds+=time.perf_counter()-t;self.events+=len(normalized)
    async def event(self,kind,request_id,*,tool_id=None,tool_type=None,status=None):
        e={'event_id':str(uuid4()),'kind':kind,'request_id':request_id}
        if tool_id is not None:e['tool_id']=tool_id
        if tool_type is not None:e['tool_type']=tool_type
        if status is not None:e['status']=status
        await self.update(events=[e])
    async def heartbeat(self,interval=.1):
        while True:
            await asyncio.sleep(interval);await self.update()
    async def read(self):
        _,state=await engine_state(self.client);return state
    async def wait_drained(self,timeout=60):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            state=await self.read();ic=state.get('interval_cache',{})
            if not ic.get('pending_h2d') and ic.get('idle',False):
                # Server proves engine, both transfer directions and tickets
                # are drained. Never flush merely to cancel uncompleted DMA.
                return state
            await self.update();await asyncio.sleep(.1)
        raise TimeoutError('engine/transfers not drained; do not force flush/reset')

async def run(args):
    headers={}
    if os.getenv('ORVIX_ENGINE_API_KEY'):headers['Authorization']='Bearer '+os.environ['ORVIX_ENGINE_API_KEY']
    async with httpx.AsyncClient(base_url=args.endpoint,headers=headers,trust_env=False,timeout=30) as h:
        _,state=await engine_state(h);control=IntervalClient(h,state)
        if args.apply:
            cfg=IntervalConfig(**json.loads(args.apply.read_text())).validate()
            await control.wait_drained();await control.update(config=cfg,reset=True)
        state=await control.read();print(json.dumps(state.get('interval_cache'),ensure_ascii=False,indent=2),flush=True)
        if args.heartbeat:
            task=asyncio.create_task(control.heartbeat(args.interval))
            try:
                if args.seconds:await asyncio.wait_for(asyncio.shield(task),args.seconds)
                else:await task
            except asyncio.TimeoutError:pass
            finally:
                task.cancel()
                try:await task
                except asyncio.CancelledError:pass

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--endpoint',type=private_endpoint,default='http://127.0.0.1:32781');p.add_argument('--apply',type=Path)
    p.add_argument('--heartbeat',action='store_true');p.add_argument('--interval',type=float,default=.1);p.add_argument('--seconds',type=float)
    a=p.parse_args()
    if not .05<=a.interval<=2:p.error('interval must be 0.05..2 s')
    if a.seconds is not None and a.seconds<=0:p.error('seconds must be positive')
    try:asyncio.run(run(a))
    except KeyboardInterrupt:print('Control clock stopped. No DMA is cancelled. Drain before reset; leases are no longer refreshed.')
if __name__=='__main__':main()
