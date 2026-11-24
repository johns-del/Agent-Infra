"""CPU policy example only: no GPU simulation, no speedup numbers."""
from dataclasses import asdict
import json
from interval_core import IntervalConfig,Lifecycle,QueueItem,order_window


def demonstrate():
    cfg=IntervalConfig(mode='active');l=Lifecycle(cfg);r=l.register('A','alice')
    def e(kind,t,**kw):return dict(kind=kind,request_id='A',event_id=kind+str(t),observed_at=t,**kw)
    l.event(e('tool_start',0.,tool_id='tool1',tool_type='local-test'))
    rows=[{'clock':0.,'state':'tool_wait','keep_until':l.keep_until(r,0.),'action':'soft LRU preference, no hard lock'}]
    l.event(e('tool_end',.2,tool_id='tool1',status='ok'))
    rows.append({'clock':.2,'state':'tool_done','tools_waiting':len(r.tools),'action':'stop tool protection; cancel any planned demotion'})
    queue=[QueueItem('A','alice',0,0,25000,256,True),QueueItem('B','bob',0,0,0,30000),QueueItem('C','carol',0,10000,10000,256)]
    ready=order_window(queue,.3,cfg,{}, {},2048)
    return {'kind':'CPU control illustration, NOT GPU benchmark','events':rows,
            'while_A_is_copying':[x.rid for x in ready],
            'explanation':'A is fenced. C can fill one native prefill round. B cannot occupy the single partial slot.',
            'native_chunk_tokens_unchanged':2048,'gpu_validated':False}

def main():print(json.dumps(demonstrate(),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
