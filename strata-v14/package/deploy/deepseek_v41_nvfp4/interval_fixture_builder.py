"""Build a SYNTHETIC mechanism fixture from explicitly supplied token-ID arrays.

No heuristic production-trace splicing. Shared prefixes here are constructed
by the caller; this must never be labelled representative production traffic.
"""
import argparse,json,hashlib
from pathlib import Path
from interval_bench import validate_fixture

def build(input_ids,tool_ids,*,vocab_size,context_length,workflows=8,calls=3,output_tokens=128,tool_delay_s=.2):
    if not 1<=workflows<=256 or not 1<=calls<=32:raise ValueError('invalid workflow/call bounds')
    result={'schema':'orvix-causal-fixture-v1','vocab_size':vocab_size,'context_length':context_length,
            'label':'constructed shared-prefix mechanism fixture; NOT original production workload',
            'workflows':[]}
    for i in range(workflows):
        steps=[]
        for j in range(calls):
            call={'max_new_tokens':output_tokens}
            if j<calls-1:call['tool']={'type':'synthetic.local','delay_s':tool_delay_s,'result_input_ids':tool_ids,'status':'ok'}
            steps.append(call)
        result['workflows'].append({'id':f'w{i}','user':f'u{i}','input_ids':input_ids,'calls':steps})
    return validate_fixture(result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_ids',type=Path,help='JSON array of token IDs from the actual model tokenizer')
    p.add_argument('tool_result_ids',type=Path,help='explicit JSON array; no implicit chat-template insertion')
    p.add_argument('out',type=Path);p.add_argument('--vocab-size',type=int,required=True);p.add_argument('--context-length',type=int,required=True)
    p.add_argument('--workflows',type=int,default=8);p.add_argument('--calls',type=int,default=3);p.add_argument('--output-tokens',type=int,default=128);p.add_argument('--tool-delay-s',type=float,default=.2)
    a=p.parse_args()
    f=build(json.loads(a.input_ids.read_text()),json.loads(a.tool_result_ids.read_text()),vocab_size=a.vocab_size,context_length=a.context_length,
            workflows=a.workflows,calls=a.calls,output_tokens=a.output_tokens,tool_delay_s=a.tool_delay_s)
    with a.out.open('x') as h:json.dump(f,h,ensure_ascii=False,indent=2);h.write('\n')
    print('Constructed mechanism fixture:',a.out,'SHA256',hashlib.sha256(a.out.read_bytes()).hexdigest())
if __name__=='__main__':main()
