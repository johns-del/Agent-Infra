"""Build a NEW isolated v7/v9 runtime from qualified native/v5 source. No downloads."""
from __future__ import annotations
import argparse, ast, hashlib, json, shutil, tempfile
from pathlib import Path
from prepare_unified_runtime import derive as derive_base, qualify_source, SCHEDULER, MODULES
from chunk_runtime_patch import instrument_chunk_scheduler, qualify_admission_source
from unified_gate_patch import instrument_scheduler
from interval_patch import patch_scheduler

COMMIT='858073e4f6d246bd6e3baae1d4e07919a578a04f'
EXTRAS=('interval_core.py','interval_probe.py','interval_native.py','interval_runtime.py','forced_trace_runtime.py')
TREE_CORE = Path('sglang/srt/mem_cache/unified_cache/unified_tree_core.py')
REMOTE_BLOBS={
 'sglang/srt/managers/cache_controller.py':'357e17404e95245cc8443247585e8610f19c5508',
 'sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py':'e0a2503eac99af5d5fb305efc56d4b47bc903609',
 'sglang/srt/mem_cache/l2_transfer.py':'387501d69ddbf3322b631e473449be7b2c4cd0be'}

def blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def plan(source):
    source=Path(source).resolve(); qualify_source(source)
    s=(source/SCHEDULER).read_text()
    if '# ORVIX_MULTI_PREFILL_V6' in s: raise ValueError('use original/native or v5 source, not v6 lifecycle-patched source')
    qualify_admission_source((source/'sglang/srt/managers/schedule_policy.py').read_text())
    planned=patch_scheduler(instrument_chunk_scheduler(instrument_scheduler(s)))
    package=Path(__file__).resolve().parents[2]
    manifest=json.loads((package/'native-contract-v7.json').read_text())
    actual={}
    for name,sha in manifest['sha256'].items():
        p=source/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha: raise ValueError('unqualified native cache source: '+name)
        actual[name]=sha
    for name,expected in REMOTE_BLOBS.items():
        p=source/name
        if not p.is_file() or blob(p.read_bytes())!=expected: raise ValueError('unqualified transfer source: '+name)
        actual[name]=expected
    return planned,actual

def stabilize_equal_recency(source: str) -> str:
    """Use the shared NodeId to break a native eviction heap tie across TP ranks."""
    old = '        return self.last_access_time < other.last_access_time'
    new = (
        '        if self.last_access_time == other.last_access_time:\n'
        '            return self.id < other.id\n'
        '        return self.last_access_time < other.last_access_time'
    )
    if source.count(old) != 1:
        raise ValueError('native UnifiedTreeNode comparator does not match the pinned contract')
    return source.replace(old, new, 1)

def derive(source,target,core,check_only=False):
    source,target,core=map(lambda p:Path(p).resolve(),(source,target,core))
    if target.exists() or target.is_relative_to(source) or source.is_relative_to(target): raise ValueError('NEW target outside source required')
    _,native=plan(source); ast.parse(core.read_text())
    receipt={'schema':'orvix-interval-runtime-v7','pinned_commit':COMMIT,'source':str(source),'target':str(target),
             'native_contracts':native,'source_read_only':True,'check_only':check_only,'gpu_validated':False,
             'substrate':'native_single_partial_no_early_result_fence'}
    if check_only:return receipt
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+target.name+'-v7-',dir=target.parent)); private=stage/'runtime'
    try:
        derive_base(source,private,core)
        p=private/SCHEDULER; p.write_text(patch_scheduler(p.read_text()))
        here=Path(__file__).resolve().parent
        for name in EXTRAS:
            text=(here/name).read_text();ast.parse(text,filename=name)
            out=private/MODULES/name
            if out.is_symlink():out.unlink()
            out.write_text(text)
        tree = private / TREE_CORE
        updated = stabilize_equal_recency(tree.read_text())
        ast.parse(updated, filename=str(TREE_CORE))
        if tree.is_symlink():
            tree.unlink()
        tree.write_text(updated)
        inventory={str(p.relative_to(private)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((private/'sglang').rglob('*.py')) if '__pycache__' not in p.parts}
        receipt['runtime_python_fingerprint']=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        (private/'orvix-runtime-identity.json').write_text(json.dumps({'python_sha256':inventory},indent=2)+'\n')
        (private/'orvix-v7-derivation.json').write_text(json.dumps(receipt,indent=2)+'\n')
        private.rename(target)
    finally:shutil.rmtree(stage,ignore_errors=True)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('target',type=Path);p.add_argument('core',type=Path);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();print(json.dumps(derive(a.source,a.target,a.core,a.check_only),indent=2))
if __name__=='__main__':main()
