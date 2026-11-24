"""Create an isolated v6 runtime; never modify the source or download code.

Lifecycle files are qualified by exact Git blob hashes from the pinned runtime.
All transformations are planned and parsed BEFORE creating a target tree.
The original v5 derivation is retained as a separate rollback entry point.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from prepare_unified_runtime import derive as derive_v5, qualify_source, SCHEDULER, MODULES
from chunk_runtime_patch import instrument_chunk_scheduler, qualify_admission_source
from unified_gate_patch import instrument_scheduler
from multi_runtime_patch import BLOBS, TRANSFORMS, PINNED_COMMIT, MARKER, blob_hash, patch_scheduler

EXTRA_MODULES = ('multi_chunk_core.py', 'multi_chunk_compat.py', 'multi_chunk_runtime.py')


def plan_changes(source: Path) -> tuple[dict, dict]:
    """Read-only qualification. Hash mismatch is never bypassed in production."""
    source = source.resolve()
    qualify_source(source)
    scheduler = (source / SCHEDULER).read_text()
    if MARKER in scheduler:
        raise ValueError('source is already v6; use the original qualified source/v5 private tree')
    qualify_admission_source((source / 'sglang/srt/managers/schedule_policy.py').read_text())
    changes = {str(SCHEDULER): patch_scheduler(instrument_chunk_scheduler(instrument_scheduler(scheduler)))}
    hashes = {}
    for name, expected in BLOBS.items():
        file = source / name
        if not file.is_file():
            raise ValueError(f'missing complete lifecycle source: {name}')
        data = file.read_bytes()
        actual = blob_hash(data)
        if actual != expected:
            raise ValueError(f'unsupported lifecycle source {name}: expected blob {expected}, found {actual}; '
                             'do not overwrite it with a different version')
        changes[name] = TRANSFORMS[name](data.decode('utf-8'))
        hashes[name] = actual
    for name, text in changes.items():
        ast.parse(text, filename=name)
    return changes, hashes


def derive(source: Path, target: Path, core: Path, *, check_only: bool = False):
    source, target, core = source.resolve(), target.resolve(), core.resolve()
    if target.exists() or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('target must be a NEW directory outside the source')
    # No source-derived import is executed by the installer.
    ast.parse(core.read_text(), filename=str(core))
    changes, native_hashes = plan_changes(source)
    receipt = {'schema': 'orvix-multi-runtime-v6', 'pinned_upstream_commit': PINNED_COMMIT,
               'source': str(source), 'target': str(target), 'gpu_validated': False,
               'source_lifecycle_blobs': native_hashes, 'source_read_only': True,
               'changed_native_files': sorted(changes), 'check_only': check_only}
    if check_only:
        return receipt
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+target.name+'-v6-', dir=target.parent))
    private = stage / 'runtime'
    try:
        base = derive_v5(source, private, core)
        # The base derivation also inserts the control-plane install anchor.
        scheduler_text = (private / SCHEDULER).read_text()
        changes[str(SCHEDULER)] = patch_scheduler(scheduler_text)
        for name, text in changes.items():
            file = private / name
            if not file.parent.resolve().is_relative_to(private):
                raise ValueError('native parent escapes private tree: '+name)
            if file.is_symlink():
                file.unlink()
            file.write_text(text)
        here = Path(__file__).resolve().parent
        for name in EXTRA_MODULES:
            source_file = here / name
            ast.parse(source_file.read_text(), filename=name)
            file = private / MODULES / name
            if file.is_symlink():
                file.unlink()
            file.write_bytes(source_file.read_bytes())
        inventory = {str(f.relative_to(private)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in sorted((private / 'sglang').rglob('*.py')) if '__pycache__' not in f.parts}
        fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        (private / 'orvix-runtime-identity.json').write_text(json.dumps({'python_sha256': inventory}, indent=2)+'\n')
        receipt.update(runtime_python_fingerprint=fingerprint, qualified_admission_ast=base['qualified_admission_ast'])
        receipt['changes'] = {name: inventory[name] for name in changes}
        receipt['changes'].update({str(MODULES/name): inventory[str(MODULES/name)] for name in EXTRA_MODULES})
        (private / 'orvix-v6-derivation.json').write_text(json.dumps(receipt, indent=2)+'\n')
        # Target creation happens only after a complete staging tree exists.
        private.rename(target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('core', type=Path)
    parser.add_argument('--check-only', action='store_true', help='validate without creating a runtime')
    args=parser.parse_args()
    print(json.dumps(derive(args.source,args.target,args.core,check_only=args.check_only),indent=2))

if __name__=='__main__':
    main()
