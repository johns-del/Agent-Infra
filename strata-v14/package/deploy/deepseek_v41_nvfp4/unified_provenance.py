"""Verify the derived Python runtime once at engine installation, not in a tick."""
from hashlib import sha256
import json
from pathlib import Path


def runtime_identity(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    path = root / 'orvix-runtime-identity.json'
    if not path.is_file():
        return {'verified': False, 'reason': 'derive the private runtime with v3 prepare-runtime'}
    data = json.loads(path.read_text())
    inventory = data.get('python_sha256', {})
    if not inventory:
        return {'verified': False, 'reason': 'empty runtime inventory'}
    actual = {str(p.relative_to(root)) for p in (root / 'sglang').rglob('*.py')
              if '__pycache__' not in p.parts}
    bad = sorted(actual - inventory.keys())
    for name, expected in inventory.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            return {'verified': False, 'reason': 'invalid inventory path'}
        target = root / relative
        if not target.is_file() or sha256(target.read_bytes()).hexdigest() != expected:
            bad.append(name)
    fingerprint = sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'verified': not bad, 'python_fingerprint': fingerprint,
            'python_files': len(inventory), 'mismatches': bad[:20],
            'scope': 'derived Python source only; excludes weights, binaries, drivers and GPU correctness'}
