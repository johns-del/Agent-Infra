"""Derive an isolated runtime with ranking, lifecycle-cache and prefill adapters.

The full, qualified DeepSeek runtime is an input. ``runtime-excerpts`` in the
delivery is a source-review snapshot, not a standalone SGLang installation.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from unified_gate_patch import instrument_scheduler
from chunk_runtime_patch import instrument_chunk_scheduler, qualify_admission_source


SCHEDULER = Path("sglang/srt/managers/scheduler.py")
IPC = Path("sglang/srt/managers/io_struct.py")
MODULES = Path("sglang/srt/observability")
ANCHOR = "        self.init_schedule_policy()\n"
INSTALL = "        from sglang.srt.observability.unified_rank import install\n        install(self)\n"
ADAPTERS = ("unified_rank.py", "unified_cache_control.py", "unified_prefill.py", "unified_prediction.py", "unified_admission.py", "unified_provenance.py", "chunk_fair_core.py", "unified_chunk.py", "multi_chunk_core.py", "multi_chunk_compat.py", "multi_chunk_runtime.py")
REQUIRED_MODULES = (
    "sglang.__init__", "sglang.launch_server", "sglang.srt.server_args",
    "sglang.srt.runtime_context", "sglang.srt.managers.schedule_batch",
    "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
    "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler",
    "sglang.srt.mem_cache.unified_radix_cache",
    "sglang.srt.models.deepseek_v4", "sglang.srt.models.deepseek_v4_dspark",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify_source(source):
    missing = []
    for module in REQUIRED_MODULES:
        path = source / Path(*module.split("."))
        if not path.with_suffix(".py").is_file() and not (path / "__init__.py").is_file():
            missing.append(module)
    if missing:
        raise ValueError("full qualified runtime required; source excerpts are not runnable. Missing: " + ", ".join(missing))


def derive(source, target, core):
    here = Path(__file__).resolve().parent
    source, target = source.resolve(), target.resolve()
    if target.is_relative_to(source) or source.is_relative_to(target) or target.exists():
        raise ValueError("target must be a new directory outside the source")
    qualify_source(source)
    native_policy = source / "sglang/srt/managers/schedule_policy.py"
    if not native_policy.is_file():
        raise ValueError("qualified PrefillAdder source required")
    admission_hashes = qualify_admission_source(native_policy.read_text())
    original = (source / SCHEDULER).read_text()
    if original.count(ANCHOR) != 1 or "self.waiting_queue.replace(" not in original:
        raise ValueError("scheduler anchors differ from the qualified DeepSeek runtime")
    if "# ORVIX_FAIR_CHUNK_V5" in original:
        patched = instrument_chunk_scheduler(original)
    elif INSTALL in original:
        if original.count(INSTALL) != 1 or ANCHOR + INSTALL not in original:
            raise ValueError("unexpected existing adapter installation")
        patched = original
    elif "observability.unified_rank" in original:
        raise ValueError("unrecognized existing adapter installation")
    else:
        patched = original.replace(ANCHOR, ANCHOR + INSTALL)
    patched = instrument_scheduler(patched)
    patched = instrument_chunk_scheduler(patched)
    ipc = (source / IPC).read_text()
    classes = [n for n in ast.parse(ipc).body if isinstance(n, ast.ClassDef)
               and n.name == "SetInternalStateReq"]
    if len(classes) != 1:
        raise ValueError("unqualified scheduler control IPC")
    fields = [n for n in classes[0].body if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Name) and n.target.id == "orvix_unified"]
    if fields:
        if len(fields) != 1 or ast.unparse(fields[0].annotation) != "dict[str, Any] | None":
            raise ValueError("unqualified existing scheduler IPC field")
        patched_ipc = ipc
    else:
        lines = ipc.splitlines(keepends=True)
        lines.insert(classes[0].end_lineno, "    orvix_unified: dict[str, Any] | None = None\n")
        patched_ipc = "".join(lines)
    changes = {SCHEDULER: patched, IPC: patched_ipc}
    for path, data in changes.items():
        ast.parse(data, filename=str(path))
    shutil.copytree(source, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    for path in (SCHEDULER.parent, IPC.parent, MODULES):
        if not (target / path).resolve().is_relative_to(target):
            raise ValueError("derived source package points outside its private tree")
    for path, data in changes.items():
        # A staged source may use symlinks. Always write a private regular file.
        out = target / path
        if out.is_symlink():
            out.unlink()
        out.write_text(data)
    adapters = [(here / name, name) for name in ADAPTERS] + [(core, "unified_core.py")]
    for src, _ in adapters:
        ast.parse(src.read_text(), filename=str(src))
    (target / MODULES).mkdir(parents=True, exist_ok=True)
    for src, name in adapters:
        out = target / MODULES / name
        if out.is_symlink():
            out.unlink()
        out.write_bytes(src.read_bytes())
    inventory = {str(path.relative_to(target)): digest(path)
                 for path in sorted((target / "sglang").rglob("*.py"))
                 if "__pycache__" not in path.parts}
    (target / "orvix-runtime-identity.json").write_text(
        json.dumps({"python_sha256": inventory}, indent=2) + "\n")
    return {
        "runtime_python_fingerprint": hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source": str(source), "target": str(target),
        "source_scheduler_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "source_ipc_sha256": hashlib.sha256(ipc.encode()).hexdigest(),
        "gpu_validated": False,
        "qualified_admission_ast": admission_hashes,
        "requires_full_source_runtime": True,
        "changes": {str(path): digest(target / path) for path in
                    [*changes, *(MODULES / name for _, name in adapters)]},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("core", type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.target, args.core), indent=2))
