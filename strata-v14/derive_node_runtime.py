#!/usr/bin/env python3
"""Derive an isolated T1+Ours SGLang runtime on any cluster node without mutating source bases."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

PCIE_OLD_SNIPPET = "    rows = [r for r in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if r <= max_rows]\n"
PCIE_NEW_SNIPPET = (
    "    candidates = (1, 2, 4, 6, 8, 12, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 384, 512)\n"
    "    rows = [r for r in candidates if r <= max_rows]\n"
)

SERVING_CHAT_REL = Path("sglang/srt/entrypoints/openai/serving_chat.py")
FP8_CONFIGS_REL = Path("sglang/kernels/ops/gemm/configs")
SERVING_CHAT_ANCHOR = "        # Process messages and apply chat template\n        processed_messages = self._process_messages(request, is_multimodal)\n"
SERVING_CHAT_HOOK = (
    "        # ORVIX_CAUSAL_PREFIX_STABILIZER_V1\n"
    "        try:\n"
    "            from sglang.srt.observability.chat_prefix_hook import stabilize_chat_processed_messages\n"
    "            stabilize_chat_processed_messages(request, processed_messages)\n"
    "        except Exception:\n"
    "            pass\n"
)


def patch_pcie_ipc_ar(root: Path) -> bool:
    target = root / "sglang/srt/distributed/device_communicators/pcie_ipc_ar.py"
    text = target.read_text(encoding="utf-8")
    if "4, 6, 8, 12," in text:
        return False
    if text.count(PCIE_OLD_SNIPPET) != 1:
        raise ValueError(f"Unexpected _tune_batches anchor in {target}")
    patched = text.replace(PCIE_OLD_SNIPPET, PCIE_NEW_SNIPPET, 1)
    ast.parse(patched, filename=str(target))
    if target.is_symlink():
        target.unlink()
    target.write_text(patched, encoding="utf-8")
    return True


def install_chat_prefix_stabilizer(runtime_root: Path, pkg_deploy_dir: Path, here_dir: Path) -> None:
    obs_dir = runtime_root / "sglang/srt/observability"
    obs_dir.mkdir(parents=True, exist_ok=True)
    for src_path, dst_name in (
        (pkg_deploy_dir / "causal_prefix_stabilizer.py", "causal_prefix_stabilizer.py"),
        (here_dir / "chat_prefix_hook.py", "chat_prefix_hook.py"),
    ):
        src_text = src_path.read_text(encoding="utf-8")
        ast.parse(src_text, filename=dst_name)
        dst_path = obs_dir / dst_name
        if dst_path.is_symlink():
            dst_path.unlink()
        dst_path.write_text(src_text, encoding="utf-8")

    sc_path = runtime_root / SERVING_CHAT_REL
    sc_text = sc_path.read_text(encoding="utf-8")
    if "ORVIX_CAUSAL_PREFIX_STABILIZER_V1" not in sc_text:
        if sc_text.count(SERVING_CHAT_ANCHOR) != 1:
            raise ValueError(f"Expected single _process_messages anchor in {sc_path}")
        sc_text = sc_text.replace(SERVING_CHAT_ANCHOR, SERVING_CHAT_ANCHOR + SERVING_CHAT_HOOK, 1)
        ast.parse(sc_text, filename=str(sc_path))
        if sc_path.is_symlink():
            sc_path.unlink()
        sc_path.write_text(sc_text, encoding="utf-8")


def install_fp8_gemm_configs(runtime_root: Path, fp8_configs_tar: Path) -> dict[str, object]:
    """Install node-tuned W8A8 block FP8 GEMM Triton JSON configs into a private runtime."""
    runtime_root = runtime_root.resolve()
    tar_path = fp8_configs_tar.resolve()
    if not tar_path.is_file():
        raise FileNotFoundError(tar_path)

    cfg_dir = runtime_root / FP8_CONFIGS_REL
    if cfg_dir.is_symlink():
        real_src = cfg_dir.resolve()
        cfg_dir.unlink()
        shutil.copytree(real_src, cfg_dir, symlinks=False)
    else:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    if not cfg_dir.resolve().is_relative_to(runtime_root):
        raise ValueError(f"FP8 config directory escapes runtime root: {cfg_dir}")

    installed: list[dict[str, str]] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            raw = member.name
            rel = raw[2:] if raw.startswith("./") else raw
            if not rel or "/" in rel or "\\" in rel or ".." in rel or rel.startswith(".") or not rel.endswith(".json"):
                raise ValueError(f"Unexpected entry in FP8 config archive: {member.name!r}")
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            parsed = json.loads(data.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object in FP8 config {rel}")
            dst = cfg_dir / rel
            if dst.is_symlink():
                dst.unlink()
            dst.write_bytes(data)
            installed.append({"name": rel, "sha256": hashlib.sha256(data).hexdigest()})

    if not installed:
        raise ValueError(f"No JSON config files found in {tar_path}")

    installed.sort(key=lambda item: item["name"])
    return {
        "archive": str(tar_path),
        "archive_sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
        "target_dir": str(cfg_dir),
        "count": len(installed),
        "files": installed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-source", type=Path, default=None)
    ap.add_argument("--target-runtime", type=Path, required=True)
    ap.add_argument("--package-dir", type=Path, default=None)
    ap.add_argument("--fp8-configs-tar", type=Path, default=None)
    ap.add_argument("--install-fp8-configs-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.umask(0o077)
    target = args.target_runtime.resolve()

    if args.install_fp8_configs_only:
        if args.fp8_configs_tar is None:
            raise ValueError("--fp8-configs-tar is required with --install-fp8-configs-only")
        meta_path = target / "orvix-trinode-runtime.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Existing derived runtime metadata not found: {meta_path}")
        fp8_receipt = install_fp8_gemm_configs(target, args.fp8_configs_tar)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["fp8_gemm_configs"] = fp8_receipt
        meta.setdefault("files_verified", {})["fp8_gemm_configs"] = fp8_receipt["count"] > 0
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(fp8_receipt, indent=2))
        return

    if args.base_source is None or args.package_dir is None:
        raise ValueError("--base-source and --package-dir are required unless --install-fp8-configs-only is set")

    here_dir = Path(__file__).resolve().parent
    pkg_dir = args.package_dir.resolve()
    deploy_dir = pkg_dir / "deploy/deepseek_v41_nvfp4"
    core_path = pkg_dir / "gateway/src/trace_collector/unified_core.py"
    base_src = args.base_source.resolve()

    if not base_src.is_dir():
        raise FileNotFoundError(base_src)
    if not (deploy_dir / "prepare_interval_runtime.py").is_file() or not core_path.is_file():
        raise FileNotFoundError("Missing interval runtime derivation files in --package-dir")

    if target.exists():
        if not args.force:
            raise FileExistsError(f"{target} already exists; pass --force to rebuild")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(deploy_dir))
    import prepare_interval_runtime as pir  # noqa: E402

    with tempfile.TemporaryDirectory(prefix=".stage-src-", dir=str(target.parent)) as tmp_dir:
        staged_src = Path(tmp_dir) / "source"
        shutil.copytree(base_src, staged_src, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        pcie_patched = patch_pcie_ipc_ar(staged_src)
        receipt = pir.derive(staged_src, target, core_path)

    install_chat_prefix_stabilizer(target, deploy_dir, here_dir)
    fp8_receipt = (
        install_fp8_gemm_configs(target, args.fp8_configs_tar)
        if args.fp8_configs_tar is not None
        else None
    )

    py_files = {
        str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((target / "sglang").rglob("*.py"))
        if "__pycache__" not in p.parts
    }
    full_fp = hashlib.sha256(json.dumps(py_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    meta = {
        "schema": "orvix-trinode-t1-ours-runtime/1",
        "base_source": str(base_src),
        "target_runtime": str(target),
        "pcie_ipc_ar_patched": pcie_patched,
        "interval_derivation_receipt": receipt,
        "final_python_fingerprint": full_fp,
        "files_verified": {
            "pcie_ipc_ar": "4, 6, 8, 12," in (target / "sglang/srt/distributed/device_communicators/pcie_ipc_ar.py").read_text(),
            "serving_chat_hook": "ORVIX_CAUSAL_PREFIX_STABILIZER_V1" in (target / SERVING_CHAT_REL).read_text(),
            "causal_prefix_stabilizer": (target / "sglang/srt/observability/causal_prefix_stabilizer.py").is_file(),
            "interval_v7_patch": (target / "orvix-v7-derivation.json").is_file(),
            "fp8_gemm_configs": bool(fp8_receipt and fp8_receipt["count"] > 0),
        },
    }
    if fp8_receipt is not None:
        meta["fp8_gemm_configs"] = fp8_receipt
    (target / "orvix-trinode-runtime.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
