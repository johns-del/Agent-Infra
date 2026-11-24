"""Review or launch an isolated DeepSeek comparison on eight released GPUs.

Use the same native engine profile for B0 and policy candidates. Enabling
tool-aware cache actions additionally requires native HiCache write_back and a
successful runtime capability report; a JSON policy knob cannot create HiCache.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

MODEL = "deepseek-v41-flash-unified-dev65"
PORT = "32781"

PROFILE_OPTIONS = {
    "chunked_prefill_size": ("--chunked-prefill-size", int),
    "max_running_requests": ("--max-running-requests", int),
    "max_total_tokens": ("--max-total-tokens", int),
    "max_prefill_tokens": ("--max-prefill-tokens", int),
    "cuda_graph_max_bs_decode": ("--cuda-graph-max-bs-decode", int),
    "torch_compile_max_bs": ("--torch-compile-max-bs", int),
    "swa_prefix_tails": ("--swa-prefix-tails", int),
    "swa_full_tokens_ratio": ("--swa-full-tokens-ratio", float),
    "disable_hybrid_swa_memory": ("--disable-hybrid-swa-memory", bool),
    "mem_fraction_static": ("--mem-fraction-static", float),
    "min_free_slots_delay": ("--min-free-slots-delay", int),
    "enable_hierarchical_cache": ("--enable-hierarchical-cache", bool),
    "hicache_write_policy": ("--hicache-write-policy", str),
    "hicache_ratio": ("--hicache-ratio", float),
    "hicache_size": ("--hicache-size", int),
    "disable_overlap_schedule": ("--disable-overlap-schedule", bool),
    "enable_mixed_chunk": ("--enable-mixed-chunk", bool),
    "random_seed": ("--random-seed", int),
}

FLOAT_BOUNDS = {
    "hicache_ratio": (1.0, 16.0),
    "swa_full_tokens_ratio": (0.01, 1.0),
    "mem_fraction_static": (0.30, 0.96),
}

ZERO_ALLOWED_INTS = {"random_seed", "min_free_slots_delay", "hicache_size"}


def _set_option(command, flag, value, boolean=False):
    result = []
    cursor = 0
    while cursor < len(command):
        argument = command[cursor]
        if argument == flag:
            cursor += 1 if boolean else 2
        elif argument.startswith(flag + "="):
            cursor += 1
        else:
            result.append(argument)
            cursor += 1
    if boolean:
        if value:
            result.append(flag)
    else:
        result.extend([flag, str(value)])
    return result


def apply_engine_profile(command, profile):
    if not isinstance(profile, dict) or not profile.keys() <= PROFILE_OPTIONS.keys():
        raise ValueError("unknown native engine profile field")
    result = list(command)
    for name, value in profile.items():
        flag, kind = PROFILE_OPTIONS[name]
        if kind is float:
            lo, hi = FLOAT_BOUNDS.get(name, (0.01, 16.0))
            valid = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and lo <= float(value) <= hi
            )
        else:
            valid = type(value) is kind
        if kind is int:
            valid = valid and (value >= 0 if name in ZERO_ALLOWED_INTS else value > 0)
            if name == "max_running_requests":
                valid = valid and value <= 512
            if name in {"cuda_graph_max_bs_decode", "torch_compile_max_bs"}:
                valid = valid and value <= 512
            if name == "swa_prefix_tails":
                valid = valid and value <= 4096
        if name == "hicache_write_policy":
            valid = valid and value in {"write_back", "write_through", "write_through_selective"}
        if not valid:
            raise ValueError("invalid native engine profile value: " + name)
        result = _set_option(result, flag, value, boolean=kind is bool)
    return result


def build_command(root, original, profile=None):
    previous = json.loads(original.read_text())
    source = root.resolve() / "runtime"
    for name in (
        "unified_rank.py",
        "unified_core.py",
        "unified_cache_control.py",
        "unified_prefill.py",
        "unified_prediction.py",
        "unified_admission.py",
        "unified_provenance.py",
        "chunk_fair_core.py",
        "unified_chunk.py",
    ):
        if not (source / "sglang/srt/observability" / name).is_file():
            raise ValueError("missing private runtime adapter: " + name)
    if (source / "orvix-v6-derivation.json").is_file():
        for name in ("multi_chunk_core.py", "multi_chunk_runtime.py", "multi_chunk_compat.py"):
            if not (source / "sglang/srt/observability" / name).is_file():
                raise ValueError("missing v6 runtime adapter: " + name)
    if (source / "orvix-v7-derivation.json").is_file():
        for name in ("interval_core.py", "interval_probe.py", "interval_native.py", "interval_runtime.py", "forced_trace_runtime.py"):
            if not (source / "sglang/srt/observability" / name).is_file():
                raise ValueError("missing v7 runtime adapter: " + name)
    command = list(previous["command"])
    if "--host" not in command or command[command.index("--host") + 1] != "127.0.0.1":
        raise ValueError("the comparison must bind to loopback")
    command = _set_option(command, "--port", PORT)
    command = _set_option(command, "--served-model-name", MODEL)
    # A private launch must not randomly choose a different engine seed each time.
    # Explicit command/profile seeds take precedence. This does NOT make outputs
    # identical across different batch compositions or DSpark verification paths.
    if not any(x == "--random-seed" or x.startswith("--random-seed=") for x in command):
        command = _set_option(command, "--random-seed", 0)
    command = apply_engine_profile(command, profile or {})
    return previous, source, command


def rebase_triton_cache_groups(home: Path) -> int:
    """Point Triton's copied group indexes at their local artifacts."""
    cache = home / ".cache/sglang/triton"
    if not cache.resolve().is_relative_to(home.resolve()):
        return 0
    changed = 0
    for index in cache.glob("*/__grp__*.json"):
        if index.is_symlink() or index.parent.is_symlink():
            continue
        try:
            group = json.loads(index.read_text())
        except json.JSONDecodeError:
            continue
        children = group.get("child_paths")
        if not children:
            continue
        paths = {key: index.with_name(Path(path).name) for key, path in children.items()}
        if not all(path.is_file() for path in paths.values()):
            continue
        rebased = {key: str(path) for key, path in paths.items()}
        if rebased != children:
            group["child_paths"] = rebased
            index.write_text(json.dumps(group))
            changed += 1
    return changed


def launch(root, original, seed_home=None, profile=None, dry_run=False, *, seed_cache=None):
    root = root.resolve()
    previous, source, command = build_command(root, original, profile)
    if dry_run:
        # The original launch receipt can contain a local API key. A command
        # preview never prints it, while the actual isolated launch preserves it.
        preview = list(command)
        for index, argument in enumerate(preview):
            if argument == "--api-key" and index + 1 < len(preview):
                preview[index + 1] = "<redacted>"
            elif argument.startswith("--api-key="):
                preview[index] = "--api-key=<redacted>"
        return {
            "dry_run": True,
            "command": preview,
            "source": str(source),
            "native_engine_overrides": profile or {},
            "requires_eight_released_gpus": True,
            "gpu_validated": False,
        }
    runtime_root = Path(previous["native_python"]).parent.parent.parent
    live = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if live:
        raise RuntimeError("the comparison needs all eight GPUs released")
    with socket.socket() as probe:
        # Match Uvicorn's bind semantics so prior private streams in
        # TIME_WAIT do not block restarting the same loopback listener.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", int(PORT)))
    run = root / "engine"
    run.mkdir(mode=0o700)
    if seed_home is None:
        (run / "home").mkdir()
    else:
        shutil.copytree(seed_home, run / "home", symlinks=True)
    rebased_triton_groups = rebase_triton_cache_groups(run / "home")
    if seed_cache is None:
        (run / "cache").mkdir()
    else:
        shutil.copytree(seed_cache, run / "cache", symlinks=True)
    python_dir = Path(previous["native_python"]).parent
    cuda = runtime_root / "runtime/cu13"
    env = {
        "HOME": str(run / "home"),
        "XDG_CACHE_HOME": str(run / "cache"),
        "HF_HOME": str(run / "cache/hf"),
        "PYTHONPATH": str(source),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_HOME": str(cuda),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "PATH": f"{python_dir}:{cuda}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LC_ALL": "C.UTF-8",
        "SGLANG_SM120_FLASHMLA_BACKEND": "flashinfer",
        "ORVIX_DSV41_UNIFIED": "1",
    }
    if previous["pcie_ipc"]:
        env["SGLANG_ENABLE_PCIE_IPC_ALLREDUCE"] = "1"
        env["SGLANG_PCIE_IPC_MAX_NUMEL"] = str(previous["pcie_ipc_max_numel"])
    with (run / "launch.json").open("x") as handle:
        json.dump(
            {
                "profile": previous["expected_server"] | {"served_model_name": MODEL},
                "source_commit": previous["source_commit"],
                "model_revision": previous["model_revision"],
                "port": int(PORT),
                "source": str(source),
                "native_engine_overrides": profile or {},
                "rebased_triton_cache_groups": rebased_triton_groups,
                "gpu_validated": False,
            },
            handle,
            indent=2,
        )
    argv = ["/usr/bin/env", "-i", *(f"{k}={v}" for k, v in env.items()), *command]
    shell = "umask 077; exec " + shlex.join(argv) + " > " + shlex.quote(str(run / "engine.log")) + " 2>&1"
    subprocess.run(
        ["tmux", "-S", str(root / "tmux.sock"), "new-session", "-d", "-s", "unified-engine", "-c", str(run), shell],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("original_launch", type=Path)
    parser.add_argument("--seed-home", type=Path)
    parser.add_argument("--seed-cache", type=Path)
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="Native engine overrides; use an identical profile for B0/candidates",
    )
    parser.add_argument("--dry-run", action="store_true", help="Review command without GPU inspection or launch")
    args = parser.parse_args()
    os.umask(0o077)
    profile = json.loads(args.profile_json.read_text()) if args.profile_json else None
    result = launch(args.root, args.original_launch, args.seed_home, profile, args.dry_run, seed_cache=args.seed_cache)
    if result is not None:
        print(json.dumps(result, indent=2))
