"""Run matched B0 / fairness / cache / prefill / interval-cache ablations on one private engine.

Every arm receives the same lifecycle instrumentation. This runner uses no
oracle tool duration, future output length or remote service modifications.
Start the qualified runtime yourself, then pass its loopback endpoint.
"""

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

import httpx

from benchmark.graph import validate_graph
from chunk_fair_core import ChunkConfig
from compare_dev_replay import (
    ENDPOINT,
    engine_configuration,
    engine_state,
    payload,
    post_control,
    scale_workload_copies,
    trial,
    verify_configuration,
)
from interval_core import V11_INTERVAL_DEFAULTS, V12_INTERVAL_DEFAULTS, V13_INTERVAL_DEFAULTS, IntervalConfig
from multi_chunk_core import MultiConfig
from paired_trial import validate_pair
from run_unified_campaign import DEFAULT, PARAMS, build_files
from summarize_unified_campaign import summarize
from unified_prefill import V11_PREFILL_DEFAULTS, V12_PREFILL_DEFAULTS, V13_PREFILL_DEFAULTS, PrefillConfig
from workload_audit import analyze, locate_inputs, read_rows


INTERVAL_ARMS = {
    "O", "K", "E", "H", "KEH", "V8", "V9", "V10",
    "V11_M1", "V11_M12", "V11_M13", "V11",
    "V12_M1", "V12_M12", "V12_M13", "V12",
    "V13_M1", "V13_M12", "V13_M13", "V13", "PACER",
    "PLAS", "CTTL",
    "IS", "IR", "ID", "IP", "IF",
}

ARMS = {
    "MS": {"fairness": False, "cache": False, "prefill": False},
    "ML": {"fairness": False, "cache": False, "prefill": False},
    "MF": {"fairness": False, "cache": False, "prefill": False},
    "G": {"fairness": False, "cache": False, "prefill": False},
    "Q": {"fairness": False, "cache": False, "prefill": False},
    "FQ": {"fairness": False, "cache": False, "prefill": False},
    "QS": {"fairness": False, "cache": False, "prefill": False},
    "O": {"fairness": False, "cache": False, "prefill": False},
    "K": {"fairness": False, "cache": False, "prefill": False},
    "E": {"fairness": False, "cache": False, "prefill": False},
    "H": {"fairness": False, "cache": False, "prefill": False},
    "KEH": {"fairness": False, "cache": False, "prefill": False},
    "V8": {"fairness": False, "cache": False, "prefill": False},
    "V9": {"fairness": False, "cache": False, "prefill": False},
    "V10": {"fairness": False, "cache": False, "prefill": False},
    "V11_M1": {"fairness": False, "cache": False, "prefill": False},
    "V11_M12": {"fairness": False, "cache": False, "prefill": True},
    "V11_M13": {"fairness": False, "cache": False, "prefill": False},
    "V11": {"fairness": False, "cache": False, "prefill": True},
    "V12_M1": {"fairness": False, "cache": False, "prefill": True},
    "V12_M12": {"fairness": False, "cache": False, "prefill": True},
    "V12_M13": {"fairness": False, "cache": False, "prefill": True},
    "V12": {"fairness": False, "cache": False, "prefill": True},
    "V13_M1": {"fairness": False, "cache": False, "prefill": True},
    "V13_M12": {"fairness": False, "cache": False, "prefill": True},
    "V13_M13": {"fairness": False, "cache": False, "prefill": True},
    "V13": {"fairness": False, "cache": False, "prefill": True},
    "PACER": {"fairness": False, "cache": False, "prefill": True},
    "PLAS": {"fairness": False, "cache": False, "prefill": False},
    "CTTL": {"fairness": False, "cache": False, "prefill": False},
    "IS": {"fairness": False, "cache": False, "prefill": False},
    "IR": {"fairness": False, "cache": False, "prefill": False},
    "ID": {"fairness": False, "cache": False, "prefill": False},
    "IP": {"fairness": False, "cache": False, "prefill": False},
    "IF": {"fairness": False, "cache": False, "prefill": False},
    "B0": {"fairness": False, "cache": False, "prefill": False},
    "AA": {"fairness": False, "cache": False, "prefill": False},
    "F": {"fairness": True, "cache": False, "prefill": False},
    "C": {"fairness": False, "cache": True, "prefill": False},
    "P": {"fairness": False, "cache": False, "prefill": True},
    "FC": {"fairness": True, "cache": True, "prefill": False},
    "FP": {"fairness": True, "cache": False, "prefill": True},
    "CP": {"fairness": False, "cache": True, "prefill": True},
    "FCP": {"fairness": True, "cache": True, "prefill": True},
}


def configurations(method, args):
    arm = ARMS[method]
    params = PARAMS | {
        "fairness_mode": getattr(args, "fairness_policy", "guarded") if arm["fairness"] else "legacy",
        "cost_model": getattr(args, "output_cost", "prefill_uniform") if arm["fairness"] else "prefill_uniform",
        "fairness_slack_s": args.fairness_slack_s,
        "candidate_window_k": getattr(args, "candidate_window", 32),
        "age_guard_quota": getattr(args, "age_guard_quota", 1),
    }
    cache = {
        "mode": getattr(args, "cache_policy", "demand-aware") if arm["cache"] else "observe",
        "min_idle_s": args.cache_min_idle_s,
        "require_native_rejection": False,
    }
    if cache["mode"] == "demand-aware":
        cache["scan_interval_s"] = getattr(args, "cache_scan_interval_s", 0.5)
        cache["require_native_rejection"] = True
    if method in {"V11_M12", "V11", "V12_M1", "V12_M13"}:
        pprof = getattr(args, "prefill_profile", None)
        pdata = json.loads(Path(pprof).read_text()) if pprof is not None else {}
        if method in {"V12_M1", "V12_M13"}:
            prefill = asdict(PrefillConfig(**V11_PREFILL_DEFAULTS))
        else:
            prefill = asdict(PrefillConfig(**(V11_PREFILL_DEFAULTS | pdata)))
    elif method in {"V12_M12", "V12", "V13_M1", "V13_M13"}:
        pprof = getattr(args, "prefill_profile", None)
        pdata = json.loads(Path(pprof).read_text()) if pprof is not None else {}
        if method in {"V13_M1", "V13_M13"}:
            prefill = asdict(PrefillConfig(**V12_PREFILL_DEFAULTS))
        else:
            prefill = asdict(PrefillConfig(**(V12_PREFILL_DEFAULTS | pdata)))
    elif method in {"V13_M12", "V13", "PACER"}:
        pprof = getattr(args, "prefill_profile", None)
        pdata = json.loads(Path(pprof).read_text()) if pprof is not None else {}
        prefill = asdict(PrefillConfig(**(V13_PREFILL_DEFAULTS | pdata)))
    else:
        prefill = {
            "mode": getattr(args, "prefill_strategy", "balanced") if arm["prefill"] else "off",
            "decode_chunk_tokens": args.decode_chunk_tokens,
            "decode_min_batch": getattr(args, "decode_min_batch", 4),
            "decode_interval": getattr(args, "decode_interval", 1),
        }
        if prefill["mode"] == "balanced":
            prefill.update(
                prefill_burst=getattr(args, "prefill_burst", 4),
                max_prefill_burst=getattr(args, "max_prefill_burst", 16),
                fill_ratio=getattr(args, "fill_ratio", 0.75),
                queue_age_s=getattr(args, "queue_age_s", 5.0),
            )
    return arm["fairness"], params, cache, prefill


def chunk_configuration(method, args):
    modes = {"G": "guard", "Q": "share", "FQ": "fair", "QS": "shadow"}
    if method not in modes:
        return None
    profile = getattr(args, "chunk_profile", None)
    data = json.loads(Path(profile).read_text()) if profile is not None else {}
    data["mode"] = modes[method]
    return ChunkConfig(**data).frozen()


def multi_configuration(method, args):
    modes = {"MS": "shadow", "ML": "locality", "MF": "fair"}
    if method not in modes:
        return None
    profile = getattr(args, "multi_profile", None)
    data = json.loads(Path(profile).read_text()) if profile is not None else {}
    data["mode"] = modes[method]
    return MultiConfig(**data).frozen()


def interval_configuration(method, args, *, on_v7_engine=True):
    """Build causal IntervalConfig dict for B0/AA and O/K/E/H/KEH/V8/IS/IR/ID/IP/IF."""
    profile = getattr(args, "interval_profile", None)
    raw = json.loads(Path(profile).read_text()) if profile is not None else {}
    base = IntervalConfig(**raw).validate()
    canonical = {"IS": "observe", "IR": "K", "ID": "E", "IP": "H", "IF": "KEH"}.get(method, method)
    if canonical in {"B0", "AA"}:
        if not on_v7_engine and method not in INTERVAL_ARMS and profile is None:
            return None
        return asdict(replace(base, mode="observe").validate())
    if canonical == "observe":
        return asdict(replace(base, mode="observe").validate())
    if canonical in {"V13_M1", "V13_M12", "V13_M13", "V13", "PACER"}:
        merged = dict(V13_INTERVAL_DEFAULTS) | raw
        if canonical in {"V13_M1", "V13_M12"}:
            merged["lane0_log_aging"] = False
            merged["hrrn_age_cap_s"] = 90.0
        if canonical == "V13_M1":
            merged["phase_chain_turn_threshold"] = 0
        return asdict(IntervalConfig(**merged).validate())
    if canonical == "PLAS":
        # Agentix-PLAS (Luo et al., USENIX NSDI '26): pure native cache/batching + program cumulative service MLFQ
        plas_cfg = dict(V11_INTERVAL_DEFAULTS) | {
            "mode": "active",
            "soft_keep": False,
            "eager_backup": False,
            "demand_release": False,
            "proactive_release": False,
            "host_swa_reclaim": False,
            "hbm_pressure_free_fraction": 0.0,
            "ready_order": True,
            "tier_signal_only_order": False,
            "bimodal_critical_path_order": False,
            "plas_program_order": True,
            "max_queue_age_s": 300.0,
        }
        return asdict(IntervalConfig(**plas_cfg).validate())
    if canonical == "CTTL":
        # Continuum-TTL (Li et al., ICLR '26): causal tool-interval HBM lease + Program-Arrival Continuity FCFS
        cttl_cfg = dict(V11_INTERVAL_DEFAULTS) | {
            "mode": "active",
            "soft_keep": True,
            "keep_s": 3.0,
            "keep_max_s": 30.0,
            "eager_backup": False,
            "demand_release": False,
            "proactive_release": False,
            "host_swa_reclaim": False,
            "hbm_pressure_free_fraction": 0.0,
            "ready_order": True,
            "tier_signal_only_order": False,
            "bimodal_critical_path_order": False,
            "continuum_program_fcfs": True,
            "max_queue_age_s": 300.0,
        }
        return asdict(IntervalConfig(**cttl_cfg).validate())
    if canonical in {"V12_M1", "V12_M12", "V12_M13", "V12"}:
        merged = dict(V12_INTERVAL_DEFAULTS) | raw
        if canonical in {"V12_M1", "V12_M12"}:
            merged["hrrn_lane1_order"] = False
            merged["critical_in_batch_fold"] = False
        return asdict(IntervalConfig(**merged).validate())
    if canonical in {"V11_M1", "V11_M12", "V11_M13", "V11"}:
        merged = dict(V11_INTERVAL_DEFAULTS) | raw
        if canonical in {"V11_M1", "V11_M12"}:
            merged["ready_order"] = False
            merged["bimodal_critical_path_order"] = False
        else:
            merged["ready_order"] = True
            merged["bimodal_critical_path_order"] = True
        if canonical == "V11_M1":
            merged["hbm_pressure_free_fraction"] = 0.0
        return asdict(IntervalConfig(**merged).validate())
    if canonical not in {"O", "K", "E", "H", "KEH", "V8", "V9", "V10"}:
        return asdict(replace(base, mode="off").validate()) if on_v7_engine else None
    v9_overrides = {}
    if canonical in {"V9", "V10"}:
        v9_overrides = {
            "streaming_backup": True,
            "host_swa_reclaim": True,
            "proactive_release": True,
            "pressure_gated_backup": True,
            "block_unrelated_during_prefetch": False,
            "prefetch_only_reorder": True,
            "streaming_prefetch": True,
            "tool_end_prefetch": True,
            "countdown_prefetch": True,
            "prefetch_during_long_continuation": True,
            "allow_full_slot_prefetch": True,
        }
    cfg = replace(
        base,
        mode="active",
        soft_keep=canonical in {"K", "KEH", "V8", "V9", "V10"},
        eager_backup=canonical in {"E", "KEH", "V8", "V9", "V10"},
        demand_release=canonical in {"E", "KEH", "V8", "V9", "V10"},
        queue_prefetch=canonical in {"H", "KEH", "V8", "V9", "V10"},
        ready_order=canonical in {"O", "H", "KEH", "V8", "V9", "V10"},
        tier_signal_only_order=(True if canonical in {"V8", "V9", "V10"} else base.tier_signal_only_order),
        **v9_overrides,
    ).validate()
    return asdict(cfg)


def write_json(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def parse_items(value, allowed=None):
    values = value.split(",")
    if not values or len(set(values)) != len(values) or any(not x for x in values):
        raise argparse.ArgumentTypeError("expected unique comma-separated values")
    if allowed is not None and any(x not in allowed for x in values):
        raise argparse.ArgumentTypeError("unknown ablation arm")
    return values


async def run(args):
    if any(m in {"FC", "FP", "CP", "FCP"} for m in args.methods) and not getattr(args, "allow_combinations", False):
        raise ValueError("run isolated F/C/P first; combinations require --allow-combinations")
    if args.repeats < 2:
        raise ValueError("requires >=2 paired repeats (including reversed order)")
    campaign_id = str(uuid4())
    graphs_path, prompts_path = locate_inputs(args.root)
    base_graphs = sorted(read_rows(graphs_path), key=lambda g: g["graph_id"])
    prompt_rows = read_rows(prompts_path)
    audit = analyze(base_graphs, prompt_rows)
    if any(g["links"] or validate_graph(g, ready_only=True) for g in base_graphs):
        raise ValueError("graph failed replay admission; expanded child links are unsupported")
    base_prompts = {(r["graph_id"], r["node_id"]): r for r in prompt_rows}
    copies = getattr(args, "workload_copies", 1)
    salt_per_copy = getattr(args, "salt_per_copy", False)
    continuity_mode = getattr(args, "continuity_mode", "recorded")
    graphs, prompts = scale_workload_copies(base_graphs, base_prompts, copies, salt_per_copy=salt_per_copy)

    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output / "results").mkdir()
    (args.output / "input").mkdir()
    import gzip
    with gzip.open(args.output / "input/graphs.jsonl.gz", "wt", encoding="utf-8") as gz:
        for g in graphs:
            gz.write(json.dumps(g, separators=(",", ":")) + "\n")
    write_json(args.output / "workload_audit.json", audit)
    headers = {"Authorization": "Bearer " + os.environ["ORVIX_ENGINE_API_KEY"]} if os.environ.get("ORVIX_ENGINE_API_KEY") else {}
    async with httpx.AsyncClient(base_url=args.endpoint, headers=headers, timeout=args.timeout_s, trust_env=False) as client:
        engine, live_policy = await engine_state(client)
        version = live_policy.get("adapter_version")
        if version not in {"orvix-dsv41-fair-chunk-v5", "orvix-dsv41-multi-prefill-v6", "orvix-dsv41-interval-cache-v7"}:
            raise RuntimeError("stale experiment engine: prepare and launch the matching private runtime")
        if any(m in {"MS", "ML", "MF"} for m in args.methods) and version != "orvix-dsv41-multi-prefill-v6":
            raise RuntimeError("multi-prefill arms require the v6 lifecycle patch")
        if any(m in INTERVAL_ARMS for m in args.methods) and version != "orvix-dsv41-interval-cache-v7":
            raise RuntimeError("interval-cache arms require the v7/v8 interval runtime")
        on_v7 = version == "orvix-dsv41-interval-cache-v7"
        write_json(args.output / "plan.json", {
            "campaign_id": campaign_id,
            "methods": args.methods, "concurrency": args.concurrency, "repeats": args.repeats,
            "workload_copies": copies, "salt_per_copy": salt_per_copy, "continuity_mode": continuity_mode,
            "pair_order": "B0,candidate in odd repeats; candidate,B0 in even repeats",
            "recorded_gap_scale": args.recorded_gap_scale, "tool_scale": args.tool_scale,
            "lifecycle": "engine for every arm", "arms": ARMS,
            "stream_metrics": getattr(args, "stream_metrics", False),
            "arm_configurations": {m: configurations(m, args) for m in ["B0", *args.methods]},
            "chunk_arm_configurations": {m: chunk_configuration(m, args) for m in ["B0", *args.methods]},
            "multi_arm_configurations": {m: multi_configuration(m, args) for m in ["B0", *args.methods]},
            "interval_arm_configurations": {m: interval_configuration(m, args, on_v7_engine=on_v7) for m in ["B0", *args.methods]},
            "build_artifact_root": str(args.build_artifact_root) if getattr(args, "build_artifact_root", None) else None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "graphs_sha256": sha256(graphs_path.read_bytes()).hexdigest(),
            "prompts_sha256": sha256(prompts_path.read_bytes()).hexdigest(),
            "limitations": [
                "fixed-input fixed-length generation unless continuity_mode=causal_splice",
                "one private engine; launch configuration stays fixed within a campaign",
            ],
        })
        identity = live_policy.get("runtime_identity", {})
        if not identity.get("verified"):
            raise RuntimeError("unverified runtime: derive it with prepare-runtime")
        fingerprint = identity["python_fingerprint"]
        if any(not r["input_ids"] or r["output_budget"] <= 0
               or len(r["input_ids"]) + r["output_budget"] >= engine["context_length"]
               for r in prompts.values()):
            raise ValueError("prompt and generation budget do not fit the engine")
        write_json(args.output / "engine_before.json", engine)
        by_length = sorted(prompt_rows, key=lambda r: len(r["input_ids"]))
        warm_rows = [by_length[i * (len(by_length) - 1) // 7] for i in range(8)]

        async def warm_arm(enabled, params, cache, prefill, chunk, multi, interval):
            if on_v7:
                deadline = time.monotonic() + 90
                _, cur = await engine_state(client)
                while not cur.get("interval_cache", {}).get("idle", True):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("interval cache did not drain before warmup")
                    await post_control(client, enabled, params, DEFAULT, cache=cache, prefill=prefill, interval=interval)
                    await asyncio.sleep(0.25)
                    _, cur = await engine_state(client)
            await post_control(client, enabled, params, DEFAULT, cache=cache, prefill=prefill,
                               chunk=chunk, multi=multi, interval=interval, reset=True)
            _, state = await engine_state(client)
            verify_configuration(state, enabled, params, DEFAULT, cache, prefill, chunk, multi, interval)

            async def one(row):
                response = await client.post("/generate", json=payload(
                    row | {"output_budget": 32}, "warmup", "foreground", time.monotonic()))
                response.raise_for_status()
                if response.json()["meta_info"]["completion_tokens"] != 32:
                    raise RuntimeError("warmup token count mismatch")

            await asyncio.gather(*(one(row) for row in warm_rows))

        for offered in args.concurrency:
            for method in args.methods:
                for repeat in range(1, args.repeats + 1):
                    pair_id = str(uuid4())
                    pair_results = {}
                    order = ("B0", method) if repeat % 2 else (method, "B0")
                    for side in order:
                        enabled, params, cache, prefill = configurations(side, args)
                        chunk = chunk_configuration(side, args)
                        multi = multi_configuration(side, args)
                        interval = interval_configuration(side, args, on_v7_engine=on_v7)
                        name = f"full-{offered}-{method}-r{repeat}-{side}"
                        print("BEGIN", name, flush=True)
                        current_engine, current_policy = await engine_state(client)
                        if current_policy.get("runtime_identity", {}).get("python_fingerprint") != fingerprint:
                            raise RuntimeError("runtime changed within campaign")
                        if engine_configuration(current_engine) != engine_configuration(engine):
                            raise RuntimeError("launch configuration changed between ablation arms")
                        await warm_arm(enabled, params, cache, prefill, chunk, multi, interval)
                        build_root = getattr(args, "build_artifact_root", None)
                        builds_before = build_files(build_root) if build_root is not None else None
                        try:
                            result = await trial(
                                client, enabled, params, graphs, prompts, offered, DEFAULT,
                                recorded_gap_scale=args.recorded_gap_scale, tool_scale=args.tool_scale,
                                lifecycle="engine", cache=cache, prefill=prefill,
                                stream_metrics=getattr(args, "stream_metrics", False),
                                chunk=chunk, multi=multi, interval=interval,
                                continuity_mode=continuity_mode,
                            )
                        except Exception as exc:
                            write_json(args.output / "results" / (name + "-FAILED.json"), {
                                "name": name, "error_type": type(exc).__name__, "error": str(exc),
                                "valid_performance_result": False})
                            raise
                        builds_after = build_files(build_root) if build_root is not None else None
                        changed_builds = ([key for key, value in builds_after.items()
                                           if builds_before.get(key) != value]
                                          if builds_after is not None else [])
                        result.update(
                            paired_context={
                                "campaign_id": campaign_id, "pair_id": pair_id,
                                "repeat": repeat, "position": order.index(side),
                                "runtime_python_fingerprint": fingerprint,
                                "adapter_version": version,
                            },
                            name=name, method=side, parameters=params, knobs=DEFAULT,
                            chunk_configuration=chunk or {"mode": "off"},
                            multi_configuration=multi or {"mode": "off"},
                            interval_configuration=interval or {"mode": "off"},
                            offered_graph_concurrency=offered,
                            workload_copies=copies,
                            salt_per_copy=salt_per_copy,
                            new_build_files=changed_builds,
                            build_artifact_observation=(
                                "local engine/home and engine/cache before/after"
                                if builds_after is not None else "not collected; inspect target engine traces"
                            ),
                        )
                        write_json(args.output / "results" / (name + ".json"), result)
                        pair_results[side] = result
                        print("DONE", name, round(result["wall_s"], 3), flush=True)
                    qualification = validate_pair(pair_results["B0"], pair_results[method])
                    write_json(args.output / (f"pair-{offered}-{method}-r{repeat}.json"), qualification)
                    if not qualification["valid_paired_comparison"]:
                        raise RuntimeError("pair failed comparability checks: " + str(qualification["reasons"]))
                    summary = summarize(args.output)
                    summary["recommendation"] = {
                        "deployment_default": "B0", "canary_candidate": None,
                        "reason": "review target GPU correctness and compile stability before promotion",
                    }
                    temp = args.output / "summary.tmp.json"
                    temp.write_text(json.dumps(summary, indent=2) + "\n")
                    temp.replace(args.output / "summary.latest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--interval-profile", type=Path, help="IntervalConfig JSON (e.g., configs/v11_interval_128slot.json)")
    parser.add_argument("--prefill-profile", type=Path, help="PrefillConfig JSON for V11/V11_M12 (e.g., configs/v11_prefill_128slot.json)")
    parser.add_argument("--multi-profile", type=Path, help="v6 policy JSON: MS=observe, ML=locality, MF=user credit + locality")
    parser.add_argument("--chunk-profile", type=Path, help="v5 chunk policy JSON; modes G/Q/FQ/QS select guard/share/fair/shadow")
    parser.add_argument("--methods", type=lambda x: parse_items(x, set(ARMS) - {"B0"}), default=["AA"])
    parser.add_argument("--concurrency", type=lambda x: [int(v) for v in parse_items(x)], default=[32])
    parser.add_argument("--workload-copies", type=int, default=1,
                        help="replicate 65-workflow DAG (2->130, 4->260) to stress 64/128+ running slots")
    parser.add_argument("--salt-per-copy", action="store_true",
                        help="isolate RadixCache namespace per replica copy")
    parser.add_argument("--continuity-mode", choices=("recorded", "causal_splice", "forced_trace"), default="causal_splice")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--recorded-gap-scale", type=float, default=0.0)
    parser.add_argument("--tool-scale", type=float, default=1.0)
    parser.add_argument("--fairness-slack-s", type=float, default=2.0)
    parser.add_argument("--cache-min-idle-s", type=float, default=1.0)
    parser.add_argument("--decode-chunk-tokens", type=int, default=2048)
    parser.add_argument("--decode-interval", type=int, default=1)
    parser.add_argument("--decode-min-batch", type=int, default=4)
    parser.add_argument("--prefill-strategy", choices=("balanced", "decode_cadence", "decode_guard"), default="balanced")
    parser.add_argument("--fairness-policy", choices=("guarded", "bounded"), default="guarded")
    parser.add_argument("--output-cost", choices=("online_guarded", "online", "prefill_uniform"), default="prefill_uniform")
    parser.add_argument("--candidate-window", type=int, default=32)
    parser.add_argument("--age-guard-quota", type=int, default=1)
    parser.add_argument("--cache-policy", choices=("demand-aware", "tool-aware"), default="demand-aware")
    parser.add_argument("--cache-scan-interval-s", type=float, default=0.5)
    parser.add_argument("--prefill-burst", type=int, default=4)
    parser.add_argument("--max-prefill-burst", type=int, default=16)
    parser.add_argument("--fill-ratio", type=float, default=0.75)
    parser.add_argument("--queue-age-s", type=float, default=5.0)
    parser.add_argument("--allow-combinations", action="store_true")
    parser.add_argument("--stream-metrics", action="store_true",
                        help="enable client TTFT and delivery-gap measurements, identically for ALL arms")
    parser.add_argument("--build-artifact-root", type=Path,
                        help="experiment directory containing engine/home and engine/cache on this host")
    parser.add_argument("--timeout-s", type=float, default=1800)
    args = parser.parse_args()
    if args.repeats < 2 or any(n < 1 for n in args.concurrency) or not 1 <= args.workload_copies <= 32:
        parser.error("require repeats >=2, positive concurrency, and workload-copies in 1..32")
    if not 1 <= args.prefill_burst <= args.max_prefill_burst <= 64:
        parser.error("require 1 <= prefill-burst <= max-prefill-burst <=64")
    if not 0 < args.fill_ratio <= 1 or not 0 <= args.queue_age_s <= 3600:
        parser.error("invalid fill ratio or queue age")
    if args.candidate_window < 1 or args.age_guard_quota < 1 or args.age_guard_quota > args.candidate_window:
        parser.error("require 1 <= age-guard-quota <= candidate-window")
    if not 1 <= args.decode_interval <= 16 or args.decode_min_batch < 1 or args.cache_scan_interval_s <= 0:
        parser.error("invalid cadence, decode batch, or cache scan interval")
    if any(m in {"FC", "FP", "CP", "FCP"} for m in args.methods) and not args.allow_combinations:
        parser.error("run isolated F/C/P first; combinations require --allow-combinations")
    if args.build_artifact_root is not None and not all((args.build_artifact_root / name).is_dir()
            for name in ("engine/home", "engine/cache")):
        parser.error("build-artifact-root must contain engine/home and engine/cache")
    asyncio.run(run(args))
