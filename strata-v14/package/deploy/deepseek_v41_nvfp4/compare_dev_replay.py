"""Compare FCFS, UNIFIED rank, and v7/v8 Interval-Cache policies on one private DeepSeek engine."""

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import time
from uuid import uuid4

import httpx

from benchmark.graph import topological_nodes, validate_graph
from interval_core import IntervalConfig, validate_events
from prepare_dev_replay import read_gzip
from replay_continuity import CONTINUITY_MODES, continuity, stitch_causal_prefix
from replay_stream import generate_with_measurements
from unified_core import Parameters
from workload_audit import locate_inputs


KEY = "orvix_unified"
MODEL = "deepseek-v41-flash-unified-dev65"
ENDPOINT = "http://127.0.0.1:32781"


def payload(row, family, lane, ready_at, *, request_id=None, identity=None,
            input_ids_override=None, cache_salt=None, forced_output_ids=None):
    """Keep replay truth (future lengths/times) out of scheduling metadata."""
    metadata = {"family_id": family, "lane": lane, "ready_at": ready_at}
    if identity is not None:
        metadata.update({k: identity[k] for k in ("session_id", "scope_id", "graph_id", "round_id",
                                                  "predecessor_request_id", "predecessor_queue_s")
                         if k in identity})
    if forced_output_ids is not None:
        metadata["forced_output_ids"] = list(forced_output_ids)
    out = {
        "rid": request_id or str(uuid4()),
        "input_ids": row["input_ids"] if input_ids_override is None else input_ids_override,
        "sampling_params": {
            "temperature": 0, "max_new_tokens": row["output_budget"], "ignore_eos": True,
            "stop": [], "stop_token_ids": [],
            "custom_params": {KEY: metadata},
        },
        "stream": False,
    }
    if cache_salt is not None:
        out["cache_salt"] = str(cache_salt)
    return out


def scale_workload_copies(graphs, prompts, copies=1, *, salt_per_copy=False):
    """Deterministically scale workflow graphs to test 64/128/256 slot concurrency.

    `copies=1` returns the exact original `dev65` corpus unchanged (identical SHA256).
    `copies>1` creates independent workflow/user/session copies; when `salt_per_copy`
    is True, each copy receives a distinct `cache_salt` so replicas do not collapse
    onto shared RadixCache pages and genuinely exercise KV/HiCache pool capacity.
    """
    if type(copies) is not int or not 1 <= copies <= 32:
        raise ValueError("workload copies must be an integer in 1..32")
    if copies == 1 and not salt_per_copy:
        return graphs, prompts
    scaled_graphs = []
    scaled_prompts = {}
    for copy_idx in range(copies):
        suffix = "" if (copies == 1 and copy_idx == 0) else f"#c{copy_idx}"
        salt = f"orvix-copy-{copy_idx}" if salt_per_copy else None
        for g in graphs:
            gc = deepcopy(g)
            old_gid = gc["graph_id"]
            new_gid = old_gid + suffix
            gc["graph_id"] = new_gid
            gc["user_id"] = str(gc["user_id"]) + suffix
            gc["session_id"] = str(gc["session_id"]) + suffix
            if salt is not None:
                gc["cache_salt"] = salt
            scaled_graphs.append(gc)
            for n in g["nodes"]:
                if n["kind"] == "model":
                    prow = deepcopy(prompts[(old_gid, n["id"])])
                    prow["graph_id"] = new_gid
                    if salt is not None:
                        prow["cache_salt"] = salt
                    scaled_prompts[(new_gid, n["id"])] = prow
    scaled_graphs.sort(key=lambda x: x["graph_id"])
    return scaled_graphs, scaled_prompts


def read_inputs(root, context, *, workload_copies=1, salt_per_copy=False):
    graph_path, prompt_path = locate_inputs(root)
    graphs = sorted(read_gzip(graph_path), key=lambda g: g["graph_id"])
    rows = list(read_gzip(prompt_path))
    prompts = {(row["graph_id"], row["node_id"]): row for row in rows}
    expected = {(g["graph_id"], n["id"]) for g in graphs for n in g["nodes"] if n["kind"] == "model"}
    if len(rows) != len(prompts) or prompts.keys() != expected:
        raise ValueError("prompt corpus does not cover the graphs exactly")
    if any(g["links"] or validate_graph(g, ready_only=True) for g in graphs):
        raise ValueError("graph failed replay admission")
    if any(not r["input_ids"] or not r["output_budget"]
           or len(r["input_ids"]) + r["output_budget"] >= context for r in rows):
        raise ValueError("prompt and generation budget do not fit the engine")
    return scale_workload_copies(graphs, prompts, workload_copies, salt_per_copy=salt_per_copy)


async def post_control(client, enabled, params, knobs=None, *, events=None, cache=None, prefill=None,
                       chunk=None, multi=None, interval=None, reset=False):
    control = {"enabled": enabled, "now": time.monotonic(), "parameters": params,
               "chunk_fair": chunk or {"mode": "off"},
               "multi_chunk": multi or {"mode": "off"}}
    if interval is not None:
        control["interval_cache"] = interval
    if knobs is not None:
        control["knobs"] = knobs
    if events is not None:
        if interval is not None:
            wire = [{k: e[k] for k in ("kind", "event_id", "request_id", "observed_at", "tool_id", "tool_type", "status")
                     if k in e} for e in events]
            control["interval_events"] = validate_events(wire, control["now"])
        else:
            control["events"] = events
    if cache is not None:
        control["cache"] = cache
    if prefill is not None:
        control["prefill"] = prefill
    if reset:
        control["reset"] = True
    update = await client.post("/set_internal_state", json={
        "server_args": {}, KEY: control,
    })
    update.raise_for_status()
    if update.json() != [True]:
        raise RuntimeError("engine rejected scheduler control")


def verify_configuration(state, enabled, params, knobs=None, cache=None, prefill=None,
                         chunk=None, multi=None, interval=None):
    if (state["enabled"] is not enabled or state["parameters"] != params
            or (knobs is not None and any(state["knobs"].get(k) != v for k, v in knobs.items()))):
        raise RuntimeError("scheduler mode was not acknowledged")
    if cache is not None:
        observed = state.get("cache", {})
        if (observed.get("requested_mode") != cache["mode"]
                or any(observed.get("config", {}).get(k) != v for k, v in cache.items())):
            raise RuntimeError("cache configuration was not acknowledged")
        if observed.get("effective_mode") != cache["mode"]:
            reasons = observed.get("capabilities", {}).get("reasons", [])
            raise RuntimeError("requested cache mode is unavailable: " + "; ".join(map(str, reasons)))
    if prefill is not None:
        observed = state.get("prefill", {})
        if any(observed.get("config", {}).get(k) != v for k, v in prefill.items()):
            raise RuntimeError("prefill configuration was not acknowledged")
        if prefill["mode"] != "off" and observed.get("unavailable_reason"):
            raise RuntimeError("requested prefill mode is unavailable: " + observed["unavailable_reason"])
        if prefill["mode"] == "decode_cadence" and (
                observed.get("native_decode_interval") != prefill.get("decode_interval", 1)
                or observed.get("native_decode_min_batch") != prefill.get("decode_min_batch", 1)):
            raise RuntimeError("native prefill/decode cadence was not acknowledged")
        if prefill["mode"] == "balanced" and observed.get("native_decode_interval") != 0:
            raise RuntimeError("balanced cadence must not coexist with fixed native interval")

    adapter_ver = state.get("adapter_version")
    if chunk is None and adapter_ver in {"orvix-dsv41-fair-chunk-v5", "orvix-dsv41-multi-prefill-v6", "orvix-dsv41-interval-cache-v7"}:
        chunk = {"mode": "off"}
    if chunk is not None:
        observed = state.get("chunk_fair", {})
        if observed.get("version") != "orvix-fair-chunk-v5":
            raise RuntimeError("missing v5 chunk capability acknowledgement")
        if any(observed.get("config", {}).get(k) != v for k, v in chunk.items()):
            raise RuntimeError("chunk execution-share configuration was not acknowledged")
        if chunk.get("mode", "off") != "off" and observed.get("unavailable_reason"):
            raise RuntimeError("chunk execution-share mode unavailable: " + observed["unavailable_reason"])

    if multi is None and adapter_ver == "orvix-dsv41-multi-prefill-v6":
        multi = {"mode": "off"}
    if multi is not None and (multi.get("mode", "off") != "off" or adapter_ver == "orvix-dsv41-multi-prefill-v6"):
        observed = state.get("multi_chunk", {})
        if observed.get("version") != "orvix-multi-prefill-v6":
            raise RuntimeError("missing v6 multi-prefill capability acknowledgement")
        if any(observed.get("config", {}).get(k) != v for k, v in multi.items()):
            raise RuntimeError("multi-prefill configuration was not acknowledged")
        if multi.get("mode", "off") != "off" and observed.get("unavailable_reason"):
            raise RuntimeError("multi-prefill unavailable: " + observed["unavailable_reason"])

    if interval is not None:
        observed = state.get("interval_cache", {})
        if observed.get("version") != "orvix-interval-cache-v7":
            raise RuntimeError("missing v7 interval-cache capability acknowledgement")
        if observed.get("config", {}).get("mode") != interval["mode"] or any(
            observed.get("config", {}).get(k) != v for k, v in interval.items()
        ):
            raise RuntimeError("interval-cache configuration was not acknowledged")
        if observed.get("effective_mode") != interval["mode"]:
            reasons = observed.get("capabilities", {}).get("reasons", [])
            raise RuntimeError("interval-cache mode degraded to observe: " + "; ".join(map(str, reasons)))


def numeric_counter_delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if type(value) in (int, float) and type(before.get(key, 0)) in (int, float)}


def cache_flush_acknowledged(response):
    """Accept a native positive acknowledgement, never an HTTP-200 failure body."""
    response.raise_for_status()
    try:
        value = response.json()
    except (ValueError, json.JSONDecodeError):
        if getattr(response, "text", "").startswith("Cache flushed"):
            return {"success": True, "format": "native_plain_text"}
        raise RuntimeError("cache flush returned no recognized positive acknowledgement")
    if value is True or (isinstance(value, dict) and value.get("success") is True):
        return value
    if isinstance(value, list) and value and all(
            x is True or (isinstance(x, dict) and x.get("success") is True) for x in value):
        return value
    raise RuntimeError("engine refused cache flush or returned an ambiguous acknowledgement")


def engine_configuration(engine):
    keys = ("served_model_name", "model_path", "revision", "quantization", "dtype",
            "context_length", "max_running_requests", "max_total_tokens", "mem_fraction_static",
            "chunked_prefill_size", "tp_size", "pp_size", "enable_hierarchical_cache",
            "hicache_ratio", "hicache_size", "hicache_write_policy", "hicache_io_backend",
            "speculative_algorithm", "speculative_num_steps", "speculative_eagle_topk",
            "speculative_num_draft_tokens", "speculative_dspark_block_size",
            "enable_mixed_chunk", "disable_overlap_schedule", "page_size", "swa_full_tokens_ratio",
            "swa_prefix_tails", "disable_hybrid_swa_memory", "min_free_slots_delay",
            "attn_dp_size", "attn_cp_size", "enable_two_batch_overlap", "enable_single_batch_overlap",
            "cuda_graph_backend_prefill", "cuda_graph_backend_decode", "cuda_graph_max_bs_decode",
            "torch_compile_max_bs", "random_seed")
    return {key: engine[key] for key in keys if key in engine}


async def engine_state(client):
    response = await client.get("/server_info")
    response.raise_for_status()
    state = response.json()
    if state["served_model_name"] != MODEL or state["speculative_algorithm"] != "DSPARK":
        raise ValueError("expected the private DSpark experiment engine")
    return state, state["internal_states"][0][KEY]


def nearest_models(graph, node, *, scope_only=True):
    """Find nearest model ancestors without guessing across a join or another scope."""
    by_id = {n["id"]: n for n in graph["nodes"]}
    pending, seen, result = list(node["deps"]), set(), set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        ancestor = by_id[key]
        if ancestor["kind"] == "model":
            if not scope_only or ancestor["scope"] == node["scope"]:
                result.add(key)
            continue
        pending.extend(ancestor["deps"])
    return result


def real_tool(node):
    """A business tool node, including recorded errors; not a live tool execution."""
    return (node["kind"] == "tool"
            and not node.get("attrs", {}).get("tool_type", "").startswith("orvix_"))


async def replay_graph(client, graph, prompts, calls, rounds, *, recorded_gap_scale=0.0,
                       tool_scale=1.0, events=None, on_event=None, on_tool_end=None,
                       before_request=None, stream_metrics=False, wait_inheritance=False,
                       continuity_mode="recorded", emit_workflow_end=False):
    """Replay DAG dependencies; tool durations are driver truth, never scheduler hints.

    `continuity_mode="causal_splice"` enables Track-A Controlled Continuity Replay,
    splicing each completed parent's actual generated token IDs into its same-scope
    child input while strictly preserving total input token count and output budget.
    """
    if any(not math.isfinite(v) or v < 0 for v in (recorded_gap_scale, tool_scale)):
        raise ValueError("gap and tool scales must be finite and nonnegative")
    if continuity_mode not in CONTINUITY_MODES:
        raise ValueError("invalid continuity_mode: " + str(continuity_mode))
    graph_id = graph["graph_id"]
    graph_salt = graph.get("cache_salt")
    forced_outputs = {}
    if continuity_mode == "forced_trace":
        for node in graph["nodes"]:
            if node["kind"] != "model":
                continue
            key = graph_id, node["id"]
            row = prompts[key]
            ids = row.get("recorded_output_ids")
            if not isinstance(ids, list) or len(ids) != row["output_budget"] or any(
                type(token) is not int or token < 0 for token in ids
            ):
                raise ValueError(f"missing or invalid recorded output token sequence for {key}")
            forced_outputs[key] = ids
    gates = {row["id"]: asyncio.Event() for row in graph["rounds"]}
    tasks, completed_requests = {}, {}
    completed_token_sequences = {}
    last_request_by_scope = {}
    ancestors = {n["id"]: nearest_models(graph, n) for n in graph["nodes"]}
    node_rounds = {n["id"]: n["round_id"] for n in graph["nodes"]} if wait_inheritance else {}

    def emit(kind, node=None, *, request_id=None, **fields):
        record = {
            "event_id": str(uuid4()), "kind": kind, "observed_at": time.monotonic(),
            "graph_id": graph_id, "user_id": graph["user_id"],
            "session_id": graph["session_id"], **fields,
        }
        if node is not None:
            record.update(node_id=node["id"], scope_id=node["scope"])
        if request_id is not None:
            record["request_id"] = request_id
        if events is not None:
            events.append(record)
        if on_event is not None:
            on_event(record)
        return record

    def prior_request(node):
        candidates = ancestors[node["id"]]
        if len(candidates) == 1:
            return completed_requests.get(next(iter(candidates)))
        return None

    async def execute(node):
        for dependency in node["deps"]:
            await tasks[dependency]
        if node["round_id"] is not None and node["kind"] != "think":
            await gates[node["round_id"]].wait()
        if node["kind"] != "model":
            is_tool = real_tool(node)
            prior = prior_request(node)
            started = time.monotonic()
            attrs = node.get("attrs", {})
            ttype = attrs.get("tool_type", "unknown")
            tstatus = attrs.get("status", "ok")
            if tstatus not in {"ok", "error", "cancelled", "timeout"}:
                tstatus = "error" if attrs.get("tool_failed") else "ok"
            if is_tool:
                emit("tool_start", node, request_id=prior, tool_id=node["id"],
                     tool_type=ttype,
                     ownership="unique_model_ancestor" if prior else "unresolved")
            scale = recorded_gap_scale if node["kind"] == "think" else tool_scale
            await asyncio.sleep(node["duration_s"] * scale)
            if is_tool:
                emit("tool_end", node, request_id=prior, tool_id=node["id"],
                     tool_type=ttype, status=tstatus,
                     observed_duration_s=time.monotonic() - started,
                     ownership="unique_model_ancestor" if prior else "unresolved")
                if on_tool_end is not None and prior is not None:
                    await on_tool_end()
            return
        key = graph_id, node["id"]
        row = prompts[key]
        lane = node["purpose"]
        if lane not in ("foreground", "subagent", "background"):
            lane = "unknown"
        request_id, prior = str(uuid4()), prior_request(node)
        start = time.monotonic()
        if prior is not None:
            emit("request_ready", node, request_id=prior)
        emit("model_ready", node, request_id=request_id)
        if before_request is not None:
            await before_request()
        sent = time.monotonic()
        emit("model_sent", node, request_id=request_id)
        identity = {"session_id": graph["session_id"], "scope_id": node["scope"], "graph_id": graph_id}
        predecessors = ancestors[node["id"]]
        parent = next(iter(predecessors)) if len(predecessors) == 1 else None
        inheritance = {}
        if wait_inheritance:
            inheritance["round_id"] = node["round_id"]
            if prior is not None and node["round_id"] is not None and node_rounds[parent] == node["round_id"]:
                previous = calls[graph_id, parent]
                inheritance.update(predecessor_request_id=prior, predecessor_queue_s=previous["queue_s"])

        effective_input_ids = row["input_ids"]
        forced_ids = None
        stitch_receipt = {"stitched": False, "reason": "recorded_mode"}
        if continuity_mode == "causal_splice":
            effective_input_ids, stitch_receipt = stitch_causal_prefix(
                completed_token_sequences.get(parent), row["input_ids"], node["scope"]
            )
        elif continuity_mode == "forced_trace":
            forced_ids = forced_outputs.get(key)
            # Align child input prefix onto parent's deterministic trajectory sequence
            # (no-op when child already shares recorded parent input + forced output verbatim)
            effective_input_ids, stitch_receipt = stitch_causal_prefix(
                completed_token_sequences.get(parent), row["input_ids"], node["scope"]
            )
            if forced_ids is not None:
                stitch_receipt = {
                    **stitch_receipt,
                    "forced_trace": True,
                    "forced_output_tokens": len(forced_ids),
                }

        body = payload(
            row, graph["user_id"], lane, start,
            request_id=request_id, identity=identity | inheritance,
            input_ids_override=effective_input_ids,
            cache_salt=row.get("cache_salt", graph_salt),
            forced_output_ids=forced_ids,
        )
        delivery = {}
        if stream_metrics:
            result, delivery = await generate_with_measurements(client, body, sent)
        else:
            response = await client.post("/generate", json=body)
            response.raise_for_status()
            result = response.json()
        meta = result["meta_info"]
        ids = result.get("output_ids")
        if continuity_mode == "forced_trace" and (forced_ids is None or ids != forced_ids):
            first_difference = next(
                (i for i, (actual, expected) in enumerate(zip(ids or (), forced_ids or ())) if actual != expected),
                None,
            )
            raise ValueError(f"server output differs from forced trajectory for {key}, "
                             f"first differing position: {first_difference}")
        if meta["completion_tokens"] != row["output_budget"] or (
                ids is not None and len(ids) != row["output_budget"]):
            raise ValueError("generation length differs from the frozen budget")
        completed_requests[node["id"]] = request_id
        last_request_by_scope[node["scope"]] = request_id
        ended = time.monotonic()
        emit("model_complete", node, request_id=request_id)
        output = ids if ids is not None else result["text"]
        calls[key] = {
            "user_id": graph["user_id"], "session_id": graph["session_id"],
            "scope_id": node["scope"], "request_id": request_id,
            "ready_at": start, "sent_at": sent, "completed_at": ended,
            "control_dispatch_s": sent - start, "elapsed_s": ended - start,
            "http_elapsed_s": ended - sent,
            "input_tokens": len(effective_input_ids), "output_tokens": row["output_budget"],
            "cached_tokens": meta.get("cached_tokens"),
            "queue_s": next((meta[k] for k in ("scheduler_queue_time", "queue_time_s", "queue_time")
                             if isinstance(meta.get(k), (int, float))), None),
            "spec_verify_ct": meta.get("spec_verify_ct"),
            "cached_tokens_details": meta.get("cached_tokens_details"),
            **delivery,
            "input_sha256": sha256(json.dumps(effective_input_ids, separators=(",", ":")).encode()).hexdigest(),
            "output_sha256": sha256(json.dumps(output).encode()).hexdigest(),
            "output_hash_basis": "ids" if ids is not None else "text",
            **({"realized_output_ids": list(ids)} if isinstance(ids, list) else {}),
            **({"causal_stitch": stitch_receipt} if continuity_mode != "recorded" else {}),
            **inheritance,
        }
        calls[key]["replay_prefix_continuity"] = continuity(
            completed_token_sequences.get(parent), effective_input_ids, node["scope"])
        completed_token_sequences[node["id"]] = {
            "input_ids": effective_input_ids,
            "recorded_input_ids": row["input_ids"],
            "output_ids": ids,
            "scope_id": node["scope"],
        }

    async with asyncio.TaskGroup() as group:
        for node in topological_nodes(graph):
            tasks[node["id"]] = group.create_task(execute(node))
        for row in sorted(graph["rounds"], key=lambda r: r["submit_s"]):
            for entry in row["entry_nodes"]:
                for dep in next(n for n in graph["nodes"] if n["id"] == entry)["deps"]:
                    if next(n for n in graph["nodes"] if n["id"] == dep)["kind"] == "think":
                        await tasks[dep]
            start = time.monotonic()
            gates[row["id"]].set()
            for end in row["exit_nodes"]:
                await tasks[end]
            rounds[(graph_id, row["id"])] = time.monotonic() - start
    if emit_workflow_end:
        for scope_name, final_rid in sorted(last_request_by_scope.items()):
            emit("workflow_end", request_id=final_rid, scope_id=scope_name)
    emit("session_complete")


def overlap_metrics(events):
    """Driver overlap, not GPU occupancy: /generate may still be queued."""
    active_models, active_tools = set(), set()
    previous = None
    overlap = request_time = tool_time = 0.0
    peak_models = peak_tools = 0
    for event in sorted(events, key=lambda e: e["observed_at"]):
        now = event["observed_at"]
        if previous is not None:
            elapsed = max(0.0, now - previous)
            overlap += elapsed * bool(active_models and active_tools)
            request_time += elapsed * bool(active_models)
            tool_time += elapsed * bool(active_tools)
        kind = event["kind"]
        key = (event.get("graph_id"), event.get("node_id"))
        if kind == "model_sent":
            active_models.add(key)
        elif kind == "model_complete":
            active_models.discard(key)
        elif kind == "tool_start":
            active_tools.add(key)
        elif kind == "tool_end":
            active_tools.discard(key)
        peak_models, peak_tools = max(peak_models, len(active_models)), max(peak_tools, len(active_tools))
        previous = now
    return {
        "request_tool_overlap_s": overlap, "any_request_inflight_s": request_time,
        "any_real_tool_active_s": tool_time, "peak_requests_inflight": peak_models,
        "peak_tools_active": peak_tools,
        "semantics": "client request includes engine queue and tools are replayed sleeps; not GPU utilization",
    }


def p95(values):
    values = sorted(values)
    return values[math.ceil(len(values) * 0.95) - 1] if values else None


async def trial(client, enabled, params, graphs, prompts, concurrency, knobs=None, *,
                recorded_gap_scale=0.0, tool_scale=1.0, lifecycle="local", cache=None, prefill=None,
                stream_metrics=False, chunk=None, multi=None, interval=None,
                continuity_mode="recorded", measure_pool_pressure=False):
    cache = {"mode": "observe"} if cache is None else cache
    prefill = {"mode": "off"} if prefill is None else prefill
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if lifecycle not in {"off", "local", "engine"}:
        raise ValueError("lifecycle must be off, local, or engine")
    if continuity_mode not in CONTINUITY_MODES:
        raise ValueError("continuity_mode must be recorded, causal_splice, or forced_trace")
    if cache and cache.get("mode") in {"tool-aware", "demand-aware"} and lifecycle != "engine":
        raise ValueError("tool-aware/demand-aware cache requires engine lifecycle events")
    if interval and interval.get("mode", "off") != "off" and lifecycle != "engine":
        raise ValueError("interval_cache requires engine lifecycle events")

    engine_before, before = await engine_state(client)
    is_v7_engine = before.get("adapter_version") == "orvix-dsv41-interval-cache-v7"
    if is_v7_engine and interval is None:
        raise RuntimeError(
            "Use interval-bench or pass explicit interval configuration on v7; "
            "legacy FCP replay cannot establish v7 lifecycle/config isolation"
        )
    engine_config = engine_configuration(engine_before)
    if is_v7_engine and "interval_cache" in before:
        deadline = time.monotonic() + 90
        while not before["interval_cache"].get("idle", True):
            if time.monotonic() >= deadline:
                raise RuntimeError("interval cache/engine did not drain before reset")
            await post_control(client, before["enabled"], before["parameters"], before["knobs"],
                               interval=before["interval_cache"]["config"])
            await asyncio.sleep(0.25)
            _, before = await engine_state(client)
    flushed = await client.post("/flush_cache")
    flush_receipt = cache_flush_acknowledged(flushed)
    await post_control(client, enabled, params, knobs, cache=cache, prefill=prefill,
                       chunk=chunk, multi=multi, interval=interval, reset=True)
    _, state = await engine_state(client)
    verify_configuration(state, enabled, params, knobs, cache, prefill, chunk, multi, interval)

    peak_running, peak_waiting = 0, 0
    pool_pressure_samples = 0
    pool_pressure_max = {}
    events, pending_events = [], []
    control_lock = asyncio.Lock()
    controls, control_seconds = 0, 0.0
    interval_active_obs = bool(interval and interval.get("mode", "off") != "off")
    immediate_tool_end_flush = bool(
        interval
        and interval.get("mode", "off") == "active"
        and interval.get("queue_prefetch", True)
        and interval.get("tool_end_prefetch", True)
    )
    allowed_wire_kinds = (
        {"tool_start", "tool_end", "request_ready", "workflow_end"}
        if is_v7_engine
        else {"tool_start", "tool_end", "request_ready"}
    )

    def queue_event(event):
        if lifecycle == "engine" and event["kind"] in allowed_wire_kinds:
            if event.get("request_id"):
                if interval_active_obs:
                    pending_events.append(event)
                elif event["kind"] in {"tool_start", "tool_end", "request_ready"}:
                    pending_events.append(event)

    async def send_control(*, only_if_pending=False):
        nonlocal controls, control_seconds
        async with control_lock:
            if only_if_pending and not pending_events:
                return
            count = len(pending_events)
            batch = pending_events[:count] if lifecycle == "engine" else None
            started = time.monotonic()
            await post_control(client, enabled, params, knobs, events=batch, cache=cache,
                               prefill=prefill, chunk=chunk, multi=multi, interval=interval)
            control_seconds += time.monotonic() - started
            controls += 1
            del pending_events[:count]

    async def heartbeat():
        nonlocal peak_running, peak_waiting, pool_pressure_samples
        tick = 0
        while True:
            await asyncio.sleep(0.25)
            await send_control()
            tick += 1
            if tick % 4 == 0:
                sample = await client.get("/v1/loads")
                sample.raise_for_status()
                load = sample.json()["loads"][0]
                peak_running = max(peak_running, load["num_running_reqs"])
                peak_waiting = max(peak_waiting, load["num_waiting_reqs"])
                if measure_pool_pressure:
                    pool_pressure_samples += 1
                    for name in ("num_used_tokens", "num_total_tokens", "num_active_tokens",
                                 "num_waiting_uncached_tokens", "token_usage"):
                        value = load.get(name)
                        if type(value) in (int, float):
                            pool_pressure_max[name] = max(value, pool_pressure_max.get(name, 0))

    calls, rounds, graph_times = {}, {}, []
    slots = asyncio.Semaphore(concurrency)

    async def run_one(graph):
        offered = time.monotonic()
        async with slots:
            admitted = time.monotonic()
            await replay_graph(
                client, graph, prompts, calls, rounds, recorded_gap_scale=recorded_gap_scale,
                tool_scale=tool_scale, events=events if lifecycle != "off" else None,
                on_event=queue_event if lifecycle == "engine" else None,
                on_tool_end=(lambda: send_control(only_if_pending=True))
                    if (lifecycle == "engine" and immediate_tool_end_flush) else None,
                before_request=(lambda: send_control(only_if_pending=True))
                    if lifecycle == "engine" else None, stream_metrics=stream_metrics,
                wait_inheritance=enabled and params.get("fairness_mode") == "wait_inheritance",
                continuity_mode=continuity_mode,
                emit_workflow_end=is_v7_engine)
            completed = time.monotonic()
            graph_times.append({
                "graph_id": graph["graph_id"], "user_id": graph["user_id"],
                "offered_at": offered, "admitted_at": admitted, "completed_at": completed,
                "admission_wait_s": admitted - offered,
                "active_elapsed_s": completed - admitted,
                "offered_to_completion_s": completed - offered,
            })

    async def run_graphs():
        async with asyncio.TaskGroup() as group:
            for graph in graphs:
                group.create_task(run_one(graph))

    start = time.monotonic()
    work, clock = asyncio.create_task(run_graphs()), asyncio.create_task(heartbeat())
    try:
        done, _ = await asyncio.wait((work, clock), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in (work, clock):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, clock, return_exceptions=True)
    elapsed = time.monotonic() - start
    await send_control(only_if_pending=True)
    if len(calls) != len(prompts) or len(rounds) != sum(len(g["rounds"]) for g in graphs):
        raise ValueError("incomplete trial")
    engine_after, after = await engine_state(client)
    if engine_configuration(engine_after) != engine_config:
        raise RuntimeError("engine configuration changed during the trial")
    verify_configuration(after, enabled, params, knobs, cache, prefill, chunk, multi, interval)
    queues = [row["queue_s"] for row in calls.values() if row["queue_s"] is not None]
    counts = numeric_counter_delta(state.get("counts", {}), after.get("counts", {}))
    delivery_ttft = [r["client_ttft_s"] for r in calls.values() if r.get("client_ttft_s") is not None]
    delivery_tpot = [r["client_post_first_chunk_s_per_token"] for r in calls.values()
                     if r.get("client_post_first_chunk_s_per_token") is not None]
    delivery_gaps = [r["client_max_delivery_gap_s"] for r in calls.values()
                     if r.get("client_max_delivery_gap_s") is not None]
    ic_before = state.get("interval_cache", {})
    ic_after = after.get("interval_cache", {})
    wb_wall_delta = (
        ic_after.get("native_writeback_wall_s", 0.0) - ic_before.get("native_writeback_wall_s", 0.0)
        if ic_after.get("native_writeback_instrumented")
        else None
    )
    wb_calls_delta = (
        ic_after.get("native_writeback_calls", 0) - ic_before.get("native_writeback_calls", 0)
        if ic_after.get("native_writeback_instrumented")
        else None
    )
    identity = sorted((g, n, len(r["input_ids"]), r["output_budget"], r.get("cache_salt"),
                       sha256(json.dumps(r["input_ids"], separators=(",", ":")).encode()).hexdigest())
                      for (g, n), r in prompts.items())
    replay_opts = {
        "recorded_gap_scale": recorded_gap_scale,
        "tool_scale": tool_scale,
        "lifecycle": lifecycle,
        "stream_metrics": stream_metrics,
        "arrival_model": "closed_loop_graph_slots",
    }
    if continuity_mode != "recorded":
        replay_opts["continuity_mode"] = continuity_mode
    result = {
        "mode": "UNIFIED-rank" if enabled else "B0", "wall_s": elapsed,
        "round_p50_s": median(rounds.values()), "round_p95_s": p95(rounds.values()),
        "queue_p95_s": p95(queues), "queue_measured_calls": len(queues),
        "client_ttft_p95_s": p95(delivery_ttft), "client_ttft_measured_calls": len(delivery_ttft),
        "client_post_first_chunk_s_per_token_p95": p95(delivery_tpot),
        "client_post_first_chunk_measured_calls": len(delivery_tpot),
        "client_max_delivery_gap_s": max(delivery_gaps, default=None),
        "client_delivery_gap_p95_s": p95(delivery_gaps),
        "native_writeback_wall_s_delta": wb_wall_delta,
        "native_writeback_calls_delta": wb_calls_delta,
        "interval_native_writeback_wall_s": wb_wall_delta,
        "interval_native_actions": numeric_counter_delta(
            ic_before.get("native_actions", {}), ic_after.get("native_actions", {})
        ),
        "interval_counters": numeric_counter_delta(ic_before.get("counts", {}), ic_after.get("counts", {})),
        "stream_measurement_semantics": "client-delivery TTFT and chunk-adjusted mean; not true token ITL or GPU time",
        "adapter_cpu_s_delta": after.get("adapter_cpu_s", 0.) - state.get("adapter_cpu_s", 0.),
        "model_calls": len(calls), "rounds": len(rounds), "rank_counters": counts,
        "peak_running_sampled": peak_running, "peak_waiting_sampled": peak_waiting,
        "engine_peak_candidates_cumulative": after["counts"].get("peak_candidates"),
        "engine_state_after": after, "engine_configuration": engine_config,
        "cache_flush_receipt": flush_receipt,
        "native_admission_counters": numeric_counter_delta(state.get("native_admission", {}).get("counts", {}),
                                                           after.get("native_admission", {}).get("counts", {})),
        "cache_counters": numeric_counter_delta(state.get("cache", {}).get("counts", {}),
                                                  after.get("cache", {}).get("counts", {})),
        "prefill_counters": numeric_counter_delta(state.get("prefill", {}).get("counts", {}),
                                                    after.get("prefill", {}).get("counts", {})),
        "workload_sha256": sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest(),
        "replay_options": replay_opts,
        "cache_configuration": cache, "prefill_configuration": prefill,
        "chunk_configuration": chunk or {"mode": "off"},
        "multi_configuration": multi or {"mode": "off"},
        "multi_counters": numeric_counter_delta(state.get("multi_chunk", {}).get("counts", {}),
                                                  after.get("multi_chunk", {}).get("counts", {})),
        "chunk_counters": numeric_counter_delta(state.get("chunk_fair", {}).get("counts", {}),
                                                 after.get("chunk_fair", {}).get("counts", {})),
        "control_requests": controls, "control_request_elapsed_s": control_seconds,
        "control_dispatch_total_s": sum(r["control_dispatch_s"] for r in calls.values()),
        "lifecycle_counts": dict(Counter(e["kind"] for e in events)),
        "overlap": overlap_metrics(events), "events": events, "graph_times": graph_times,
        "calls": [{"graph_id": g, "node_id": n, **row} for (g, n), row in calls.items()],
        "round_times": [{"graph_id": g, "round_id": r, "elapsed_s": value}
                        for (g, r), value in rounds.items()],
    }
    if measure_pool_pressure:
        result.update(pool_pressure_samples=pool_pressure_samples, pool_pressure_max=pool_pressure_max)
    if interval is not None and "interval_cache" in after:
        before_ic = state.get("interval_cache", {})
        after_ic = after.get("interval_cache", {})
        result.update(
            interval_configuration=interval,
            interval_counters=numeric_counter_delta(before_ic.get("counts", {}), after_ic.get("counts", {})),
            interval_lifecycle=numeric_counter_delta(before_ic.get("lifecycle", {}), after_ic.get("lifecycle", {})),
            interval_native_actions=numeric_counter_delta(before_ic.get("native_actions", {}), after_ic.get("native_actions", {})),
            interval_native_writeback_instrumented=bool(after_ic.get("native_writeback_instrumented")),
            interval_native_writeback_calls=after_ic.get("native_writeback_calls", 0) - before_ic.get("native_writeback_calls", 0),
            interval_native_writeback_wall_s=after_ic.get("native_writeback_wall_s", 0.0) - before_ic.get("native_writeback_wall_s", 0.0),
        )
    return result


async def main(args):
    async with httpx.AsyncClient(base_url=args.endpoint, timeout=1800, trust_env=False) as client:
        state, _ = await engine_state(client)
        graphs, prompts = read_inputs(
            args.root, state["context_length"],
            workload_copies=args.workload_copies, salt_per_copy=args.salt_per_copy,
        )
        params = Parameters(theta_fg=20, theta_bg=60, prefill_tokens_per_s=11500,
                            decode_tokens_per_s=250, uniform_output_tokens=197).frozen()
        interval_active = None
        interval_b0 = None
        if args.interval_profile is not None:
            interval_active = asdict(IntervalConfig(**json.loads(args.interval_profile.read_text())).validate())
            interval_b0 = asdict(IntervalConfig(mode="observe"))
        args.output.mkdir(mode=0o700)
        by_length = sorted(prompts.values(), key=lambda row: len(row["input_ids"]))
        warmup = [by_length[i * (len(by_length) - 1) // 7] for i in range(8)]

        async def warm(row):
            req = payload({**row, "output_budget": 32}, "warmup", "foreground", time.monotonic())
            response = await client.post("/generate", json=req)
            response.raise_for_status()
            if response.json()["meta_info"]["completion_tokens"] != 32:
                raise RuntimeError("warmup token count mismatch")

        await asyncio.gather(*(warm(row) for row in warmup))
        print("warmup complete", flush=True)
        for index, method in enumerate(args.order, 1):
            use_interval = None
            use_enabled = method == "U"
            if interval_active is not None:
                use_enabled = False
                use_interval = interval_active if method == "U" else interval_b0
            result = await trial(
                client, use_enabled, params, graphs, prompts, args.concurrency,
                recorded_gap_scale=args.recorded_gap_scale,
                tool_scale=args.tool_scale, lifecycle=args.lifecycle,
                cache={"mode": args.cache_mode}, stream_metrics=args.stream_metrics,
                interval=use_interval, continuity_mode=args.continuity_mode,
            )
            result.update(
                offered_graph_concurrency=args.concurrency,
                workload_copies=args.workload_copies,
                salt_per_copy=args.salt_per_copy,
                order=args.order,
                parameters=params,
            )
            with (args.output / f"{index:02d}-{method}.json").open("x") as handle:
                json.dump(result, handle, indent=2)
            print(index, method, result["wall_s"], result["round_p95_s"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--order", choices=("BU", "BUUB"), default="BUUB")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--workload-copies", type=int, default=1,
                        help="deterministically replicate the 65-workflow DAG (e.g., 2->130, 4->260) for 64/128+ slot tests")
    parser.add_argument("--salt-per-copy", action="store_true",
                        help="attach distinct cache_salt per copy so replicated workflows do not share RadixCache pages")
    parser.add_argument("--interval-profile", type=Path,
                        help="optional v7/v8 IntervalConfig JSON for B0(observe) vs U(active) replay")
    parser.add_argument("--continuity-mode", choices=CONTINUITY_MODES, default="causal_splice")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--recorded-gap-scale", type=float, default=0.0)
    parser.add_argument("--tool-scale", type=float, default=1.0)
    parser.add_argument("--lifecycle", choices=("off", "local", "engine"), default="local")
    parser.add_argument("--cache-mode", choices=("observe", "tool-aware", "demand-aware"), default="observe")
    parser.add_argument("--stream-metrics", action="store_true")
    asyncio.run(main(parser.parse_args()))
