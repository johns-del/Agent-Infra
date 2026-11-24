#!/usr/bin/env python3
"""Single-node Prefill/Decode (PD) speed benchmark + DSpark stochastic/tail-graph warmup across .14/.13/.12."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics
import time

import httpx
from tokenizers import Tokenizer


WORDS = (
    "scheduler", "prefill", "decode", "interval", "cache", "prefix", "radix", "token",
    "batch", "worker", "tensor", "expert", "router", "speculative", "draft", "verify",
    "kernel", "graph", "memory", "window", "attention", "dispatch", "latency", "throughput",
    "context", "session", "tool", "parser", "stream", "checkpoint", "trace", "metric",
    "adapter", "governor", "balance", "guard", "queue", "frame", "shard", "block",
)


def make_prompt_ids(tokenizer: Tokenizer, seed: int, length: int, *, coding: bool = False) -> list[int]:
    rng = random.Random(seed)
    lines = [
        f"RunSeed={seed} Deployment=tri-node-ours Benchmark={'decode-code' if coding else 'prefill-events'}"
    ]
    if coding:
        lines.append(
            "Task: Implement Python code to parse, validate, index, and deduplicate agent trace events. "
            "Include strict dataclasses, type annotations, docstrings, and self-tests."
        )
    else:
        lines.append(
            "Task: Analyze the following synthetic execution trace records and report summary counts concisely."
        )
    needed_est = max(24, length // 8 + 48)
    for idx in range(needed_est):
        w0, w1, w2, w3 = (rng.choice(WORDS) for _ in range(4))
        v0 = rng.randrange(100000, 999999)
        v1 = rng.randrange(10, 999)
        lines.append(f"[{idx:05d}] svc={w0}_{w1} op={w2}.{w3} req={v0:06d} chunks={v1} status=ready")
    ids = tokenizer.encode("\n".join(lines), add_special_tokens=False).ids
    while len(ids) < length:
        extra = " ".join(rng.choice(WORDS) + f"_{rng.randrange(1000, 9999)}" for _ in range(64))
        ids.extend(tokenizer.encode("\n" + extra, add_special_tokens=False).ids)
    return ids[:length]


async def stream_generate_probe(
    client: httpx.AsyncClient,
    base_url: str,
    label: str,
    input_ids: list[int],
    output_tokens: int,
    *,
    stochastic: bool = False,
    session_id: str | None = None,
) -> dict:
    sampling: dict = {"temperature": 0.0, "max_new_tokens": output_tokens, "ignore_eos": True}
    if stochastic:
        sampling.update(temperature=0.7, top_p=0.95, top_k=40, sampling_seed=41920)
    payload: dict = {"input_ids": input_ids, "stream": True, "sampling_params": sampling}
    headers = {"Content-Type": "application/json"}
    if session_id:
        payload["session_id"] = session_id
        headers["x-orvix-session-id"] = session_id

    first_time: float | None = None
    last_time: float | None = None
    first_tokens = 0
    final: dict | None = None
    done = False

    t0 = time.monotonic()
    async with client.stream("POST", f"{base_url}/generate", headers=headers, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                done = True
                break
            obj = json.loads(data_str)
            meta = obj.get("meta_info") or {}
            comp = int(meta.get("completion_tokens") or 0)
            if comp <= 0:
                continue
            now = time.monotonic()
            last_time = now
            if first_time is None:
                first_time = now
                first_tokens = comp
            final = meta
    end_t = time.monotonic()

    if not done or final is None or first_time is None or last_time is None:
        raise RuntimeError(f"{label}: incomplete stream (done={done}, final={final})")

    ttft_s = first_time - t0
    wall_s = end_t - t0
    cached = int(final.get("cached_tokens") or 0)
    ptok = int(final.get("prompt_tokens") or len(input_ids))
    ctok = int(final.get("completion_tokens") or output_tokens)
    out: dict = {
        "case": label,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "cached_tokens": cached,
        "ttft_ms": round(ttft_s * 1000.0, 2),
        "wall_ms": round(wall_s * 1000.0, 2),
        "prompt_tok_per_ttft_s": round(ptok / ttft_s, 2) if ttft_s > 0 else None,
        "spec_accept_length": final.get("spec_accept_length"),
        "spec_verify_ct": final.get("spec_verify_ct"),
    }
    if ctok > first_tokens and last_time > first_time:
        dec = (ctok - first_tokens) / (last_time - first_time)
        out["decode_tok_s"] = round(dec, 2)
        out["tpot_ms"] = round(1000.0 / dec, 2)
    else:
        out["decode_tok_s"] = None
        out["tpot_ms"] = None
    return out


async def bench_gateway_chat_stream(client: httpx.AsyncClient, compat_url: str, api_key: str, seed: int) -> dict:
    prompt_text = (
        f"[gw-pd-{seed}] Write clean Python dataclasses and unit tests for a priority queue scheduler. "
        "Keep code well-structured and detailed."
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "ds",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
    first_t: float | None = None
    last_t: float | None = None
    usage: dict | None = None
    chunks = 0
    relay_route = None
    worker_node = None
    async with client.stream("POST", f"{compat_url}/chat/completions", headers=headers, json=body) as resp:
        resp.raise_for_status()
        relay_route = resp.headers.get("X-Orvix-Relay-Route")
        worker_node = resp.headers.get("X-Orvix-Worker-Node")
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or ()
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.monotonic()
                    if first_t is None:
                        first_t = now
                    last_t = now
                    chunks += 1
    end_t = time.monotonic()
    assert first_t is not None and last_t is not None and usage is not None
    ctok = int(usage.get("completion_tokens") or 0)
    dec_span = max(last_t - first_t, 1e-4)
    return {
        "model": "ds",
        "relay_route": relay_route,
        "worker_node": worker_node,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": ctok,
        "chunks": chunks,
        "ttft_ms": round((first_t - t0) * 1000.0, 2),
        "wall_ms": round((end_t - t0) * 1000.0, 2),
        "decode_tok_s_after_ttft": round(max(ctok - 1, 1) / dec_span, 2) if ctok > 1 else None,
        "e2e_output_tok_s": round(ctok / max(end_t - t0, 1e-4), 2),
    }


async def benchmark_node(
    client: httpx.AsyncClient,
    tokenizer: Tokenizer,
    name: str,
    base_url: str,
    node_seed: int,
) -> dict:
    # 1. Warmup: DSpark stochastic sampler + decoder-tail-graph replay warmup (4K) + short greedy warmup
    warmup_stoch = await stream_generate_probe(
        client,
        base_url,
        "warmup_stochastic_4k",
        make_prompt_ids(tokenizer, node_seed + 1, 4096),
        32,
        stochastic=True,
        session_id=f"pd-warm-stoch-{name}-{node_seed}",
    )
    assert (warmup_stoch.get("spec_verify_ct") or 0) > 0, f"{name} stochastic warmup lacked verify steps"
    warmup_greedy = await stream_generate_probe(
        client,
        base_url,
        "warmup_greedy_512",
        make_prompt_ids(tokenizer, node_seed + 2, 512, coding=True),
        32,
        stochastic=False,
        session_id=f"pd-warm-greedy-{name}-{node_seed}",
    )

    # 2. Cold Prefill 4K
    cold_4k = await stream_generate_probe(
        client,
        base_url,
        "cold_prefill_4k",
        make_prompt_ids(tokenizer, node_seed + 10, 4096),
        1,
        session_id=f"pd-cold4k-{name}-{node_seed}",
    )
    assert cold_4k["cached_tokens"] == 0, f"{name} cold_4k had cached={cold_4k['cached_tokens']}"

    # 3. Cold Prefill 16K + 4. Warm Prefill 16K
    ids_16k = make_prompt_ids(tokenizer, node_seed + 20, 16384)
    sess_16k = f"pd-16k-{name}-{node_seed}"
    cold_16k = await stream_generate_probe(
        client,
        base_url,
        "cold_prefill_16k",
        ids_16k,
        1,
        session_id=sess_16k,
    )
    assert cold_16k["cached_tokens"] == 0, f"{name} cold_16k had cached={cold_16k['cached_tokens']}"

    warm_ids_16k = ids_16k + make_prompt_ids(tokenizer, node_seed + 21, 16)
    warm_16k = await stream_generate_probe(
        client,
        base_url,
        "warm_prefill_16k",
        warm_ids_16k,
        1,
        session_id=sess_16k,
    )
    assert warm_16k["cached_tokens"] >= 15872, f"{name} warm_16k cache hit too low: {warm_16k}"

    # 5. Sustained Single-Stream Decode C=1 (512 -> 384)
    decode_c1 = await stream_generate_probe(
        client,
        base_url,
        "decode_c1_code_384",
        make_prompt_ids(tokenizer, node_seed + 30, 512, coding=True),
        384,
        session_id=f"pd-dec1-{name}-{node_seed}",
    )

    # 6. Concurrent Decode C=4 (4 x (512 -> 256))
    c4_prompts = [make_prompt_ids(tokenizer, node_seed + 40 + i, 512, coding=True) for i in range(4)]
    c4_t0 = time.monotonic()
    c4_runs = await asyncio.gather(
        *(
            stream_generate_probe(
                client,
                base_url,
                f"decode_c4_stream_{i}",
                c4_prompts[i],
                256,
                session_id=f"pd-decc4-{name}-{node_seed}-{i}",
            )
            for i in range(4)
        )
    )
    c4_wall_s = time.monotonic() - c4_t0
    dec_rates = [r["decode_tok_s"] for r in c4_runs if r.get("decode_tok_s") is not None]
    accept_lens = [float(r["spec_accept_length"]) for r in c4_runs if r.get("spec_accept_length") is not None]
    decode_c4 = {
        "streams": 4,
        "total_output_tokens": sum(r["completion_tokens"] for r in c4_runs),
        "wall_ms": round(c4_wall_s * 1000.0, 2),
        "aggregate_total_tok_s": round(sum(r["completion_tokens"] for r in c4_runs) / c4_wall_s, 2),
        "mean_per_stream_decode_tok_s": round(statistics.mean(dec_rates), 2) if dec_rates else None,
        "mean_spec_accept_length": round(statistics.mean(accept_lens), 3) if accept_lens else None,
        "per_stream": c4_runs,
    }

    return {
        "warmup_stochastic_4k": warmup_stoch,
        "warmup_greedy_512": warmup_greedy,
        "cold_prefill_4k": cold_4k,
        "cold_prefill_16k": cold_16k,
        "warm_prefill_16k": warm_16k,
        "decode_c1_code_384": decode_c1,
        "decode_c4_code_256": decode_c4,
    }


async def run_suite(args: argparse.Namespace) -> dict:
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    base_seed = args.seed or int(time.time())
    receipt: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": base_seed,
        "nodes": {},
        "gateway_chat_stream_e2e": {},
    }
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
    timeout = httpx.Timeout(300.0, connect=10.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:
        for idx, (name, base_url) in enumerate(args.nodes.items()):
            receipt["nodes"][name] = await benchmark_node(
                client, tokenizer, name, base_url, base_seed + (idx + 1) * 1000
            )
        if args.api_key:
            receipt["gateway_chat_stream_e2e"] = await bench_gateway_chat_stream(
                client, args.compat_url, args.api_key, base_seed + 9999
            )
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        name: {
            "cold_4k_tok_s": data["cold_prefill_4k"]["prompt_tok_per_ttft_s"],
            "cold_4k_ttft_ms": data["cold_prefill_4k"]["ttft_ms"],
            "cold_16k_tok_s": data["cold_prefill_16k"]["prompt_tok_per_ttft_s"],
            "cold_16k_ttft_ms": data["cold_prefill_16k"]["ttft_ms"],
            "warm_16k_cached": data["warm_prefill_16k"]["cached_tokens"],
            "warm_16k_ttft_ms": data["warm_prefill_16k"]["ttft_ms"],
            "decode_c1_tok_s": data["decode_c1_code_384"]["decode_tok_s"],
            "decode_c1_tpot_ms": data["decode_c1_code_384"]["tpot_ms"],
            "decode_c1_accept": data["decode_c1_code_384"]["spec_accept_length"],
            "decode_c4_agg_tok_s": data["decode_c4_code_256"]["aggregate_total_tok_s"],
            "decode_c4_stream_tok_s": data["decode_c4_code_256"]["mean_per_stream_decode_tok_s"],
        }
        for name, data in receipt["nodes"].items()
    }
    print(json.dumps({"summary": summary, "gateway_chat_stream_e2e": receipt["gateway_chat_stream_e2e"]}, indent=2))
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
    )
    for label in ("dot14", "dot13", "dot12"):
        ap.add_argument(f"--{label}-url")
    ap.add_argument("--compat-url", default="http://127.0.0.1:32682/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("pd_benchmark_receipt.json"))
    args = ap.parse_args()
    args.nodes = {
        label: url for label in ("dot14", "dot13", "dot12")
        if (url := getattr(args, f"{label}_url"))
    }
    if not args.nodes:
        ap.error("provide at least one node URL")
    asyncio.run(run_suite(args))


if __name__ == "__main__":
    main()
