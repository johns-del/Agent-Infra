#!/usr/bin/env python3
"""
STRATA v14 Comprehensive 7-Phase Load-Balancing Evaluation Runner (C = 1..24)
Replays real multi-turn OpenAI chat traces from 20261120-unified directly against
.13 gateway (http://127.0.0.1:32862/v1) and collects:
  - Client streaming metrics: TTFT (C1 >120s), TBT P99 & max gap (C2 >2.0s), JCT, Q, Slowdown, Jain fairness, M1 makespan
  - Cache efficiency (G2): per-turn cold vs warm hit rates, >=90% / 0% counts, Prometheus prompt/cached/hicache deltas, SystemPrefixPatcher stats
  - Load-balancing telemetry: .13 relay priority/sticky/router spillover/evictions/failovers, .14/.12 worker splits, 1Hz peak resource curves
  - Incremental per-phase checkpointing + automatic Markdown report + safety circuit breaker
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import httpx


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cst_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def nearest_rank(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
    return round(float(s[idx]), 4)


def jain_index(vals: list[float]) -> float | None:
    pos = [v for v in vals if v > 0]
    if not pos:
        return None
    s1 = sum(pos)
    s2 = sum(v * v for v in pos)
    return round((s1 * s1) / (len(pos) * s2), 4) if s2 > 0 else None


def load_api_key(key_file: Path | None) -> str:
    if token := os.environ.get("OPENAI_API_KEY", "").strip():
        return token
    candidates = [key_file] if key_file else []
    for p in candidates:
        if not p.is_file():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k in ("api_key", "token", "key", "model"):
                if isinstance(raw.get(k), str) and raw[k].strip():
                    return raw[k].strip()
            # tokens.json may be {"tokens": ["..."]} or {"<token>": {...}}
            if isinstance(raw.get("tokens"), list) and raw["tokens"]:
                t0 = raw["tokens"][0]
                if isinstance(t0, str) and t0.strip():
                    return t0.strip()
                if isinstance(t0, dict):
                    for k in ("token", "api_key", "key"):
                        if isinstance(t0.get(k), str) and t0[k].strip():
                            return t0[k].strip()
            if isinstance(raw.get("tokens"), dict) and raw["tokens"]:
                return next(iter(raw["tokens"].keys()))
            for k, v in raw.items():
                if isinstance(v, dict) and any(x in v for x in ("role", "scopes", "models", "name", "user")):
                    return str(k)
        elif isinstance(raw, list) and raw and isinstance(raw[0], str):
            return raw[0].strip()
    raise FileNotFoundError("Set OPENAI_API_KEY or pass --api-key-file for a live run")


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    sums = {
        "prompt_tokens_total": 0.0,
        "cached_tokens_total": 0.0,
        "generation_tokens_total": 0.0,
        "num_requests_total": 0.0,
    }
    maxes = {
        "token_usage": 0.0,
        "hicache_host_used_tokens": 0.0,
        "num_queue_reqs": 0.0,
        "num_running_reqs": 0.0,
    }
    for line in text.splitlines():
        if not line or line.startswith("#") or not line.startswith("sglang:"):
            continue
        m = re.match(r"^sglang:([a-zA-Z0-9_]+)(?:\{[^}]*\})?\s+([0-9eE.+-]+)$", line.strip())
        if not m:
            continue
        name, val_s = m.group(1), m.group(2)
        try:
            val = float(val_s)
        except ValueError:
            continue
        if name in sums:
            sums[name] += val
        if name in maxes:
            maxes[name] = max(maxes[name], val)
    out.update(sums)
    out.update(maxes)
    return out


async def snapshot_cluster(client: httpx.AsyncClient, relay_url: str, dot14_url: str, dot12_url: str) -> dict[str, Any]:
    async def fetch_json(url: str) -> dict | None:
        try:
            r = await client.get(url, timeout=5.0)
            return r.json() if r.status_code == 200 else {"http_status": r.status_code}
        except Exception as exc:
            return {"error": repr(exc)}

    async def fetch_prom(url: str) -> dict[str, float] | None:
        try:
            r = await client.get(url, timeout=5.0)
            return parse_prometheus_metrics(r.text) if r.status_code == 200 else None
        except Exception:
            return None

    relay, d14, d12, p14, p12 = await asyncio.gather(
        fetch_json(f"{relay_url}/orvix_relay_status"),
        fetch_json(f"{dot14_url}/orvix_daemon_status"),
        fetch_json(f"{dot12_url}/orvix_daemon_status"),
        fetch_prom(f"{dot14_url}/metrics"),
        fetch_prom(f"{dot12_url}/metrics"),
    )
    return {
        "ts": time.time(),
        "relay": relay,
        "dot14_daemon": d14,
        "dot12_daemon": d12,
        "dot14_prom": p14,
        "dot12_prom": p12,
    }


def _sanitize_json_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "properties" and isinstance(v, dict):
                out[k] = {
                    pk: (_sanitize_json_schema(pv) if isinstance(pv, (dict, bool)) else {"type": "string", "description": "redacted"})
                    for pk, pv in v.items()
                }
            elif k in ("items", "additionalProperties") and not isinstance(v, (dict, bool, list, type(None))):
                out[k] = {"type": "string", "description": "redacted"}
            else:
                out[k] = _sanitize_json_schema(v)
        return out
    if isinstance(node, list):
        return [_sanitize_json_schema(x) for x in node]
    return node


def inject_prefix_nonce(body: dict, nonce: str) -> dict:
    """Return a shallow-copied chat body with a tiny deterministic prefix nonce at msg[0] start."""
    out = dict(body)
    msgs = copy.deepcopy(body["messages"])
    tag = f"[ORVIX-EVAL:{nonce}]\n"
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") in ("system", "developer"):
        c = msgs[0].get("content")
        if isinstance(c, str):
            msgs[0]["content"] = tag + c
        elif isinstance(c, list):
            msgs[0]["content"] = [{"type": "text", "text": tag}] + c
    else:
        msgs.insert(0, {"role": "system", "content": tag.strip()})
    out["messages"] = msgs
    if isinstance(body.get("tools"), list):
        out["tools"] = _sanitize_json_schema(body["tools"])
    return out


async def send_streaming_turn(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    body: dict,
    max_tokens: int,
    session_id: str | None,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    payload = dict(body)
    payload["model"] = model
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    payload["max_tokens"] = int(max_tokens)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if session_id:
        headers["X-Orvix-Session-Id"] = session_id

    t0 = time.perf_counter()
    ttft_s: float | None = None
    ttft_visible_s: float | None = None
    last_tok_t: float | None = None
    gaps_s: list[float] = []
    chunks = 0
    usage: dict | None = None
    status = 0
    err: str | None = None

    try:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=15.0),
        ) as resp:
            status = resp.status_code
            if status != 200:
                err_body = (await resp.aread()).decode("utf-8", errors="replace")[:300]
                err = f"HTTP {status}: {err_body}"
            else:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        vis_tok = delta.get("content") or ("T" if delta.get("tool_calls") else "")
                        any_tok = vis_tok or delta.get("reasoning_content") or ""
                        if any_tok:
                            now = time.perf_counter()
                            if ttft_s is None:
                                ttft_s = now - t0
                            elif last_tok_t is not None:
                                gaps_s.append(now - last_tok_t)
                            if vis_tok and ttft_visible_s is None:
                                ttft_visible_s = now - t0
                            last_tok_t = now
                            chunks += 1
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"

    total_s = time.perf_counter() - t0
    pt = int((usage or {}).get("prompt_tokens") or 0)
    ct = int(((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    comp = int((usage or {}).get("completion_tokens") or 0)
    max_gap_s = max(gaps_s) if gaps_s else 0.0
    tbt_ms = [g * 1000.0 for g in gaps_s]
    decode_s = max(0.0, total_s - ttft_s) if ttft_s is not None else 0.0
    tpot_ms = round((decode_s * 1000.0) / max(1, comp - 1), 2) if (ttft_s is not None and comp > 1 and decode_s > 0) else None
    decode_tok_s = round((comp - 1) / decode_s, 2) if (ttft_s is not None and comp > 1 and decode_s > 0.001) else None

    return {
        "ok": status == 200 and err is None,
        "status": status,
        "error": err,
        "ttft_s": round(ttft_s, 4) if ttft_s is not None else None,
        "ttft_visible_s": round(ttft_visible_s, 4) if ttft_visible_s is not None else None,
        "decode_s": round(decode_s, 4),
        "total_s": round(total_s, 4),
        "tpot_ms": tpot_ms,
        "decode_tok_s": decode_tok_s,
        "max_gap_s": round(max_gap_s, 4),
        "stall_gt_0_5s": sum(1 for g in gaps_s if g > 0.5),
        "stall_gt_1_0s": sum(1 for g in gaps_s if g > 1.0),
        "stall_gt_2_0s": sum(1 for g in gaps_s if g > 2.0),
        "tbt_mean_ms": round(sum(tbt_ms) / len(tbt_ms), 2) if tbt_ms else None,
        "tbt_p50_ms": nearest_rank(tbt_ms, 0.50),
        "tbt_p95_ms": nearest_rank(tbt_ms, 0.95),
        "tbt_p99_ms": nearest_rank(tbt_ms, 0.99),
        "chunks": chunks,
        "prompt_tokens": pt,
        "cached_tokens": ct,
        "hit_rate_pct": round(100.0 * ct / pt, 2) if pt else 0.0,
        "completion_tokens": comp,
    }


def diff_snapshots(before: dict, after: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rb = (before.get("relay") or {}).get("stats") or {}
    ra = (after.get("relay") or {}).get("stats") or {}
    out["relay_delta"] = {k: ra.get(k, 0) - rb.get(k, 0) for k in ra if isinstance(ra.get(k), (int, float))}

    for node in ("dot14", "dot12"):
        db = before.get(f"{node}_daemon") or {}
        da = after.get(f"{node}_daemon") or {}
        sb = db.get("stats") or {}
        sa = da.get("stats") or {}
        pb = db.get("system_prefix_patcher") or {}
        pa = da.get("system_prefix_patcher") or {}
        out[f"{node}_req_delta"] = sa.get("requests_total", 0) - sb.get("requests_total", 0)
        out[f"{node}_patcher_delta"] = {
            k: pa.get(k, 0) - pb.get(k, 0)
            for k in ("sessions_initialized", "patched_new_delta", "patched_steady_replay", "guardrail_refreshes", "frozen_prefix_chars_preserved", "delta_chars_relocated")
            if isinstance(pa.get(k), int)
        }
        prb = before.get(f"{node}_prom") or {}
        pra = after.get(f"{node}_prom") or {}
        if prb and pra:
            pt_d = max(0.0, pra.get("prompt_tokens_total", 0.0) - prb.get("prompt_tokens_total", 0.0))
            ct_d = max(0.0, pra.get("cached_tokens_total", 0.0) - prb.get("cached_tokens_total", 0.0))
            gt_d = max(0.0, pra.get("generation_tokens_total", 0.0) - prb.get("generation_tokens_total", 0.0))
            out[f"{node}_prom_delta"] = {
                "prompt_tokens": round(pt_d),
                "cached_tokens": round(ct_d),
                "uncached_tokens": round(max(0.0, pt_d - ct_d)),
                "generation_tokens": round(gt_d),
                "hit_rate_pct": round(100.0 * ct_d / pt_d, 2) if pt_d > 0 else None,
            }
    return out


class BackgroundSampler:
    """1Hz background telemetry sampler for peak inflight / KV usage / HiCache / queue depth."""

    def __init__(self, client: httpx.AsyncClient, relay_url: str, dot14_url: str, dot12_url: str):
        self.client = client
        self.relay_url = relay_url
        self.dot14_url = dot14_url
        self.dot12_url = dot12_url
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.samples: list[dict] = []

    async def _loop(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                r_req = self.client.get(f"{self.relay_url}/orvix_relay_status", timeout=2.0)
                p14_req = self.client.get(f"{self.dot14_url}/metrics", timeout=2.0)
                p12_req = self.client.get(f"{self.dot12_url}/metrics", timeout=2.0)
                r_res, p14_res, p12_res = await asyncio.gather(r_req, p14_req, p12_req, return_exceptions=True)
                rec: dict[str, Any] = {"t": round(time.time(), 2)}
                if isinstance(r_res, httpx.Response) and r_res.status_code == 200:
                    tiers = (r_res.json() or {}).get("tiers") or {}
                    rec["dot14_inflight"] = (tiers.get("dot14") or {}).get("inflight", 0)
                    rec["dot12_inflight"] = (tiers.get("dot12") or {}).get("inflight", 0)
                if isinstance(p14_res, httpx.Response) and p14_res.status_code == 200:
                    m14 = parse_prometheus_metrics(p14_res.text)
                    rec["dot14_kv"] = m14.get("token_usage", 0.0)
                    rec["dot14_hicache_tok"] = m14.get("hicache_host_used_tokens", 0.0)
                    rec["dot14_q"] = m14.get("num_queue_reqs", 0.0)
                if isinstance(p12_res, httpx.Response) and p12_res.status_code == 200:
                    m12 = parse_prometheus_metrics(p12_res.text)
                    rec["dot12_kv"] = m12.get("token_usage", 0.0)
                    rec["dot12_hicache_tok"] = m12.get("hicache_host_used_tokens", 0.0)
                    rec["dot12_q"] = m12.get("num_queue_reqs", 0.0)
                self.samples.append(rec)
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=max(0.1, 1.0 - elapsed))
            except asyncio.TimeoutError:
                pass

    def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._task:
            await self._task
        if not self.samples:
            return {"count": 0}
        def peak(k):
            vals = [s[k] for s in self.samples if k in s]
            return round(max(vals), 4) if vals else None
        def avg(k):
            vals = [s[k] for s in self.samples if k in s]
            return round(sum(vals) / len(vals), 4) if vals else None
        return {
            "count": len(self.samples),
            "peak_dot14_inflight": peak("dot14_inflight"),
            "peak_dot12_inflight": peak("dot12_inflight"),
            "peak_dot14_kv": peak("dot14_kv"),
            "peak_dot12_kv": peak("dot12_kv"),
            "avg_dot14_kv": avg("dot14_kv"),
            "avg_dot12_kv": avg("dot12_kv"),
            "peak_dot14_hicache_tok": peak("dot14_hicache_tok"),
            "peak_dot12_hicache_tok": peak("dot12_hicache_tok"),
            "peak_dot14_queue": peak("dot14_q"),
            "peak_dot12_queue": peak("dot12_q"),
        }


async def run_scenario(
    *,
    name: str,
    sessions: list[dict],
    turn_slice: tuple[int, int],
    nonce_mode: str,  # "scenario" | "session"
    nonce_base: str,
    initial_delays_s: list[float],
    turn_gap_scale: float,
    turn_gap_cap_s: float,
    max_tokens_bounds: tuple[int, int],
    send_session_header: bool,
    reversed_t2_barrier: bool,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    relay_url: str,
    dot14_url: str,
    dot12_url: str,
    solo_jct_ref: dict[str, float] | None = None,
    serial: bool = False,
) -> dict[str, Any]:
    before = await snapshot_cluster(client, relay_url, dot14_url, dot12_url)
    sampler = BackgroundSampler(client, relay_url, dot14_url, dot12_url)
    sampler.start()

    t_start_all = time.perf_counter()
    start_turn, end_turn = turn_slice
    min_mt, max_mt = max_tokens_bounds

    # Optional barrier for reversed_t2 ordering test (all finish T1, then latter half launches T2 0.25s BEFORE first half!)
    t1_done_count = 0
    t1_all_done = asyncio.Event()
    total_sessions = len(sessions)

    async def run_one_session(idx: int, sess: dict) -> dict[str, Any]:
        nonlocal t1_done_count
        delay = initial_delays_s[idx] if idx < len(initial_delays_s) else 0.0
        if delay > 0:
            await asyncio.sleep(delay)

        sid = sess["session_id"]
        nonce = f"{nonce_base}-s{idx:02d}-{sid}" if nonce_mode == "session" else nonce_base
        hdr_sid = f"eval-{nonce_base}-s{idx:02d}-{sid}" if send_session_header else None

        sess_t0 = time.perf_counter()
        turns_out = []
        selected_turns = sess["turns"][start_turn:end_turn]
        for local_ti, tdata in enumerate(selected_turns):
            if local_ti > 0:
                if reversed_t2_barrier and local_ti == 1:
                    await t1_all_done.wait()
                    # Sessions in the second half (which spilled to dot12!) go FIRST; first 6 go 0.25s LATER
                    if idx < 6:
                        await asyncio.sleep(0.25)
                elif turn_gap_scale > 0:
                    raw_gap = float(tdata.get("inter_turn_gap_s") or 2.0)
                    wait_s = min(turn_gap_cap_s, max(0.15, raw_gap * turn_gap_scale))
                    await asyncio.sleep(wait_s)

            rec_ct = int(tdata.get("recorded_completion_tokens") or 160)
            mt = max(min_mt, min(max_mt, rec_ct))
            req_body = inject_prefix_nonce(tdata["body"], nonce)
            res = await send_streaming_turn(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                body=req_body,
                max_tokens=mt,
                session_id=hdr_sid,
            )
            res["turn_index"] = start_turn + local_ti
            res["local_turn"] = local_ti
            res["recorded_pt"] = tdata.get("recorded_prompt_tokens", 0)
            res["target_max_tokens"] = mt
            turns_out.append(res)

            if reversed_t2_barrier and local_ti == 0:
                t1_done_count += 1
                if t1_done_count >= total_sessions:
                    t1_all_done.set()

            if not res["ok"]:
                # Still unblock barrier if a turn failed
                if reversed_t2_barrier and local_ti == 0 and not t1_all_done.is_set():
                    t1_all_done.set()

        sess_jct = time.perf_counter() - sess_t0
        # Compute model-only JCT (excluding simulated tool waits) and per-session inter-emission wait
        model_jct = sum(t["total_s"] for t in turns_out)
        warm_turn_waits = [t["ttft_s"] for t in turns_out[1:] if t["ok"] and t["ttft_s"] is not None]
        return {
            "session_id": sid,
            "task_id": sess["task_id"],
            "bucket": sess["bucket"],
            "mutating": sess["mutating"],
            "sys_versions": sess["sys_versions"],
            "jct_wall_s": round(sess_jct, 4),
            "jct_model_s": round(model_jct, 4),
            "inter_emission_wait_mean_s": round(sum(warm_turn_waits) / len(warm_turn_waits), 4) if warm_turn_waits else None,
            "turns": turns_out,
        }

    if serial:
        sess_results = []
        for idx, sess in enumerate(sessions):
            sess_results.append(await run_one_session(idx, sess))
    else:
        sess_results = list(await asyncio.gather(*(run_one_session(i, s) for i, s in enumerate(sessions))))

    wall_s = time.perf_counter() - t_start_all
    telemetry = await sampler.stop()
    after = await snapshot_cluster(client, relay_url, dot14_url, dot12_url)
    cluster_diff = diff_snapshots(before, after)

    # Aggregate client-side metrics
    all_turns = [t for s in sess_results for t in s["turns"]]
    ok_turns = [t for t in all_turns if t["ok"]]
    err_turns = [t for t in all_turns if not t["ok"]]
    cold_turns = [t for t in ok_turns if t["turn_index"] == 0]
    warm_turns = [t for t in ok_turns if t["turn_index"] > 0]
    # Inter-emission waits = TTFT of turns after the first local turn in this scenario (strictly excluding tool-wait sleep!)
    inter_emission_turns = [t for t in ok_turns if t["local_turn"] > 0 and t["ttft_s"] is not None]
    inter_waits = [t["ttft_s"] for t in inter_emission_turns]
    inter_waits_hit90 = [t["ttft_s"] for t in inter_emission_turns if t["hit_rate_pct"] >= 90.0]
    inter_waits_miss = [t["ttft_s"] for t in inter_emission_turns if t["hit_rate_pct"] < 90.0]

    ttfts = [t["ttft_s"] for t in ok_turns if t["ttft_s"] is not None]
    ttfts_vis = [t["ttft_visible_s"] for t in ok_turns if t["ttft_visible_s"] is not None]
    cold_ttfts = [t["ttft_s"] for t in cold_turns if t["ttft_s"] is not None]
    warm_ttfts = [t["ttft_s"] for t in warm_turns if t["ttft_s"] is not None]
    tpots = [t["tpot_ms"] for t in ok_turns if t["tpot_ms"] is not None]
    dec_speeds = [t["decode_tok_s"] for t in ok_turns if t["decode_tok_s"] is not None]
    max_gaps = [t["max_gap_s"] for t in ok_turns]
    tbt_means = [t["tbt_mean_ms"] for t in ok_turns if t["tbt_mean_ms"] is not None]
    tbt_p99s = [t["tbt_p99_ms"] for t in ok_turns if t["tbt_p99_ms"] is not None]

    # Per-bucket (S/M/L/XL) user experience breakdown
    by_bucket_ux: dict[str, Any] = {}
    for b in ("S", "M", "L", "XL"):
        bsess = [s for s in sess_results if s["bucket"] == b]
        if not bsess:
            continue
        bturns = [t for s in bsess for t in s["turns"] if t["ok"]]
        b_cold = [t["ttft_s"] for t in bturns if t["turn_index"] == 0 and t["ttft_s"] is not None]
        b_inter = [t["ttft_s"] for t in bturns if t["local_turn"] > 0 and t["ttft_s"] is not None]
        b_spd = [t["decode_tok_s"] for t in bturns if t["decode_tok_s"] is not None]
        b_wpt = sum(t["prompt_tokens"] for t in bturns if t["turn_index"] > 0)
        b_wct = sum(t["cached_tokens"] for t in bturns if t["turn_index"] > 0)
        by_bucket_ux[b] = {
            "sessions": len(bsess),
            "cold_ttft_mean_s": round(sum(b_cold) / len(b_cold), 3) if b_cold else None,
            "inter_emission_wait_mean_s": round(sum(b_inter) / len(b_inter), 3) if b_inter else None,
            "inter_emission_wait_p95_s": nearest_rank(b_inter, 0.95),
            "decode_tok_s_p50": nearest_rank(b_spd, 0.50),
            "warm_hit_rate_pct": round(100.0 * b_wct / b_wpt, 2) if b_wpt else None,
        }

    def dist_summary(vals: list[float]) -> dict[str, float | int | None]:
        if not vals:
            return {"n": 0, "mean": None, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
        return {
            "n": len(vals),
            "mean": round(sum(vals) / len(vals), 4),
            "min": round(min(vals), 4),
            "p50": nearest_rank(vals, 0.50),
            "p90": nearest_rank(vals, 0.90),
            "p95": nearest_rank(vals, 0.95),
            "p99": nearest_rank(vals, 0.99),
            "max": round(max(vals), 4),
        }

    jcts = [s["jct_model_s"] for s in sess_results]
    slowdowns = []
    q_waits = []
    if solo_jct_ref:
        for s in sess_results:
            ref = solo_jct_ref.get(s["session_id"])
            if ref and ref > 0:
                # Scale solo ref proportionally if turn count differs (P0 runs 6 turns)
                scale = len(s["turns"]) / 6.0
                ref_scaled = ref * scale
                slowdowns.append(s["jct_model_s"] / ref_scaled)
                q_waits.append(max(0.0, s["jct_model_s"] - ref_scaled))

    def hit_stats(tlist: list[dict]) -> dict[str, Any]:
        pt = sum(t["prompt_tokens"] for t in tlist)
        ct = sum(t["cached_tokens"] for t in tlist)
        return {
            "count": len(tlist),
            "prompt_tokens": pt,
            "cached_tokens": ct,
            "uncached_tokens": max(0, pt - ct),
            "weighted_hit_rate_pct": round(100.0 * ct / pt, 2) if pt else 0.0,
            "ge90_count": sum(1 for t in tlist if t["hit_rate_pct"] >= 90.0),
            "zero_hit_count": sum(1 for t in tlist if t["cached_tokens"] == 0),
        }

    total_pt = sum(t["prompt_tokens"] for t in ok_turns)
    total_ct = sum(t["cached_tokens"] for t in ok_turns)
    total_comp = sum(t["completion_tokens"] for t in ok_turns)
    c1_over_120s = sum(1 for x in ttfts if x > 120.0)
    c2_over_2s = sum(1 for g in max_gaps if g > 2.0)
    stalls_05 = sum(t.get("stall_gt_0_5s", 0) for t in ok_turns)
    stalls_10 = sum(t.get("stall_gt_1_0s", 0) for t in ok_turns)
    stalls_20 = sum(t.get("stall_gt_2_0s", 0) for t in ok_turns)

    return {
        "scenario": name,
        "started_at_utc": utc_now_iso(),
        "concurrency": 1 if serial else len(sessions),
        "sessions_count": len(sessions),
        "turns_per_session": end_turn - start_turn,
        "nonce_mode": nonce_mode,
        "send_session_header": send_session_header,
        "total_calls": len(all_turns),
        "ok_calls": len(ok_turns),
        "error_calls": len(err_turns),
        "error_rate_pct": round(100.0 * len(err_turns) / max(1, len(all_turns)), 2),
        "sample_errors": [t["error"] for t in err_turns[:5]],
        "makespan_wall_s": round(wall_s, 3),
        "c1_ttft_over_120s": c1_over_120s,
        "c2_max_gap_over_2s": c2_over_2s,
        "c2_max_gap_s": round(max(max_gaps), 4) if max_gaps else 0.0,
        "stalls_within_turn": {"gt_0_5s": stalls_05, "gt_1_0s": stalls_10, "gt_2_0s": stalls_20},
        "tbt_mean_avg_ms": round(sum(tbt_means) / len(tbt_means), 2) if tbt_means else None,
        "tbt_p99_max_ms": round(max(tbt_p99s), 2) if tbt_p99s else None,
        "tpot_ms": dist_summary(tpots),
        "per_stream_decode_tok_s": dist_summary(dec_speeds),
        "effective_tok_s_excl_tool": compute_effective_tok_s_excl_tool(sess_results),
        "inter_emission_wait_s": {
            "all": dist_summary(inter_waits),
            "cache_hit_ge90": dist_summary(inter_waits_hit90),
            "cache_miss_lt90": dist_summary(inter_waits_miss),
        },
        "ttft_all_s": dist_summary(ttfts),
        "ttft_visible_s": dist_summary(ttfts_vis),
        "ttft_cold_s": dist_summary(cold_ttfts),
        "ttft_warm_s": dist_summary(warm_ttfts),
        "by_bucket_ux": by_bucket_ux,
        "jct_model_s": dist_summary(jcts),
        "slowdown_vs_solo": {
            "n_matched": len(slowdowns),
            "p50": nearest_rank(slowdowns, 0.50),
            "p95": nearest_rank(slowdowns, 0.95),
            "max": round(max(slowdowns), 4) if slowdowns else None,
            "jain_fairness": jain_index(slowdowns),
            "queue_q_p95_s": nearest_rank(q_waits, 0.95),
        },
        "cache_all": hit_stats(ok_turns),
        "cache_cold_t1": hit_stats(cold_turns),
        "cache_warm_t2plus": hit_stats(warm_turns),
        "throughput": {
            "prompt_tok_s": round(total_pt / wall_s, 1) if wall_s > 0 else 0.0,
            "uncached_prefill_tok_s": round(max(0, total_pt - total_ct) / wall_s, 1) if wall_s > 0 else 0.0,
            "output_tok_s": round(total_comp / wall_s, 2) if wall_s > 0 else 0.0,
            "total_output_tokens": total_comp,
        },
        "cluster_diff": cluster_diff,
        "telemetry_1hz": telemetry,
        "sessions": sess_results,
    }


def compute_effective_tok_s_excl_tool(sess_results: list[dict]) -> dict[str, Any]:
    """Compute effective output token rate strictly excluding tool wait: completion_tokens / total_s (TTFT + decode)."""
    all_t = [t for s in sess_results for t in (s.get("turns") or []) if t.get("ok") and (t.get("total_s") or 0) > 0]
    warm_t = [t for t in all_t if t.get("local_turn", t.get("turn_index", 0)) > 0]
    all_tok = sum(int(t.get("completion_tokens") or 0) for t in all_t)
    all_s = sum(float(t.get("total_s") or 0.0) for t in all_t)
    warm_tok = sum(int(t.get("completion_tokens") or 0) for t in warm_t)
    warm_s = sum(float(t.get("total_s") or 0.0) for t in warm_t)
    sess_all: list[float] = []
    sess_warm: list[float] = []
    for s in sess_results:
        ts = [t for t in (s.get("turns") or []) if t.get("ok") and (t.get("total_s") or 0) > 0]
        if ts:
            ss = sum(float(t.get("total_s") or 0.0) for t in ts)
            if ss > 0:
                sess_all.append(sum(int(t.get("completion_tokens") or 0) for t in ts) / ss)
        wt = [t for t in ts if t.get("local_turn", t.get("turn_index", 0)) > 0]
        if wt:
            ws = sum(float(t.get("total_s") or 0.0) for t in wt)
            if ws > 0:
                sess_warm.append(sum(int(t.get("completion_tokens") or 0) for t in wt) / ws)

    def _s(pooled_tok: int, pooled_s: float, rates: list[float]) -> dict[str, Any]:
        return {
            "pooled_tok_s": round(pooled_tok / pooled_s, 2) if pooled_s > 0 else None,
            "per_session_mean": round(sum(rates) / len(rates), 2) if rates else None,
            "per_session_p50": round(nearest_rank(rates, 0.50), 2) if rates else None,
            "per_session_p10": round(nearest_rank(rates, 0.10), 2) if rates else None,
            "per_session_min": round(min(rates), 2) if rates else None,
            "total_completion_tokens": pooled_tok,
            "total_model_time_s": round(pooled_s, 3),
        }

    return {
        "all_turns": _s(all_tok, all_s, sess_all),
        "warm_t2plus": _s(warm_tok, warm_s, sess_warm),
    }


def evaluate_ux_tier(r: dict) -> tuple[str, list[str]]:
    """Classify each concurrency run into Green(丝滑) / Yellow(稳定可用) / Orange(轻度退化) / Red(超载)."""
    reasons = []
    if r["error_calls"] > 0:
        reasons.append(f"报错 {r['error_calls']} 次")
    if r["c1_ttft_over_120s"] > 0:
        reasons.append(f"C1违例({r['c1_ttft_over_120s']}笔>120s)")
    if r["c2_max_gap_over_2s"] > 0:
        reasons.append(f"C2卡顿({r['c2_max_gap_over_2s']}笔>2s, max={r['c2_max_gap_s']:.2f}s)")

    iw = (r.get("inter_emission_wait_s") or {}).get("all") or {}
    iw_mean = iw.get("mean")
    iw_p95 = iw.get("p95")
    dec_p50 = (r.get("per_stream_decode_tok_s") or {}).get("p50")
    warm_hit = (r.get("cache_warm_t2plus") or {}).get("weighted_hit_rate_pct", 0.0)
    sd_p95 = (r.get("slowdown_vs_solo") or {}).get("p95")

    if r["error_calls"] > 0 or r["c1_ttft_over_120s"] > 0 or (iw_mean is not None and iw_mean > 8.0):
        return "🔴 超载", reasons or [f"续轮等待均值 {iw_mean:.2f}s > 8s"]
    if (
        r["c2_max_gap_over_2s"] == 0
        and (iw_mean is None or iw_mean <= 2.0)
        and (iw_p95 is None or iw_p95 <= 4.0)
        and (dec_p50 is None or dec_p50 >= 40.0)
        and warm_hit >= 90.0
        and (sd_p95 is None or sd_p95 <= 2.0)
    ):
        return "🟢 极致丝滑", ["全部体感指标达标"]
    if (
        r["c2_max_gap_over_2s"] == 0
        and (iw_mean is None or iw_mean <= 4.0)
        and (iw_p95 is None or iw_p95 <= 8.0)
        and (dec_p50 is None or dec_p50 >= 25.0)
        and (sd_p95 is None or sd_p95 <= 3.0)
    ):
        notes = []
        if iw_mean and iw_mean > 2.0:
            notes.append(f"吐字间隔均值 {iw_mean:.2f}s")
        if dec_p50 and dec_p50 < 40.0:
            notes.append(f"单流速度 {dec_p50:.1f} tok/s")
        if warm_hit < 90.0:
            notes.append(f"热命中 {warm_hit:.1f}%")
        return "🟡 稳定可用", notes or ["符合稳定体感线"]
    if iw_mean and iw_mean > 4.0:
        reasons.append(f"吐字间隔均值 {iw_mean:.2f}s")
    if iw_p95 and iw_p95 > 8.0:
        reasons.append(f"吐字间隔P95 {iw_p95:.2f}s")
    if dec_p50 and dec_p50 < 25.0:
        reasons.append(f"单流速度仅 {dec_p50:.1f} tok/s")
    if sd_p95 and sd_p95 > 3.0:
        reasons.append(f"慢化P95 {sd_p95:.2f}x")
    if warm_hit < 85.0:
        reasons.append(f"热命中降至 {warm_hit:.1f}%")
    return "🟠 体感退化", reasons or ["部分指标偏离稳定线"]


def render_markdown_report(run_meta: dict, results: list[dict]) -> str:
    lines = [
        "# STRATA v14 双机负载均衡全量测评报告（`20261120-unified`）",
        "",
        f"- **运行 ID**：`{run_meta['run_id']}`",
        f"- **开始时间（CST）**：`{run_meta['started_cst']}` | **完成/更新时间（CST）**：`{cst_now_iso()}`",
        f"- **测试入口**：`{run_meta['base_url']}`（`model={run_meta['model']}`）",
        f"- **中止状态**：`{run_meta.get('aborted_reason') or '正常完成'}`",
        "",
        "## 一、双机（.14 + .12）承载并发人数与体感稳定性判定（P0 + P1 并发梯度）",
        "",
        "> **口径说明**：**“两次吐字间等待（不含工具时间）”** 即同一会话上一轮 Agent 吐完最后一个字、扣除外部工具耗时后，到下一轮吐出首字的纯系统等待时间（含路由+排队+增量 Prefill/缓存恢复）。**“除去工具等待有效吐字速率（全轮 / 热T2+）”** 为 `completion_tokens / 模型侧总耗时(含排队+Prefill+Decode，严格不含工具耗时)`，直接反映用户真实端到端吐字体验。**单流吐字速度**为去掉 TTFT 后的纯打字机解码速率（tok/s）。",
        "",
        "| 并发 `C` | 体感评级 | 有效吐字(不含工具) 全轮/热T2+ (tok/s) | 两次吐字间等待 Mean / P50 / P95 (s) | 命中 vs 未命中等待 Mean (s) | 首轮冷等待 P50 / P95 (s) | 单流纯解码 P50 / P10 (tok/s) | TPOT P50 (ms) | 最大停顿 `C2` (s) | 慢化比 P50 / P95 (Jain) | 热缓存命中率 | 路由 `.14`/`.12` (sticky/prio/router) | 判定说明 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    scaling_runs = [
        r for r in results
        if r["scenario"] == "p0_solo_baseline" or r["scenario"].startswith("p1a_scaling_") or r["scenario"].startswith("p1b_repeat_")
    ]
    max_green_c = 0
    max_stable_c = 0
    for r in scaling_runs:
        tier_label, notes = evaluate_ux_tier(r)
        if r["scenario"].startswith("p1a_scaling_") or r["scenario"] == "p0_solo_baseline":
            if "🟢" in tier_label:
                max_green_c = max(max_green_c, r["concurrency"])
                max_stable_c = max(max_stable_c, r["concurrency"])
            elif "🟡" in tier_label:
                max_stable_c = max(max_stable_c, r["concurrency"])
        iw_all = (r.get("inter_emission_wait_s") or {}).get("all") or {}
        iw_hit = (r.get("inter_emission_wait_s") or {}).get("cache_hit_ge90") or {}
        iw_miss = (r.get("inter_emission_wait_s") or {}).get("cache_miss_lt90") or {}
        cold = r.get("ttft_cold_s") or {}
        spd = r.get("per_stream_decode_tok_s") or {}
        tpot = r.get("tpot_ms") or {}
        sd = r.get("slowdown_vs_solo") or {}
        wc = r.get("cache_warm_t2plus") or {}
        cd = r.get("cluster_diff") or {}
        rd = cd.get("relay_delta") or {}
        def fmt(v, nd=2):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"
        p10_spd = nearest_rank(
            [t["decode_tok_s"] for s in r.get("sessions", []) for t in s.get("turns", []) if t.get("ok") and t.get("decode_tok_s") is not None],
            0.10,
        )
        eff = r.get("effective_tok_s_excl_tool") or compute_effective_tok_s_excl_tool(r.get("sessions") or [])
        eff_all = (eff.get("all_turns") or {}).get("pooled_tok_s")
        eff_warm = (eff.get("warm_t2plus") or {}).get("pooled_tok_s")
        tag = f"**{r['concurrency']}**" if not r["scenario"].startswith("p1b_") else f"{r['concurrency']}(副集)"
        lines.append(
            f"| {tag} | {tier_label} | {fmt(eff_all, 1)} / **{fmt(eff_warm, 1)}** "
            f"| **{fmt(iw_all.get('mean'))}** / {fmt(iw_all.get('p50'))} / {fmt(iw_all.get('p95'))} "
            f"| {fmt(iw_hit.get('mean'))} vs {fmt(iw_miss.get('mean'))} "
            f"| {fmt(cold.get('p50'))} / {fmt(cold.get('p95'))} "
            f"| {fmt(spd.get('p50'), 1)} / {fmt(p10_spd, 1)} | {fmt(tpot.get('p50'), 1)} "
            f"| {fmt(r.get('c2_max_gap_s'))} ({r.get('c2_max_gap_over_2s', 0)}违) "
            f"| {fmt(sd.get('p50'))}x / {fmt(sd.get('p95'))}x ({fmt(sd.get('jain_fairness'), 3)}) "
            f"| **{fmt(wc.get('weighted_hit_rate_pct'), 2)}%** ({wc.get('zero_hit_count', 0)}零) "
            f"| {cd.get('dot14_req_delta',0)}/{cd.get('dot12_req_delta',0)} ({rd.get('sticky_routed_total',0)}/{rd.get('dot14_priority_routed',0)+rd.get('dot12_priority_routed',0)}/{rd.get('router_spillover_routed',0)}) "
            f"| {'；'.join(notes)} |"
        )

    lines.extend([
        "",
        f"- **自动容量结论（基于主分层集 P0+P1A）**：双机（`.14 + .12`）实测 **极致丝滑最大并发** $C_{{\\text{{smooth}}}}^* = {max_green_c}$，**稳定体感最大并发** $C_{{\\text{{stable}}}}^* = {max_stable_c}$。",
        f"- **换算在线同时使用人数（参考）**：`20261120-unified` 真实工具等待中位数 `9.0s`、单轮模型热执行约 `3–5s`（模型占空比约 `30%–40%`），上述 **无工具等待背靠背瞬时并发 {max_stable_c} 路** 折合支持约 **`{int(round(max_stable_c * 2.5))}–{int(round(max_stable_c * 3.2))}` 名活跃 Agent 用户同时在线使用**并保持体感稳定（亦可对照下方 `P5` 的 `short_gap / medium_gap` 短工具间隙实测数据）。",
        "",
        "## 二、按上下文长度分桶（S / M / L / XL）的关键并发体感拆解",
        "",
        "| 阶段 | 档位 S (<40K) 等待/速度/命中 | 档位 M (40-100K) 等待/速度/命中 | 档位 L (100-180K) 等待/速度/命中 | 档位 XL (180-270K) 等待/速度/命中 |",
        "|---|---|---|---|---|",
    ])
    for r in [x for x in results if x["scenario"] in ("p0_solo_baseline", "p1a_scaling_c06", "p1a_scaling_c08", "p1a_scaling_c12", "p1a_scaling_c16", "p1a_scaling_c24", "p5_shape_c24_medium_gap")]:
        bux = r.get("by_bucket_ux") or {}
        cells = []
        for b in ("S", "M", "L", "XL"):
            bd = bux.get(b)
            if not bd:
                cells.append("-")
            else:
                iw = f"{bd['inter_emission_wait_mean_s']:.2f}s" if bd.get("inter_emission_wait_mean_s") is not None else "-"
                sp = f"{bd['decode_tok_s_p50']:.0f}t/s" if bd.get("decode_tok_s_p50") is not None else "-"
                ht = f"{bd['warm_hit_rate_pct']:.1f}%" if bd.get("warm_hit_rate_pct") is not None else "-"
                cells.append(f"{iw} / {sp} / {ht}")
        lines.append(f"| `{r['scenario']}` | {' | '.join(cells)} |")

    lines.extend([
        "",
        "## 三、全部 10 阶段完整指标总表",
        "",
        "| 阶段 | 并发 `C` | 成功/总计 | 排空 `M1`(s) | 冷TTFT P50/Max(s) | 续轮吐字间隔 Mean/P50/P95(s) | 单流吐字 P50(tok/s) | 热缓存命中率 | `C1`(>120s)/`C2`(>2s) | 聚合输出(tok/s) | 路由 `.14`/`.12` (st/pr/rt) | 峰值KV `.14`/`.12` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for r in results:
        cd = r.get("cluster_diff") or {}
        rd = cd.get("relay_delta") or {}
        t1 = r.get("telemetry_1hz") or {}
        c_ttft = r.get("ttft_cold_s") or {}
        iw = (r.get("inter_emission_wait_s") or {}).get("all") or {}
        spd = r.get("per_stream_decode_tok_s") or {}
        w_cache = r.get("cache_warm_t2plus") or {}
        c_hit_str = f"{w_cache.get('weighted_hit_rate_pct', 0):.2f}% ({w_cache.get('ge90_count',0)}/{w_cache.get('count',0)} ≥90%, {w_cache.get('zero_hit_count',0)} 零)" if w_cache.get("count") else "N/A"
        st_cnt = rd.get("sticky_routed_total", 0)
        pr_cnt = rd.get("dot14_priority_routed", 0) + rd.get("dot12_priority_routed", 0)
        rt_cnt = rd.get("router_spillover_routed", 0)
        w14 = cd.get("dot14_req_delta", 0)
        w12 = cd.get("dot12_req_delta", 0)
        kv14 = t1.get("peak_dot14_kv")
        kv12 = t1.get("peak_dot12_kv")
        kv_str = f"{kv14:.2f}/{kv12:.2f}" if (kv14 is not None and kv12 is not None) else "N/A"
        def f2(v, nd=2):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"
        lines.append(
            f"| `{r['scenario']}` | {r['concurrency']} | {r['ok_calls']}/{r['total_calls']} | {r['makespan_wall_s']:.1f} "
            f"| {f2(c_ttft.get('p50'))}/{f2(c_ttft.get('max'))} "
            f"| {f2(iw.get('mean'))}/{f2(iw.get('p50'))}/{f2(iw.get('p95'))} "
            f"| {f2(spd.get('p50'), 1)} "
            f"| **{c_hit_str}** | {r['c1_ttft_over_120s']} / {r['c2_max_gap_over_2s']} (max {r['c2_max_gap_s']:.2f}s) "
            f"| {r['throughput']['output_tok_s']:.1f} | {w14}/{w12} ({st_cnt}/{pr_cnt}/{rt_cnt}) | {kv_str} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


async def run_rolling_soak(
    *,
    name: str,
    pool: list[dict],
    concurrency: int,
    duration_s: float,
    hard_stop_epoch: float,
    turns_per_user: int,
    nonce_prefix: str,
    max_tokens_bounds: tuple[int, int],
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    relay_url: str,
    dot14_url: str,
    dot12_url: str,
) -> dict[str, Any]:
    """Open-loop rolling steady-state soak: maintains constant C concurrent multi-turn users with continuous cold churn."""
    before = await snapshot_cluster(client, relay_url, dot14_url, dot12_url)
    sampler = BackgroundSampler(client, relay_url, dot14_url, dot12_url)
    sampler.start()

    t_start_wall = time.perf_counter()
    stop_at = min(time.time() + duration_s, hard_stop_epoch)
    min_mt, max_mt = max_tokens_bounds
    completed_sessions: list[dict] = []
    cycle_counter = 0

    async def worker_loop(wid: int):
        nonlocal cycle_counter
        # Stagger initial arrivals gently over 4s so soak starts smoothly
        await asyncio.sleep((wid / max(1, concurrency)) * 4.0)
        local_cycle = 0
        while time.time() < stop_at:
            sess = pool[(wid + local_cycle * concurrency) % len(pool)]
            local_cycle += 1
            cycle_counter += 1
            nonce = f"{nonce_prefix}-w{wid}-c{local_cycle}"
            hdr_sid = f"eval-{nonce}-{sess['session_id']}"
            turns_out = []
            sess_t0 = time.perf_counter()
            for ti, tdata in enumerate(sess["turns"][:turns_per_user]):
                if time.time() >= stop_at:
                    break
                if ti > 0:
                    raw_gap = float(tdata.get("inter_turn_gap_s") or 2.0)
                    wait_s = min(1.0, max(0.05, raw_gap * 0.05))
                    await asyncio.sleep(wait_s)
                    if time.time() >= stop_at:
                        break
                rec_ct = int(tdata.get("recorded_completion_tokens") or 180)
                mt = max(min_mt, min(max_mt, rec_ct))
                req_body = inject_prefix_nonce(tdata["body"], nonce)
                res = await send_streaming_turn(
                    client=client, base_url=base_url, api_key=api_key, model=model,
                    body=req_body, max_tokens=mt, session_id=hdr_sid,
                )
                res["turn_index"] = ti
                res["local_turn"] = ti
                res["completed_at_offset_s"] = round(time.perf_counter() - t_start_wall, 1)
                turns_out.append(res)
                if not res["ok"]:
                    break
            if turns_out:
                warm_w = [t["ttft_s"] for t in turns_out[1:] if t["ok"] and t["ttft_s"] is not None]
                completed_sessions.append({
                    "session_id": f"{sess['session_id']}#{local_cycle}",
                    "task_id": sess["task_id"],
                    "bucket": sess["bucket"],
                    "mutating": sess["mutating"],
                    "sys_versions": sess["sys_versions"],
                    "jct_wall_s": round(time.perf_counter() - sess_t0, 4),
                    "jct_model_s": round(sum(t["total_s"] for t in turns_out), 4),
                    "inter_emission_wait_mean_s": round(sum(warm_w) / len(warm_w), 4) if warm_w else None,
                    "turns": turns_out,
                })

    await asyncio.gather(*(worker_loop(w) for w in range(concurrency)))
    wall_s = time.perf_counter() - t_start_wall
    telemetry = await sampler.stop()
    after = await snapshot_cluster(client, relay_url, dot14_url, dot12_url)
    cluster_diff = diff_snapshots(before, after)

    all_turns = [t for s in completed_sessions for t in s["turns"]]
    ok_turns = [t for t in all_turns if t["ok"]]
    err_turns = [t for t in all_turns if not t["ok"]]
    cold_turns = [t for t in ok_turns if t["turn_index"] == 0]
    warm_turns = [t for t in ok_turns if t["turn_index"] > 0]
    inter_waits = [t["ttft_s"] for t in warm_turns if t["ttft_s"] is not None]
    inter_hit = [t["ttft_s"] for t in warm_turns if t["ttft_s"] is not None and t["hit_rate_pct"] >= 90.0]
    inter_miss = [t["ttft_s"] for t in warm_turns if t["ttft_s"] is not None and t["hit_rate_pct"] < 90.0]

    def dist_s(vals):
        if not vals:
            return {"n": 0, "mean": None, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
        return {"n": len(vals), "mean": round(sum(vals)/len(vals), 4), "min": round(min(vals), 4), "p50": nearest_rank(vals, 0.50), "p90": nearest_rank(vals, 0.90), "p95": nearest_rank(vals, 0.95), "p99": nearest_rank(vals, 0.99), "max": round(max(vals), 4)}

    def hit_s(tl):
        pt = sum(t["prompt_tokens"] for t in tl)
        ct = sum(t["cached_tokens"] for t in tl)
        return {"count": len(tl), "prompt_tokens": pt, "cached_tokens": ct, "uncached_tokens": max(0, pt-ct), "weighted_hit_rate_pct": round(100.0*ct/pt, 2) if pt else 0.0, "ge90_count": sum(1 for t in tl if t["hit_rate_pct"] >= 90.0), "zero_hit_count": sum(1 for t in tl if t["cached_tokens"] == 0)}

    # 5-minute time-window trend buckets
    trend_5min = []
    num_win = max(1, int(math.ceil(wall_s / 300.0)))
    for wi in range(num_win):
        lo, hi = wi * 300.0, (wi + 1) * 300.0
        wt = [t for t in warm_turns if lo <= t.get("completed_at_offset_s", 0.0) < hi]
        if wt:
            w_iw = [t["ttft_s"] for t in wt if t["ttft_s"] is not None]
            w_sp = [t["decode_tok_s"] for t in wt if t["decode_tok_s"] is not None]
            w_pt = sum(t["prompt_tokens"] for t in wt)
            w_ct = sum(t["cached_tokens"] for t in wt)
            trend_5min.append({
                "window_min": f"{wi*5}-{(wi+1)*5}",
                "warm_calls": len(wt),
                "inter_wait_mean_s": round(sum(w_iw)/len(w_iw), 3) if w_iw else None,
                "decode_tok_s_p50": nearest_rank(w_sp, 0.50),
                "warm_hit_pct": round(100.0*w_ct/w_pt, 2) if w_pt else None,
            })

    ttfts = [t["ttft_s"] for t in ok_turns if t["ttft_s"] is not None]
    max_gaps = [t["max_gap_s"] for t in ok_turns]
    total_pt = sum(t["prompt_tokens"] for t in ok_turns)
    total_ct = sum(t["cached_tokens"] for t in ok_turns)
    total_comp = sum(t["completion_tokens"] for t in ok_turns)

    return {
        "scenario": name,
        "started_at_utc": utc_now_iso(),
        "concurrency": concurrency,
        "sessions_count": len(completed_sessions),
        "turns_per_session": turns_per_user,
        "nonce_mode": "rolling_fresh_user",
        "send_session_header": True,
        "total_calls": len(all_turns),
        "ok_calls": len(ok_turns),
        "error_calls": len(err_turns),
        "error_rate_pct": round(100.0 * len(err_turns) / max(1, len(all_turns)), 2),
        "sample_errors": [t["error"] for t in err_turns[:5]],
        "makespan_wall_s": round(wall_s, 3),
        "c1_ttft_over_120s": sum(1 for x in ttfts if x > 120.0),
        "c2_max_gap_over_2s": sum(1 for g in max_gaps if g > 2.0),
        "c2_max_gap_s": round(max(max_gaps), 4) if max_gaps else 0.0,
        "tpot_ms": dist_s([t["tpot_ms"] for t in ok_turns if t["tpot_ms"] is not None]),
        "per_stream_decode_tok_s": dist_s([t["decode_tok_s"] for t in ok_turns if t["decode_tok_s"] is not None]),
        "effective_tok_s_excl_tool": compute_effective_tok_s_excl_tool(completed_sessions),
        "inter_emission_wait_s": {"all": dist_s(inter_waits), "cache_hit_ge90": dist_s(inter_hit), "cache_miss_lt90": dist_s(inter_miss)},
        "ttft_all_s": dist_s(ttfts),
        "ttft_cold_s": dist_s([t["ttft_s"] for t in cold_turns if t["ttft_s"] is not None]),
        "ttft_warm_s": dist_s(inter_waits),
        "trend_5min": trend_5min,
        "cache_all": hit_s(ok_turns),
        "cache_cold_t1": hit_s(cold_turns),
        "cache_warm_t2plus": hit_s(warm_turns),
        "throughput": {
            "prompt_tok_s": round(total_pt / wall_s, 1) if wall_s > 0 else 0.0,
            "uncached_prefill_tok_s": round(max(0, total_pt - total_ct) / wall_s, 1) if wall_s > 0 else 0.0,
            "output_tok_s": round(total_comp / wall_s, 2) if wall_s > 0 else 0.0,
            "total_output_tokens": total_comp,
        },
        "cluster_diff": cluster_diff,
        "telemetry_1hz": telemetry,
        "sessions": completed_sessions,
    }


async def async_main(args: argparse.Namespace) -> None:
    bundle_dir = args.bundle_dir
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    sessions_by_id: dict[str, dict] = {}
    with gzip.open(bundle_dir / manifest["bundle_file"], "rt", encoding="utf-8") as gz:
        for line in gz:
            if not line.strip():
                continue
            obj = json.loads(line)
            sessions_by_id[obj["session_id"]] = obj

    cohort_24a = [sessions_by_id[sid] for sid in manifest.get("cohort_24a", manifest["cohort_24"])]
    cohort_24b = [sessions_by_id[sid] for sid in manifest.get("cohort_24b", manifest["cohort_24"])]
    solo_12 = [sessions_by_id[sid] for sid in manifest.get("solo_12", manifest["solo_8"])]
    deep_12 = [sessions_by_id[sid] for sid in manifest.get("deep_12", manifest["deep_mut_8"])]
    mix_short_16 = [sessions_by_id[sid] for sid in manifest.get("mix_short_16", manifest["cohort_24"][:16])]
    mix_long_16 = [sessions_by_id[sid] for sid in manifest.get("mix_long_16", manifest["cohort_24"][:16])]
    mix_mut_12 = [sessions_by_id[sid] for sid in manifest.get("mix_mut_12", manifest["deep_mut_8"])]
    all_48_pool = cohort_24a + cohort_24b

    deadline_epoch = datetime.fromisoformat(args.deadline_cst).timestamp() if args.deadline_cst else (time.time() + 8 * 3600)

    run_id = args.run_id or f"lb24-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)

    if args.dry_run:
        print(
            f"[DRY-RUN] bundle={bundle_dir} total={len(sessions_by_id)} "
            f"24A={len(cohort_24a)} 24B={len(cohort_24b)} SOLO12={len(solo_12)} "
            f"DEEP12={len(deep_12)} SHORT16={len(mix_short_16)} LONG16={len(mix_long_16)} MUT12={len(mix_mut_12)}"
        )
        print(f"  deadline_cst={args.deadline_cst}")
        print("[DRY-RUN] OK")
        return

    api_key = load_api_key(args.api_key_file)
    limits = httpx.Limits(max_connections=96, max_keepalive_connections=64)
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        snap0 = await snapshot_cluster(client, args.relay_url, args.dot14_url, args.dot12_url)
        d14_ok = bool((snap0.get("dot14_daemon") or {}).get("ours_verified"))
        d12_ok = bool((snap0.get("dot12_daemon") or {}).get("ours_verified"))
        print(f"[{run_id}] Pre-flight health: dot14_verified={d14_ok}, dot12_verified={d12_ok}, dot14_prom={bool(snap0.get('dot14_prom'))}, dot12_prom={bool(snap0.get('dot12_prom'))}, deadline={args.deadline_cst}", flush=True)
        if not d14_ok:
            raise RuntimeError("dot14 is not healthy/ours_verified before launch!")

        if args.smoke:
            print(f"[{run_id}] Running 2-call SMOKE test (1 session x 2 turns)...", flush=True)
            s_res = await run_scenario(
                name="smoke_c1_2turn",
                sessions=[solo_12[0]],
                turn_slice=(0, 2),
                nonce_mode="session",
                nonce_base=f"{run_id}-smoke",
                initial_delays_s=[0.0],
                turn_gap_scale=0.0,
                turn_gap_cap_s=0.0,
                max_tokens_bounds=(32, 64),
                send_session_header=True,
                reversed_t2_barrier=False,
                client=client,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                relay_url=args.relay_url,
                dot14_url=args.dot14_url,
                dot12_url=args.dot12_url,
                serial=True,
            )
            print(
                f"[SMOKE] ok={s_res['ok_calls']}/{s_res['total_calls']} t1_hit={s_res['cache_cold_t1']['weighted_hit_rate_pct']}% "
                f"t2_hit={s_res['cache_warm_t2plus']['weighted_hit_rate_pct']}% "
                f"inter_wait={s_res['inter_emission_wait_s']['all']['mean']}s "
                f"decode_p50={s_res['per_stream_decode_tok_s']['p50']}tok/s",
                flush=True,
            )
            (out_dir / "smoke_receipt.json").write_text(json.dumps(s_res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return

        run_meta = {
            "run_id": run_id,
            "started_utc": utc_now_iso(),
            "started_cst": cst_now_iso(),
            "deadline_cst": args.deadline_cst,
            "base_url": args.base_url,
            "model": args.model,
            "preflight": {
                "dot14_verified": d14_ok,
                "dot12_verified": d12_ok,
                "relay_priority_order": (snap0.get("relay") or {}).get("priority_order"),
                "relay_sticky_affinity": (snap0.get("relay") or {}).get("sticky_affinity"),
            },
            "aborted_reason": None,
        }

        all_results: list[dict] = []
        solo_jct_ref: dict[str, float] = {}

        def checkpoint():
            (out_dir / "full_eval_receipt.json").write_text(
                json.dumps({"meta": run_meta, "results": all_results}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (out_dir / "REPORT_20261124_LB24.md").write_text(
                render_markdown_report(run_meta, all_results),
                encoding="utf-8",
            )

        def remaining_s() -> float:
            return deadline_epoch - time.time()

        async def execute_and_record(cfg: dict) -> bool:
            if remaining_s() < 300:
                print(f"[{cst_now_iso()}] Deadline approaching ({remaining_s():.0f}s left); skipping {cfg['name']} to enter P9 recovery.", flush=True)
                return False
            print(f"\n[{cst_now_iso()}] >>> Starting {cfg['name']} (C={cfg.get('display_c', len(cfg['sessions']))}, turns={cfg['turn_slice']}, rem={remaining_s()/60:.1f}m)...", flush=True)
            res = await run_scenario(
                name=cfg["name"],
                sessions=cfg["sessions"],
                turn_slice=cfg["turn_slice"],
                nonce_mode=cfg["nonce_mode"],
                nonce_base=cfg["nonce_base"],
                initial_delays_s=cfg["initial_delays_s"],
                turn_gap_scale=cfg["turn_gap_scale"],
                turn_gap_cap_s=cfg["turn_gap_cap_s"],
                max_tokens_bounds=cfg["max_tokens_bounds"],
                send_session_header=cfg["send_session_header"],
                reversed_t2_barrier=cfg.get("reversed_t2_barrier", False),
                client=client,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                relay_url=args.relay_url,
                dot14_url=args.dot14_url,
                dot12_url=args.dot12_url,
                solo_jct_ref=solo_jct_ref if solo_jct_ref else None,
                serial=cfg.get("serial", False),
            )
            all_results.append(res)
            (out_dir / f"phase_{len(all_results):02d}_{cfg['name']}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            checkpoint()
            wc = res["cache_warm_t2plus"]
            iw = res["inter_emission_wait_s"]["all"]
            print(
                f"<<< Done {cfg['name']}: ok={res['ok_calls']}/{res['total_calls']} wall={res['makespan_wall_s']}s "
                f"inter_wait_mean={iw['mean']}s/p95={iw['p95']}s dec_p50={res['per_stream_decode_tok_s']['p50']}t/s "
                f"warm_hit={wc['weighted_hit_rate_pct']}% (0%:{wc['zero_hit_count']}) out={res['throughput']['output_tok_s']}t/s "
                f"w14/w12={res['cluster_diff'].get('dot14_req_delta')}/{res['cluster_diff'].get('dot12_req_delta')}",
                flush=True,
            )
            if res["error_rate_pct"] >= 30.0:
                # Re-check cluster health before declaring hard abort
                await asyncio.sleep(15.0)
                chk = await snapshot_cluster(client, args.relay_url, args.dot14_url, args.dot12_url)
                if not ((chk.get("dot14_daemon") or {}).get("ours_verified")):
                    run_meta["aborted_reason"] = f"Circuit breaker tripped at {cfg['name']}: error_rate={res['error_rate_pct']}% and dot14 unhealthy"
                    checkpoint()
                    return False
            await asyncio.sleep(1.5)
            return True

        # ========== Build Comprehensive 10-Phase Overnight Suite (P0..P7 structured + P8 adaptive soak + P9 recovery) ==========
        phases: list[dict] = []

        # P0: 12-session Solo baseline (S/M/L/XL x 3, 6 turns, realistic decode [32, 512])
        phases.append({
            "name": "p0_solo_baseline",
            "display_c": 1,
            "sessions": solo_12,
            "turn_slice": (0, 6),
            "nonce_mode": "session",
            "nonce_base": f"{run_id}-p0",
            "initial_delays_s": [0.0] * len(solo_12),
            "turn_gap_scale": 0.0,
            "turn_gap_cap_s": 0.0,
            "max_tokens_bounds": (32, 512),
            "send_session_header": True,
            "serial": True,
        })

        # P1A: Fine-grained 12-point Scaling Curve (C = 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24) on COHORT-24A x 5 turns
        # Zero artificial tool sleep so scaling measures pure continuous concurrency without idle wait bloat.
        for c in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24):
            phases.append({
                "name": f"p1a_scaling_c{c:02d}",
                "sessions": cohort_24a[:c],
                "turn_slice": (0, 5),
                "nonce_mode": "scenario",
                "nonce_base": f"{run_id}-p1ac{c}",
                "initial_delays_s": [round(i * 0.10, 2) for i in range(c)],
                "turn_gap_scale": 0.0,
                "turn_gap_cap_s": 0.0,
                "max_tokens_bounds": (32, 384),
                "send_session_header": True,
            })

        # P1B: Independent Disjoint Cohort Repeat (COHORT-24B) at 8 key points for cross-sample statistical confidence
        for c in (4, 6, 8, 10, 12, 16, 20, 24):
            phases.append({
                "name": f"p1b_repeat_c{c:02d}",
                "sessions": cohort_24b[:c],
                "turn_slice": (0, 4),
                "nonce_mode": "scenario",
                "nonce_base": f"{run_id}-p1bc{c}",
                "initial_delays_s": [round(i * 0.10, 2) for i in range(c)],
                "turn_gap_scale": 0.0,
                "turn_gap_cap_s": 0.0,
                "max_tokens_bounds": (32, 384),
                "send_session_header": True,
            })

        # P2: Cold vs Warm 3-way isolation across C = 6, 8, 12, 16, 24
        for c in (6, 8, 12, 16, 24):
            sub = cohort_24a[:c]
            shared_nonce = f"{run_id}-p2c{c}-shared"
            phases.extend([
                {
                    "name": f"p2_strict_cold_c{c:02d}",
                    "sessions": sub,
                    "turn_slice": (0, 3),
                    "nonce_mode": "session",
                    "nonce_base": f"{run_id}-p2c{c}-strict",
                    "initial_delays_s": [0.0] * c,
                    "turn_gap_scale": 0.0,
                    "turn_gap_cap_s": 0.0,
                    "max_tokens_bounds": (32, 220),
                    "send_session_header": True,
                },
                {
                    "name": f"p2_shared_cold_c{c:02d}",
                    "sessions": sub,
                    "turn_slice": (0, 3),
                    "nonce_mode": "scenario",
                    "nonce_base": shared_nonce,
                    "initial_delays_s": [0.0] * c,
                    "turn_gap_scale": 0.0,
                    "turn_gap_cap_s": 0.0,
                    "max_tokens_bounds": (32, 220),
                    "send_session_header": True,
                },
                {
                    "name": f"p2_warm_burst_c{c:02d}",
                    "sessions": sub,
                    "turn_slice": (3, 5),
                    "nonce_mode": "scenario",
                    "nonce_base": shared_nonce,
                    "initial_delays_s": [0.0] * c,
                    "turn_gap_scale": 0.0,
                    "turn_gap_cap_s": 0.0,
                    "max_tokens_bounds": (32, 220),
                    "send_session_header": True,
                },
            ])

        # P3: 5 Arrival Orderings & Head-of-Line sensitivity at C = 8, 12, 16
        for c in (8, 12, 16):
            sub = cohort_24a[:c]
            orig = sorted(sub, key=lambda s: (s["first_ts"], s["session_id"]))
            heavy = sorted(sub, key=lambda s: (-s["max_pt"], s["session_id"]))
            light = sorted(sub, key=lambda s: (s["max_pt"], s["session_id"]))
            # Alternating XL/L and S/M
            alt = []
            lo, hi = 0, len(heavy) - 1
            while lo <= hi:
                alt.append(heavy[lo])
                if lo != hi:
                    alt.append(heavy[hi])
                lo += 1; hi -= 1
            for oname, olist, delays, rev_t2 in (
                ("original", orig, [round(i * 0.15, 2) for i in range(c)], False),
                ("heavy_first", heavy, [0.0 if i < max(2, c // 3) else 0.30 for i in range(c)], False),
                ("light_first", light, [0.0 if i < max(2, c // 3) else 0.30 for i in range(c)], False),
                ("alternating", alt, [round(i * 0.08, 2) for i in range(c)], False),
                ("reversed_t2", sub, [round(i * 0.05, 2) for i in range(c)], True),
            ):
                phases.append({
                    "name": f"p3_order_c{c:02d}_{oname}",
                    "sessions": olist,
                    "turn_slice": (0, 4),
                    "nonce_mode": "scenario",
                    "nonce_base": f"{run_id}-p3c{c}-{oname}",
                    "initial_delays_s": delays,
                    "turn_gap_scale": 0.0,
                    "turn_gap_cap_s": 0.0,
                    "max_tokens_bounds": (32, 256),
                    "send_session_header": True,
                    "reversed_t2_barrier": rev_t2,
                })

        # P4: Workload Mix sensitivity (Pure Short vs Pure Long vs All-Mutating vs Mixed) at C = 12 & 16
        for c in (12, 16):
            mix_specs = [
                ("pure_short", mix_short_16[:c]),
                ("pure_long", mix_long_16[:c]),
                ("mixed_stratified", cohort_24a[:c]),
            ]
            if c == 12:
                mix_specs.append(("all_mutating", mix_mut_12[:12]))
            for mname, msess in mix_specs:
                phases.append({
                    "name": f"p4_mix_c{c:02d}_{mname}",
                    "sessions": msess,
                    "turn_slice": (0, 4),
                    "nonce_mode": "scenario",
                    "nonce_base": f"{run_id}-p4c{c}-{mname}",
                    "initial_delays_s": [round(i * 0.10, 2) for i in range(len(msess))],
                    "turn_gap_scale": 0.0,
                    "turn_gap_cap_s": 0.0,
                    "max_tokens_bounds": (32, 320),
                    "send_session_header": True,
                })

        # P5: Compact Traffic Shapes & Short/Medium Tool-Gap Sensitivity (capped <= 2.5s) at C = 16 & C = 24
        for c in (16, 24):
            sub = cohort_24a[:c]
            phases.extend([
                {
                    "name": f"p5_shape_c{c:02d}_sync_burst",
                    "sessions": sub,
                    "turn_slice": (0, 4),
                    "nonce_mode": "scenario",
                    "nonce_base": f"{run_id}-p5c{c}-sync",
                    "initial_delays_s": [0.0] * c,
                    "turn_gap_scale": 0.0,
                    "turn_gap_cap_s": 0.0,
                    "max_tokens_bounds": (32, 256),
                    "send_session_header": True,
                },
                {
                    "name": f"p5_shape_c{c:02d}_staggered_3wave",
                    "sessions": sub,
                    "turn_slice": (0, 4),
                    "nonce_mode": "scenario",
                    "nonce_base": f"{run_id}-p5c{c}-wave",
                    "initial_delays_s": [float((i // max(1, c // 3)) * 1.5) for i in range(c)],
                    "turn_gap_scale": 0.05,
                    "turn_gap_cap_s": 1.0,
                    "max_tokens_bounds": (32, 256),
                    "send_session_header": True,
                },
                {
                    "name": f"p5_shape_c{c:02d}_short_gap",
                    "sessions": sub,
                    "turn_slice": (0, 5),
                    "nonce_mode": "scenario",
                    "nonce_base": f"{run_id}-p5c{c}-sgap",
                    "initial_delays_s": [round(i * 0.15, 2) for i in range(c)],
                    "turn_gap_scale": 0.10,
                    "turn_gap_cap_s": 1.5,
                    "max_tokens_bounds": (32, 384),
                    "send_session_header": True,
                },
                {
                    "name": f"p5_shape_c{c:02d}_medium_gap",
                    "sessions": sub,
                    "turn_slice": (0, 5),
                    "nonce_mode": "scenario",
                    "nonce_base": f"{run_id}-p5c{c}-mgap",
                    "initial_delays_s": [round(i * 0.25, 2) for i in range(c)],
                    "turn_gap_scale": 0.20,
                    "turn_gap_cap_s": 2.5,
                    "max_tokens_bounds": (32, 384),
                    "send_session_header": True,
                },
            ])

        # P6: Deep 16-turn chain + With-Header vs Headerless at C = 8 & C = 12 (zero tool sleep)
        for c in (8, 12):
            for hname, use_hdr in (("with_header", True), ("headerless", False)):
                phases.append({
                    "name": f"p6_deep16_c{c:02d}_{hname}",
                    "sessions": deep_12[:c],
                    "turn_slice": (0, 16),
                    "nonce_mode": "scenario",
                    "nonce_base": f"{run_id}-p6c{c}-{hname}",
                    "initial_delays_s": [round(i * 0.10, 2) for i in range(c)],
                    "turn_gap_scale": 0.0,
                    "turn_gap_cap_s": 0.0,
                    "max_tokens_bounds": (32, 256),
                    "send_session_header": use_hdr,
                })

        # P7: Heavy Long-Output Decode & PD Interleaving stress (max_tokens [128, 768]) at C = 8, 12, 16 (zero tool sleep)
        for c in (8, 12, 16):
            phases.append({
                "name": f"p7_long_decode_c{c:02d}",
                "sessions": cohort_24a[:c],
                "turn_slice": (0, 4),
                "nonce_mode": "scenario",
                "nonce_base": f"{run_id}-p7c{c}",
                "initial_delays_s": [round(i * 0.10, 2) for i in range(c)],
                "turn_gap_scale": 0.0,
                "turn_gap_cap_s": 0.0,
                "max_tokens_bounds": (128, 768),
                "send_session_header": True,
            })

        # Run structured phases P0..P7 (respecting hard deadline)
        aborted = False
        for idx, cfg in enumerate(phases):
            ok = await execute_and_record(cfg)
            if idx == 0 and all_results:
                for s in all_results[0]["sessions"]:
                    solo_jct_ref[s["session_id"]] = s["jct_model_s"]
            if not ok:
                if run_meta.get("aborted_reason"):
                    aborted = True
                break

        # P8: Adaptive Rolling Steady-State Soak filling all remaining overnight time cleanly up to (deadline - 5 min)
        if not aborted and remaining_s() > 420:
            soak_tiers = [12, 16, 20, 24]
            soak_round = 0
            while remaining_s() > 420:
                soak_round += 1
                for c_soak in soak_tiers:
                    rem = remaining_s() - 330  # reserve 5.5 min for drain + P9 recovery
                    if rem < 180:
                        break
                    slot_dur = min(1500.0, max(180.0, rem / max(1, len(soak_tiers) if soak_round == 1 else 2)))
                    sname = f"p8_soak_r{soak_round}_c{c_soak:02d}"
                    print(f"\n[{cst_now_iso()}] >>> Starting Rolling Soak {sname} (C={c_soak}, dur={slot_dur/60:.1f}m, rem_to_deadline={remaining_s()/60:.1f}m)...", flush=True)
                    s_res = await run_rolling_soak(
                        name=sname,
                        pool=all_48_pool,
                        concurrency=c_soak,
                        duration_s=slot_dur,
                        hard_stop_epoch=deadline_epoch - 300,
                        turns_per_user=6,
                        nonce_prefix=f"{run_id}-{sname}",
                        max_tokens_bounds=(32, 384),
                        client=client,
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        relay_url=args.relay_url,
                        dot14_url=args.dot14_url,
                        dot12_url=args.dot12_url,
                    )
                    all_results.append(s_res)
                    (out_dir / f"phase_{len(all_results):02d}_{sname}.json").write_text(
                        json.dumps(s_res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                    checkpoint()
                    print(
                        f"<<< Done {sname}: sessions={s_res['sessions_count']} calls={s_res['ok_calls']}/{s_res['total_calls']} "
                        f"inter_wait_mean={s_res['inter_emission_wait_s']['all']['mean']}s "
                        f"warm_hit={s_res['cache_warm_t2plus']['weighted_hit_rate_pct']}%",
                        flush=True,
                    )
                    if s_res["error_rate_pct"] >= 30.0:
                        run_meta["aborted_reason"] = f"Soak circuit breaker tripped at {sname}: err={s_res['error_rate_pct']}%"
                        aborted = True
                        break
                    await asyncio.sleep(5.0)
                if aborted or remaining_s() <= 420:
                    break

        # P9: Post-test recovery & cluster cleanliness check
        await asyncio.sleep(15.0)
        print(f"\n[{cst_now_iso()}] >>> Starting P9 Post-Test Recovery Check...", flush=True)
        p9_res = await run_scenario(
            name="p9_post_recovery_c02",
            sessions=solo_12[:2],
            turn_slice=(0, 2),
            nonce_mode="session",
            nonce_base=f"{run_id}-p9-rec",
            initial_delays_s=[0.0, 0.0],
            turn_gap_scale=0.0,
            turn_gap_cap_s=0.0,
            max_tokens_bounds=(32, 128),
            send_session_header=True,
            reversed_t2_barrier=False,
            client=client,
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            relay_url=args.relay_url,
            dot14_url=args.dot14_url,
            dot12_url=args.dot12_url,
            serial=False,
        )
        all_results.append(p9_res)
        (out_dir / f"phase_{len(all_results):02d}_p9_post_recovery_c02.json").write_text(
            json.dumps(p9_res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        run_meta["finished_utc"] = utc_now_iso()
        run_meta["finished_cst"] = cst_now_iso()
        checkpoint()
        print(f"\n[{run_id}] ALL DONE at {cst_now_iso()}! Report saved to {out_dir / 'REPORT_20261124_LB24.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:32862/v1")
    ap.add_argument("--model", default="DeepSeek")
    ap.add_argument("--relay-url", default="http://127.0.0.1:32859")
    ap.add_argument("--dot14-url", default="http://node14.example.invalid:32683")
    ap.add_argument("--dot12-url", default="http://node12.example.invalid:32683")
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--deadline-cst", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
