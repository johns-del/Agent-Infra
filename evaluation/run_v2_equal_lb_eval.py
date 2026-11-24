#!/usr/bin/env python3
"""
STRATA v14 V2 Equal-Weight (dot14 == dot12, strict 1:1 + full Session-Sticky)
Targeted Key-Point & Multi-Workload Comparison Suite against V1 (14->12 Priority).

Reuses the exact bundle, metrics, 1Hz sampler, and schema sanitizer from run_20261120_lb24_eval.py
so V1 vs V2 comparisons are strictly apples-to-apples, and auto-generates:
  - REPORT_V2_EQUAL_LB.md
  - COMPARE_V1_PRIORITY_VS_V2_EQUAL.md (side-by-side delta table against --v1-receipt)
  - full_eval_receipt.json + per-phase phase_XX_<name>.json
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_20261120_lb24_eval as base  # noqa: E402


def _fmt(v: Any, nd: int = 2) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def _fmt_delta(v1: float | None, v2: float | None, *, nd: int = 2, suffix: str = "", lower_is_better: bool = True) -> str:
    if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
        return "-"
    d = v2 - v1
    sign = "+" if d > 0 else ""
    better = (d < 0) if lower_is_better else (d > 0)
    mark = "✅" if better and abs(d) >= (0.05 if nd >= 2 else 0.5) else ("⚠️" if (not better) and abs(d) >= (0.05 if nd >= 2 else 0.5) else "➖")
    return f"{mark} {sign}{d:.{nd}f}{suffix}"


def _route_summary(r: dict) -> str:
    cd = r.get("cluster_diff") or {}
    rd = cd.get("relay_delta") or {}
    w14 = cd.get("dot14_req_delta", 0)
    w12 = cd.get("dot12_req_delta", 0)
    st = rd.get("sticky_routed_total", 0)
    new_d = rd.get("equal_routed_total", 0) + rd.get("dot14_priority_routed", 0) + rd.get("dot12_priority_routed", 0)
    rt = rd.get("router_spillover_routed", 0)
    return f"{w14}/{w12} ({st}/{new_d}/{rt})"


def _eff_pair(r: dict) -> tuple[float | None, float | None, float | None]:
    eff = r.get("effective_tok_s_excl_tool") or base.compute_effective_tok_s_excl_tool(r.get("sessions") or [])
    ea = (eff.get("all_turns") or {}).get("pooled_tok_s")
    ew = (eff.get("warm_t2plus") or {}).get("pooled_tok_s")
    ew_p10 = (eff.get("warm_t2plus") or {}).get("per_session_p10")
    return ea, ew, ew_p10


def render_v2_report(run_meta: dict, results: list[dict]) -> str:
    lines = [
        "# STRATA v14 V2 双机均等负载均衡（12 = 14 严格 1:1 + 全程粘性）测评报告",
        "",
        f"- **运行 ID**：`{run_meta['run_id']}`",
        f"- **调度模式**：`dispatch_mode={run_meta.get('relay_dispatch_mode', 'equal')}`（无优先级，`.14` 与 `.12` 最小并发 + 轮询 1:1 均分 + 全程 Session-Sticky）",
        f"- **开始时间（CST）**：`{run_meta['started_cst']}` | **更新时间（CST）**：`{base.cst_now_iso()}`",
        f"- **测试入口**：`{run_meta['base_url']}`（`model={run_meta['model']}`）",
        f"- **状态**：`{run_meta.get('aborted_reason') or '正常推进/完成'}`",
        "",
        "## 一、均等模式下关键并发梯度（主集 `COHORT-24A` + 副集 `COHORT-24B`，0s 工具等待）",
        "",
        "| 并发 `C` | 体感评级 | 有效吐字(不含工具) 全轮/热T2+ (tok/s) | 吐字间隔 Mean / P50 / P95 (s) | 冷首字 P50 / P95 (s) | 单流纯解码 P50 / P10 (tok/s) | TPOT P50 (ms) | 最大停顿 `C2` (s) | 慢化 P50 / P95 (Jain) | 热缓存命中率 | 路由 `.14`/`.12` (st/eq/rt) | 峰值KV `.14`/`.12` |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    scaling = [r for r in results if r["scenario"].startswith("p1a_scaling_") or r["scenario"].startswith("p1b_repeat_")]
    for r in scaling:
        tier_label, _ = base.evaluate_ux_tier(r)
        ea, ew, _ = _eff_pair(r)
        iw = (r.get("inter_emission_wait_s") or {}).get("all") or {}
        cold = r.get("ttft_cold_s") or {}
        spd = r.get("per_stream_decode_tok_s") or {}
        tpot = r.get("tpot_ms") or {}
        sd = r.get("slowdown_vs_solo") or {}
        wc = r.get("cache_warm_t2plus") or {}
        t1 = r.get("telemetry_1hz") or {}
        p10_spd = base.nearest_rank(
            [t["decode_tok_s"] for s in r.get("sessions", []) for t in s.get("turns", []) if t.get("ok") and t.get("decode_tok_s") is not None],
            0.10,
        )
        kv14, kv12 = t1.get("peak_dot14_kv"), t1.get("peak_dot12_kv")
        kv_s = f"{kv14:.2f}/{kv12:.2f}" if (kv14 is not None and kv12 is not None) else "-"
        tag = f"**{r['concurrency']}**" if r["scenario"].startswith("p1a_") else f"{r['concurrency']}(副集)"
        lines.append(
            f"| {tag} | {tier_label} | {_fmt(ea, 1)} / **{_fmt(ew, 1)}** | **{_fmt(iw.get('mean'))}** / {_fmt(iw.get('p50'))} / {_fmt(iw.get('p95'))} "
            f"| {_fmt(cold.get('p50'))} / {_fmt(cold.get('p95'))} "
            f"| {_fmt(spd.get('p50'), 1)} / {_fmt(p10_spd, 1)} | {_fmt(tpot.get('p50'), 1)} "
            f"| {_fmt(r.get('c2_max_gap_s'))} ({r.get('c2_max_gap_over_2s', 0)}违) "
            f"| {_fmt(sd.get('p50'))}x / {_fmt(sd.get('p95'))}x ({_fmt(sd.get('jain_fairness'), 3)}) "
            f"| **{_fmt(wc.get('weighted_hit_rate_pct'), 2)}%** ({wc.get('zero_hit_count', 0)}零) "
            f"| {_route_summary(r)} | {kv_s} |"
        )

    lines.extend([
        "",
        "## 二、全部 V2 场景完整明细表",
        "",
        "| 场景 | `C` | 成功/总计 | 排空 `M1`(s) | 冷TTFT P50/Max(s) | 续轮吐字间隔 Mean/P50/P95(s) | 单流 P50(tok/s) | 热缓存命中率 | `C1`/`C2` | 输出(tok/s) | 路由 `.14`/`.12` (st/eq/rt) | 峰值KV `.14`/`.12` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for r in results:
        t1 = r.get("telemetry_1hz") or {}
        c_ttft = r.get("ttft_cold_s") or {}
        iw = (r.get("inter_emission_wait_s") or {}).get("all") or {}
        spd = r.get("per_stream_decode_tok_s") or {}
        wc = r.get("cache_warm_t2plus") or {}
        kv14, kv12 = t1.get("peak_dot14_kv"), t1.get("peak_dot12_kv")
        kv_s = f"{kv14:.2f}/{kv12:.2f}" if (kv14 is not None and kv12 is not None) else "-"
        c_hit = f"{wc.get('weighted_hit_rate_pct', 0):.2f}% ({wc.get('ge90_count',0)}/{wc.get('count',0)} ≥90%, {wc.get('zero_hit_count',0)}零)" if wc.get("count") else "N/A"
        lines.append(
            f"| `{r['scenario']}` | {r['concurrency']} | {r['ok_calls']}/{r['total_calls']} | {r['makespan_wall_s']:.1f} "
            f"| {_fmt(c_ttft.get('p50'))}/{_fmt(c_ttft.get('max'))} "
            f"| {_fmt(iw.get('mean'))}/{_fmt(iw.get('p50'))}/{_fmt(iw.get('p95'))} "
            f"| {_fmt(spd.get('p50'), 1)} | **{c_hit}** "
            f"| {r['c1_ttft_over_120s']}/{r['c2_max_gap_over_2s']} (max {r['c2_max_gap_s']:.2f}s) "
            f"| {r['throughput']['output_tok_s']:.1f} | {_route_summary(r)} | {kv_s} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_v1_vs_v2_comparison(v1_results: list[dict], v2_results: list[dict], run_meta: dict) -> str:
    v1_map = {r["scenario"]: r for r in v1_results}
    lines = [
        "# STRATA v14 负载均衡 A/B 对照报告：V1（14→12 优先级） vs V2（12/14 严格 1:1 均等 + 全程粘性）",
        "",
        f"- **更新时间（CST）**：`{base.cst_now_iso()}` | **V2 运行 ID**：`{run_meta['run_id']}`",
        "- **对比基线（V1）**：`dot14 (soft_cap=6) -> dot12 (soft_cap=4) -> sglang-router`",
        "- **优化方案（V2）**：`dispatch_mode=equal`（`.14` 与 `.12` 无优先级，最小 `inflight` + 轮询 1:1 分发，`soft_cap=16 / sticky_burst=8 / max_kv=0.85`，全程 Session-Sticky）",
        "",
        "## 一、同场景逐项实测对比（仅列出 V1 与 V2 严格同样本、同参数的配对场景）",
        "",
        "| 场景 | `C` | 有效吐字(不含工具) 全轮 (V1→V2, Δ) | 有效吐字(不含工具) 热T2+ (V1→V2, Δ) | 吐字间隔 Mean (V1→V2, Δ) | 吐字间隔 P95 (V1→V2, Δ) | 热缓存命中率 (V1→V2, Δ) | 零命中 (V1→V2) | 路由 `.14/.12` (V1 → V2) |",
        "|---|---:|---|---|---|---|---|---:|---|",
    ]
    matched = 0
    for r2 in v2_results:
        sc = r2["scenario"]
        r1 = v1_map.get(sc)
        if not r1:
            continue
        matched += 1
        ea1, ew1, _ = _eff_pair(r1)
        ea2, ew2, _ = _eff_pair(r2)
        iw1 = (r1.get("inter_emission_wait_s") or {}).get("all") or {}
        iw2 = (r2.get("inter_emission_wait_s") or {}).get("all") or {}
        w1 = r1.get("cache_warm_t2plus") or {}
        w2 = r2.get("cache_warm_t2plus") or {}
        lines.append(
            f"| `{sc}` | {r2['concurrency']} "
            f"| {_fmt(ea1, 1)} → **{_fmt(ea2, 1)}** ({_fmt_delta(ea1, ea2, nd=1, lower_is_better=False)}) "
            f"| {_fmt(ew1, 1)} → **{_fmt(ew2, 1)}** ({_fmt_delta(ew1, ew2, nd=1, lower_is_better=False)}) "
            f"| {_fmt(iw1.get('mean'))}s → **{_fmt(iw2.get('mean'))}s** ({_fmt_delta(iw1.get('mean'), iw2.get('mean'), suffix='s', lower_is_better=True)}) "
            f"| {_fmt(iw1.get('p95'))}s → **{_fmt(iw2.get('p95'))}s** ({_fmt_delta(iw1.get('p95'), iw2.get('p95'), suffix='s', lower_is_better=True)}) "
            f"| {_fmt(w1.get('weighted_hit_rate_pct'), 1)}% → **{_fmt(w2.get('weighted_hit_rate_pct'), 1)}%** ({_fmt_delta(w1.get('weighted_hit_rate_pct'), w2.get('weighted_hit_rate_pct'), nd=1, suffix='pp', lower_is_better=False)}) "
            f"| {w1.get('zero_hit_count', 0)} → **{w2.get('zero_hit_count', 0)}** "
            f"| `{_route_summary(r1)}` → `{_route_summary(r2)}` |"
        )
    if matched == 0:
        lines.append("| *(尚无匹配场景，等待 V2 首个配对场景落盘)* | - | - | - | - | - | - | - | - | - |")

    # Also summarize V2-only new workload stress scenarios
    v2_only = [r for r in v2_results if r["scenario"] not in v1_map]
    if v2_only:
        lines.extend([
            "",
            "## 二、V2 均等模式新增多负载容量验证场景（针对目标人数档位）",
            "",
            "| 场景 | `C` | 成功/总计 | 排空(s) | 吐字间隔 Mean/P50/P95(s) | 单流 P50(tok/s) | 热缓存命中率 | `C1`/`C2` | 输出(tok/s) | 路由 `.14/.12` (st/eq/rt) | 峰值KV `.14/.12` |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for r in v2_only:
            iw = (r.get("inter_emission_wait_s") or {}).get("all") or {}
            spd = r.get("per_stream_decode_tok_s") or {}
            wc = r.get("cache_warm_t2plus") or {}
            t1 = r.get("telemetry_1hz") or {}
            kv14, kv12 = t1.get("peak_dot14_kv"), t1.get("peak_dot12_kv")
            kv_s = f"{kv14:.2f}/{kv12:.2f}" if (kv14 is not None and kv12 is not None) else "-"
            lines.append(
                f"| `{r['scenario']}` | {r['concurrency']} | {r['ok_calls']}/{r['total_calls']} | {r['makespan_wall_s']:.1f} "
                f"| **{_fmt(iw.get('mean'))}**/{_fmt(iw.get('p50'))}/{_fmt(iw.get('p95'))} "
                f"| {_fmt(spd.get('p50'), 1)} | **{_fmt(wc.get('weighted_hit_rate_pct'), 2)}%** ({wc.get('zero_hit_count', 0)}零) "
                f"| {r['c1_ttft_over_120s']}/{r['c2_max_gap_over_2s']} (max {r['c2_max_gap_s']:.2f}s) "
                f"| {r['throughput']['output_tok_s']:.1f} | {_route_summary(r)} | {kv_s} |"
            )

    lines.append("")
    return "\n".join(lines) + "\n"


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

    cohort_24a = [sessions_by_id[sid] for sid in manifest["cohort_24a"]]
    cohort_24b = [sessions_by_id[sid] for sid in manifest["cohort_24b"]]
    solo_12 = [sessions_by_id[sid] for sid in manifest["solo_12"]]
    deep_12 = [sessions_by_id[sid] for sid in manifest["deep_12"]]
    mix_short_16 = [sessions_by_id[sid] for sid in manifest["mix_short_16"]]
    mix_long_16 = [sessions_by_id[sid] for sid in manifest["mix_long_16"]]
    mix_mut_12 = [sessions_by_id[sid] for sid in manifest["mix_mut_12"]]
    all_48_pool = cohort_24a + cohort_24b

    # Load V1 results if provided (for apples-to-apples solo_jct_ref + live side-by-side comparison)
    v1_results: list[dict] = []
    solo_jct_ref: dict[str, float] = {}
    if args.v1_receipt and args.v1_receipt.is_file():
        v1_data = json.loads(args.v1_receipt.read_text(encoding="utf-8"))
        v1_results = v1_data.get("results") or []
        for r in v1_results:
            if r.get("scenario") == "p0_solo_baseline":
                for s in r.get("sessions") or []:
                    if s.get("session_id") and s.get("jct_model_s"):
                        solo_jct_ref[s["session_id"]] = s["jct_model_s"]
                break

    deadline_epoch = datetime.fromisoformat(args.deadline_cst).timestamp() if args.deadline_cst else time.time() + 8 * 3600
    run_id = args.run_id or f"lb24-v2-equal-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)

    # Build targeted V2 Equal-Weight + Multi-Workload phases
    phases: list[dict] = []

    if args.quick:
        # Compact ~25-30 min daytime-safe A/B suite (21 highest-value V1-matched scenarios, max C=16)
        for c in (4, 6, 8, 10, 12, 14, 16):
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
        for c in (6, 12):
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
        for c in (6, 12):
            sub = cohort_24a[:c]
            shared_nonce = f"{run_id}-p2c{c}-shared"
            phases.extend([
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
        for wname, wsess in (
            ("p4_mix_c12_pure_short", mix_short_16[:12]),
            ("p4_mix_c12_all_mutating", mix_mut_12[:12]),
            ("p4_mix_c12_pure_long", mix_long_16[:12]),
            ("p4_mix_c16_pure_short", mix_short_16[:16]),
        ):
            phases.append({
                "name": wname,
                "sessions": wsess,
                "turn_slice": (0, 4),
                "nonce_mode": "scenario",
                "nonce_base": f"{run_id}-{wname}",
                "initial_delays_s": [round(i * 0.10, 2) for i in range(len(wsess))],
                "turn_gap_scale": 0.0,
                "turn_gap_cap_s": 0.0,
                "max_tokens_bounds": (32, 320),
                "send_session_header": True,
            })
        phases.extend([
            {
                "name": "p3_order_c12_reversed_t2",
                "sessions": cohort_24a[:12],
                "turn_slice": (0, 4),
                "nonce_mode": "scenario",
                "nonce_base": f"{run_id}-p3c12-rev",
                "initial_delays_s": [round(i * 0.05, 2) for i in range(12)],
                "turn_gap_scale": 0.0,
                "turn_gap_cap_s": 0.0,
                "max_tokens_bounds": (32, 256),
                "send_session_header": True,
                "reversed_t2_barrier": True,
            },
            {
                "name": "p6_deep16_c12_headerless",
                "sessions": deep_12[:12],
                "turn_slice": (0, 16),
                "nonce_mode": "scenario",
                "nonce_base": f"{run_id}-p6c12-hdrless",
                "initial_delays_s": [round(i * 0.10, 2) for i in range(12)],
                "turn_gap_scale": 0.0,
                "turn_gap_cap_s": 0.0,
                "max_tokens_bounds": (32, 256),
                "send_session_header": False,
            },
            {
                "name": "p7_long_decode_c12",
                "sessions": cohort_24a[:12],
                "turn_slice": (0, 4),
                "nonce_mode": "scenario",
                "nonce_base": f"{run_id}-p7c12",
                "initial_delays_s": [round(i * 0.10, 2) for i in range(12)],
                "turn_gap_scale": 0.0,
                "turn_gap_cap_s": 0.0,
                "max_tokens_bounds": (128, 768),
                "send_session_header": True,
            },
            {
                "name": "p5_shape_c16_medium_gap",
                "sessions": cohort_24a[:16],
                "turn_slice": (0, 5),
                "nonce_mode": "scenario",
                "nonce_base": f"{run_id}-p5c16-mgap",
                "initial_delays_s": [round(i * 0.25, 2) for i in range(16)],
                "turn_gap_scale": 0.20,
                "turn_gap_cap_s": 2.5,
                "max_tokens_bounds": (32, 384),
                "send_session_header": True,
            },
        ])
    else:
        # Full 61-phase comparison suite
        for c in (4, 6, 8, 10, 12, 14, 16, 18, 20):
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

        # Block 2: Independent Disjoint Cohort Repeat on COHORT-24B (C = 6, 8, 10, 12, 16)
        for c in (6, 8, 10, 12, 16):
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

        # Block 3: Cold vs Warm 3-Way Isolation at key points (C = 6, 8, 12, 16)
        for c in (6, 8, 12, 16):
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

        # Block 4: Key Ordering & Anti-Drift checks at C = 8, 12, 16
        for c in (8, 12, 16):
            sub = cohort_24a[:c]
            heavy = sorted(sub, key=lambda s: (-s["max_pt"], s["session_id"]))
            light = sorted(sub, key=lambda s: (s["max_pt"], s["session_id"]))
            for oname, olist, delays, rev_t2 in (
                ("heavy_first", heavy, [0.0 if i < max(2, c // 3) else 0.30 for i in range(c)], False),
                ("light_first", light, [0.0 if i < max(2, c // 3) else 0.30 for i in range(c)], False),
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

        # Block 5: Diverse Workload Profiles around Recommended Capacity (C = 8, 10, 12, 16)
        workload_matrix = [
            ("p4_mix_c08_pure_long", mix_long_16[:8], (32, 320)),
            ("p4_mix_c08_all_mutating", mix_mut_12[:8], (32, 320)),
            ("p4_mix_c10_pure_long", mix_long_16[:10], (32, 320)),
            ("p4_mix_c10_mixed_stratified", cohort_24a[:10], (32, 320)),
            ("p4_mix_c12_pure_short", mix_short_16[:12], (32, 320)),
            ("p4_mix_c12_pure_long", mix_long_16[:12], (32, 320)),
            ("p4_mix_c12_mixed_stratified", cohort_24a[:12], (32, 320)),
            ("p4_mix_c12_all_mutating", mix_mut_12[:12], (32, 320)),
            ("p4_mix_c16_pure_short", mix_short_16[:16], (32, 320)),
            ("p4_mix_c16_pure_long", mix_long_16[:16], (32, 320)),
            ("p4_mix_c16_mixed_stratified", cohort_24a[:16], (32, 320)),
        ]
        for wname, wsess, mt_b in workload_matrix:
            phases.append({
                "name": wname,
                "sessions": wsess,
                "turn_slice": (0, 4),
                "nonce_mode": "scenario",
                "nonce_base": f"{run_id}-{wname}",
                "initial_delays_s": [round(i * 0.10, 2) for i in range(len(wsess))],
                "turn_gap_scale": 0.0,
                "turn_gap_cap_s": 0.0,
                "max_tokens_bounds": mt_b,
                "send_session_header": True,
            })

        # Block 6: Realistic Compact Pacing (short <=1.5s & medium <=2.5s tool gaps) at C = 12, 16, 20, 24
        for c in (12, 16, 20, 24):
            sub = cohort_24a[:c]
            phases.extend([
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

        # Block 7: Deep 16-Turn Chain (Header vs Headerless) at C = 8 & 12
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

        # Block 8: Heavy Long-Output Decode & PD Interleaving (max_tokens [128, 768]) at C = 8, 12, 16
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

    if args.dry_run:
        print(
            f"[DRY-RUN V2-EQUAL] phases={len(phases)} v1_loaded={len(v1_results)} "
            f"solo_refs={len(solo_jct_ref)} deadline={args.deadline_cst}"
        )
        print("  sample:", [p["name"] for p in phases[:8]], "...", [p["name"] for p in phases[-4:]])
        print("[DRY-RUN V2-EQUAL] OK")
        return

    api_key = base.load_api_key(args.api_key_file)
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=64)
    timeout = httpx.Timeout(360.0, connect=15.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:
        init_snap = await base.snapshot_cluster(client, args.relay_url, args.dot14_url, args.dot12_url)
        relay_mode = (init_snap.get("relay") or {}).get("dispatch_mode", "unknown")
        run_meta = {
            "run_id": run_id,
            "started_utc": base.utc_now_iso(),
            "started_cst": base.cst_now_iso(),
            "deadline_cst": args.deadline_cst,
            "base_url": args.base_url,
            "model": args.model,
            "relay_dispatch_mode": relay_mode,
            "v1_receipt": str(args.v1_receipt) if args.v1_receipt else None,
            "initial_cluster_snapshot": init_snap,
            "aborted_reason": None,
        }
        print(
            f"[{run_id}] Pre-flight: dispatch_mode={relay_mode}, "
            f"dot14_verified={(init_snap.get('dot14_daemon') or {}).get('ours_verified')}, "
            f"dot12_verified={(init_snap.get('dot12_daemon') or {}).get('ours_verified')}, "
            f"v1_scenarios={len(v1_results)}, deadline={args.deadline_cst}",
            flush=True,
        )
        if relay_mode != "equal":
            print(f"[WARN] Expected relay dispatch_mode='equal' but got '{relay_mode}'!", flush=True)

        all_results: list[dict] = []

        def checkpoint() -> None:
            (out_dir / "full_eval_receipt.json").write_text(
                json.dumps({"meta": run_meta, "results": all_results}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (out_dir / "REPORT_V2_EQUAL_LB.md").write_text(
                render_v2_report(run_meta, all_results), encoding="utf-8"
            )
            if v1_results:
                (out_dir / "COMPARE_V1_PRIORITY_VS_V2_EQUAL.md").write_text(
                    render_v1_vs_v2_comparison(v1_results, all_results, run_meta), encoding="utf-8"
                )

        def remaining_s() -> float:
            return deadline_epoch - time.time()

        aborted = False
        for cfg in phases:
            if remaining_s() < 240:
                print(f"[{base.cst_now_iso()}] Deadline guard: {remaining_s():.0f}s left, skipping {cfg['name']}", flush=True)
                break
            c = len(cfg["sessions"])
            print(f"\n[{base.cst_now_iso()}] >>> Starting {cfg['name']} (C={c}, turns={cfg['turn_slice']}, rem={remaining_s()/60:.1f}m)...", flush=True)
            res = await base.run_scenario(
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
                serial=False,
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
                f"route={_route_summary(res)}",
                flush=True,
            )
            if res["error_rate_pct"] >= 30.0:
                await asyncio.sleep(15.0)
                chk = await base.snapshot_cluster(client, args.relay_url, args.dot14_url, args.dot12_url)
                if not ((chk.get("dot14_daemon") or {}).get("ours_verified")):
                    run_meta["aborted_reason"] = f"Circuit breaker tripped at {cfg['name']}: err={res['error_rate_pct']}%"
                    aborted = True
                    checkpoint()
                    break
            await asyncio.sleep(1.5)

        # Adaptive Rolling Steady-State Soak at target equal-mode capacities (C = 10, 12, 14, 16) up to deadline - 5 min
        if (not args.skip_soak) and (not aborted) and remaining_s() > 420:
            soak_tiers = [10, 12, 14, 16]
            soak_round = 0
            while remaining_s() > 420:
                soak_round += 1
                for c_soak in soak_tiers:
                    rem = remaining_s() - 330
                    if rem < 180:
                        break
                    slot_dur = min(1500.0, max(180.0, rem / max(1, len(soak_tiers) if soak_round == 1 else 2)))
                    sname = f"p8_equal_soak_r{soak_round}_c{c_soak:02d}"
                    print(f"\n[{base.cst_now_iso()}] >>> Starting Equal Soak {sname} (C={c_soak}, dur={slot_dur/60:.1f}m)...", flush=True)
                    s_res = await base.run_rolling_soak(
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
                        f"warm_hit={s_res['cache_warm_t2plus']['weighted_hit_rate_pct']}% route={_route_summary(s_res)}",
                        flush=True,
                    )
                    if s_res["error_rate_pct"] >= 30.0:
                        run_meta["aborted_reason"] = f"Soak circuit breaker tripped at {sname}: err={s_res['error_rate_pct']}%"
                        aborted = True
                        break
                    await asyncio.sleep(3.0)
                if aborted or remaining_s() <= 420:
                    break

        # P9: Post-test recovery check
        await asyncio.sleep(10.0)
        print(f"\n[{base.cst_now_iso()}] >>> Starting P9 Equal Post-Test Recovery Check...", flush=True)
        p9 = await base.run_scenario(
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
        all_results.append(p9)
        (out_dir / f"phase_{len(all_results):02d}_p9_post_recovery_c02.json").write_text(
            json.dumps(p9, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        run_meta["finished_utc"] = base.utc_now_iso()
        run_meta["finished_cst"] = base.cst_now_iso()
        checkpoint()
        print(f"\n[{run_id}] ALL DONE at {base.cst_now_iso()}! Compare report: {out_dir / 'COMPARE_V1_PRIORITY_VS_V2_EQUAL.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--v1-receipt", type=Path, default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:32862/v1")
    ap.add_argument("--model", default="DeepSeek")
    ap.add_argument("--relay-url", default="http://127.0.0.1:32859")
    ap.add_argument("--dot14-url", default="http://node14.example.invalid:32683")
    ap.add_argument("--dot12-url", default="http://node12.example.invalid:32683")
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--deadline-cst", default="")
    ap.add_argument("--quick", action="store_true", default=False)
    ap.add_argument("--skip-soak", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
