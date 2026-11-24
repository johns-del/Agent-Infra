"""Summarize matched scheduler trials, including incomplete or excluded pairs."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
from statistics import median
import time

from paired_trial import validate_pair


def mean(values):
    return sum(values) / len(values)


def ratio(candidate, baseline):
    return 100 * (candidate / baseline - 1) if baseline else None


def completed_pairs(results, user_by_graph, expected_calls=None, expected_rounds=None):
    groups = defaultdict(dict)
    for name, row in results.items():
        match = re.fullmatch(r"full-(\d+)-(.+)-r(\d+)-(.+)", name)
        if match:
            n, candidate, rep, method = match.groups()
            groups[(int(n), candidate, int(rep))][method] = row
    pairs = []
    modern_versions = {
        "orvix-dsv41-regression-fix-v3",
        "orvix-dsv41-diagnostic-v4",
        "orvix-dsv41-fair-chunk-v5",
        "orvix-dsv41-multi-prefill-v6",
        "orvix-dsv41-interval-cache-v7",
    }
    for (n, candidate, rep), items in sorted(groups.items()):
        row = {"offered": n, "candidate": candidate, "repeat": rep}
        pairs.append(row)
        if set(items) != {"B0", candidate}:
            row.update(eligible=False, reason="missing paired pass", present=sorted(items))
            continue
        baseline, treatment = items["B0"], items[candidate]
        is_modern = any(
            x.get("paired_context")
            or x.get("adapter_version") in modern_versions
            or x.get("engine_state_after", {}).get("adapter_version") in modern_versions
            for x in (baseline, treatment)
        )
        if is_modern:
            qualification = validate_pair(baseline, treatment)
            row["pair_qualification"] = qualification
            if not qualification["valid_paired_comparison"]:
                row.update(eligible=False, reason="paired provenance failed: " + "; ".join(qualification["reasons"]))
                continue
        b = {(x["graph_id"], x["node_id"]): x for x in baseline["calls"]}
        u = {(x["graph_id"], x["node_id"]): x for x in treatment["calls"]}
        br = {(x["graph_id"], x["round_id"]): x["elapsed_s"] for x in baseline["round_times"]}
        ur = {(x["graph_id"], x["round_id"]): x["elapsed_s"] for x in treatment["round_times"]}
        row["build_changes"] = [len(x.get("new_build_files", [])) for x in (baseline, treatment)]
        coverage = (
            len(b) == len(baseline["calls"]) and len(u) == len(treatment["calls"])
            and len(br) == len(baseline["round_times"]) and len(ur) == len(treatment["round_times"])
            and b.keys() == u.keys() and br.keys() == ur.keys() and bool(b) and bool(br)
            and (expected_calls is None or b.keys() == expected_calls)
            and (expected_rounds is None or br.keys() == expected_rounds)
            and all((b[k]["input_tokens"], b[k]["output_tokens"]) ==
                    (u[k]["input_tokens"], u[k]["output_tokens"]) for k in b)
        )
        if not coverage:
            row.update(eligible=False, reason="workload coverage mismatch")
            continue
        is_causal_splice = (baseline.get("replay_options") or {}).get("continuity_mode") in {"causal_splice", "forced_trace"}
        if not is_causal_splice and baseline.get("workload_sha256") != treatment.get("workload_sha256"):
            row.update(eligible=False, reason="target token workload fingerprint mismatch")
            continue
        if baseline.get("replay_options") != treatment.get("replay_options"):
            row.update(eligible=False, reason="replay arrival, lifecycle or time-scale mismatch")
            continue
        if baseline.get("engine_configuration") != treatment.get("engine_configuration"):
            row.update(eligible=False, reason="engine or HiCache launch configuration mismatch")
            continue
        row["eligible"] = not any(row["build_changes"])
        if not row["eligible"]:
            row["reason"] = "new or modified private build artifact"
        if any(x.get("build_artifact_observation", "measured").startswith("not collected")
               for x in (baseline, treatment)):
            row.update(eligible=False, reason="GPU compiler stability not observed; exploratory pair only")
        b_users, u_users = defaultdict(list), defaultdict(list)
        for k in br:
            user = user_by_graph[k[0]]
            b_users[user].append(br[k])
            u_users[user].append(ur[k])
        row["worst_user_mean_delta_pct"] = ratio(
            max(map(mean, u_users.values())), max(map(mean, b_users.values())))
        row["median_paired_user_mean_delta_pct"] = median(
            ratio(mean(u_users[k]), mean(b_users[k])) for k in b_users)
        paired_users = [
            {"user_id": user, "baseline_mean_s": mean(b_users[user]),
             "treatment_mean_s": mean(u_users[user]),
             "mean_delta_pct": ratio(mean(u_users[user]), mean(b_users[user]))}
            for user in sorted(b_users)
        ]
        row["paired_users"] = paired_users
        row["round_weighted_mean_s"] = [mean(br.values()), mean(ur.values())]
        row["user_equal_mean_s"] = [mean([mean(v) for v in b_users.values()]),
                                    mean([mean(v) for v in u_users.values()])]
        row["aggregation_semantics"] = "round-weighted efficiency vs user-equal latency; neither proves token-service fairness"
        row["comparison_kind"] = "AA_same_policy_noise_control" if candidate == "AA" else "AB_candidate"

        row["max_paired_user_mean_delta_pct"] = max(x["mean_delta_pct"] for x in paired_users)
        row["users_harmed_by_more_than_10pct"] = sum(x["mean_delta_pct"] > 10 for x in paired_users)
        row["same_output_hashes"] = sum(
            b[k]["output_sha256"] == u[k]["output_sha256"]
            and b[k].get("output_hash_basis") == u[k].get("output_hash_basis") for k in b)
        row["model_calls"] = len(b)
        row["paired_output_token_ids_verified"] = (
            row["same_output_hashes"] == len(b)
            and all(b[k].get("output_hash_basis") == "ids" for k in b))
        row["work_fidelity"] = ("token-identical" if row["paired_output_token_ids_verified"] else
                                  "fixed-input and fixed-length only; DSpark work can differ")
        row["paired_rounds_faster_by_at_least_1s"] = sum(br[k] - ur[k] >= 1 for k in br)
        row["paired_rounds_slower_by_at_least_1s"] = sum(ur[k] - br[k] >= 1 for k in br)
        metrics = ("wall_s", "round_p50_s", "round_p95_s", "queue_p95_s",
                   "peak_running_sampled", "peak_waiting_sampled")
        row["B0"] = {k: baseline[k] for k in metrics}
        row["treatment"] = {k: treatment[k] for k in metrics}
        row["delta_pct"] = {k: ratio(treatment[k], baseline[k]) for k in metrics[:4]
                            if treatment[k] is not None and baseline[k] is not None}
        row["reordered_cycles"] = treatment.get("rank_counters", {}).get("reordered_cycles", 0)
        row["chunk_counters"] = {"B0": baseline.get("chunk_counters", {}),
                                 "treatment": treatment.get("chunk_counters", {})}
        row["chunk_configuration"] = {"B0": baseline.get("chunk_configuration"),
                                      "treatment": treatment.get("chunk_configuration")}
        if "interval_counters" in treatment or "interval_counters" in baseline:
            row["interval_counters"] = {"B0": baseline.get("interval_counters", {}),
                                        "treatment": treatment.get("interval_counters", {})}
            row["interval_native_actions"] = {"B0": baseline.get("interval_native_actions", {}),
                                              "treatment": treatment.get("interval_native_actions", {})}
            row["interval_writeback_wall_s"] = [
                baseline.get("interval_native_writeback_wall_s"),
                treatment.get("interval_native_writeback_wall_s"),
            ]
        row["chunk_accounting_semantics"] = (
            "executed prefill / accepted output service proxy; not GPU seconds; "
            "prefill sharing is not proof of native mixed decode or speedup")
        row["cached_tokens"] = [sum(x["cached_tokens"] or 0 for x in r["calls"])
                                for r in (baseline, treatment)]
        row["spec_verify_count"] = [sum(x["spec_verify_ct"] or 0 for x in r["calls"])
                                    for r in (baseline, treatment)]
    return pairs


def recommendation(pairs, shortlist):
    high = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        if pair["eligible"] and pair["offered"] >= 32 and pair["candidate"] != "AA":
            high[pair["candidate"]][pair["offered"]].append(pair)
    for method, levels in high.items():
        if not all(len(levels[n]) >= 2 for n in (32, 48)):
            continue
        evidence = levels[32] + levels[48]
        if all(p["delta_pct"]["round_p95_s"] < 0 and p["delta_pct"]["wall_s"] <= 5
               and p["delta_pct"]["round_p50_s"] <= 5
               and p["max_paired_user_mean_delta_pct"] <= 10
               and p["same_output_hashes"] == p["model_calls"]
               for p in evidence):
            return {"deployment_default": "B0", "canary_candidate": method,
                    "qualified_high_load_pairs": len(evidence)}
    return {"deployment_default": "B0", "canary_candidate": None,
            "screening_finalist": shortlist.get("selected")}


def summarize(root):
    results = {}
    for path in (root / "results").glob("*.json"):
        try:
            results[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    with gzip.open(root / "input/graphs.jsonl.gz", "rt") as f:
        graphs = [json.loads(line) for line in f]
    pairs = completed_pairs(
        results, {g["graph_id"]: g["user_id"] for g in graphs},
        {(g["graph_id"], n["id"]) for g in graphs for n in g["nodes"] if n["kind"] == "model"},
        {(g["graph_id"], r["id"]) for g in graphs for r in g["rounds"]})
    shortlist = results.get("shortlist", {})
    closeout = {name: (root / name).read_text().strip()
                for name in ("maintenance.exit", "restore.exit", "private-gpu-released.utc",
                             "restored-run.txt") if (root / name).is_file()}
    return {
        "protocol": "development fixed-input fixed-output-length, DSpark on",
        "complete_burst_trials": sum(k.startswith("burst-") for k in results),
        "complete_full_passes": sum(k.startswith("full-") for k in results),
        "shortlist": shortlist, "full_pairs": pairs, "closeout": closeout,
        "recommendation": recommendation(pairs, shortlist),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        summary = summarize(args.root)
        output = args.root / "summary.latest.json"
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(summary, indent=2) + "\n")
        temp.replace(output)
        if not args.watch or (args.root / "maintenance.exit").is_file():
            break
        time.sleep(30)
