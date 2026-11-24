"""Summarize workload opportunity without exposing private prompts or token IDs.

This is a structural audit, not a GPU performance model. Prefix statistics are
exact token-input comparisons; they do not imply those tokens are resident or
that the DeepSeek SWA/Full/compression state can be restored at that boundary.
"""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median


def read_rows(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "sum": 0, "min": None, "p50": None, "p90": None,
                "p95": None, "max": None}
    return {
        "count": len(values), "sum": sum(values), "min": values[0],
        "p50": median(values), "p90": values[math.ceil(len(values) * .90) - 1],
        "p95": values[math.ceil(len(values) * .95) - 1], "max": values[-1],
    }


def common_prefix(left, right):
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def unique_prefix_tokens(sequences):
    """Number of nodes in the input-only token trie, without building the trie."""
    ordered = sorted(sequences)
    return sum(map(len, ordered)) - sum(common_prefix(a, b)
                                       for a, b in zip(ordered, ordered[1:]))


def is_real_tool(node):
    return (node["kind"] == "tool"
            and not node.get("attrs", {}).get("tool_type", "").startswith("orvix_"))


def locate_inputs(root, prompts_path=None):
    graph_choices = [root / "graphs.jsonl.gz", root / "source/graphs.jsonl.gz",
                     root / "input/graphs.jsonl.gz", root / "workload/source/graphs.jsonl.gz"]
    prompt_choices = ([prompts_path] if prompts_path else [
        root / "private/deepseek-prompts.jsonl.gz",
        root / "deepseek-replay/private/deepseek-prompts.jsonl.gz",
        root / "input/private/deepseek-prompts.jsonl.gz",
        root / "workload/deepseek-replay/private/deepseek-prompts.jsonl.gz"])
    graph_path = next((p for p in graph_choices if p.is_file()), None)
    prompt_path = next((p for p in prompt_choices if p.is_file()), None)
    if graph_path is None or prompt_path is None:
        raise ValueError("expected graph corpus and target-tokenizer prompt corpus")
    return graph_path, prompt_path


def analyze(graphs, prompt_rows):
    prompts = {(r["graph_id"], r["node_id"]): r for r in prompt_rows}
    expected = {(g["graph_id"], n["id"]) for g in graphs
                for n in g["nodes"] if n["kind"] == "model"}
    if len(prompts) != len(prompt_rows) or prompts.keys() != expected:
        raise ValueError("prompt corpus has duplicate, missing or unexpected model rows")
    if len({g["graph_id"] for g in graphs}) != len(graphs):
        raise ValueError("duplicate graph IDs")
    nodes = [n for g in graphs for n in g["nodes"]]
    tools = [n for n in nodes if is_real_tool(n)]
    gaps = [n for n in nodes if n["kind"] == "think"]
    by_tool = defaultdict(list)
    for node in tools:
        by_tool[node.get("attrs", {}).get("tool_type", "unknown")].append(node["duration_s"])

    first_inputs, successive_lcp, successive_fraction, successive_tail = [], [], [], []
    recorded_return_gaps, full_prior_prefix = [], 0
    calls_per_user = Counter()
    for graph in graphs:
        models = sorted((n for n in graph["nodes"] if n["kind"] == "model"),
                        key=lambda n: (n["start_s"], n["id"]))
        calls_per_user[graph["user_id"]] += len(models)
        if models:
            first_inputs.append(prompts[(graph["graph_id"], models[0]["id"])] ["input_ids"])
        # Same-scope order is only an input reuse opportunity, not a scheduler
        # oracle: no information from these pairs is sent to the runtime.
        previous = {}
        for node in models:
            prior = previous.get(node["scope"])
            if prior is not None:
                left = prompts[(graph["graph_id"], prior["id"])]["input_ids"]
                right = prompts[(graph["graph_id"], node["id"])]["input_ids"]
                count = common_prefix(left, right)
                successive_lcp.append(count)
                successive_fraction.append(count / len(right) if right else 0)
                successive_tail.append(len(right) - count)
                full_prior_prefix += count == len(left)
                recorded_return_gaps.append(max(0, node["start_s"] - prior["end_s"]))
            previous[node["scope"]] = node

    first_inputs.sort()
    all_inputs = [r["input_ids"] for r in prompt_rows]
    result = {
        "scope": "offline structure only; no target-engine performance measurements",
        "coverage": {
            "graphs": len(graphs), "rounds": sum(len(g["rounds"]) for g in graphs),
            "model_calls": len(prompts), "users": len({g["user_id"] for g in graphs}),
            "sessions": len({g["session_id"] for g in graphs}),
            "node_kinds": dict(Counter(n["kind"] for n in nodes)),
            "model_lanes": dict(Counter(n.get("purpose") for n in nodes if n["kind"] == "model")),
            "graphs_with_subagents": sum(any(n.get("purpose") == "subagent"
                                             for n in g["nodes"]) for g in graphs),
            "graphs_with_cross_scope_links": sum(bool(g["links"]) for g in graphs),
            "graphs_with_real_tools": sum(any(is_real_tool(n) for n in g["nodes"])
                                           for g in graphs),
            "graphs_with_one_model": sum(sum(n["kind"] == "model" for n in g["nodes"]) == 1
                                          for g in graphs),
            "scopes_per_graph": dict(Counter(len({n["scope"] for n in g["nodes"]}) for g in graphs)),
        },
        "tool_node_semantics": "business tool nodes excluding application gaps; includes errors and fixed-result projections, not proof of live successful execution",
        "tool_execution_status_counts": dict(Counter(n.get("attrs", {}).get("execution_status", "unspecified") for n in tools)),
        "tool_seconds_by_status": {
            status: distribution(n["duration_s"] for n in tools
                                 if n.get("attrs", {}).get("execution_status", "unspecified") == status)
            for status in sorted({n.get("attrs", {}).get("execution_status", "unspecified") for n in tools})},
        "tool_ge5s_by_status": dict(Counter(n.get("attrs", {}).get("execution_status", "unspecified")
                                             for n in tools if n["duration_s"] >= 5)),
        "tool_explicit_fixed_result_projection_count": sum(n.get("attrs", {}).get("executes_real_tool") is False for n in tools),
        "real_tool_seconds": distribution(n["duration_s"] for n in tools),
        "real_tool_seconds_by_type": {k: distribution(v) for k, v in sorted(by_tool.items())},
        "real_tool_duration_threshold_counts": {
            str(t): sum(n["duration_s"] >= t for n in tools) for t in (.01, .1, 1, 5, 10, 30, 60)},
        "synthetic_application_gap_nodes": sum(n["kind"] == "tool" and not is_real_tool(n)
                                                for n in nodes),
        "recorded_client_gap_seconds": distribution(n["duration_s"] for n in gaps),
        "recorded_gap_semantics": dict(Counter(n.get("attrs", {}).get("wait_semantics", "unspecified")
                                                for n in gaps)),
        "historical_saved_gap_seconds": distribution(n["attrs"]["historical_saved_gap_s"]
                                                       for n in gaps if "historical_saved_gap_s" in n["attrs"]),
        "input_tokens": distribution(map(len, all_inputs)),
        "output_budget_tokens": distribution(r["output_budget"] for r in prompt_rows),
        "calls_per_graph": distribution(sum(n["kind"] == "model" for n in g["nodes"]) for g in graphs),
        "calls_per_user": distribution(calls_per_user.values()),
        "prefix_opportunity": {
            "all_first_requests_common_tokens": min((common_prefix(first_inputs[0], x)
                                                       for x in first_inputs[1:]), default=0),
            "successive_same_scope_input_lcp_tokens": distribution(successive_lcp),
            "successive_next_input_reusable_fraction": distribution(successive_fraction),
            "successive_next_input_new_tail_tokens": distribution(successive_tail),
            "successive_pairs_with_entire_prior_input": full_prior_prefix,
            "all_input_tokens_sum": sum(map(len, all_inputs)),
            "all_unique_input_prefix_tokens": unique_prefix_tokens(all_inputs),
            "first_request_unique_prefix_tokens": unique_prefix_tokens(first_inputs),
            "semantics": "exact target input token prefixes; not KV bytes, residency or restored complete state",
        },
        "recorded_between_model_gap_seconds": distribution(recorded_return_gaps),
        "limitations": [
            "The supplied analysis bundle contains inputs, not DeepSeek trial results or engine traces.",
            "Input-only prefix opportunity is not measured cache hit rate; generated DeepSeek outputs differ from the source recording.",
            "Fixed-input fixed-output-length replay does not enforce token-identical generation; DSpark verification work may differ.",
            "Recorded client gaps can be programmatic. Historical saved gaps must not silently replace recorded durations.",
            "Tool wall times cannot be summed into end-to-end critical path when tools overlap.",
            "GPU utilization, transfer bandwidth, cache pressure and offload benefit need target-engine measurements.",
        ],
    }
    return result


def main(args):
    graphs_path, prompts_path = locate_inputs(args.root, args.prompts)
    result = analyze(read_rows(graphs_path), read_rows(prompts_path))
    result["input_sha256"] = {"graphs": sha256(graphs_path.read_bytes()).hexdigest(),
                              "target_prompts": sha256(prompts_path.read_bytes()).hexdigest()}
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
