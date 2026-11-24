"""Prepare target-tokenizer prompts for a fixed-input, fixed-length replay."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median


def read_gzip(path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def replay_prompt(original):
    fields = ("messages", "tools", "tool_choice", "response_format", "continue_final_message")
    prompt = {key: original[key] for key in fields if original.get(key) is not None}
    kwargs = original.get("chat_template_kwargs") or {}
    thinking = kwargs.get("enable_thinking", kwargs.get("thinking", True))
    if type(thinking) is not bool:
        raise ValueError("invalid thinking mode")
    prompt["thinking"] = thinking
    for message in prompt["messages"]:
        if message.get("content") is None:
            message["content"] = ""
    return prompt


def prepare(source, model, output, context):
    from tokenizers import Tokenizer
    from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe

    tokenizer_file = model / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    renderer = DeepSeekRecipe("dsv41", tokenizer)
    graphs = list(read_gzip(source / "graphs.jsonl.gz"))
    manifest = json.loads((source / "manifest.json").read_text())
    expected = {(g["graph_id"], n["id"]) for g in graphs for n in g["nodes"] if n["kind"] == "model"}
    if len(graphs) != manifest["total_graphs"] or len(expected) != manifest["model_calls"]:
        raise ValueError("graph coverage differs from the source manifest")
    records, errors = {}, Counter()
    for record in read_gzip(source / "private/tokens.private.jsonl.gz"):
        key = record["graph_id"], record["node_id"]
        if key not in expected or key in records:
            raise ValueError("unexpected or duplicate model call")
        try:
            images = []
            ids = renderer.encode(replay_prompt(record["original_chat_request"]), image_urls=images)
            if images:
                raise ValueError("unexpected image in text replay")
        except (ValueError, TypeError, KeyError) as exc:
            # Recipe errors may include private request content.
            errors[type(exc).__name__] += 1
            continue
        records[key] = {
            "graph_id": key[0], "node_id": key[1], "input_ids": ids,
            "output_budget": len(record["output_ids"]),
            "recorded_output_ids": [int(t) for t in record["output_ids"]],
        }
    if errors or records.keys() != expected:
        raise ValueError(f"target render coverage {len(records)}/{len(expected)}; errors={dict(errors)}")
    lengths = [len(row["input_ids"]) for row in records.values()]
    totals = [len(row["input_ids"]) + row["output_budget"] for row in records.values()]
    over_context = sum(total >= context for total in totals)
    with gzip.open(output, "xt") as handle:
        for row in records.values():
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "mode": "target-rendered-input-and-source-output-length-budget",
        "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        "graphs": len(graphs), "rounds": sum(len(g["rounds"]) for g in graphs),
        "model_calls": len(records), "prompt_tokens_min": min(lengths),
        "prompt_tokens_median": median(lengths), "prompt_tokens_max": max(lengths),
        "input_and_output_tokens_max": max(totals), "over_context": over_context,
        "context": context,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context", type=int, default=65536)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.model, args.output, args.context), indent=2))
