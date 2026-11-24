#!/usr/bin/env python3
"""Build a deterministic LB24 replay bundle from authored synthetic content."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from statistics import mean


BUCKETS = {"S": 16, "M": 48, "L": 96, "XL": 192}
TOOL = {
    "type": "function",
    "function": {
        "name": "read_fixture",
        "description": "Read a row from the fictional inventory in this benchmark.",
        "parameters": {
            "type": "object",
            "properties": {"row": {"type": "integer"}},
            "required": ["row"],
        },
    },
}
BUNDLE_NAME = "lb24-synthetic.jsonl.gz"


def make_reference(rows):
    return "\n".join(
        f"Row {i:05d} | item F-{i:05d} | stock {(i * 29) % 997:03d} | aisle {i % 23:02d} | open"
        for i in range(rows)
    )


def make_session(index, bucket, reference):
    label = f"synthetic-{index + 1:02d}"
    mutating = index % 12 < 4 or index == 4
    turn_count = 16 if index % 12 < 3 else 13 if index >= 42 else 14
    stable_system = (
        "Use the fictional warehouse inventory to answer each request. "
        "Return the item code and available stock.\n" + reference
    )
    history = []
    turns = []
    for turn in range(turn_count):
        system = stable_system + (f"\nProgress revision: {turn}." if mutating else "")
        row = 1 + (index * 11 + turn * 7) % 15
        history.append({
            "role": "user",
            "content": f"Worksheet {label}, step {turn + 1}: check inventory row {row}.",
        })
        messages = [{"role": "system", "content": system}, *history]
        turns.append({
            "recorded_prompt_tokens": sum(len(message.get("content") or "") for message in messages) // 4,
            "recorded_completion_tokens": 96 + 32 * (turn % 6),
            "inter_turn_gap_s": 0 if turn == 0 else 1 + (index + turn * 3) % 15,
            "sys_hash": hashlib.sha256(system.encode()).hexdigest(),
            "body": {"messages": messages, "tools": [TOOL]},
        })
        if turn % 3 == 1:
            call_id = f"call-{label}-{turn}"
            history.extend([
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_fixture", "arguments": json.dumps({"row": row})},
                }]},
                {"role": "tool", "tool_call_id": call_id, "content": f"Row {row}: verified."},
            ])
        history.append({
            "role": "assistant",
            "content": f"F-{row:05d} has {(row * 29) % 997} units available.",
        })

    prompt_sizes = [turn["recorded_prompt_tokens"] for turn in turns]
    return {
        "session_id": label,
        "task_id": f"fixture-task-{index + 1:02d}",
        "source_role": "root" if index < 21 else "child",
        "bucket": bucket,
        "min_pt": min(prompt_sizes),
        "max_pt": max(prompt_sizes),
        "mean_pt": round(mean(prompt_sizes)),
        "sys_versions": turn_count if mutating else 1,
        "mutating": mutating,
        "first_ts": index * 30.0,
        "turns": turns,
    }


def build(output_dir, context_scale):
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = output_dir / BUNDLE_NAME
    summary = []
    with bundle.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output:
                for bucket_index, (bucket, rows) in enumerate(BUCKETS.items()):
                    reference = make_reference(rows * context_scale)
                    for index in range(bucket_index * 12, (bucket_index + 1) * 12):
                        session = make_session(index, bucket, reference)
                        output.write(json.dumps(session, separators=(",", ":")) + "\n")
                        turns = session.pop("turns")
                        summary.append({**session, "n_turns": len(turns)})

    ids = [session["session_id"] for session in summary]
    cohort_a, cohort_b = ids[::2], ids[1::2]
    solo = [label for index, label in enumerate(ids) if index % 12 in (3, 6, 9)]
    deep = [session["session_id"] for session in summary if session["n_turns"] == 16]
    mutable = [session["session_id"] for session in summary if session["mutating"]]
    manifest = {
        "fixture": "synthetic-lb24",
        "context_scale": context_scale,
        "bundle_file": BUNDLE_NAME,
        "bundle_bytes": bundle.stat().st_size,
        "total_selected_sessions": len(summary),
        "total_turns_in_bundle": sum(session["n_turns"] for session in summary),
        "cohort_24": cohort_a,
        "cohort_24a": cohort_a,
        "cohort_24b": cohort_b,
        "solo_8": solo[:8],
        "solo_12": solo,
        "deep_mut_8": mutable[:8],
        "deep_12": deep,
        "mix_short_16": ids[:8] + ids[12:20],
        "mix_long_16": ids[24:32] + ids[36:44],
        "mix_mut_12": mutable[:12],
        "sessions_summary": summary,
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-scale", type=int, default=1, help="multiply the number of inventory rows")
    args = parser.parse_args()
    if args.context_scale < 1:
        parser.error("--context-scale must be positive")
    build(args.output_dir, args.context_scale)


if __name__ == "__main__":
    main()
