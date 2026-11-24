"""Stage an already-tokenized analysis bundle without rerendering private text."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from compare_dev_replay import read_inputs
from workload_audit import locate_inputs


def prepare(root, output, context=65536):
    graph_path, prompt_path = locate_inputs(root)
    graphs, prompts = read_inputs(root, context)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / "private").mkdir(mode=0o700)
    copies = [(graph_path, output / "graphs.jsonl.gz"),
              (prompt_path, output / "private/deepseek-prompts.jsonl.gz")]
    receipt = {"mode": "copy_existing_target_tokens_no_retokenization", "context": context,
               "graphs": len(graphs), "rounds": sum(len(g["rounds"]) for g in graphs),
               "model_calls": len(prompts), "files": {}}
    for source, target in copies:
        shutil.copyfile(source, target)
        target.chmod(0o600)
        expected = sha256(source.read_bytes()).hexdigest()
        observed = sha256(target.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError("fixture copy hash mismatch")
        receipt["files"][str(target.relative_to(output))] = {"sha256": observed,
                                                             "bytes": target.stat().st_size}
    (output / "FIXTURE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context", type=int, default=65536)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output, args.context), indent=2))
