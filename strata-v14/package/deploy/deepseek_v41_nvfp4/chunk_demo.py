"""Run an inspectable policy demonstration. No model/GPU or speedup simulation."""
import argparse
import json
from pathlib import Path

from chunk_fair_core import Candidate, ChunkConfig, ServiceLedger, plan_shares


def demonstration():
    config = ChunkConfig(mode="fair", age_guard_s=0., fairness_slack_units=0.)
    ledger = ServiceLedger(config)
    ledger.sync(["user_A", "user_B"])
    a = Candidate("A", "user_A", 8192, 0., 0, True)
    b = Candidate("B", "user_B", 256, 0., 1)
    first = plan_shares([a, b], 2048, 256, 1, config,
                        {u: ledger.effective(u) for u in ledger.active}, 1.)
    assert [(s.rid, s.cap) for s in first.shares] == [("A", 1792), ("B", 256)]
    # Illustrative execution receipt: these are DECLARED example counts, not
    # measured model work. A native execution test exists separately in tests/.
    ledger.reserve(1, [("user_A", 1792, 0), ("user_B", 256, 1)])
    ledger.settle(1, [("user_A", 1792, 0), ("user_B", 256, 1)])
    before_wait = ledger.snapshot()
    full = plan_shares([a, b], 2048, 256, 0, config, {}, 1.)
    two_longs = plan_shares([a, Candidate("B-long", "user_B", 8192, 0., 1)],
                           2048, 256, 1, config, {}, 1.)
    return {
        "kind": "policy demonstration, NOT GPU measurement or performance simulation",
        "budget": 2048, "alignment": 256,
        "example_A_1792_B_256": first.frozen(),
        "A_remaining_after_this_example": 8192-1792,
        "B_status": "prefill complete; request row remains occupied until native completion",
        "example_ledger_after_declared_execution": before_wait,
        "no_free_request_slot": full.frozen(),
        "two_long_prefills_one_native_partial": two_longs.frozen(),
        "tool_wait": "no compute share; native KV kept or reclaimed by native policy",
        "gpu_validated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write a NEW example receipt")
    args = parser.parse_args()
    value = demonstration()
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if args.json:
        with args.json.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
