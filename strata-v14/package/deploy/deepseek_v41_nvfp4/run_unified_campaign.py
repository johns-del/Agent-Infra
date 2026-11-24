"""Screen and repeat scheduling policies on the private eight-GPU engine."""

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import time

import httpx

from compare_dev_replay import ENDPOINT, engine_state, p95, payload, post_control, read_inputs, trial
from unified_core import Parameters


DEFAULT = {"native": "fcfs", "min_candidates": 2, "interleave": 0}
METHODS = {
    "B0": (False, DEFAULT),
    "SP": (False, DEFAULT | {"native": "shortest-prefill-first"}),
    "U": (True, DEFAULT),
    "U4": (True, DEFAULT | {"min_candidates": 4}),
    "I1": (False, DEFAULT | {"interleave": 1}),
    "U4-I1": (True, DEFAULT | {"min_candidates": 4, "interleave": 1}),
}
PARAMS = Parameters(theta_fg=20, theta_bg=60, prefill_tokens_per_s=11500,
                    decode_tokens_per_s=250, uniform_output_tokens=197,
                    levers={"interleave": False, "short_first": False, "latch_reset": False}).frozen()
BUILD_SUFFIXES = {".so", ".cubin", ".hsaco", ".fatbin", ".ptx"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def build_files(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
            for d in (root / "engine/home", root / "engine/cache")
            for p in d.rglob("*") if p.suffix in BUILD_SUFFIXES and p.is_file()}


def write(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def initial_requests(graphs, prompts, count):
    by_graph = sorted(graphs, key=lambda g: sha256(g["graph_id"].encode()).digest())
    selected = []
    for graph in by_graph[:count]:
        first = min((n for n in graph["nodes"] if n["kind"] == "model"), key=lambda n: n["start_s"])
        selected.append((graph["user_id"], prompts[(graph["graph_id"], first["id"])]))
    return selected


async def burst(client, method, rows, *, flush=True, output_cap=None):
    enabled, knobs = METHODS[method]
    if flush:
        (await client.post("/flush_cache")).raise_for_status()
    await post_control(client, enabled, PARAMS, knobs, reset=True,
                       cache={"mode": "observe"}, prefill={"mode": "off"})
    _, state = await engine_state(client)
    if state["enabled"] is not enabled or any(state["knobs"].get(k) != v for k, v in knobs.items()):
        raise ValueError("burst policy was not acknowledged")

    peak_running, peak_waiting = 0, 0

    async def clock():
        nonlocal peak_running, peak_waiting
        tick = 0
        while True:
            await asyncio.sleep(.25)
            await post_control(client, enabled, PARAMS, knobs,
                               cache={"mode": "observe"}, prefill={"mode": "off"})
            tick += 1
            if tick % 4 == 0:
                sample = await client.get("/v1/loads")
                sample.raise_for_status()
                load = sample.json()["loads"][0]
                peak_running = max(peak_running, load["num_running_reqs"])
                peak_waiting = max(peak_waiting, load["num_waiting_reqs"])

    async def call(family, row):
        if output_cap is not None:
            row = row | {"output_budget": min(row["output_budget"], output_cap)}
        start = time.monotonic()
        resp = await client.post("/generate", json=payload(row, family, "foreground", start))
        resp.raise_for_status()
        value = resp.json()
        info = value["meta_info"]
        if info["completion_tokens"] != row["output_budget"]:
            raise ValueError("burst output length differs")
        queue = next((info[k] for k in ("scheduler_queue_time", "queue_time_s", "queue_time")
                      if isinstance(info.get(k), (int, float))), None)
        return {"elapsed_s": time.monotonic() - start, "output_tokens": info["completion_tokens"],
                "queue_s": queue, "cached_tokens": info.get("cached_tokens"),
                "verify_ct": info.get("spec_verify_ct"),
                "output_hash": sha256(json.dumps(value.get("output_ids", value.get("text"))).encode()).hexdigest()}

    start = time.monotonic()
    task = asyncio.ensure_future(asyncio.gather(*(call(family, row) for family, row in rows)))
    ticker = asyncio.create_task(clock())
    try:
        done, _ = await asyncio.wait((task, ticker), return_when=asyncio.FIRST_COMPLETED)
        for ended in done:
            ended.result()
        calls = task.result()
    finally:
        for live in (task, ticker):
            if not live.done():
                live.cancel()
        await asyncio.gather(task, ticker, return_exceptions=True)
    wall = time.monotonic() - start
    _, after = await engine_state(client)
    grouped = defaultdict(list)
    for (family, _), record in zip(rows, calls):
        grouped[family].append(record["elapsed_s"])
    return {
        "method": method, "requests": len(rows), "wall_s": wall,
        "request_p50_s": median(row["elapsed_s"] for row in calls),
        "request_p95_s": p95(row["elapsed_s"] for row in calls),
        "worst_family_mean_s": max(sum(v) / len(v) for v in grouped.values()),
        "output_tokens_per_s": sum(r["output_tokens"] for r in calls) / wall,
        "peak_running_sampled": peak_running, "peak_waiting_sampled": peak_waiting,
        "reordered_cycles": after["counts"].get("reordered_cycles", 0) -
                            state["counts"].get("reordered_cycles", 0),
        "calls": calls, "knobs": knobs,
    }


class Campaign:
    def __init__(self, root, client, graphs, prompts, deadline):
        self.root, self.client, self.graphs, self.prompts = root, client, graphs, prompts
        self.deadline = deadline
        self.out = root / "results"
        self.out.mkdir(mode=0o700)
        self.screens = []

    def remaining(self):
        return self.deadline - time.time()

    async def measure(self, name, action):
        before = build_files(self.root)
        started = utc()
        print(started, "BEGIN", name, flush=True)
        result = await action()
        after = build_files(self.root)
        changes = [p for p, v in after.items() if before.get(p) != v]
        result.update(name=name, started_utc=started, ended_utc=utc(), new_build_files=changes)
        write(self.out / (name + ".json"), result)
        print(result["ended_utc"], "DONE", name,
              round(result["wall_s"], 2), len(changes), flush=True)
        return result

    async def screen(self):
        rows = initial_requests(self.graphs, self.prompts, 48)
        await self.measure("warmup-48", lambda: burst(self.client, "B0", rows, output_cap=96))
        for rep in (1, 2):
            for offered in (16, 32, 48):
                choices = list(METHODS)
                if rep == 2:
                    choices.reverse()
                for method in choices:
                    if self.remaining() < 7800:
                        return
                    name = f"burst-{offered}-r{rep}-{method}"
                    result = await self.measure(name, lambda: burst(self.client, method, rows[:offered]))
                    self.screens.append((offered, rep, method, result))

    def shortlist(self):
        rows = {(n, r, m): v for n, r, m, v in self.screens}
        scores = {}
        for method in METHODS:
            if method == "B0":
                continue
            ratios = []
            for n in (32, 48):
                for r in (1, 2):
                    b, u = rows.get((n, r, "B0")), rows.get((n, r, method))
                    if b and u and not (b["new_build_files"] or u["new_build_files"]):
                        ratios.append({
                            "p95": u["request_p95_s"] / b["request_p95_s"],
                            "wall": u["wall_s"] / b["wall_s"],
                            "worst_family": u["worst_family_mean_s"] / b["worst_family_mean_s"],
                        })
            if len(ratios) >= 2:
                scores[method] = {k: median(row[k] for row in ratios) for k in ratios[0]} | {
                    "qualified_pairs": len(ratios)}
        eligible = [m for m in scores if scores[m]["wall"] <= 1.05 and scores[m]["worst_family"] <= 1.10]
        choice = min(eligible, key=lambda m: scores[m]["p95"]) if eligible else "U"
        write(self.out / "shortlist.json", {"selected": choice, "burst_median_ratios": scores})
        return choice

    async def full_pair(self, n, method, number, reversed_order, budget):
        if self.remaining() < budget:
            print(utc(), "SKIP full", n, method, number, "remaining", int(self.remaining()), flush=True)
            return False
        for side in ((method, "B0") if reversed_order else ("B0", method)):
            enabled, knobs = METHODS[side]
            name = f"full-{n}-{method}-r{number}-{side}"
            async def replay():
                result = await trial(self.client, enabled, PARAMS, self.graphs, self.prompts, n, knobs)
                return result | {"method": side, "knobs": knobs, "offered_graph_concurrency": n}
            await self.measure(name, replay)
        return True

    async def run(self):
        await self.screen()
        pick = self.shortlist()
        print(utc(), "FINALIST", pick, flush=True)
        # Recheck the original result, then validate the finalist at two high
        # offered loads. Reverse the second pair to control method order.
        conditions = [(32, "U", 3300), (48, pick, 3300)]
        if pick != "U":
            conditions.append((32, pick, 3300))
        conditions.append((16, pick, 4100))
        for n, method, budget in conditions:
            for number in (1, 2):
                if not await self.full_pair(n, method, number, number == 2, budget):
                    break
        # Use remaining time for more high-pressure evidence.
        for number in (3, 4):
            if not await self.full_pair(48, pick, number, number % 2 == 0, 3300):
                break


async def run(args):
    async with httpx.AsyncClient(base_url=ENDPOINT, timeout=1800, trust_env=False) as client:
        engine, _ = await engine_state(client)
        graphs, prompts = read_inputs(args.root / "input", engine["context_length"])
        campaign = Campaign(args.root, client, graphs, prompts, args.deadline)
        async with asyncio.timeout(max(0, args.deadline - time.time())):
            await campaign.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("deadline", type=float)
    asyncio.run(run(parser.parse_args()))
