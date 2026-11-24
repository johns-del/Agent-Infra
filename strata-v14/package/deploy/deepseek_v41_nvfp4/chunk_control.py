"""Inspect/activate v5 on ONE private loopback engine; optionally supply its clock.

Do not run alongside ablate/compare or another controller. Administrative
control is not a public API. Initial activation requires a drained engine.
The heartbeat updates only the common logical clock after activation.
"""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import httpx
from chunk_fair_core import ChunkConfig
from compare_dev_replay import engine_state, verify_configuration, KEY
from run_unified_campaign import DEFAULT


def private_endpoint(value):
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise argparse.ArgumentTypeError("use a private loopback endpoint (or a local SSH tunnel)")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("endpoint must be a loopback origin without credentials/path")
    return value


async def update(client, payload):
    response = await client.post("/set_internal_state", json={"server_args": {}, KEY: payload})
    response.raise_for_status()
    if response.json() != [True]:
        raise RuntimeError("engine rejected v5 control; drain requests / inspect compatibility")


async def run(args):
    headers = {}
    key = os.environ.get("ORVIX_ENGINE_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    async with httpx.AsyncClient(base_url=args.endpoint, headers=headers, trust_env=False, timeout=30.) as client:
        _, state = await engine_state(client)
        if state.get("adapter_version") != "orvix-dsv41-fair-chunk-v5":
            raise RuntimeError("v5 private runtime required")
        identity = state.get("runtime_identity", {})
        if not identity.get("verified"):
            raise RuntimeError("unverified runtime; use prepare-runtime")
        start = time.monotonic()
        base = max(start, float(state.get("clock", state.get("now", 0.))))
        if args.apply:
            config = ChunkConfig(**json.loads(args.apply.read_text())).frozen()
            await update(client, {"now": base, "enabled": False, "parameters": state["parameters"],
                                  "knobs": DEFAULT, "cache": {"mode": "observe"},
                                  "prefill": {"mode": "off"}, "chunk_fair": config})
            _, state = await engine_state(client)
            verify_configuration(state, False, state["parameters"], DEFAULT,
                                 {"mode": "observe"}, {"mode": "off"}, config)
        print(json.dumps({"adapter_version": state["adapter_version"],
                          "runtime_fingerprint": identity["python_fingerprint"],
                          "chunk_fair": state.get("chunk_fair")}, ensure_ascii=False, indent=2), flush=True)
        if args.heartbeat:
            if state.get("chunk_fair", {}).get("config", {}).get("mode", "off") == "off":
                raise ValueError("activate a v5 policy before starting its clock")
            while args.seconds is None or time.monotonic() - start < args.seconds:
                await asyncio.sleep(args.interval)
                # Native broadcast applies the SAME number on all TP ranks.
                await update(client, {"now": base + time.monotonic() - start,
                                      "enabled": False, "parameters": state["parameters"],
                                      "knobs": DEFAULT})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=private_endpoint, default="http://127.0.0.1:32781")
    parser.add_argument("--apply", type=Path, help="policy JSON; missing means read-only")
    parser.add_argument("--heartbeat", action="store_true", help="run one authoritative clock owner")
    parser.add_argument("--interval", type=float, default=.25)
    parser.add_argument("--seconds", type=float, help="optional finite clock duration; does not disable policy on exit")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or not .05 <= args.interval <= 10:
        parser.error("interval must be in [0.05, 10] seconds")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("seconds must be positive and finite")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Clock stopped; policy is NOT reset. Stop/drain requests before deactivation; do not rely on stale age guards.")


if __name__ == "__main__":
    main()
