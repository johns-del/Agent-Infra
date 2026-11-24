#!/usr/bin/env python3
"""Unit & integration tests for Session-Sticky Affinity in dot14_priority_relay.py."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HAS_WEB_STACK = all(
    importlib.util.find_spec(mod) is not None for mod in ("fastapi", "httpx", "uvicorn")
)

ROOT = Path(__file__).resolve().parent
if not (ROOT / "dot14_priority_relay.py").is_file() and (ROOT.parents[1] / "dot14_priority_relay.py").is_file():
    ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

if HAS_WEB_STACK:
    import httpx
    import dot14_priority_relay as relay


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        listen_host="127.0.0.1",
        listen_port=32859,
        router_url="http://router.local",
        dot14_url="http://dot14.local",
        dot14_soft_cap=6,
        dot14_max_token_usage=0.80,
        dot14_max_waiting=8,
        dot12_url="http://dot12.local",
        dot12_soft_cap=4,
        dot12_max_token_usage=0.80,
        dot12_max_waiting=4,
        dot13_url="",
        dot13_soft_cap=0,
        dot13_max_token_usage=0.80,
        dot13_max_waiting=8,
        disable_sticky_affinity=False,
        sticky_burst=4,
        sticky_ttl_s=3600.0,
        sticky_max_sessions=4096,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _mock_client(node_name: str, *, delay_s: float = 0.0, fail_status: int | None = None, sse: bool = False):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, text="ok")
        if request.url.path == "/get_load":
            return httpx.Response(200, json={"token_usage": 0.1, "num_waiting_reqs": 0})
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        if fail_status is not None:
            return httpx.Response(fail_status, json={"error": "simulated"})
        if sse:
            class _SSEStream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield f'data: {{"node":"{node_name}"}}\n\n'.encode()
                    yield b"data: [DONE]\n\n"
            return httpx.Response(200, stream=_SSEStream(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"node": node_name, "choices": [{"message": {"content": "ok"}}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=f"http://{node_name}.local")


@unittest.skipUnless(HAS_WEB_STACK, "fastapi/httpx/uvicorn not installed in this Python env")
class StickyRelayTests(unittest.TestCase):
    def test_extract_sticky_keys_header_and_headerless_and_toolcalls(self):
        b1 = relay.extract_sticky_keys({"X-Orvix-Session-Id": "sess-abc"}, b"not-json", "/v1/chat/completions")
        self.assertEqual(b1.lookup_keys, ("hdr:sess-abc",))
        self.assertEqual(b1.bind_keys, ("hdr:sess-abc",))

        # Short Chinese Turn 1 prompt still binds t1: key; does not lookup on Turn 1
        static_sys_head = "You are Orvix assistant. " + ("Follow deterministic engineering rules. " * 20)
        t1_body = json.dumps({
            "messages": [
                {"role": "system", "content": static_sys_head + "\n<task>initial</task>"},
                {"role": "user", "content": "帮我看下这个报错"},
            ]
        }).encode()
        b_t1 = relay.extract_sticky_keys({}, t1_body, "/v1/chat/completions")
        self.assertEqual(b_t1.lookup_keys, ())
        self.assertEqual(len(b_t1.bind_keys), 1)
        self.assertTrue(b_t1.bind_keys[0].startswith("t1:"))

        # Turn 2 with pure tool_calls assistant message & mutated system tail
        t2_body = json.dumps({
            "messages": [
                {"role": "system", "content": static_sys_head + "\n<task>updated</task>"},
                {"role": "user", "content": "帮我看下这个报错"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_log", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "traceback..."},
            ]
        }).encode()
        b_t2 = relay.extract_sticky_keys({}, t2_body, "/v1/chat/completions")
        self.assertIn(b_t1.bind_keys[0], b_t2.lookup_keys)
        self.assertTrue(any(k.startswith("cv:") for k in b_t2.bind_keys))
        self.assertTrue(any(k.startswith("cvu:") for k in b_t2.bind_keys))

        # Turn 3 even if system[:512] changed completely still matches via cvu: key
        t3_body = json.dumps({
            "messages": [
                {"role": "system", "content": "COMPLETELY DIFFERENT HEADER " * 30},
                {"role": "user", "content": "帮我看下这个报错"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_log", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "traceback..."},
                {"role": "assistant", "content": "Found bug."},
                {"role": "user", "content": "修一下"},
            ]
        }).encode()
        b_t3 = relay.extract_sticky_keys({}, t3_body, "/v1/chat/completions")
        cvu_keys = [k for k in b_t2.bind_keys if k.startswith("cvu:")]
        self.assertEqual(len(cvu_keys), 1)
        self.assertIn(cvu_keys[0], b_t3.lookup_keys)

    def test_affinity_table_ttl_and_lru(self):
        now = [100.0]
        tbl = relay.SessionAffinityTable(max_entries=64, ttl_s=30.0, clock=lambda: now[0])
        tbl.bind(("k1",), "dot12")
        self.assertEqual(tbl.lookup(("k1",)), "dot12")
        now[0] = 135.0
        self.assertIsNone(tbl.lookup(("k1",)))
        self.assertEqual(len(tbl), 0)

        for i in range(80):
            tbl.bind((f"s{i}",), "dot14")
        self.assertEqual(len(tbl), 64)
        self.assertIsNone(tbl.lookup(("s0",)))
        self.assertEqual(tbl.lookup(("s79",)), "dot14")

    def test_8way_reversed_arrival_order_zero_cross_node_drift(self):
        async def run():
            state = relay.PriorityRelayState(_make_args())
            state.tier_map["dot14"].client = _mock_client("dot14", delay_s=0.06)
            state.tier_map["dot12"].client = _mock_client("dot12", delay_s=0.06)
            state.router_client = _mock_client("router", delay_s=0.01)
            for t in state.tiers:
                t.healthy = True
                t.resource_ok = True
            app = relay.build_app(state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://relay.local") as client:
                async def send_turn1(i):
                    body = {
                        "model": "DeepSeek",
                        "messages": [
                            {"role": "system", "content": f"Spec w{i} " + ("rule " * 50)},
                            {"role": "user", "content": f"Turn 1 unique task prompt for worker {i}"},
                        ],
                    }
                    r = await client.post("/v1/chat/completions", json=body)
                    return i, r.json()["node"], r.headers.get("X-Orvix-Relay-Route")

                r1 = await asyncio.gather(*(send_turn1(i) for i in range(8)))
                home = {i: node for i, node, _ in r1}
                self.assertEqual(list(home.values()).count("dot14"), 6)
                self.assertEqual(list(home.values()).count("dot12"), 2)

                # Round 2: dot12 sessions arrive FIRST, then dot14 sessions reversed!
                dot12_ids = [i for i, n in home.items() if n == "dot12"]
                dot14_ids = [i for i, n in home.items() if n == "dot14"]
                order = dot12_ids + list(reversed(dot14_ids))

                async def send_turn2(i, delay):
                    await asyncio.sleep(delay)
                    body = {
                        "model": "DeepSeek",
                        "messages": [
                            {"role": "system", "content": f"Spec w{i} " + ("rule " * 50) + "\n<task>mutated</task>"},
                            {"role": "user", "content": f"Turn 1 unique task prompt for worker {i}"},
                            {"role": "assistant", "content": f"Reply from turn 1 for worker {i}"},
                            {"role": "user", "content": f"Turn 2 follow-up for worker {i}"},
                        ],
                    }
                    r = await client.post("/v1/chat/completions", json=body)
                    return i, r.json()["node"], r.headers.get("X-Orvix-Relay-Route")

                r2 = await asyncio.gather(*(send_turn2(i, idx * 0.005) for idx, i in enumerate(order)))
                for i, node, route in r2:
                    self.assertEqual(node, home[i], f"Worker {i} drifted from {home[i]} to {node}!")
                    self.assertEqual(route, f"{home[i]}-sticky")
                self.assertEqual(state.stats["sticky_routed_total"], 8)
                self.assertEqual(state.stats["dot14_sticky_routed"], 6)
                self.assertEqual(state.stats["dot12_sticky_routed"], 2)
                self.assertEqual(state.stats["sticky_evicted_overload"], 0)
            await state.tier_map["dot14"].client.aclose()
            await state.tier_map["dot12"].client.aclose()
            await state.router_client.aclose()

        asyncio.run(run())

    def test_10way_mixed_warm_and_new_sessions(self):
        """6 warm dot14 + 2 warm dot12 + 2 brand new -> warm stick cleanly, 2 new spill to dot12."""
        async def run():
            state = relay.PriorityRelayState(_make_args())
            state.tier_map["dot14"].client = _mock_client("dot14", delay_s=0.06)
            state.tier_map["dot12"].client = _mock_client("dot12", delay_s=0.06)
            state.router_client = _mock_client("router", delay_s=0.01)
            for t in state.tiers:
                t.healthy = True
                t.resource_ok = True
            for i in range(6):
                state.affinity.bind((f"hdr:old14-{i}",), "dot14")
            for i in range(2):
                state.affinity.bind((f"hdr:old12-{i}",), "dot12")

            app = relay.build_app(state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://relay.local") as client:
                async def call(sid, delay):
                    await asyncio.sleep(delay)
                    r = await client.post(
                        "/v1/chat/completions",
                        headers={"X-Orvix-Session-Id": sid},
                        json={"messages": [{"role": "user", "content": sid}]},
                    )
                    return sid, r.json()["node"], r.headers.get("X-Orvix-Relay-Route")

                tasks = (
                    [call(f"old14-{i}", 0.0) for i in range(6)]
                    + [call(f"old12-{i}", 0.0) for i in range(2)]
                    + [call(f"new-{i}", 0.015) for i in range(2)]
                )
                res = dict((sid, (node, route)) for sid, node, route in await asyncio.gather(*tasks))
                for i in range(6):
                    self.assertEqual(res[f"old14-{i}"], ("dot14", "dot14-sticky"))
                for i in range(2):
                    self.assertEqual(res[f"old12-{i}"], ("dot12", "dot12-sticky"))
                for i in range(2):
                    self.assertEqual(res[f"new-{i}"], ("dot12", "dot12-priority"))
                self.assertEqual(state.tier_map["dot14"].inflight, 0)
                self.assertEqual(state.tier_map["dot12"].inflight, 0)
            await state.tier_map["dot14"].client.aclose()
            await state.tier_map["dot12"].client.aclose()
            await state.router_client.aclose()

        asyncio.run(run())

    def test_sticky_burst_cap_and_failover(self):
        async def run():
            state = relay.PriorityRelayState(_make_args(sticky_burst=2))
            state.tier_map["dot14"].client = _mock_client("dot14")
            state.tier_map["dot12"].client = _mock_client("dot12")
            state.router_client = _mock_client("router")
            for t in state.tiers:
                t.healthy = True
                t.resource_ok = True

            # Case A: home dot12 unhealthy -> evict to dot14 & rebind
            state.affinity.bind(("hdr:sess-x",), "dot12")
            state.tier_map["dot12"].healthy = False
            app = relay.build_app(state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://relay.local") as client:
                r = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Orvix-Session-Id": "sess-x"},
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                self.assertEqual(r.json()["node"], "dot14")
                self.assertEqual(r.headers.get("X-Orvix-Relay-Route"), "dot14-priority")
                self.assertEqual(state.stats["sticky_evicted_overload"], 1)

                # Case B: home dot14 at sticky_cap (6+2=8) -> evict to healthy dot12
                state.tier_map["dot12"].healthy = True
                state.tier_map["dot12"].resource_ok = True
                state.tier_map["dot14"].inflight = 8
                r2 = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Orvix-Session-Id": "sess-x"},
                    json={"messages": [{"role": "user", "content": "hi2"}]},
                )
                self.assertEqual(r2.json()["node"], "dot12")
                self.assertEqual(r2.headers.get("X-Orvix-Relay-Route"), "dot12-priority")
                self.assertEqual(state.stats["sticky_evicted_overload"], 2)
                state.tier_map["dot14"].inflight = 0
            await state.tier_map["dot14"].client.aclose()
            await state.tier_map["dot12"].client.aclose()
            await state.router_client.aclose()

        asyncio.run(run())

    def test_sse_streaming_releases_sticky_slot(self):
        async def run():
            state = relay.PriorityRelayState(_make_args())
            state.tier_map["dot14"].client = _mock_client("dot14", sse=True)
            state.tier_map["dot12"].client = _mock_client("dot12", sse=True)
            state.router_client = _mock_client("router", sse=True)
            for t in state.tiers:
                t.healthy = True
                t.resource_ok = True
            state.affinity.bind(("hdr:sse-1",), "dot12")
            app = relay.build_app(state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://relay.local") as client:
                r = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Orvix-Session-Id": "sse-1"},
                    json={"messages": [{"role": "user", "content": "stream"}], "stream": True},
                )
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.headers.get("X-Orvix-Relay-Route"), "dot12-sticky")
                self.assertIn("dot12", r.text)
                self.assertEqual(state.tier_map["dot12"].inflight, 0)
            await state.tier_map["dot14"].client.aclose()
            await state.tier_map["dot12"].client.aclose()
            await state.router_client.aclose()

        asyncio.run(run())

    def test_equal_mode_strict_1to1_concurrent_and_sequential(self):
        async def run():
            state = relay.PriorityRelayState(
                _make_args(dispatch_mode="equal", dot14_soft_cap=16, dot12_soft_cap=16, sticky_burst=8)
            )
            state.tier_map["dot14"].client = _mock_client("dot14", delay_s=0.05)
            state.tier_map["dot12"].client = _mock_client("dot12", delay_s=0.05)
            state.router_client = _mock_client("router", delay_s=0.01)
            for t in state.tiers:
                t.healthy = True
                t.resource_ok = True

            app = relay.build_app(state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://relay.local") as client:
                st = (await client.get("/orvix_relay_status")).json()
                self.assertEqual(st["dispatch_mode"], "equal")

                # Concurrent C=6 new sessions must split strictly 3:3
                res6 = await asyncio.gather(*(
                    client.post(
                        "/v1/chat/completions",
                        headers={"X-Orvix-Session-Id": f"eq-c6-{i}"},
                        json={"messages": [{"role": "user", "content": f"hello {i}"}]},
                    )
                    for i in range(6)
                ))
                nodes6 = [r.json()["node"] for r in res6]
                self.assertEqual(nodes6.count("dot14"), 3)
                self.assertEqual(nodes6.count("dot12"), 3)
                self.assertTrue(all(r.headers.get("X-Orvix-Relay-Route", "").endswith("-equal") for r in res6))

                # Sequential new sessions (both inflight=0) must strictly alternate 1:1
                seq_nodes = []
                for i in range(4):
                    r = await client.post(
                        "/v1/chat/completions",
                        headers={"X-Orvix-Session-Id": f"eq-seq-{i}"},
                        json={"messages": [{"role": "user", "content": f"seq {i}"}]},
                    )
                    seq_nodes.append(r.json()["node"])
                self.assertEqual(seq_nodes.count("dot14"), 2)
                self.assertEqual(seq_nodes.count("dot12"), 2)
                self.assertEqual(seq_nodes[0], seq_nodes[2])
                self.assertEqual(seq_nodes[1], seq_nodes[3])
                self.assertNotEqual(seq_nodes[0], seq_nodes[1])
                self.assertEqual(state.stats["router_spillover_routed"], 0)

            await state.tier_map["dot14"].client.aclose()
            await state.tier_map["dot12"].client.aclose()
            await state.router_client.aclose()

        asyncio.run(run())

    def test_equal_mode_c16_full_sticky_zero_router_spillover(self):
        async def run():
            state = relay.PriorityRelayState(
                _make_args(dispatch_mode="equal", dot14_soft_cap=16, dot12_soft_cap=16, sticky_burst=8)
            )
            state.tier_map["dot14"].client = _mock_client("dot14", delay_s=0.05)
            state.tier_map["dot12"].client = _mock_client("dot12", delay_s=0.05)
            state.router_client = _mock_client("router", delay_s=0.01)
            for t in state.tiers:
                t.healthy = True
                t.resource_ok = True

            app = relay.build_app(state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://relay.local") as client:
                t1 = await asyncio.gather(*(
                    client.post(
                        "/v1/chat/completions",
                        headers={"X-Orvix-Session-Id": f"eq16-{i}"},
                        json={"messages": [{"role": "user", "content": f"t1-{i}"}]},
                    )
                    for i in range(16)
                ))
                homes = {f"eq16-{i}": t1[i].json()["node"] for i in range(16)}
                self.assertEqual(list(homes.values()).count("dot14"), 8)
                self.assertEqual(list(homes.values()).count("dot12"), 8)

                # Turn 2 in reversed order: 16/16 must stick to their exact home node with 0 router spillover
                t2 = await asyncio.gather(*(
                    client.post(
                        "/v1/chat/completions",
                        headers={"X-Orvix-Session-Id": f"eq16-{i}"},
                        json={"messages": [{"role": "user", "content": f"t2-{i}"}]},
                    )
                    for i in reversed(range(16))
                ))
                for idx_rev, r in enumerate(t2):
                    sid = f"eq16-{15 - idx_rev}"
                    self.assertEqual(r.json()["node"], homes[sid])
                    self.assertEqual(r.headers.get("X-Orvix-Relay-Route"), f"{homes[sid]}-sticky")
                self.assertEqual(state.stats["router_spillover_routed"], 0)
                self.assertEqual(state.stats["sticky_routed_total"], 16)

            await state.tier_map["dot14"].client.aclose()
            await state.tier_map["dot12"].client.aclose()
            await state.router_client.aclose()

        asyncio.run(run())

    def test_equal_mode_failover_and_kv_pressure_rebalance(self):
        async def run():
            state = relay.PriorityRelayState(
                _make_args(dispatch_mode="equal", dot14_soft_cap=16, dot12_soft_cap=16, sticky_burst=8)
            )
            state.tier_map["dot14"].client = _mock_client("dot14")
            state.tier_map["dot12"].client = _mock_client("dot12")
            state.router_client = _mock_client("router")
            for t in state.tiers:
                t.healthy = True
                t.resource_ok = True

            # dot12 under KV pressure -> all new sessions go to dot14
            state.tier_map["dot12"].resource_ok = False
            app = relay.build_app(state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://relay.local") as client:
                for i in range(4):
                    r = await client.post(
                        "/v1/chat/completions",
                        headers={"X-Orvix-Session-Id": f"kv-press-{i}"},
                        json={"messages": [{"role": "user", "content": "hi"}]},
                    )
                    self.assertEqual(r.json()["node"], "dot14")
                    self.assertEqual(r.headers.get("X-Orvix-Relay-Route"), "dot14-equal")

                # dot12 recovers with 0 inflight while dot14 has 5 inflight -> next new session picks dot12
                state.tier_map["dot12"].resource_ok = True
                state.tier_map["dot14"].inflight = 5
                r_bal = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Orvix-Session-Id": "kv-recover"},
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                self.assertEqual(r_bal.json()["node"], "dot12")
                self.assertEqual(r_bal.headers.get("X-Orvix-Relay-Route"), "dot12-equal")
                state.tier_map["dot14"].inflight = 0

            await state.tier_map["dot14"].client.aclose()
            await state.tier_map["dot12"].client.aclose()
            await state.router_client.aclose()

        asyncio.run(run())

    def test_equal_mode_weighted_3to2_concurrent_and_sequential(self):
        async def run():
            state = relay.PriorityRelayState(
                _make_args(
                    dispatch_mode="equal",
                    dot14_soft_cap=16,
                    dot12_soft_cap=16,
                    dot14_weight=3.0,
                    dot12_weight=2.0,
                    sticky_burst=8,
                )
            )
            state.tier_map["dot14"].client = _mock_client("dot14", delay_s=0.05)
            state.tier_map["dot12"].client = _mock_client("dot12", delay_s=0.05)
            state.router_client = _mock_client("router", delay_s=0.01)
            for t in state.tiers:
                t.healthy = True
                t.resource_ok = True

            app = relay.build_app(state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://relay.local") as client:
                # Concurrent C=10 -> strictly 6 on dot14, 4 on dot12 (3:2)
                res10 = await asyncio.gather(*(
                    client.post(
                        "/v1/chat/completions",
                        headers={"X-Orvix-Session-Id": f"w32-c10-{i}"},
                        json={"messages": [{"role": "user", "content": f"hello {i}"}]},
                    )
                    for i in range(10)
                ))
                nodes10 = [r.json()["node"] for r in res10]
                self.assertEqual(nodes10.count("dot14"), 6)
                self.assertEqual(nodes10.count("dot12"), 4)

                # Sequential 5 requests -> strictly 3 on dot14, 2 on dot12
                seq5 = []
                for i in range(5):
                    r = await client.post(
                        "/v1/chat/completions",
                        headers={"X-Orvix-Session-Id": f"w32-seq-{i}"},
                        json={"messages": [{"role": "user", "content": f"s {i}"}]},
                    )
                    seq5.append(r.json()["node"])
                self.assertEqual(seq5.count("dot14"), 3)
                self.assertEqual(seq5.count("dot12"), 2)
                self.assertEqual(state.stats["router_spillover_routed"], 0)

            await state.tier_map["dot14"].client.aclose()
            await state.tier_map["dot12"].client.aclose()
            await state.router_client.aclose()

        asyncio.run(run())

    def test_probe_tier_falls_back_to_prometheus_metrics_kv_guard(self):
        async def run():
            state = relay.PriorityRelayState(
                _make_args(dispatch_mode="equal", dot12_max_token_usage=0.80, dot12_max_waiting=12)
            )

            async def prom_handler(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/health":
                    return httpx.Response(200, text="OK")
                if request.url.path == "/get_load":
                    return httpx.Response(404, json={"detail": "Not Found"})
                if request.url.path == "/metrics":
                    body = (
                        'sglang:token_usage{tp_rank="0"} 0.87\n'
                        'sglang:num_queue_reqs{tp_rank="0"} 2.0\n'
                    )
                    return httpx.Response(200, text=body)
                return httpx.Response(404)

            await state.tier_map["dot12"].client.aclose()
            state.tier_map["dot12"].client = httpx.AsyncClient(
                transport=httpx.MockTransport(prom_handler), base_url="http://dot12.local"
            )
            await state.probe_tier_once(state.tier_map["dot12"])
            self.assertTrue(state.tier_map["dot12"].healthy)
            self.assertFalse(state.tier_map["dot12"].resource_ok)
            self.assertEqual(state.tier_map["dot12"].last_load["source"], "metrics")
            self.assertAlmostEqual(state.tier_map["dot12"].last_load["token_usage"], 0.87, places=2)

            await state.tier_map["dot14"].client.aclose()
            await state.tier_map["dot12"].client.aclose()
            await state.router_client.aclose()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
