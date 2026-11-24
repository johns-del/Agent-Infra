#!/usr/bin/env python3
"""Cache-aware session-sticky + capacity-aware 14 -> 12 -> 13 priority dispatcher layered in front of sglang-router.

Routing policy:
1. Session-Sticky Affinity (enabled by default; disable with --disable-sticky-affinity):
   - Identifies conversations via explicit session headers (X-Orvix-Session-Id / X-Trace-Task-Id /
     X-Trace-Run-Id / X-Session-Id / body.session_id) or, for standard headerless OpenAI clients,
     via stable content fingerprints (first system head + first user message + first assistant reply).
   - Multi-turn follow-ups (or explicit-header sessions) with an existing healthy home node whose
     resource metrics are OK and `inflight < soft_cap + sticky_burst` are routed straight back to
     their home node, preventing cross-node RadixCache / SystemPrefixPatcher eviction.
2. Priority order is dot14 -> dot12 -> dot13 for new sessions (or when a sticky home is unhealthy,
   resource-saturated, or at its sticky burst cap). A request goes directly to the highest-priority
   node that is healthy, below its KV usage / queue threshold, and below its `--<node>-soft-cap`.
3. When all priority tiers reach their soft caps (or any node is degraded), overflow requests
   route through sglang-router (`--policy power_of_two`).
"""
from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import hashlib
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
import uvicorn


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}

MAX_JSON_INSPECT_BYTES = 8 * 1024 * 1024


def _extract_load_metrics(payload) -> tuple[float | None, int]:
    """Extract max (token_usage, waiting_reqs) from SGLang /get_load JSON payload."""
    records = payload if isinstance(payload, list) else ([payload] if isinstance(payload, dict) else [])
    usages: list[float] = []
    max_wait = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        load = item.get("load") if isinstance(item.get("load"), dict) else item
        if isinstance(load, dict):
            for key in ("token_usage", "kv_cache_usage", "cache_usage"):
                val = load.get(key)
                if isinstance(val, (int, float)):
                    usages.append(float(val))
                    break
            for wkey in ("num_waiting_reqs", "waiting_reqs", "queue_size", "num_queue_reqs"):
                wval = load.get(wkey)
                if isinstance(wval, (int, float)):
                    max_wait = max(max_wait, int(wval))
                    break
    return (max(usages) if usages else None), max_wait


def _extract_prom_metrics(text: str) -> tuple[float | None, int]:
    """Extract max (token_usage, waiting_reqs) from SGLang Prometheus /metrics text."""
    usages: list[float] = []
    max_wait = 0
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("sglang:token_usage{") or line.startswith("sglang:token_usage "):
            with suppress(Exception):
                usages.append(float(line.rsplit(" ", 1)[-1]))
        elif line.startswith("sglang:num_queue_reqs{") or line.startswith("sglang:num_queue_reqs "):
            with suppress(Exception):
                max_wait = max(max_wait, int(float(line.rsplit(" ", 1)[-1])))
    return (max(usages) if usages else None), max_wait


def _norm_msg_text(msg: dict, limit: int = 1024) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        acc = 0
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                acc += len(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if isinstance(t, str) and t:
                    parts.append(t)
                    acc += len(t)
            if acc >= limit:
                break
        text = "\n".join(parts)
    else:
        text = ""
    if not text and isinstance(msg.get("tool_calls"), list):
        # First assistant turn might be a pure tool-call turn with empty content
        with suppress(Exception):
            text = json.dumps(msg["tool_calls"][:2], sort_keys=True, separators=(",", ":"))
    return " ".join(text[: limit * 2].split())[:limit]


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:24]


@dataclass(frozen=True)
class StickyKeyBundle:
    """Ordered lookup keys and write-back binding keys extracted from one request."""

    lookup_keys: tuple[str, ...]
    bind_keys: tuple[str, ...]


def extract_sticky_keys(headers: dict[str, str], body: bytes, path: str) -> StickyKeyBundle:
    """Extract session-affinity lookup and bind keys without mutating the request."""
    lower_hdrs = {k.lower(): v for k, v in headers.items()}
    hdr_sess = (
        lower_hdrs.get("x-orvix-session-id")
        or lower_hdrs.get("x-trace-task-id")
        or lower_hdrs.get("x-trace-run-id")
        or lower_hdrs.get("x-session-id")
        or lower_hdrs.get("x-conversation-id")
    )
    if hdr_sess:
        k = f"hdr:{str(hdr_sess).strip()[:128]}"
        return StickyKeyBundle(lookup_keys=(k,), bind_keys=(k,))

    if path != "/v1/chat/completions" or not body or len(body) > MAX_JSON_INSPECT_BYTES:
        return StickyKeyBundle(lookup_keys=(), bind_keys=())

    try:
        obj = json.loads(body)
    except Exception:
        return StickyKeyBundle(lookup_keys=(), bind_keys=())
    if not isinstance(obj, dict):
        return StickyKeyBundle(lookup_keys=(), bind_keys=())

    body_sess = obj.get("session_id")
    if isinstance(body_sess, str) and body_sess.strip():
        k = f"bsid:{body_sess.strip()[:128]}"
        return StickyKeyBundle(lookup_keys=(k,), bind_keys=(k,))

    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return StickyKeyBundle(lookup_keys=(), bind_keys=())

    sys_head = ""
    u1 = ""
    a1 = ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer") and not sys_head:
            sys_head = _norm_msg_text(m, limit=512)
        elif role == "user" and not u1:
            u1 = _norm_msg_text(m, limit=1024)
        elif role == "assistant" and not a1:
            a1 = _norm_msg_text(m, limit=512)
            break

    if not u1:
        return StickyKeyBundle(lookup_keys=(), bind_keys=())

    t1_key = f"t1:{_short_hash(sys_head + chr(0) + u1)}" if u1 else None
    u1_key = f"u1:{_short_hash(u1)}" if len(u1) >= 24 else None

    if a1:
        # Turn >= 2: conversation already has history (and KV cache) on its Turn-1 node.
        conv_key = f"cv:{_short_hash(sys_head + chr(0) + u1 + chr(0) + a1)}"
        conv_nosys_key = f"cvu:{_short_hash(u1 + chr(0) + a1)}" if len(a1) >= 16 else None
        lookups = tuple(k for k in (conv_key, conv_nosys_key, t1_key, u1_key) if k)
        # Bind conversation-specific keys (with and without sys_head, surviving even sys[:512] edits)
        binds = tuple(k for k in (conv_key, conv_nosys_key) if k)
        return StickyKeyBundle(lookup_keys=lookups, bind_keys=binds)

    # Turn 1 (no assistant turn yet): do NOT look up content-only keys so two users sending
    # an identical first prompt stay independent; only bind Turn-1 anchors for Turn 2 to find.
    binds = tuple(k for k in (t1_key, u1_key) if k)
    return StickyKeyBundle(lookup_keys=(), bind_keys=binds)


class SessionAffinityTable:
    """Bounded TTL-LRU table mapping session keys to tier names."""

    def __init__(self, *, max_entries: int = 4096, ttl_s: float = 3600.0, clock: Callable[[], float] = time.monotonic):
        self.max_entries = max(64, int(max_entries))
        self.ttl_s = max(10.0, float(ttl_s))
        self._clock = clock
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self.ttl_s
        while self._entries:
            oldest_key, (_, ts) = next(iter(self._entries.items()))
            if ts < cutoff:
                self._entries.pop(oldest_key, None)
            else:
                break

    def lookup(self, keys: tuple[str, ...]) -> str | None:
        if not keys:
            return None
        now = self._clock()
        self._evict_expired(now)
        for k in keys:
            item = self._entries.get(k)
            if item is None:
                continue
            tier_name, ts = item
            if now - ts > self.ttl_s:
                self._entries.pop(k, None)
                continue
            self._entries.move_to_end(k)
            self._entries[k] = (tier_name, now)
            return tier_name
        return None

    def bind(self, keys: tuple[str, ...], tier_name: str) -> None:
        if not keys or not tier_name:
            return
        now = self._clock()
        self._evict_expired(now)
        for k in keys:
            self._entries[k] = (tier_name, now)
            self._entries.move_to_end(k)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class TierState:
    """Live health, capacity limits, and in-flight state for one priority tier."""

    name: str
    url: str
    soft_cap: int
    sticky_burst: int
    max_token_usage: float
    max_waiting: int
    client: httpx.AsyncClient
    weight: float = 1.0
    healthy: bool = False
    resource_ok: bool = False
    inflight: int = 0
    last_load: dict | list | None = None

    @property
    def sticky_cap(self) -> int:
        return self.soft_cap + max(0, self.sticky_burst)

    def to_status_dict(self) -> dict:
        return {
            "url": self.url,
            "weight": self.weight,
            "healthy": self.healthy,
            "resource_ok": self.resource_ok,
            "inflight": self.inflight,
            "soft_cap": self.soft_cap,
            "sticky_cap": self.sticky_cap,
            "max_token_usage": self.max_token_usage,
            "max_waiting": self.max_waiting,
            "last_load": self.last_load,
        }


def _claim_tier_slot(tier: TierState) -> Callable[[], None]:
    """Synchronously claim one in-flight slot and return an idempotent releaser."""
    tier.inflight += 1
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            tier.inflight = max(0, tier.inflight - 1)

    return release


class PriorityRelayState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dispatch_mode = str(getattr(args, "dispatch_mode", "priority") or "priority").strip().lower()
        if self.dispatch_mode not in ("priority", "equal"):
            raise ValueError(f"Unsupported dispatch_mode: {self.dispatch_mode}")
        self.sticky_enabled = not getattr(args, "disable_sticky_affinity", False)
        sticky_burst = max(0, int(getattr(args, "sticky_burst", 4)))
        self._rr_cursor: int = 0
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=256)
        timeout = httpx.Timeout(3600.0, connect=10.0)
        tier_specs = (
            ("dot14", args.dot14_url, args.dot14_soft_cap, args.dot14_max_token_usage, args.dot14_max_waiting, float(getattr(args, "dot14_weight", 1.0))),
            ("dot12", args.dot12_url, args.dot12_soft_cap, args.dot12_max_token_usage, args.dot12_max_waiting, float(getattr(args, "dot12_weight", 1.0))),
            ("dot13", args.dot13_url, args.dot13_soft_cap, args.dot13_max_token_usage, args.dot13_max_waiting, float(getattr(args, "dot13_weight", 1.0))),
        )
        self.tiers: list[TierState] = [
            TierState(
                name=name,
                url=url.rstrip("/"),
                soft_cap=soft_cap,
                sticky_burst=sticky_burst,
                max_token_usage=max_usage,
                max_waiting=max_wait,
                weight=max(0.1, weight),
                client=httpx.AsyncClient(base_url=url.rstrip("/"), limits=limits, timeout=timeout, trust_env=False),
            )
            for name, url, soft_cap, max_usage, max_wait, weight in tier_specs
            if url and soft_cap > 0
        ]
        self.tier_map: dict[str, TierState] = {t.name: t for t in self.tiers}
        self._swrr_cw: dict[str, float] = {t.name: 0.0 for t in self.tiers}
        self.affinity = SessionAffinityTable(
            max_entries=int(getattr(args, "sticky_max_sessions", 4096)),
            ttl_s=float(getattr(args, "sticky_ttl_s", 3600.0)),
        )
        self.router_client = httpx.AsyncClient(
            base_url=args.router_url.rstrip("/"), limits=limits, timeout=timeout, trust_env=False
        )
        self.stats = {
            "requests_total": 0,
            "dot14_priority_routed": 0,
            "dot12_priority_routed": 0,
            "dot13_priority_routed": 0,
            "dot14_equal_routed": 0,
            "dot12_equal_routed": 0,
            "dot13_equal_routed": 0,
            "equal_routed_total": 0,
            "dot14_sticky_routed": 0,
            "dot12_sticky_routed": 0,
            "dot13_sticky_routed": 0,
            "sticky_routed_total": 0,
            "sticky_evicted_overload": 0,
            "router_spillover_routed": 0,
            "dot14_failovers": 0,
            "dot12_failovers": 0,
            "dot13_failovers": 0,
            "other_proxied": 0,
            "errors": 0,
        }
        self._mon_task: asyncio.Task | None = None

    def select_equal_tier(self, excluded: set[str]) -> TierState | None:
        """Pick an eligible tier with minimum inflight/weight, tie-breaking smoothly by configured weights (e.g. 3:2)."""
        eligible = [
            t for t in self.tiers
            if t.name not in excluded and t.healthy and t.resource_ok and t.inflight < t.soft_cap
        ]
        if not eligible:
            return None
        # Advance smooth weighted round-robin credits for eligible peers
        total_w = sum(t.weight for t in eligible)
        for t in eligible:
            self._swrr_cw[t.name] = self._swrr_cw.get(t.name, 0.0) + t.weight
        # Primary: lowest weighted inflight ratio; tie-break: highest smooth WRR credit
        best_tier = min(
            eligible,
            key=lambda t: (round(t.inflight / t.weight, 6), -self._swrr_cw.get(t.name, 0.0)),
        )
        self._swrr_cw[best_tier.name] -= total_w
        return best_tier

    async def probe_tier_once(self, tier: TierState) -> None:
        try:
            h = await tier.client.get("/health", timeout=2.5)
            healthy = h.status_code == 200
        except Exception:
            healthy = False

        if not healthy:
            tier.healthy = False
            tier.resource_ok = False
            return

        resource_ok = True
        load_snap = None
        max_usage: float | None = None
        max_wait = 0
        with suppress(Exception):
            lr = await tier.client.get("/get_load", timeout=2.0)
            if lr.status_code == 200:
                load_snap = lr.json()
                max_usage, max_wait = _extract_load_metrics(load_snap)
        if max_usage is None:
            with suppress(Exception):
                mr = await tier.client.get("/metrics", timeout=2.0)
                if mr.status_code == 200:
                    max_usage, max_wait = _extract_prom_metrics(mr.text)
                    if max_usage is not None or max_wait > 0:
                        load_snap = {
                            "token_usage": round(max_usage, 4) if max_usage is not None else None,
                            "num_waiting_reqs": max_wait,
                            "source": "metrics",
                        }
        if max_usage is not None:
            resource_ok = (max_usage < tier.max_token_usage) and (max_wait <= tier.max_waiting)
        elif max_wait > tier.max_waiting:
            resource_ok = False
        tier.healthy = True
        tier.resource_ok = resource_ok
        if load_snap is not None:
            tier.last_load = load_snap

    async def probe_all_once(self) -> None:
        if self.tiers:
            await asyncio.gather(*(self.probe_tier_once(t) for t in self.tiers))

    async def monitor_loop(self) -> None:
        while True:
            with suppress(Exception):
                await self.probe_all_once()
            await asyncio.sleep(3.0)


def _filtered_headers(raw_headers) -> dict[str, str]:
    out = {k: v for k, v in raw_headers.items() if k.lower() not in HOP_BY_HOP}
    out["accept-encoding"] = "identity"
    return out


def build_app(state: PriorityRelayState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await state.probe_all_once()
        state._mon_task = asyncio.create_task(state.monitor_loop())
        try:
            yield
        finally:
            if state._mon_task and not state._mon_task.done():
                state._mon_task.cancel()
                with suppress(asyncio.CancelledError):
                    await state._mon_task
            for tier in state.tiers:
                await tier.client.aclose()
            await state.router_client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/orvix_relay_status")
    async def relay_status():
        dot14 = state.tier_map.get("dot14")
        return {
            "dispatch_mode": state.dispatch_mode,
            "priority_order": [t.name for t in state.tiers],
            "sticky_affinity": {
                "enabled": state.sticky_enabled,
                "tracked_keys": len(state.affinity),
                "max_entries": state.affinity.max_entries,
                "ttl_s": state.affinity.ttl_s,
                "sticky_burst": int(getattr(state.args, "sticky_burst", 4)),
            },
            "dot14_url": state.args.dot14_url,
            "router_url": state.args.router_url,
            "dot14_healthy": dot14.healthy if dot14 else False,
            "dot14_resource_ok": dot14.resource_ok if dot14 else False,
            "dot14_inflight": dot14.inflight if dot14 else 0,
            "dot14_soft_cap": dot14.soft_cap if dot14 else 0,
            "dot14_max_token_usage": dot14.max_token_usage if dot14 else 0.0,
            "last_dot14_load": dot14.last_load if dot14 else None,
            "tiers": {t.name: t.to_status_dict() for t in state.tiers},
            "stats": state.stats,
        }

    async def _forward(
        client: httpx.AsyncClient,
        method: str,
        path: str,
        query_params,
        headers: dict[str, str],
        body: bytes,
        route_tag: str,
        *,
        on_release: Callable[[], None] | None = None,
    ) -> Response:
        req = client.build_request(
            method,
            path,
            params=query_params,
            headers=headers,
            content=body if body else None,
        )
        upstream = await client.send(req, stream=True)
        if on_release is not None and upstream.status_code >= 500:
            await upstream.aclose()
            raise httpx.HTTPStatusError(
                f"{route_tag} returned {upstream.status_code}",
                request=req,
                response=upstream,
            )

        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
        resp_headers["X-Orvix-Relay-Route"] = route_tag
        media_type = upstream.headers.get("content-type")

        if "text/event-stream" in (media_type or "").lower():
            async def sse_gen():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    with suppress(Exception):
                        await upstream.aclose()
                    if on_release is not None:
                        on_release()

            return StreamingResponse(
                sse_gen(),
                status_code=upstream.status_code,
                headers=resp_headers,
                media_type=media_type,
            )

        try:
            raw = await upstream.aread()
        finally:
            with suppress(Exception):
                await upstream.aclose()
            if on_release is not None:
                on_release()
        return Response(content=raw, status_code=upstream.status_code, headers=resp_headers, media_type=media_type)

    async def _handle_inference(request: Request, path: str):
        state.stats["requests_total"] += 1
        body = await request.body()
        headers = _filtered_headers(request.headers)

        key_bundle = (
            extract_sticky_keys(headers, body, path)
            if state.sticky_enabled
            else StickyKeyBundle(lookup_keys=(), bind_keys=())
        )
        tried_sticky_name: str | None = None

        # Step 1: Session-sticky fast path for existing multi-turn / header-tracked sessions
        if state.sticky_enabled and key_bundle.lookup_keys:
            home_name = state.affinity.lookup(key_bundle.lookup_keys)
            if home_name is not None:
                home_tier = state.tier_map.get(home_name)
                if home_tier is not None:
                    if home_tier.healthy and home_tier.resource_ok and home_tier.inflight < home_tier.sticky_cap:
                        tried_sticky_name = home_tier.name
                        release = _claim_tier_slot(home_tier)
                        state.stats[f"{home_tier.name}_sticky_routed"] += 1
                        state.stats["sticky_routed_total"] += 1
                        if key_bundle.bind_keys:
                            state.affinity.bind(key_bundle.bind_keys, home_tier.name)
                        try:
                            return await _forward(
                                home_tier.client,
                                request.method,
                                path,
                                request.query_params,
                                headers,
                                body,
                                f"{home_tier.name}-sticky",
                                on_release=release,
                            )
                        except (httpx.HTTPError, httpx.HTTPStatusError):
                            release()
                            home_tier.healthy = False
                            home_tier.resource_ok = False
                            state.stats[f"{home_tier.name}_failovers"] += 1
                    else:
                        state.stats["sticky_evicted_overload"] += 1

        # Step 2: New-session / fallback dispatch (equal least-inflight 1:1 OR priority order)
        if state.dispatch_mode == "equal":
            tried_equal: set[str] = {tried_sticky_name} if tried_sticky_name else set()
            while True:
                tier = state.select_equal_tier(tried_equal)
                if tier is None:
                    break
                tried_equal.add(tier.name)
                release = _claim_tier_slot(tier)
                state.stats[f"{tier.name}_equal_routed"] += 1
                state.stats["equal_routed_total"] += 1
                if state.sticky_enabled and key_bundle.bind_keys:
                    state.affinity.bind(key_bundle.bind_keys, tier.name)
                try:
                    return await _forward(
                        tier.client,
                        request.method,
                        path,
                        request.query_params,
                        headers,
                        body,
                        f"{tier.name}-equal",
                        on_release=release,
                    )
                except (httpx.HTTPError, httpx.HTTPStatusError):
                    release()
                    tier.healthy = False
                    tier.resource_ok = False
                    state.stats[f"{tier.name}_failovers"] += 1
        else:
            for tier in state.tiers:
                if tier.name == tried_sticky_name:
                    continue
                if tier.healthy and tier.resource_ok and tier.inflight < tier.soft_cap:
                    release = _claim_tier_slot(tier)
                    state.stats[f"{tier.name}_priority_routed"] += 1
                    if state.sticky_enabled and key_bundle.bind_keys:
                        state.affinity.bind(key_bundle.bind_keys, tier.name)
                    try:
                        return await _forward(
                            tier.client,
                            request.method,
                            path,
                            request.query_params,
                            headers,
                            body,
                            f"{tier.name}-priority",
                            on_release=release,
                        )
                    except (httpx.HTTPError, httpx.HTTPStatusError):
                        release()
                        tier.healthy = False
                        tier.resource_ok = False
                        state.stats[f"{tier.name}_failovers"] += 1

        # Step 3: Spillover to sglang-router (--policy power_of_two)
        state.stats["router_spillover_routed"] += 1
        try:
            return await _forward(
                state.router_client,
                request.method,
                path,
                request.query_params,
                headers,
                body,
                "sglang-router",
            )
        except httpx.HTTPError as exc:
            state.stats["errors"] += 1
            return JSONResponse({"error": "upstream_router_unavailable", "detail": str(exc)}, status_code=503)

    @app.post("/v1/chat/completions")
    async def relay_chat(request: Request):
        return await _handle_inference(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def relay_completions(request: Request):
        return await _handle_inference(request, "/v1/completions")

    @app.post("/generate")
    async def relay_generate(request: Request):
        return await _handle_inference(request, "/generate")

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
    async def catch_all(request: Request, full_path: str):
        state.stats["other_proxied"] += 1
        body = await request.body()
        headers = _filtered_headers(request.headers)
        try:
            return await _forward(
                state.router_client,
                request.method,
                "/" + full_path,
                request.query_params,
                headers,
                body,
                "sglang-router",
            )
        except httpx.HTTPError as exc:
            state.stats["errors"] += 1
            return JSONResponse({"error": "upstream_router_unavailable", "detail": str(exc)}, status_code=503)

    return app


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=32859)
    ap.add_argument("--router-url", default="http://127.0.0.1:32860")
    ap.add_argument("--dot14-url", default="http://node14.example.invalid:32683")
    ap.add_argument("--dot14-soft-cap", type=int, default=6)
    ap.add_argument("--dot14-max-token-usage", type=float, default=0.80)
    ap.add_argument("--dot14-max-waiting", type=int, default=8)
    ap.add_argument("--dot14-weight", type=float, default=1.0)
    ap.add_argument("--dot12-url", default="http://node12.example.invalid:32683")
    ap.add_argument("--dot12-soft-cap", type=int, default=4)
    ap.add_argument("--dot12-max-token-usage", type=float, default=0.80)
    ap.add_argument("--dot12-max-waiting", type=int, default=4)
    ap.add_argument("--dot12-weight", type=float, default=1.0)
    ap.add_argument("--dot13-url", default="http://127.0.0.1:32683")
    ap.add_argument("--dot13-soft-cap", type=int, default=6)
    ap.add_argument("--dot13-max-token-usage", type=float, default=0.80)
    ap.add_argument("--dot13-max-waiting", type=int, default=8)
    ap.add_argument("--dot13-weight", type=float, default=1.0)
    ap.add_argument("--dispatch-mode", choices=("priority", "equal"), default="priority")
    ap.add_argument("--disable-sticky-affinity", action="store_true", default=False)
    ap.add_argument("--sticky-burst", type=int, default=4)
    ap.add_argument("--sticky-ttl-s", type=float, default=3600.0)
    ap.add_argument("--sticky-max-sessions", type=int, default=4096)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    state = PriorityRelayState(args)
    uvicorn.run(build_app(state), host=args.listen_host, port=args.listen_port, access_log=False)


if __name__ == "__main__":
    main()
