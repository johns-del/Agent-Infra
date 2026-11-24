#!/usr/bin/env python3
"""Per-node T1+Ours scheduler controller & lightweight session-aware OpenAI/SGLang proxy (:32683).

Responsibilities:
1. Activates and maintains our 3-pillar policy (`IntervalConfig.ours()` + `PrefillConfig(V13_PREFILL_DEFAULTS)`)
   via `/set_internal_state` on the local SGLang engine (`127.0.0.1:32681`).
2. Runs a 0.25s monotonic logical-clock heartbeat delivering causal lifecycle events
   (`tool_start`, `tool_end`, `request_ready`), and re-applies configuration automatically if the
   engine restarts.
3. Proxies OpenAI/SGLang requests from localhost and `.13` (`sglang-router`) to `127.0.0.1:32681`,
   inferring stable multi-turn conversation `session_id`s, attaching `orvix_unified` metadata,
   and observing tool-call boundaries cleanly without mutating client response payloads.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
import uvicorn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _norm_text_for_hash(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("type") or ""))
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        text = str(content or "")
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        text += "|" + json.dumps(tool_calls, sort_keys=True, separators=(",", ":"), ensure_ascii=False)[:512]
    return text


def _msg_digest(part: str) -> str:
    return sha256(part.encode("utf-8", errors="ignore")).hexdigest()[:20]


MODEL_ALIASES = {
    "ds",
    "DeepSeek",
    "deepseek",
    "deepseek-v41-flash-nvfp4",
    "deepseek-v41-flash-nvfp4-t1",
    "deepseek-v41-flash-nvfp4-dot12",
    "deepseek-v41-flash-nvfp4-dot13",
    "deepseek-v41-flash-unified-dev65",
}

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


class OursNodeService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        pkg_root = args.package_dir.resolve()
        for p in (pkg_root, pkg_root / "gateway/src/trace_collector", pkg_root / "deploy/deepseek_v41_nvfp4"):
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)

        from compare_dev_replay import KEY, verify_configuration  # noqa: E402
        from causal_prefix_stabilizer import CausalPrefixStabilizer  # noqa: E402
        from interval_core import IntervalConfig, validate_events  # noqa: E402
        from run_ablation import DEFAULT as KNOB_DEFAULT, PARAMS as BASE_PARAMS, V13_PREFILL_DEFAULTS  # noqa: E402
        from unified_prefill import PrefillConfig  # noqa: E402

        here_dir = str(Path(__file__).resolve().parent)
        if here_dir not in sys.path:
            sys.path.insert(0, here_dir)
        from system_prefix_patcher import SystemPrefixPatcher  # noqa: E402

        self.KEY = KEY
        self.verify_configuration = verify_configuration
        self.validate_events = validate_events
        self.gen_stabilizer = CausalPrefixStabilizer(enabled=True)
        self.system_patcher = SystemPrefixPatcher(
            enabled=getattr(args, "enable_system_prefix_patcher", True),
        )

        self.enabled = False
        self.params = dict(BASE_PARAMS) | {
            "fairness_mode": "legacy",
            "cost_model": "prefill_uniform",
            "fairness_slack_s": 2.0,
            "candidate_window_k": 32,
            "age_guard_quota": 1,
        }
        self.knobs = dict(KNOB_DEFAULT)
        self.cache_cfg = {
            "mode": "observe",
            "min_idle_s": 1.0,
            "require_native_rejection": False,
        }
        self.prefill_cfg = asdict(PrefillConfig(**V13_PREFILL_DEFAULTS))
        self.interval_cfg = asdict(IntervalConfig.ours().validate())
        self.chunk_cfg = {"mode": "off"}
        self.multi_cfg = {"mode": "off"}

        self.allowed_clients = {ip.strip() for ip in args.allowed_clients.split(",") if ip.strip()}
        self.control_lock = asyncio.Lock()
        self.mono_origin = time.monotonic()
        self.clock_base = 0.0
        self.last_sent_now = -1.0
        self.engine_epoch_clock = -1.0
        self.ours_verified = False
        self.resolved_max_model_len: int | None = None
        self.last_error: str | None = None

        self.pending_events: list[dict] = []
        self.sessions: OrderedDict[str, dict] = OrderedDict()
        self.conv_index: OrderedDict[tuple[str, str], str] = OrderedDict()
        self.stats = {
            "requests_total": 0,
            "chat_requests": 0,
            "generate_requests": 0,
            "completions_requests": 0,
            "tool_starts_emitted": 0,
            "tool_ends_emitted": 0,
            "request_ready_emitted": 0,
            "control_updates": 0,
            "control_errors": 0,
            "reconfigurations": 0,
        }
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=256)
        timeout = httpx.Timeout(3600.0, connect=10.0)
        self.client = httpx.AsyncClient(base_url=args.engine_url.rstrip("/"), limits=limits, timeout=timeout, trust_env=False)
        self._hb_task: asyncio.Task | None = None

    def _logical_now(self) -> float:
        now = self.clock_base + (time.monotonic() - self.mono_origin)
        if now <= self.last_sent_now:
            now = self.last_sent_now + 1e-4
        self.last_sent_now = now
        return now

    async def _apply_ours_config(self, *, reset: bool = True) -> dict:
        resp = await self.client.get("/server_info", timeout=10.0)
        resp.raise_for_status()
        info = resp.json()
        state = info["internal_states"][0][self.KEY]
        ic = state.get("interval_cache", {})
        eng_clock = float(ic.get("clock_s") or state.get("clock_s") or 0.0)
        self.clock_base = max(self.clock_base, eng_clock + 0.05) - (time.monotonic() - self.mono_origin)
        need_reset = reset and (ic.get("effective_mode") != "active")
        if need_reset and not ic.get("idle", True):
            raise RuntimeError("engine busy while waiting to reset interval_cache")
        now = self._logical_now()
        ctrl = {
            "enabled": self.enabled,
            "now": now,
            "parameters": self.params,
            "knobs": self.knobs,
            "cache": self.cache_cfg,
            "prefill": self.prefill_cfg,
            "chunk_fair": self.chunk_cfg,
            "multi_chunk": self.multi_cfg,
            "interval_cache": self.interval_cfg,
            "interval_events": [],
            "reset": need_reset,
        }
        up = await self.client.post("/set_internal_state", json={"server_args": {}, self.KEY: ctrl}, timeout=10.0)
        up.raise_for_status()
        if up.json() != [True]:
            raise RuntimeError(f"Engine rejected Ours control: {up.text}")

        resp2 = await self.client.get("/server_info", timeout=10.0)
        resp2.raise_for_status()
        info2 = resp2.json()
        state2 = info2["internal_states"][0][self.KEY]
        self.verify_configuration(
            state2,
            self.enabled,
            self.params,
            self.knobs,
            self.cache_cfg,
            self.prefill_cfg,
            self.chunk_cfg,
            self.multi_cfg,
            self.interval_cfg,
        )
        resolved_max_len: int | None = self.resolved_max_model_len
        with suppress(Exception):
            mr = await self.client.get("/v1/models", timeout=10.0)
            if mr.status_code == 200:
                mdata = mr.json().get("data") or []
                if mdata and isinstance(mdata[0], dict):
                    ml = mdata[0].get("max_model_len")
                    if isinstance(ml, int) and ml > 0:
                        resolved_max_len = ml
        self.resolved_max_model_len = resolved_max_len
        self.engine_epoch_clock = float(state2.get("clock_s") or now)
        self.ours_verified = True
        self.last_error = None
        self.stats["reconfigurations"] += 1
        receipt = {
            "node_label": self.args.node_label,
            "activated_utc": _utc_now(),
            "engine_url": self.args.engine_url,
            "served_model_name": info2.get("served_model_name"),
            "configured_context_length": info2.get("context_length") or (info2.get("server_args") or {}).get("context_length") or "auto",
            "max_model_len": self.resolved_max_model_len,
            "tp_size": info2.get("tp_size"),
            "ep_size": info2.get("ep_size"),
            "max_running_requests": info2.get("max_running_requests"),
            "speculative_algorithm": info2.get("speculative_algorithm"),
            "enable_hierarchical_cache": info2.get("enable_hierarchical_cache"),
            "adapter_version": state2.get("adapter_version"),
            "interval_mode": state2.get("interval_cache", {}).get("effective_mode"),
            "crowd_aware_order": state2.get("interval_cache", {}).get("config", {}).get("crowd_aware_order"),
            "turn0_ttft_guard_s": state2.get("interval_cache", {}).get("config", {}).get("turn0_ttft_guard_s"),
            "prefill_mode": state2.get("prefill", {}).get("config", {}).get("mode"),
            "ours_verified": True,
        }
        if self.args.state_file:
            _atomic_json(self.args.state_file, receipt)
        return receipt

    async def flush_control(self, *, only_if_pending: bool = False) -> None:
        async with self.control_lock:
            if not self.ours_verified:
                try:
                    await self._apply_ours_config(reset=True)
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self.stats["control_errors"] += 1
                return
            if only_if_pending and not self.pending_events:
                return
            count = min(len(self.pending_events), 512)
            batch = self.pending_events[:count]
            now = self._logical_now()
            wire = []
            for ev in batch:
                item = dict(ev)
                item["observed_at"] = min(now, max(0.0, float(item["observed_at"])))
                wire.append(item)
            wire = self.validate_events(wire, now) if wire else []
            ctrl = {
                "enabled": self.enabled,
                "now": now,
                "parameters": self.params,
                "knobs": self.knobs,
                "cache": self.cache_cfg,
                "prefill": self.prefill_cfg,
                "chunk_fair": self.chunk_cfg,
                "multi_chunk": self.multi_cfg,
                "interval_cache": self.interval_cfg,
                "interval_events": wire,
            }
            try:
                up = await self.client.post("/set_internal_state", json={"server_args": {}, self.KEY: ctrl}, timeout=5.0)
                up.raise_for_status()
                if up.json() != [True]:
                    # Engine might have restarted cleanly; check clock/mode and re-apply
                    self.ours_verified = False
                    await self._apply_ours_config(reset=True)
                else:
                    del self.pending_events[:count]
                    self.stats["control_updates"] += 1
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.stats["control_errors"] += 1
                with suppress(Exception):
                    info = (await self.client.get("/server_info", timeout=3.0)).json()
                    ic_mode = info["internal_states"][0][self.KEY].get("interval_cache", {}).get("effective_mode")
                    if ic_mode != "active":
                        self.ours_verified = False

    async def heartbeat_loop(self) -> None:
        while True:
            try:
                if not self.ours_verified:
                    h = await self.client.get("/health", timeout=3.0)
                    if h.status_code == 200:
                        await self.flush_control(only_if_pending=False)
                    await asyncio.sleep(1.0)
                    continue
                await self.flush_control(only_if_pending=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.stats["control_errors"] += 1
            await asyncio.sleep(0.25)

    def _queue_event(self, kind: str, rid: str, **extra) -> None:
        if len(self.pending_events) >= 4096:
            del self.pending_events[:512]
        obs = self.clock_base + (time.monotonic() - self.mono_origin)
        ev = {
            "kind": kind,
            "event_id": str(uuid4()),
            "request_id": rid,
            "observed_at": max(0.0, obs),
            **extra,
        }
        self.pending_events.append(ev)
        if kind == "tool_start":
            self.stats["tool_starts_emitted"] += 1
        elif kind == "tool_end":
            self.stats["tool_ends_emitted"] += 1
        elif kind == "request_ready":
            self.stats["request_ready_emitted"] += 1

    def _resolve_session(self, headers, body: dict, rid: str) -> tuple[str, str, str | None, bool, tuple[str, str] | None]:
        hdr_sess = (
            headers.get("x-orvix-session-id")
            or headers.get("x-trace-task-id")
            or headers.get("x-trace-run-id")
            or headers.get("x-session-id")
        )
        family_id = (
            headers.get("x-orvix-user-id")
            or headers.get("x-trace-user-id")
            or body.get("user")
            or ""
        )
        turn1_key = None
        conv_pair = None
        messages = body.get("messages")
        if hdr_sess:
            session_id = str(hdr_sess)[:128]
        elif body.get("session_id"):
            session_id = str(body["session_id"])[:128]
        elif isinstance(messages, list) and messages:
            sys_part = ""
            u1_part = ""
            a1_part = ""
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if role in ("system", "developer") and not sys_part:
                    sys_part = _norm_text_for_hash(m)[:1024]
                elif role == "user" and not u1_part:
                    u1_part = _norm_text_for_hash(m)[:1024]
                elif role == "assistant" and not a1_part:
                    a1_part = _norm_text_for_hash(m)[:512]
                    break
            turn1_key = _msg_digest(sys_part + "\x00" + u1_part)
            if a1_part:
                a1_dig = _msg_digest(a1_part)
                session_id = self.conv_index.get((turn1_key, a1_dig))
                if session_id:
                    self.conv_index.move_to_end((turn1_key, a1_dig))
                else:
                    session_id = f"auto-{turn1_key[:12]}-{a1_dig[:8]}"
            else:
                session_id = f"auto-{rid[:16]}"
            conv_pair = (turn1_key, a1_part)
        else:
            session_id = f"req-{rid[:16]}"

        if not family_id:
            family_id = f"fam-{session_id[:20]}"

        sess = self.sessions.get(session_id)
        if sess is not None:
            self.sessions.move_to_end(session_id)
        else:
            sess = {
                "last_rid": None,
                "in_tool_wait": False,
                "tool_id": None,
                "turn_count": 0,
            }
            if len(self.sessions) >= 2048:
                self.sessions.popitem(last=False)
            self.sessions[session_id] = sess

        prior_rid = sess["last_rid"]
        resumed = prior_rid is not None
        if resumed:
            ends_with_tool = bool(
                isinstance(messages, list)
                and messages
                and isinstance(messages[-1], dict)
                and messages[-1].get("role") == "tool"
            )
            if sess["in_tool_wait"] or ends_with_tool:
                tool_id = sess["tool_id"] or f"tool-{prior_rid}"
                self._queue_event("tool_end", prior_rid, tool_id=tool_id, status="ok")
                sess["in_tool_wait"] = False
            self._queue_event("request_ready", prior_rid)

        return session_id, str(family_id)[:128], prior_rid, resumed, conv_pair

    def _record_completion(
        self,
        session_id: str,
        rid: str,
        conv_pair: tuple[str, str] | None,
        assistant_preview: str,
        saw_tool_calls: bool,
    ) -> None:
        sess = self.sessions.get(session_id)
        if sess is None:
            sess = {"last_rid": rid, "in_tool_wait": False, "tool_id": None, "turn_count": 0}
            if len(self.sessions) >= 2048:
                self.sessions.popitem(last=False)
            self.sessions[session_id] = sess
        sess["last_rid"] = rid
        sess["turn_count"] += 1
        if saw_tool_calls:
            tool_id = f"tool-{rid}"
            sess["in_tool_wait"] = True
            sess["tool_id"] = tool_id
            self._queue_event("tool_start", rid, tool_id=tool_id, tool_type="agent_tool")
        if conv_pair is not None and not conv_pair[1] and assistant_preview:
            turn1_key = conv_pair[0]
            a1_dig = _msg_digest(assistant_preview[:512])
            if len(self.conv_index) >= 4096:
                self.conv_index.popitem(last=False)
            self.conv_index[(turn1_key, a1_dig)] = session_id


def build_app(service: OursNodeService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        service._hb_task = asyncio.create_task(service.heartbeat_loop())
        try:
            yield
        finally:
            if service._hb_task and not service._hb_task.done():
                service._hb_task.cancel()
                with suppress(asyncio.CancelledError):
                    await service._hb_task
            await service.client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def allow_peers(request: Request, call_next):
        client_host = (request.client.host if request.client else "").removeprefix("::ffff:")
        if service.allowed_clients and client_host not in service.allowed_clients:
            return JSONResponse({"error": "peer not allowed"}, status_code=403)
        return await call_next(request)

    @app.get("/orvix_daemon_status")
    async def daemon_status():
        srv = None
        try:
            r = await service.client.get("/server_info", timeout=3.0)
            if r.status_code == 200:
                j = r.json()
                st = j["internal_states"][0][service.KEY]
                srv = {
                    "served_model_name": j.get("served_model_name"),
                    "configured_context_length": j.get("context_length") or (j.get("server_args") or {}).get("context_length") or "auto",
                    "max_model_len": service.resolved_max_model_len,
                    "adapter_version": st.get("adapter_version"),
                    "interval_mode": st.get("interval_cache", {}).get("effective_mode"),
                    "crowd_aware_order": st.get("interval_cache", {}).get("config", {}).get("crowd_aware_order"),
                    "prefill_mode": st.get("prefill", {}).get("config", {}).get("mode"),
                    "interval_counts": st.get("interval_cache", {}).get("counts"),
                    "prefill_counts": st.get("prefill", {}).get("counts"),
                }
        except Exception as exc:
            srv = {"error": str(exc)}
        return {
            "node_label": service.args.node_label,
            "ours_verified": service.ours_verified,
            "last_error": service.last_error,
            "stats": service.stats,
            "system_prefix_patcher": service.system_patcher.stats_snapshot(),
            "sessions_tracked": len(service.sessions),
            "pending_events": len(service.pending_events),
            "engine_state": srv,
        }

    async def forward_simple(request: Request, path: str):
        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in HOP_BY_HOP
        }
        headers["accept-encoding"] = "identity"
        req = service.client.build_request(
            request.method,
            path,
            params=request.query_params,
            headers=headers,
            content=body if body else None,
        )
        try:
            upstream = await service.client.send(req, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": "engine_unavailable", "detail": str(exc)}, status_code=503)
        resp_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in HOP_BY_HOP
        }
        resp_headers["X-Orvix-Worker-Node"] = service.args.node_label
        media_type = upstream.headers.get("content-type")
        if "text/event-stream" in (media_type or "").lower():
            async def gen():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()
            return StreamingResponse(gen(), status_code=upstream.status_code, headers=resp_headers, media_type=media_type)
        try:
            raw = await upstream.aread()
        finally:
            await upstream.aclose()
        return Response(content=raw, status_code=upstream.status_code, headers=resp_headers, media_type=media_type)

    @app.api_route("/health", methods=["GET", "HEAD"])
    async def health_check(request: Request):
        if not service.ours_verified:
            return JSONResponse({"status": "initializing_ours", "node_label": service.args.node_label}, status_code=503)
        return await forward_simple(request, "/health")

    @app.post("/v1/chat/completions")
    async def proxy_chat(request: Request):
        service.stats["requests_total"] += 1
        service.stats["chat_requests"] += 1
        raw_bytes = await request.body()
        try:
            body = json.loads(raw_bytes)
        except Exception:
            return await forward_simple(request, "/v1/chat/completions")

        rid = str(body.get("rid") or f"chatcmpl-{uuid4().hex}")
        body["rid"] = rid
        if body.get("model") in MODEL_ALIASES or not body.get("model"):
            body["model"] = service.args.engine_model

        session_id, family_id, prior_rid, resumed, conv_pair = service._resolve_session(request.headers, body, rid)
        if resumed and service.pending_events:
            await service.flush_control(only_if_pending=True)

        body, patch_decision = service.system_patcher.apply(session_id, body)

        lane = str(request.headers.get("x-trace-purpose") or "foreground")
        if lane not in {"foreground", "subagent", "background"}:
            lane = "foreground"

        custom = dict(body.get("custom_params") or {})
        unified = dict(custom.get(service.KEY) or {})
        unified.setdefault("family_id", family_id)
        unified.setdefault("session_id", session_id)
        unified.setdefault("graph_id", session_id)
        unified.setdefault("scope_id", f"{session_id}-s0")
        unified.setdefault("lane", lane)
        unified["ready_at"] = service.clock_base + (time.monotonic() - service.mono_origin)
        if prior_rid:
            unified.setdefault("predecessor_request_id", prior_rid)
        custom[service.KEY] = unified
        body["custom_params"] = custom

        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["accept-encoding"] = "identity"
        headers["content-type"] = "application/json"

        up_req = service.client.build_request("POST", "/v1/chat/completions", headers=headers, json=body)
        try:
            upstream = await service.client.send(up_req, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": "engine_unavailable", "detail": str(exc)}, status_code=503)
        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
        resp_headers["X-Orvix-Worker-Node"] = service.args.node_label
        media_type = upstream.headers.get("content-type") or "application/json"

        if "text/event-stream" in media_type.lower():
            async def sse_stream():
                buf = bytearray()
                preview_parts: list[str] = []
                tool_calls_seen: list[dict] = []
                saw_tool_calls = False
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                        if len(buf) < 65536:
                            buf.extend(chunk)
                            while b"\n" in buf:
                                line, _, rest = buf.partition(b"\n")
                                buf = bytearray(rest)
                                if line.startswith(b"data:"):
                                    payload = line[5:].strip()
                                    if payload and payload != b"[DONE]":
                                        with suppress(Exception):
                                            obj = json.loads(payload)
                                            for ch in obj.get("choices") or ():
                                                if ch.get("finish_reason") == "tool_calls":
                                                    saw_tool_calls = True
                                                d = ch.get("delta") or {}
                                                if d.get("tool_calls"):
                                                    saw_tool_calls = True
                                                    if len(tool_calls_seen) < 4:
                                                        tool_calls_seen.append(d["tool_calls"][0])
                                                if len(preview_parts) < 32 and isinstance(d.get("content"), str):
                                                    preview_parts.append(d["content"])
                finally:
                    await upstream.aclose()
                    prev_text = "".join(preview_parts)
                    if not prev_text and tool_calls_seen:
                        prev_text = json.dumps(tool_calls_seen, sort_keys=True, ensure_ascii=False)[:512]
                    service._record_completion(session_id, rid, conv_pair, prev_text, saw_tool_calls)

            return StreamingResponse(sse_stream(), status_code=upstream.status_code, headers=resp_headers, media_type=media_type)

        try:
            raw = await upstream.aread()
        finally:
            await upstream.aclose()
        saw_tool_calls = False
        prev_text = ""
        with suppress(Exception):
            obj = json.loads(raw)
            for ch in obj.get("choices") or ():
                if ch.get("finish_reason") == "tool_calls":
                    saw_tool_calls = True
                msg = ch.get("message") or {}
                if msg.get("tool_calls"):
                    saw_tool_calls = True
                prev_text = _norm_text_for_hash(msg)
                break
        service._record_completion(session_id, rid, conv_pair, prev_text, saw_tool_calls)
        return Response(content=raw, status_code=upstream.status_code, headers=resp_headers, media_type=media_type)

    @app.post("/generate")
    async def proxy_generate(request: Request):
        service.stats["requests_total"] += 1
        service.stats["generate_requests"] += 1
        raw_bytes = await request.body()
        try:
            body = json.loads(raw_bytes)
        except Exception:
            return await forward_simple(request, "/generate")

        rid = str(body.get("rid") or f"gen-{uuid4().hex}")
        body["rid"] = rid
        sp = dict(body.get("sampling_params") or {})
        custom = dict(sp.get("custom_params") or {})
        unified = dict(custom.get(service.KEY) or {})
        sess_id = str(
            unified.get("session_id")
            or body.get("session_id")
            or request.headers.get("x-orvix-session-id")
            or ""
        )
        if sess_id:
            graph_id = str(unified.get("graph_id") or sess_id)
            scope_id = str(unified.get("scope_id") or f"{sess_id}-s0")
            if isinstance(body.get("input_ids"), list) and len(body["input_ids"]) >= 512:
                eff_ids, _ = service.gen_stabilizer.transform(graph_id, scope_id, body["input_ids"])
                body["input_ids"] = eff_ids
                if len(service.gen_stabilizer.scope_spines) > 1024:
                    oldest_key = next(iter(service.gen_stabilizer.scope_spines))
                    service.gen_stabilizer.scope_spines.pop(oldest_key, None)
            unified.setdefault("family_id", f"fam-{sess_id[:20]}")
            unified.setdefault("session_id", sess_id)
            unified.setdefault("graph_id", graph_id)
            unified.setdefault("scope_id", scope_id)
            unified.setdefault("lane", "foreground")
            unified["ready_at"] = service.clock_base + (time.monotonic() - service.mono_origin)
            custom[service.KEY] = unified
            sp["custom_params"] = custom
            body["sampling_params"] = sp

        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["accept-encoding"] = "identity"
        headers["content-type"] = "application/json"
        up_req = service.client.build_request("POST", "/generate", headers=headers, json=body)
        try:
            upstream = await service.client.send(up_req, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": "engine_unavailable", "detail": str(exc)}, status_code=503)
        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
        resp_headers["X-Orvix-Worker-Node"] = service.args.node_label
        media_type = upstream.headers.get("content-type")
        if "text/event-stream" in (media_type or "").lower():
            async def gen():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()
            return StreamingResponse(gen(), status_code=upstream.status_code, headers=resp_headers, media_type=media_type)
        try:
            raw = await upstream.aread()
        finally:
            await upstream.aclose()
        return Response(content=raw, status_code=upstream.status_code, headers=resp_headers, media_type=media_type)

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
    async def catch_all(request: Request, full_path: str):
        if full_path in {"generate", "v1/completions"} and request.method == "POST":
            service.stats["requests_total"] += 1
            if full_path == "generate":
                service.stats["generate_requests"] += 1
            else:
                service.stats["completions_requests"] += 1
        return await forward_simple(request, "/" + full_path)

    return app


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node-label", required=True)
    ap.add_argument("--listen-host", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, default=32683)
    ap.add_argument("--engine-url", default="http://127.0.0.1:32681")
    ap.add_argument("--engine-model", default="deepseek-v41-flash-nvfp4")
    ap.add_argument("--package-dir", type=Path, required=True)
    ap.add_argument("--state-file", type=Path, default=None)
    ap.add_argument("--allowed-clients", default="127.0.0.1,::1")
    ap.add_argument(
        "--disable-system-prefix-patcher",
        dest="enable_system_prefix_patcher",
        action="store_false",
        default=True,
        help="Disable semantic block-level system prompt freezing and trailing update patching.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    svc = OursNodeService(args)
    app = build_app(svc)
    uvicorn.run(app, host=args.listen_host, port=args.listen_port, access_log=False)


if __name__ == "__main__":
    main()
