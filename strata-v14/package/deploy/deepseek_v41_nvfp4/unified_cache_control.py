"""Experimental tool-aware demotion through the qualified native HiCache stack.

This module never copies tensors or frees allocator slots itself. It queues the
native, lock-protected D2H backup, lets the native TP-consensus acknowledgement
path finish it, and only then asks the native cache to demote a still-idle leaf.
H2D remains demand-driven through native prefix matching/admission. There is no
claim of predicting tool completion or of prefetching before a request exists.

All decision time comes from the control message broadcast to every TP rank.
Unsupported runtimes stay in observe mode and keep serving requests normally.
"""

from __future__ import annotations

from array import array
from collections import Counter, OrderedDict, deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import math
import time


LOGGER = logging.getLogger(__name__)
KEY = "orvix_unified"


class CachePlanMismatch(RuntimeError):
    """The TP collective completed with different plans on different ranks."""


@dataclass(frozen=True)
class CacheConfig:
    mode: str = "observe"
    require_native_rejection: bool = False
    pressure_free_ratio: float = 0.15
    min_idle_s: float = 1.0
    max_backup_tokens: int = 65536
    max_actions_per_tick: int = 2
    max_records: int = 4096
    max_events: int = 32768
    max_tools_per_request: int = 128
    # v2 demand-aware policy: amortize control work and bound each native batch.
    scan_interval_s: float = 0.5
    max_bundle_leaves: int = 8
    max_waiter_scan: int = 16
    max_full_copy_per_swa_token: float = 8.0


@dataclass
class _Record:
    rid: str
    scope: tuple[str, ...]
    node_id: int
    node_object: object
    fingerprint: str
    token_count: int
    namespace: tuple[object, object]
    tools: dict[str, float] = field(default_factory=dict)
    ended_tools: set[str] = field(default_factory=set)
    ready: bool = False
    backup_queued: bool = False
    demoted: bool = False


def _concat_ids(origin, output):
    from array import array as _array
    if isinstance(origin, _array):
        out = _array(origin.typecode, origin)
        if output:
            out.extend(output)
        return out
    if isinstance(output, _array):
        out = _array(output.typecode, origin)
        out.extend(output)
        return out
    return list(origin) + list(output)


def _name(value):
    return getattr(value, "name", str(value)).upper()


def _metadata(req):
    sampling = getattr(req, "sampling_params", None)
    custom = getattr(sampling, "custom_params", None)
    value = custom.get(KEY) if isinstance(custom, dict) else None
    return value if isinstance(value, dict) else {}


def _scope(req):
    meta = _metadata(req)
    # A caller that has not supplied a session/scope gets a request-private
    # identity; it cannot accidentally collapse two agents into one owner.
    session = meta.get("session_id", getattr(req, "session_id", None))
    scope = meta.get("scope_id", meta.get("scope"))
    if session is None and scope is None:
        return ("request", str(req.rid))
    return tuple(str(v or "") for v in (
        meta.get("family_id", meta.get("user_id")),
        meta.get("graph_id"), session, scope,
    ))


def _fingerprint(tokens, namespace):
    digest = hashlib.sha256()
    digest.update(repr(namespace).encode("utf-8"))
    digest.update(b"\x00")
    if isinstance(tokens, array) and tokens.typecode == "q":
        digest.update(memoryview(tokens).cast("B"))
    else:
        digest.update(array("q", tokens).tobytes())
    return digest.hexdigest()


def _node_identity(node, root):
    path = []
    cursor = node
    while cursor is not root:
        if cursor is None or cursor.key is None or len(path) >= 4096:
            raise ValueError("unqualified node path")
        path.append(cursor)
        cursor = cursor.parent
    tokens = array("q")
    bigram = bool(getattr(node.key, "is_bigram", False))
    for cursor in reversed(path):
        key = cursor.key
        if bool(getattr(key, "is_bigram", False)) != bigram:
            raise ValueError("mixed radix key representations")
        raw = key.raw_token_ids() if callable(getattr(key, "raw_token_ids", None)) else key.token_ids[:len(key) + int(bigram)]
        if bigram and tokens:
            # Native bigram edge slices share one raw boundary token. Hash the
            # original raw prefix, independent of subsequent radix edge splits.
            if not raw or tokens[-1] != raw[0]:
                raise ValueError("noncontiguous bigram cache path")
            tokens.extend(raw[1:])
        else:
            tokens.extend(raw)
    namespace = (getattr(node.key, "extra_key", None), getattr(node.key, "cache_salt", None))
    return _fingerprint(tokens, namespace), len(tokens), namespace


def inspect_cache_capabilities(scheduler):
    """Read capabilities without allocating/copying GPU tensors or importing torch."""
    cache = scheduler.tree_cache
    core = getattr(cache, "tree_core", None)
    controller = getattr(cache, "cache_controller", None)
    allocator = getattr(cache, "token_to_kv_pool_allocator", None)
    reasons = []
    components = sorted(_name(k) for k in getattr(cache, "components", {}))
    if type(cache).__name__ != "UnifiedRadixCache":
        reasons.append("cache_is_not_unified_radix")
    if getattr(cache, "_tree_core_backend", None) != "python":
        reasons.append("tree_backend_is_not_python")
    if getattr(cache, "disable", False):
        reasons.append("prefix_cache_disabled")
    if not components or not set(components) <= {"FULL", "SWA"} or "FULL" not in components:
        reasons.append("only_full_swa_components_qualified")
    if getattr(cache, "enable_session_radix_cache", False):
        reasons.append("session_reference_cache_not_qualified")
    if getattr(cache, "linker", None) is not None:
        reasons.append("external_linker_not_qualified")
    if getattr(cache, "pp_size", 1) != 1:
        reasons.append("pipeline_parallel_cache_not_qualified")
    if controller is None:
        reasons.append("native_hicache_not_enabled")
    if getattr(cache, "host_memory_mode", "cache") != "cache":
        reasons.append("resident_host_cache_required")
    if controller is not None and getattr(controller, "write_policy", None) != "write_back":
        reasons.append("native_write_back_policy_required")
    required_methods = {
        "cache": (cache, ("checkpoint_into_tree", "_execute_and_commit_kv_backup", "_evict_device_leaf", "flush_pending_backups", "check_hicache_events", "_all_reduce")),
        "tree": (core, ("node_by_id", "_is_device_leaf")),
        "scheduler": (scheduler, ("_process_hicache_events",)),
    }
    for group, (obj, names) in required_methods.items():
        if any(not callable(getattr(obj, name, None)) for name in names):
            reasons.append(group + "_native_contract_missing")
    host_group = getattr(cache, "host_pool_group", None)
    entries = getattr(host_group, "entry_map", {}) if host_group is not None else {}
    host_pools = sorted(_name(k) for k in entries)
    sidecars = sorted(_name(s.pool_name) for s in getattr(cache, "sidecar_pool_specs", ()))
    device_pools = []
    state_contract = "unknown"
    try:
        pool = allocator.get_kvcache()
        describe = getattr(pool, "get_device_pool_infos", None)
        if not callable(describe):
            reasons.append("device_pool_inventory_unavailable")
        else:
            infos = describe()
            device_pools = sorted(_name(i.pool_name) for i in infos)
            missing = set(device_pools) - set(host_pools)
            if missing:
                reasons.append("missing_host_pools:" + ",".join(sorted(missing)))
            missing_sidecars = set(device_pools) - {"KV", "SWA"} - set(sidecars)
            if missing_sidecars:
                reasons.append("missing_sidecar_transfers:" + ",".join(sorted(missing_sidecars)))
            if not device_pools:
                reasons.append("empty_device_pool_inventory")
            # This is the model's existing prefix-restore contract, not an
            # invented save-all-running-state implementation. The native stack
            # describes page-owned compressor/indexer stores as sidecars and
            # reconstructs request-private state while re-prefilling its tail.
            if type(pool).__name__ == "DeepSeekV4TokenToKVPool":
                if not callable(getattr(pool, "clear_request_scoped_state", None)):
                    reasons.append("request_private_state_rebuild_contract_missing")
                if not callable(getattr(cache, "swa_reprefill_tail_tokens", None)):
                    reasons.append("native_swa_replay_contract_missing")
                state_contract = "native_dsv4_prefix_restore_and_tail_replay"
            else:
                reasons.append("only_dsv4_pool_contract_qualified")
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        reasons.append("device_pool_inventory_failed:" + type(exc).__name__)
    if "KV" not in host_pools:
        reasons.append("host_anchor_pool_missing")
    return {
        "native_hicache": controller is not None,
        "tool_demotion_supported": not reasons,
        "reasons": reasons,
        "components": components,
        "device_pools": device_pools,
        "host_pools": host_pools,
        "sidecars": sidecars,
        "state_contract": state_contract,
        "h2d_policy": "native_demand_load_back",
        "gpu_validated": False,
    }


class CacheControl:
    def __init__(self, scheduler, *, backup_factory=None, capabilities=None, consensus=None):
        self.scheduler = scheduler
        self.cache = scheduler.tree_cache
        self.core = self.cache.tree_core
        self.backup_factory = backup_factory
        self.consensus = consensus or (lambda value: None)
        self.capabilities = capabilities if capabilities is not None else inspect_cache_capabilities(scheduler)
        self.config = CacheConfig()
        self.now = 0.0
        self.epoch = 0
        self.counts = Counter()
        self.records = OrderedDict()
        self.latest_by_scope = {}
        self.seen_events = OrderedDict()
        self.terminal_requests = OrderedDict()
        self.last_tick_now = None
        self.failed = False
        self.tracking_exhausted = False
        self.last_pressure = {}
        self.prefix_probe = None
        self.last_admission_room = {}
        self.action_log = deque(maxlen=128)
        self.last_reason = None
        self.host_control_elapsed_s = 0.0

    @property
    def effective_mode(self):
        if self.config.mode in {"tool-aware", "demand-aware"} and (not self.capabilities["tool_demotion_supported"] or self.failed or self.tracking_exhausted):
            return "observe"
        return self.config.mode

    def validate_control(self, value):
        """Stage the complete message; validation must precede rank mutation."""
        now = value["now"]
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now) or now < self.now:
            raise ValueError("invalid broadcast cache clock")
        reset = value.get("reset", False)
        if type(reset) is not bool:
            raise ValueError("reset must be boolean")
        raw = value.get("cache", {})
        if not isinstance(raw, dict) or not raw.keys() <= asdict(CacheConfig()).keys():
            raise ValueError("unknown cache configuration")
        config = CacheConfig(**(asdict(self.config) | raw))
        if config.mode not in {"off", "observe", "tool-aware", "demand-aware"}:
            raise ValueError("invalid cache mode")
        for name in ("pressure_free_ratio", "min_idle_s", "scan_interval_s", "max_full_copy_per_swa_token"):
            number = getattr(config, name)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError("invalid cache threshold")
        if not 0 < config.pressure_free_ratio <= 0.5 or not 0 <= config.min_idle_s <= 3600:
            raise ValueError("cache thresholds outside qualified range")
        if not 0 <= config.scan_interval_s <= 10 or config.max_full_copy_per_swa_token < 1:
            raise ValueError("invalid demand-aware bounds")
        for name, lo, hi in (("max_bundle_leaves", 1, 32), ("max_waiter_scan", 1, 128), ("max_backup_tokens", 1, 1048576), ("max_actions_per_tick", 1, 16), ("max_records", 1, 32768), ("max_events", 1, 262144), ("max_tools_per_request", 1, 1024)):
            number = getattr(config, name)
            if type(number) is not int or not lo <= number <= hi:
                raise ValueError("invalid cache budget: " + name)
        if type(config.require_native_rejection) is not bool:
            raise ValueError("require_native_rejection must be boolean")
        if config.require_native_rejection and config.mode == "demand-aware":
            feedback = getattr(self.scheduler, "_orvix_admission_feedback", None)
            if feedback is None or not feedback.available:
                raise ValueError("native admission feedback unavailable")
        events = value.get("events", [])
        if not isinstance(events, list) or len(events) > 4096:
            raise ValueError("events must be a bounded list")
        checked = []
        for event in events:
            if not isinstance(event, dict) or event.get("kind") not in {"tool_start", "tool_end", "request_ready"}:
                raise ValueError("invalid lifecycle event")
            for name in ("event_id", "request_id"):
                if not isinstance(event.get(name), str) or not 0 < len(event[name]) <= 512:
                    raise ValueError("invalid lifecycle identifier")
            if event["kind"] != "request_ready" and (not isinstance(event.get("tool_id"), str) or not 0 < len(event["tool_id"]) <= 512):
                raise ValueError("tool events require tool_id")
            observed = event.get("observed_at", now)
            if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed) or observed > now:
                raise ValueError("invalid event observation time")
            # Future durations/ETAs are intentionally not accepted as policy inputs.
            if any(name in event for name in ("duration_s", "expected_duration_s", "finish_at", "eta")):
                raise ValueError("future tool duration is not a causal input")
            checked.append({name: event[name] for name in ("event_id", "request_id", "kind", "tool_id") if name in event} | {"observed_at": float(observed)})
        return {"now": float(now), "reset": reset, "config": config, "events": checked}

    def apply_control(self, prepared):
        self.now = prepared["now"]
        self.config = prepared["config"]
        if prepared["reset"]:
            self.reset_tracking()
        for event in prepared["events"]:
            event_id = event["event_id"]
            if event_id in self.seen_events:
                self.counts["duplicate_events"] += 1
                continue
            self.seen_events[event_id] = None
            if len(self.seen_events) > self.config.max_events:
                self.seen_events.popitem(last=False)
            self._event(event)
        if len(self.records) > self.config.max_records:
            self.tracking_exhausted = True
        # No GPU/cache actions run inside the IPC setter. The scheduler's normal
        # event round first processes native acknowledgements, then calls tick.

    def reset_tracking(self):
        self.records.clear()
        self.latest_by_scope.clear()
        self.seen_events.clear()
        self.terminal_requests.clear()
        self.last_tick_now = None
        self.action_log.clear()
        self.last_reason = None
        self.tracking_exhausted = False
        self.epoch += 1
        # Native in-flight copies retain their native locks and ack ownership.
        # Clearing our intentions must never cancel/free those operations.

    def _terminal(self, rid):
        self.terminal_requests[rid] = None
        if len(self.terminal_requests) > self.config.max_records:
            self.terminal_requests.popitem(last=False)

    def _event(self, event):
        self.counts[event["kind"] + "_events"] += 1
        rid = event["request_id"]
        record = self.records.get(rid)
        if event["kind"] == "request_ready":
            self._terminal(rid)
            if record is not None:
                record.ready = True
                record.tools.clear()
            return
        if record is None:
            self.counts["events_without_completed_request"] += 1
            # An unmatched end/start is not permission to act later.
            self._terminal(rid)
            return
        tool_id = event["tool_id"]
        if event["kind"] == "tool_end":
            record.tools.pop(tool_id, None)
            if len(record.ended_tools) >= self.config.max_tools_per_request:
                record.ready = True
                self._terminal(rid)
            else:
                record.ended_tools.add(tool_id)
            return
        if rid in self.terminal_requests or record.ready or tool_id in record.ended_tools:
            self.counts["late_starts_ignored"] += 1
            return
        if len(record.tools) >= self.config.max_tools_per_request:
            record.ready = True
            record.tools.clear()
            self._terminal(rid)
            self.counts["tool_tracking_limit"] += 1
            return
        record.tools.setdefault(tool_id, event["observed_at"])

    def completed(self, req):
        if self.config.mode == "off" or not req.finished() or not _metadata(req):
            return
        rid = str(req.rid)
        if rid in self.records:
            return
        if len(self.records) >= self.config.max_records:
            # Losing an owner could make a shared leaf look exclusive. Stop
            # proactive actions rather than guess after bookkeeping overflow.
            self.tracking_exhausted = True
            self.counts["record_tracking_limit"] += 1
            return
        try:
            node_id = req.last_node
            node = self.core.node_by_id(node_id)
            if node is self.core.root_node:
                return
            fingerprint, count, namespace = _node_identity(node, self.core.root_node)
            record = _Record(rid, _scope(req), node_id, node, fingerprint, count, namespace, ready=rid in self.terminal_requests)
        except (AttributeError, KeyError, TypeError, ValueError):
            self.counts["unqualified_completed_anchor"] += 1
            return
        previous = self.latest_by_scope.get(record.scope)
        if previous in self.records:
            self.records[previous].ready = True
            self.records[previous].tools.clear()
        self.latest_by_scope[record.scope] = rid
        self.records[rid] = record
        self.counts["completed_anchors"] += 1

    def _pressure(self):
        if not len(self.scheduler.waiting_queue):
            return {}
        running = getattr(self.scheduler, "running_batch", None)
        get_room = getattr(self.scheduler, "get_num_allocatable_reqs", None)
        maximum = getattr(self.scheduler, "max_running_requests", None)
        if running is None or not callable(get_room) or not isinstance(maximum, int):
            self.last_admission_room = {"known": False}
            self.counts["unknown_request_slot_skips"] += 1
            return {}
        count = len(running.reqs)
        room = get_room(count, running_batch=running)
        self.last_admission_room = {"known": True, "running_requests": count, "max_running_requests": maximum, "allocatable_request_rows": room}
        if count >= maximum or room <= 0:
            self.counts["no_request_slot_skips"] += 1
            return {}
        alloc = self.cache.token_to_kv_pool_allocator
        result = {}
        for name, size_name, getter in (("full", "size_full", "full_available_size"), ("swa", "size_swa", "swa_available_size")):
            capacity = getattr(alloc, size_name, None)
            available = getattr(alloc, getter, None)
            if name == "full" and capacity is None:
                capacity = getattr(self.scheduler, "max_total_num_tokens", None)
            if name == "full" and not callable(available):
                available = getattr(alloc, "available_size", None)
            if not isinstance(capacity, int) or capacity <= 0 or not callable(available):
                continue
            free = available()
            if not isinstance(free, (int, float)) or not math.isfinite(free) or free < 0:
                continue
            evictable_fn = getattr(self.cache, name + "_evictable_size", None)
            evictable = evictable_fn() if callable(evictable_fn) else None
            result[name] = {"free": free, "capacity": capacity,
                            "native_evictable": evictable,
                            "pressure": free < capacity * self.config.pressure_free_ratio}
        return result

    def _helps_waiting_chunk(self, component_sizes, pressure):
        """Conservative immediate-chunk estimate, never a native-admission proof.

        Native admission still accounts for generation reserve, beam groups,
        auxiliary pools and concurrent batches. Those constraints remain native.
        Missing estimate evidence keeps the explicit pressure/slot gates; the
        snapshot states that no footprint was established.
        """
        page = getattr(self.cache, "page_size", None)
        if not callable(self.prefix_probe) or not isinstance(page, int) or page <= 0:
            self.counts["footprint_estimate_unavailable"] += 1
            return True
        chunk = getattr(self.scheduler, "chunked_prefill_size", None)
        governor = getattr(self.scheduler, "_orvix_prefill_governor", None)
        planned = getattr(governor, "planned_budget", None)
        if callable(planned):
            chunk = planned(self.scheduler.running_batch)
        prefill = getattr(self.scheduler, "max_prefill_tokens", None)
        limit = min((v for v in (chunk, prefill) if isinstance(v, int) and v > 0), default=None)
        window = getattr(self.cache, "sliding_window_size", None)
        for req in self.scheduler.waiting_queue:
            if getattr(req, "beam_group", None) is not None:
                continue
            tokens = _concat_ids(req.origin_input_ids, req.output_ids)
            match = self.prefix_probe(req, tokens)
            uncached = max(0, len(tokens) - int(match.device_prefix_len))
            immediate = min(uncached, limit) if limit is not None else uncached
            # One extra page provides headroom for the next decoding token.
            full = ((immediate + page - 1) // page + 1) * page
            needed = {"full": full}
            if "swa" in pressure and isinstance(window, int) and window > 0:
                needed["swa"] = min(full, ((window + page - 1) // page + 1) * page)
            deficient = [name for name, n in needed.items() if name in pressure and n > pressure[name]["free"]]
            if deficient and all(needed[name] <= pressure[name]["free"] + component_sizes.get(name, 0) for name in deficient):
                self.counts["estimated_chunk_room_opportunities"] += 1
                return True
        self.counts["no_estimated_chunk_room_benefit"] += 1
        return False

    def _waiting_uses(self, record):
        tcount = record.token_count
        rns = record.namespace
        rfp = record.fingerprint
        for req in self.scheduler.waiting_queue:
            if _scope(req) == record.scope:
                return True
            namespace = (getattr(req, "extra_key", None), getattr(req, "cache_salt", None))
            if namespace != rns:
                continue
            total = len(req.origin_input_ids) + len(req.output_ids)
            if total < tcount:
                continue
            fp_cache = getattr(req, "_orvix_fp_cache", None)
            cache_key = (total, tcount, namespace)
            if isinstance(fp_cache, dict) and cache_key in fp_cache:
                fp = fp_cache[cache_key]
            else:
                ids = _concat_ids(req.origin_input_ids, req.output_ids)
                fp = _fingerprint(ids[:tcount], namespace)
                if not isinstance(fp_cache, dict):
                    fp_cache = {}
                    try:
                        req._orvix_fp_cache = fp_cache
                    except AttributeError:
                        fp_cache = None
                if isinstance(fp_cache, dict) and len(fp_cache) < 64:
                    fp_cache[cache_key] = fp
            if fp == rfp:
                return True
        return False

    def _eligible(self, record, pressure, *, check_footprint=True):
        if not record.tools or record.ready or record.demoted or self.latest_by_scope.get(record.scope) != record.rid:
            return None
        if self.now - min(record.tools.values()) < self.config.min_idle_s:
            return None
        try:
            node = self.core.node_by_id(record.node_id)
        except KeyError:
            self.counts["stale_anchors"] += 1
            return None
        if node is not record.node_object or node is self.core.root_node or node.children:
            return None
        if any(cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in node.component_data):
            return None
        if node.write_through_pending_id is not None or node.load_back_pending_id is not None:
            return None
        if getattr(node, "write_back_queued", False) or not self.core._is_device_leaf(node):
            return None
        if _node_identity(node, self.core.root_node)[0] != record.fingerprint:
            self.counts["changed_anchors"] += 1
            return None
        # Cross-agent exact-prefix sharing is kept even when both agents sleep.
        if any(other.node_object is node and other.scope != record.scope for other in self.records.values()):
            self.counts["shared_leaf_skips"] += 1
            return None
        if self._waiting_uses(record):
            self.counts["waiting_prefix_skips"] += 1
            return None
        component_sizes = { _name(ct).lower(): len(node.component_data[ct].value) if node.component_data[ct].value is not None else 0 for ct in self.cache.components }
        if not any(info["pressure"] and component_sizes.get(name, 0) > 0 for name, info in pressure.items()):
            return None
        if component_sizes.get("full", 0) > self.config.max_backup_tokens:
            self.counts["backup_budget_skips"] += 1
            return None
        if check_footprint and not self._helps_waiting_chunk(component_sizes, pressure):
            return None
        return node

    def _demand(self, req, pressure):
        """Next-chunk pool demand, explicitly NOT a proof of native admission."""
        page = getattr(self.cache, "page_size", None)
        if not callable(self.prefix_probe) or type(page) is not int or page <= 0:
            return None
        if getattr(req, "beam_group", None) is not None:
            return None
        tokens = _concat_ids(req.origin_input_ids, req.output_ids)
        hit = int(self.prefix_probe(req, tokens).device_prefix_len)
        uncached = max(0, len(tokens) - hit)
        if not uncached:
            return None
        chunk = getattr(self.scheduler, "chunked_prefill_size", None)
        governor = getattr(self.scheduler, "_orvix_prefill_governor", None)
        if governor is not None:
            chunk = governor.planned_budget(self.scheduler.running_batch)
        limits = [v for v in (chunk, getattr(self.scheduler, "max_prefill_tokens", None))
                  if type(v) is int and v > 0]
        if not limits:
            return None
        full = ((min(uncached, min(limits)) + page - 1) // page + 1) * page
        needed = {"full": full}
        if "swa" in pressure:
            window = getattr(self.cache, "sliding_window_size", None)
            if type(window) is not int or window <= 0:
                return None
            needed["swa"] = min(full, ((window + page - 1) // page + 1) * page)
        return {name: max(0, count - pressure[name]["free"])
                for name, count in needed.items() if name in pressure}

    def _backup_closure(self, node):
        """Use the native parent-before-child action, not a naked leaf copy.

        Shared/locked ancestors make this conservative policy decline. We do
        not evict those ancestors. Full/SWA/index/compressor transfers remain
        native. The budget is logical Full tokens, not physical transfer bytes.
        """
        if node.backuped and all(cd.value is None or cd.host_value is not None
                                 for cd in node.component_data):
            return [], 0
        build = getattr(self.core, "build_backup_kv_action", None)
        if not callable(build):
            return None, 0
        ids = list(build(node.id).node_ids)
        if not ids:
            # A Full host hit alone is insufficient when a resident auxiliary
            # component is not backed. Do not interpret an empty action as ACK.
            return None, 0
        full = next(ct for ct in self.cache.components if _name(ct) == "FULL")
        count = 0
        for node_id in ids:
            parent = self.core.node_by_id(node_id)
            if (parent.write_through_pending_id is not None or parent.load_back_pending_id is not None
                    or getattr(parent, "write_back_queued", False)
                    or any(cd.lock_ref or cd.host_lock_ref for cd in parent.component_data)):
                return None, 0
            cd = parent.component_data[full]
            if cd.value is not None and cd.host_value is None:
                count += len(cd.value)
        return ids, count

    def _plan_demand_bundle(self, pressure):
        if not any(p["pressure"] for p in pressure.values()):
            self.last_reason = "no_pool_pressure_with_request_slot"
            return None
        candidates = []
        seen = set()
        for record in self.records.values():
            node = self._eligible(record, pressure, check_footprint=False)
            if node is None or node.id in seen:
                continue
            seen.add(node.id)
            closure, backup_tokens = self._backup_closure(node)
            if closure is None:
                self.counts["backup_closure_unavailable_or_locked"] += 1
                continue
            sizes = {_name(ct).lower(): len(node.component_data[ct].value)
                     if node.component_data[ct].value is not None else 0
                     for ct in self.cache.components}
            candidates.append((record, node, closure, backup_tokens, sizes))
        if not candidates:
            self.last_reason = "no_safe_idle_leaf"
            return None
        for req in list(self.scheduler.waiting_queue)[:self.config.max_waiter_scan]:
            if self.config.require_native_rejection:
                feedback = self.scheduler._orvix_admission_feedback
                if not feedback.recently_capacity_rejected(req):
                    self.counts["no_recent_native_rejection"] += 1
                    continue
            deficits = self._demand(req, pressure)
            if deficits is None:
                self.counts["demand_evidence_missing"] += 1
                continue
            deficits = {k: v for k, v in deficits.items() if v > 0}
            if not deficits:
                continue
            # Prefer already-backed victims, then least Full-copy work per
            # deficient-pool token. This is not a transfer-time predictor.
            def key(item):
                useful = max((min(v, item[4].get(k, 0)) for k, v in deficits.items()), default=0)
                return (bool(item[2]), item[3] / max(1, useful), -useful, item[0].rid)
            selected, copied, backups, relief = [], set(), [], {k: 0 for k in deficits}
            full_tokens = 0
            full_type = next(ct for ct in self.cache.components if _name(ct) == "FULL")
            for item in sorted(candidates, key=key):
                record, node, closure, _, sizes = item
                if not any(relief[k] < deficits[k] and sizes.get(k, 0) for k in deficits):
                    continue
                extra = [n for n in closure if n not in copied]
                tokens = sum(len(n.component_data[full_type].value)
                             for n in (self.core.node_by_id(i) for i in extra)
                             if n.component_data[full_type].value is not None
                             and n.component_data[full_type].host_value is None)
                if full_tokens + tokens > self.config.max_backup_tokens:
                    self.counts["bundle_backup_budget_skips"] += 1
                    continue
                if set(deficits) == {"swa"} and tokens > self.config.max_full_copy_per_swa_token * max(1, sizes.get("swa", 0)):
                    self.counts["swa_copy_amplification_skips"] += 1
                    continue
                selected.append(item)
                full_tokens += tokens
                backups.extend(extra)
                copied.update(extra)
                for k in deficits:
                    relief[k] += sizes.get(k, 0)
                if all(relief[k] >= deficits[k] for k in deficits):
                    return {"action": "backup" if backups else "demote",
                            "beneficiary": str(req.rid), "deficits": deficits,
                            "victims": [item[0].rid for item in selected],
                            "nodes": [item[1].id for item in selected],
                            "fingerprints": [item[0].fingerprint for item in selected],
                            "backup_nodes": backups, "backup_full_tokens": full_tokens,
                            "estimated_pool_relief": relief}
                if len(selected) >= self.config.max_bundle_leaves:
                    break
        self.last_reason = "no_bundle_can_help_a_waiting_chunk"
        self.counts["no_useful_bundle"] += 1
        return None

    def _tick_demand_aware(self):
        # Only the broadcast clock throttles planning; all ranks take the same
        # cadence. Every due tick participates in ONE common plan collective,
        # including empty plans and local errors. Never skip it on local memory.
        if (self.last_tick_now is not None
                and self.now - self.last_tick_now < self.config.scan_interval_s):
            self.counts["throttled_ticks"] += 1
            return
        start = time.perf_counter()
        plan = error = None
        try:
            if self.effective_mode == "demand-aware":
                self.last_pressure = self._pressure()
                plan = self._plan_demand_bundle(self.last_pressure)
        except Exception as exc:
            error = type(exc).__name__
        self.consensus({"phase": "bundle_plan", "epoch": self.epoch, "now": self.now,
                        "config": asdict(self.config), "mode": self.effective_mode,
                        "plan": plan, "error": error})
        self.counts["plan_consensus_calls"] += 1
        self.last_tick_now = self.now
        self.counts["ticks"] += 1
        if error:
            raise RuntimeError("demand-aware planning failed: " + error)
        if plan is None:
            self.counts["no_action_ticks"] += 1
            self.host_control_elapsed_s += time.perf_counter() - start
            return
        receipt = {"at": self.now, **plan, "error": None}
        try:
            if plan["action"] == "backup":
                self.cache._execute_and_commit_kv_backup(
                    self.backup_factory(node_ids=plan["backup_nodes"]), write_back=False)
                accepted = []
                for rid in plan["victims"]:
                    record = self.records[rid]
                    node = self.core.node_by_id(record.node_id)
                    record.backup_queued = bool(node.backuped or node.write_through_pending_id is not None)
                    accepted.append(record.backup_queued)
                self.cache.flush_pending_backups()
                receipt["accepted"] = accepted
                self.counts["backup_batches"] += 1
                self.counts["backup_intents_queued"] += sum(accepted)
                self.counts["planned_backup_full_tokens"] += plan["backup_full_tokens"]
            else:
                freed = {"full": 0, "swa": 0}
                for rid, node_id in zip(plan["victims"], plan["nodes"]):
                    tracker = {ct: 0 for ct in self.cache.tree_components}
                    if self.cache._evict_device_leaf(node_id, tracker) is not None:
                        raise RuntimeError("native demotion unexpectedly needs a backup")
                    self.records[rid].demoted = True
                    self.counts["demoted_leaves"] += 1
                    for ct, count in tracker.items():
                        name = _name(ct).lower()
                        if name in freed:
                            freed[name] += int(count)
                            self.counts["demoted_" + name + "_tokens"] += int(count)
                receipt["actual_pool_tokens_freed"] = freed
                self.counts["demotion_batches"] += 1
            # A beneficiary in this receipt is an estimated opportunity, NOT
            # proof that it was admitted or that a transfer shortened its wait.
        except Exception as exc:
            receipt["error"] = type(exc).__name__
        self.consensus({"phase": "bundle_outcome", "receipt": receipt})
        self.counts["outcome_consensus_calls"] += 1
        self.action_log.append(receipt)
        self.last_reason = plan["action"]
        self.host_control_elapsed_s += time.perf_counter() - start
        if receipt["error"]:
            raise RuntimeError("demand-aware native action failed: " + receipt["error"])

    def tick(self):
        if self.config.mode == "demand-aware":
            try:
                self._tick_demand_aware()
            except Exception:
                self.failed = True
                self.counts["controller_failures"] += 1
                raise
            return
        # mode/clock are broadcast controls. last_tick_now starts equal and is
        # advanced only after the preceding gate agreed on that control epoch.
        if self.config.mode == "off" or self.last_tick_now == self.now:
            return
        if self.config.mode == "observe":
            self.last_tick_now = self.now
            self.counts["ticks"] += 1
            try:
                self.last_pressure = self._pressure()
                if any(info["pressure"] for info in self.last_pressure.values()):
                    self.counts["pressure_ticks_with_waiters"] += 1
            except Exception:
                self.counts["observation_failures"] += 1
            return
        try:
            # Rank-local capability/bookkeeping state must not cause one rank
            # to skip the plan collective that another rank enters.
            self.consensus({"phase": "gate", "epoch": self.epoch, "now": self.now,
                            "config": asdict(self.config), "effective_mode": self.effective_mode})
            self.last_tick_now = self.now
            self.counts["ticks"] += 1
            if self.effective_mode != "tool-aware":
                return
            handled = set()
            for action_index in range(self.config.max_actions_per_tick):
                record = node = None
                plan = planning_error = None
                try:
                    pressure = self._pressure()
                    self.last_pressure = pressure
                    if any(info["pressure"] for info in pressure.values()):
                        if action_index == 0:
                            self.counts["pressure_ticks_with_waiters"] += 1
                        for candidate in self.records.values():
                            if candidate.rid in handled:
                                continue
                            leaf = self._eligible(candidate, pressure)
                            if leaf is not None:
                                record, node = candidate, leaf
                                break
                    plan = None if node is None else {
                        "node": node.id,
                        "fingerprint": record.fingerprint,
                        "action": "demote" if record.backup_queued and node.backuped else "backup",
                    }
                except Exception as exc:
                    planning_error = type(exc).__name__
                # Empty plans also participate. A rank-local capacity outcome
                # or read-only probe error cannot bypass the same collective.
                self.consensus({"phase": "plan", "epoch": self.epoch, "now": self.now,
                                "plan": plan, "planning_error": planning_error})
                if planning_error is not None:
                    raise RuntimeError("cache planning failed: " + planning_error)
                if plan is None:
                    break
                handled.add(record.rid)
                # A native write may already back this leaf. Never demote a
                # newly assigned host slot until its native ack released locks.
                outcome = {"phase": "outcome", "node": node.id, "error": None}
                try:
                    if plan["action"] == "demote":
                        tracker = {ct: 0 for ct in self.cache.tree_components}
                        remaining = self.cache._evict_device_leaf(node.id, tracker)
                        if remaining is not None:
                            raise RuntimeError("native demotion unexpectedly requires backup")
                        record.demoted = True
                        self.counts["demoted_leaves"] += 1
                        self.counts["demoted_full_tokens"] += int(tracker.get(next(ct for ct in self.cache.components if _name(ct) == "FULL"), 0))
                        self.counts["demoted_swa_tokens"] += sum(int(v) for ct, v in tracker.items() if _name(ct) == "SWA")
                        outcome["evicted"] = {_name(ct): int(v) for ct, v in tracker.items()}
                    else:
                        self.cache._execute_and_commit_kv_backup(self.backup_factory(node_ids=[node.id]), write_back=False)
                        record.backup_queued = bool(node.backuped or node.write_through_pending_id is not None)
                        if record.backup_queued:
                            self.cache.flush_pending_backups()
                            self.counts["backup_intents_queued"] += 1
                        else:
                            self.counts["backup_allocation_declined"] += 1
                        outcome["backup_queued"] = record.backup_queued
                except Exception as exc:
                    outcome["error"] = type(exc).__name__
                self.consensus(outcome)
                if outcome["error"] is not None:
                    raise RuntimeError("native cache action failed: " + outcome["error"])
        except Exception as exc:
            # Keep native acknowledgement ownership intact. No synthetic free,
            # rollback or unlock is safe after an unexpected native exception.
            self.failed = True
            self.counts["controller_failures"] += 1
            LOGGER.exception("Tool cache controller failed; refusing divergent native cache execution")
            raise RuntimeError("tool cache native contract/TP consistency failed") from exc

    def on_admitted(self, requests):
        """Attach observed native admission to a bounded demotion receipt.

        This temporal association is NOT a counterfactual causal speedup. The
        broadcast clock has heartbeat granularity, so report it as coarse.
        """
        if self.config.mode != "demand-aware":
            return
        rows = {str(getattr(req, "rid", "")): req for req in requests}
        for receipt in self.action_log:
            if (receipt["action"] != "demote" or receipt.get("error") is not None
                    or receipt.get("beneficiary_admitted") or receipt["beneficiary"] not in rows):
                continue
            req = rows[receipt["beneficiary"]]
            receipt["beneficiary_admitted"] = True
            receipt["beneficiary_admitted_at_broadcast_clock"] = self.now
            receipt["demotion_to_admission_s_coarse"] = max(0., self.now - receipt["at"])
            receipt["admission_prefix_len"] = int(getattr(req, "prefix_len", 0))
            self.counts["demotion_receipts_followed_by_native_admission"] += 1

    def snapshot(self):
        return {
            "requested_mode": self.config.mode,
            "effective_mode": self.effective_mode,
            "config": asdict(self.config),
            "capabilities": self.capabilities,
            "epoch": self.epoch,
            "clock_s": self.now,
            "counts": dict(self.counts),
            "tracked_requests": len(self.records),
            "sleeping_requests": sum(bool(r.tools) and not r.ready for r in self.records.values()),
            "pending_backup_intents": sum(r.backup_queued and not r.demoted for r in self.records.values()),
            "tracking_exhausted": self.tracking_exhausted,
            "controller_failed": self.failed,
            "last_pressure": self.last_pressure,
            "last_admission_room": self.last_admission_room,
            "footprint_model": "immediate_chunk_estimate_not_native_admission_guarantee",
            "last_reason": self.last_reason,
            "host_control_elapsed_s": self.host_control_elapsed_s,
            "action_log": list(self.action_log),
            "pending_intent_semantics": "backed_but_not_demoted; not a DMA-inflight count",
        }


def install_cache_control(scheduler):
    capabilities = inspect_cache_capabilities(scheduler)
    factory = None
    consensus = None
    cache = scheduler.tree_cache
    # Choose transport from the shared scheduler topology, never from the
    # rank-local capability verdict. PP>1 is unqualified everywhere and performs
    # no tool-aware actions; the native PP event loop owns its communication.
    if getattr(cache, "pp_size", 1) == 1 and callable(getattr(cache, "_all_reduce", None)):
        import torch
        enabled = torch.tensor([int(capabilities["tool_demotion_supported"])], dtype=torch.int64, device="cpu")
        cache._all_reduce(enabled, torch.distributed.ReduceOp.MIN)
        capabilities["capability_consensus_applied"] = True
        if not int(enabled[0]):
            capabilities["tool_demotion_supported"] = False
            capabilities["reasons"] = [*capabilities["reasons"], "capability_unavailable_on_one_or_more_tp_ranks"]
        def consensus(value):
            digest = int.from_bytes(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).digest()[:8], "little") & ((1 << 62) - 1)
            agreed = torch.tensor([digest, -digest], dtype=torch.int64, device="cpu")
            scheduler.tree_cache._all_reduce(agreed, torch.distributed.ReduceOp.MIN)
            if int(agreed[0]) != -int(agreed[1]):
                raise CachePlanMismatch("tool-aware cache decisions diverged across TP ranks")
    if capabilities["tool_demotion_supported"]:
        from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
        factory = BackupKV
    policy = CacheControl(scheduler, backup_factory=factory, capabilities=capabilities, consensus=consensus)
    if callable(getattr(cache, "checkpoint_into_tree", None)):
        original_checkpoint = cache.checkpoint_into_tree
        def checkpoint(req, *args, **kwargs):
            result = original_checkpoint(req, *args, **kwargs)
            policy.completed(req)
            return result
        cache.checkpoint_into_tree = checkpoint
    if callable(getattr(scheduler, "_process_hicache_events", None)):
        original_events = scheduler._process_hicache_events
        def process_events(*args, **kwargs):
            result = original_events(*args, **kwargs)
            policy.tick()
            return result
        scheduler._process_hicache_events = process_events
    if callable(getattr(cache, "reset", None)):
        original_reset = cache.reset
        def reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            policy.reset_tracking()
            return result
        cache.reset = reset
    scheduler._orvix_cache_control = policy
    return policy
