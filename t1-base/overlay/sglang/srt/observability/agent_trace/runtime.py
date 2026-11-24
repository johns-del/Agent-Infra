"""Fail-open, bounded local spool. No GPU calls, HTTP, or business imports.

The local spool is relayed separately to the existing authenticated Collector.
All configuration is opt-in; worker identities survive PID reuse via UUIDs.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import socket
import threading
import time
import uuid
from datetime import datetime, timezone


def point():
    return {"ts": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "mono_ns": str(time.monotonic_ns()), "clock_domain_id": _clock_domain}


_clock_domain = uuid.uuid4().hex
_writers = {}
DEFAULT_MAX_STORED_BYTES = 4 * 1024**3
_log = logging.getLogger(__name__)


def stored_byte_budget(value=None):
    value = os.environ.get('ORVIX_TRACE_MAX_STORED_BYTES', DEFAULT_MAX_STORED_BYTES) if value is None else value
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError('ORVIX_TRACE_MAX_STORED_BYTES requires a positive byte budget') from None
    if isinstance(value, bool) or not 1 <= result <= 1024**4:
        raise ValueError('ORVIX_TRACE_MAX_STORED_BYTES must be between 1 byte and 1 TiB')
    return result


def enabled():
    return os.environ.get("ORVIX_ENGINE_TRACE", "0") == "1"


def flag(name, default=False):
    return os.environ.get(name, "1" if default else "0") == "1"


def integer(name, default, lo=1, hi=10_000_000):
    return max(lo, min(hi, int(os.environ.get(name, default))))


def number(name, default, lo=0.05, hi=3600.0):
    return max(lo, min(hi, float(os.environ.get(name, default))))


def primitive(value, depth=0):
    """Never stringify tensors or arbitrary application objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 6:
        return {"unavailable": "depth_limit"}
    if isinstance(value, dict):
        return {str(k): primitive(v, depth+1) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [primitive(v, depth+1) for v in value]
    return {"unavailable": "non_scalar", "native_type": type(value).__name__}


class Writer:
    def __init__(self, role, rank=None):
        # Missing explicit configuration disables initialization, never inference.
        self.instance = os.environ["ORVIX_TRACE_INSTANCE_ID"]
        self.deployment = os.environ["ORVIX_TRACE_DEPLOYMENT_ID"]
        directory = Path(os.environ["ORVIX_TRACE_DIR"])
        if not directory.is_absolute() or directory.is_symlink():
            raise ValueError("trace directory must be an absolute non-symlink path")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.source = uuid.uuid4().hex
        self.role, self.rank = role, primitive(rank or {})
        self.boot = point()["ts"]
        self.seq = self.dropped = self.write_errors = self.callback_errors = 0
        self.attempted_events = self.stored_budget_drops = self.low_disk_drops = 0
        self.quota_warnings = {}; self.quota_first_observed_at = {}
        self.pending_bytes = self.peak_pending_bytes = 0
        self.oversized_events = self.queue_byte_drops = self.queue_event_drops = self.closed_drops = 0
        self.max_observed_event_bytes = 0
        self.spool_bytes = self.spool_quota_drops = 0
        self.max_spool_bytes = integer("ORVIX_TRACE_SPOOL_BYTES", 2*1024*1024*1024, hi=1024*1024*1024*1024)
        self.max_stored_bytes = stored_byte_budget()
        self.stored_bytes = 0
        self.last_storage_refresh = 0.
        self.min_free_bytes = integer('ORVIX_TRACE_MIN_FREE_BYTES', 0, lo=0, hi=1024**4)
        self.recording_stopped_reason = None
        self.callback_ns = self.callback_calls = 0
        # Hooks nest (result_after -> tick -> sampler.tick); callback_ns counts
        # only outermost calls so nested time is not charged twice.
        self._local = threading.local()
        self.callback_breakdown = {}
        self.max_bytes =integer("ORVIX_TRACE_QUEUE_BYTES", 16*1024*1024, hi=256*1024*1024)
        self.max_event = integer("ORVIX_TRACE_EVENT_BYTES", 8*1024*1024, hi=64*1024*1024)
        self.queue = queue.Queue(integer("ORVIX_TRACE_QUEUE_EVENTS", 2048, hi=100000))
        self.lock = threading.Lock()
        self.closed = False
        self.path = directory / (role + "-" + self.source + ".jsonl")
        self.health_path = self.path.with_suffix(".health.json")
        self.segment_bytes = integer("ORVIX_TRACE_SEGMENT_BYTES", 0, lo=0,
                                     hi=self.max_spool_bytes)
        self.segment = self.segment_written = self.total_written_bytes = 0
        self.base_path = self.path
        # Exclusive creation; no overwrite of old evidence, private permissions.
        self.fd = os.fdopen(os.open(self.path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), "w", encoding="utf-8")
        self.thread = threading.Thread(target=self._drain, daemon=True, name="orvix-engine-trace")
        self.thread.start()
        atexit.register(self.close)
        self.emit("engine.trace.capabilities", {
            "role": role, "rank": self.rank,
            "protocol": "orvix-engine-observation/1", "mode": "passive",
            "device_timing": "native_completed_ack_only",
            "physical_observation": "opt_in_bounded_index_snapshot",
            "continuous_physical_lifetime": False,
            "gpu_synchronization_added": False,
            "wall_clock_uncertainty_ms": None,
            "payload_capture": "agent_gateway_only",
        })

    def emit(self, kind, data, rid=None, span=None, **ids):
        with self.lock:
            self.attempted_events += 1
            stopped = self.recording_stopped_reason
        if stopped:
            self._quota_drop(stopped)
            return None
        try:
            with self.lock:
                self.seq += 1
                seq = self.seq
            p = point()
            e = {"schema_version": "1", "event_id": self.source+":"+str(seq),
                 "type": kind, "source": "engine", "source_instance_id": self.source,
                 "source_seq": seq, **p, "available_at": p.copy(),
                 "host": socket.gethostname(), "pid": os.getpid(), "boot_ts": self.boot,
                 "batch": None, "config_ref": os.environ.get("ORVIX_TRACE_CONFIG_REF"),
                 "instance_id": self.instance, "deployment_id": self.deployment,
                 "source_role": self.role, "rank": self.rank,
                 "engine_rid": str(rid) if rid is not None else None,
                 "data": primitive(data), "missing_reason": {
                     "batch": "not_applicable", "config_ref": "inline_configuration",
                 }, **ids}
            if span or kind.endswith((".start", ".end", ".accepted")):
                e["span_id"] = span or e["event_id"]
            def null_times(v, path="data"):
                if not isinstance(v, dict):
                    return
                for k, x in v.items():
                    q=path+"."+k
                    if x is None and (k.startswith("t_") or k in ("available_at", "consumer_first_use_at")):
                        e["missing_reason"][q]="not_observed"
                    elif isinstance(x, dict):
                        null_times(x,q)
                    elif isinstance(x, list):
                        for j,y in enumerate(x): null_times(y,q+f"[{j}]")
            null_times(e["data"])
            line=json.dumps(e,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n"
            size=len(line.encode())
            with self.lock:
                self.max_observed_event_bytes=max(self.max_observed_event_bytes,size)
                if self.closed or size>self.max_event or self.pending_bytes+size>self.max_bytes:
                    if self.closed:self.closed_drops+=1
                    elif size>self.max_event:self.oversized_events+=1
                    else:self.queue_byte_drops+=1
                    self.dropped+=1
                    return None
                try:
                    self.queue.put_nowait((line,size))
                except queue.Full:
                    self.queue_event_drops+=1
                    self.dropped+=1
                    return None
                self.pending_bytes+=size
                self.peak_pending_bytes=max(self.peak_pending_bytes,self.pending_bytes)
            return e["event_id"]
        except Exception:
            with self.lock: self.dropped+=1
            return None

    def safe(self, fn, *args, **kwargs):
        return self.safe_labeled(getattr(fn, "__qualname__", None) or type(fn).__name__, fn, *args, **kwargs)

    def safe_labeled(self, label, fn, *args, **kwargs):
        t=time.perf_counter_ns()
        depth=getattr(self._local,"depth",0)
        self._local.depth=depth+1
        try:
            return fn(*args,**kwargs)
        except Exception:
            self.callback_errors+=1
            return None
        finally:
            self._local.depth=depth
            dt=time.perf_counter_ns()-t
            self.callback_calls+=1
            if depth==0:self.callback_ns+=dt
            # Inclusive per-hook wall time, for locating hot spots on a real run.
            row=self.callback_breakdown.get(label)
            if row is None:row=self.callback_breakdown[label]=[0,0,0]
            row[0]+=1;row[1]+=dt
            if dt>row[2]:row[2]=dt

    def health(self):
        return {"source_instance_id": self.source, "instance_id":self.instance,
                "deployment_id":self.deployment,"last_attempted_source_seq":self.seq,
                "attempted_events":self.attempted_events,
                "dropped_events":self.dropped,"write_errors":self.write_errors,
                "stored_budget_drops":self.stored_budget_drops,"low_disk_drops":self.low_disk_drops,
                "quota_dropped_events":self.stored_budget_drops+self.low_disk_drops+self.spool_quota_drops,
                "quota_warnings":dict(self.quota_warnings),
                "quota_first_observed_at":dict(self.quota_first_observed_at),
                "oversized_events":self.oversized_events,"queue_byte_drops":self.queue_byte_drops,
                "queue_event_drops":self.queue_event_drops,"closed_drops":self.closed_drops,
                "max_observed_event_bytes":self.max_observed_event_bytes,
                "max_event_bytes":self.max_event,"max_queue_bytes":self.max_bytes,
                "callback_errors":self.callback_errors,"callback_calls":self.callback_calls,
                "spool_bytes":self.spool_bytes,"spool_quota_drops":self.spool_quota_drops,
                "max_spool_bytes":self.max_spool_bytes,
                "segment_bytes":self.segment_bytes,"total_written_bytes":self.total_written_bytes,
                "spool_accounting":"unarchived_segments" if self.segment_bytes else "lifetime_file",
                "max_stored_bytes":self.max_stored_bytes,"stored_bytes":self.stored_bytes,
                "min_free_bytes":self.min_free_bytes,
                "recording_stopped_reason":self.recording_stopped_reason,
                "callback_cpu_ns":self.callback_ns,
                "callback_ns_semantics":"outermost_hook_wall_ns_on_calling_thread_not_double_counted",
                "callback_breakdown_top":[{"hook":k,"calls":v[0],"inclusive_wall_ns":v[1],"max_wall_ns":v[2]}
                    for k,v in sorted(list(self.callback_breakdown.items()),key=lambda kv:-kv[1][1])[:12]],
                "pending_bytes":self.pending_bytes,
                "peak_pending_bytes":self.peak_pending_bytes,"closed":self.closed,
                "collection_precision":"host_observation", "observed_at":point()}

    def _quota_drop(self, reason, *, stop=False):
        """Count every lost event; alarm once per reason without logging data.

        Health is outside the payload quota, so exhaustion remains observable.
        Sticky storage/disk stops never auto-resume and hide a coverage gap.
        """
        with self.lock:
            self.dropped += 1
            if reason == 'stored_byte_budget_exhausted': self.stored_budget_drops += 1
            elif reason == 'low_disk_space': self.low_disk_drops += 1
            elif reason == 'spool_byte_budget_exhausted': self.spool_quota_drops += 1
            if stop and self.recording_stopped_reason is None:
                self.recording_stopped_reason = reason
            first = reason not in self.quota_warnings
            if first:
                self.quota_warnings[reason] = 1
                self.quota_first_observed_at[reason] = point()
        if first:
            _log.warning('ORVIX_TRACE_QUOTA_EXHAUSTED role=%s source=%s reason=%s '
                         'stored_bytes=%d max_stored_bytes=%d spool_bytes=%d max_spool_bytes=%d '
                         'min_free_bytes=%d; recording coverage is incomplete; inspect native health',
                         self.role, self.source, reason, self.stored_bytes, self.max_stored_bytes,
                         self.spool_bytes, self.max_spool_bytes, self.min_free_bytes)
            self._health_file()

    def _health_file(self):
        try:
            tmp=self.health_path.with_suffix(".tmp")
            fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            with os.fdopen(fd,"w") as f: json.dump(self.health(),f)
            os.replace(tmp,self.health_path)
        except Exception:
            self.write_errors+=1

    def _seal_segment(self):
        self.fd.flush()
        self.fd.close()
        # The relay may archive only a closed, fully acknowledged segment.
        marker = self.path.with_suffix(".closed")
        with os.fdopen(os.open(marker, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), "w") as f:
            f.write(str(self.path.stat().st_size))

    def _refresh_spool_bytes(self):
        self.fd.flush()
        size = 0
        for path in self.base_path.parent.glob(self.base_path.stem + "*.jsonl"):
            try:size += path.stat().st_size
            except FileNotFoundError:pass  # Relay archived this sealed segment.
        self.spool_bytes = size

    def _rotate_segment(self):
        self._seal_segment()
        self.segment += 1
        self.path = self.base_path.with_name(self.base_path.stem + f"-{self.segment:06d}.jsonl")
        self.fd = os.fdopen(os.open(self.path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600),
                           "w", encoding="utf-8")
        self.segment_written = 0
        self._refresh_spool_bytes()

    def _refresh_stored_bytes(self):
        total=0
        for directory in (self.base_path.parent,self.base_path.parent/'archive'):
            for path in directory.glob(self.base_path.stem+'*.jsonl*'):
                try:total+=path.stat().st_blocks*512
                except FileNotFoundError:pass
        self.stored_bytes=total;self.last_storage_refresh=time.monotonic()

    def _drain(self):
        last=0.0
        while True:
            try:
                item=self.queue.get(timeout=0.2)
            except queue.Empty:
                item=None
            if item is not None:
                line,size=item
                try:
                    if self.max_stored_bytes and time.monotonic()-self.last_storage_refresh>=15:
                        self._refresh_stored_bytes()
                    if self.max_stored_bytes and self.stored_bytes+size>self.max_stored_bytes:
                        # ACK/archive compression may have freed space since the
                        # periodic sample. Refresh before making a sticky stop.
                        self.fd.flush();self._refresh_stored_bytes()
                        if self.stored_bytes+size>self.max_stored_bytes:
                            self._quota_drop('stored_byte_budget_exhausted',stop=True)
                            continue
                    if self.min_free_bytes and shutil.disk_usage(self.path.parent).free<size+self.min_free_bytes:
                        self._quota_drop('low_disk_space',stop=True)
                        continue
                    if self.recording_stopped_reason:
                        self._quota_drop(self.recording_stopped_reason)
                        continue
                    if self.segment_bytes and self.segment_written and self.segment_written+size>self.segment_bytes:
                        self._rotate_segment()
                    if self.spool_bytes+size>self.max_spool_bytes:
                        self._quota_drop('spool_byte_budget_exhausted')
                    else:
                        self.fd.write(line);self.spool_bytes+=size
                        self.segment_written+=size;self.total_written_bytes+=size
                        self.stored_bytes+=size
                except Exception:
                    self.write_errors+=1
                finally:
                    with self.lock: self.pending_bytes-=size
                    self.queue.task_done()
            now=time.monotonic()
            if now-last>=1.0:
                try:
                    self.fd.flush()
                    if self.segment_bytes:self._refresh_spool_bytes()
                except Exception: self.write_errors+=1
                self._health_file();last=now
            if self.closed and self.queue.empty(): break
        try:
            if self.segment_bytes:self._seal_segment()
            else:self.fd.flush(); self.fd.close()
        except Exception: self.write_errors+=1
        self._health_file()

    def close(self):
        if self.closed: return
        self.emit("engine.trace.final",self.health())
        with self.lock: self.closed=True
        self.thread.join(timeout=2.0)
        self._health_file()


def get_writer(role="api",rank=None):
    if not enabled(): return None
    key=(os.getpid(),role)
    if key not in _writers:
        try: _writers[key]=Writer(role,rank)
        except Exception as exc:
            _writers[key]=None
            _log.warning('ORVIX_TRACE_INITIALIZATION_FAILED role=%s error_type=%s; recording unavailable',
                         role,type(exc).__name__)
    return _writers[key]
