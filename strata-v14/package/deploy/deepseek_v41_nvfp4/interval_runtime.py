"""Tool interval controller: one owner for leases, backup, demotion and H2D.

Default OFF. Activation requires a qualified private runtime and excludes old
F/C and v5/v6 action policies (V11 co-runs cleanly with PrefillGovernor batch
time-slicing). All clocks arrive through the broadcast control plane. Transfer
completion/lock release is owned by native HiCache TP ACKs.
"""
from __future__ import annotations
from collections import Counter, OrderedDict, deque
from dataclasses import asdict
import hashlib
import math
from .interval_core import (
    IMPLEMENTATION,
    KINDS,
    STATUSES,
    VERSION,
    ByteBucket,
    IntervalConfig,
    Lifecycle,
    QueueItem,
    order_window,
    validate_events,
)
from .interval_probe import Fences, concat_token_ids, probe_tiered_prefix
from .interval_native import NativeActions
from .unified_cache_control import CachePlanMismatch, _metadata, _name


def _normalize_fallback_events(raw_events, now):
    """Map legacy replay `events` wire entries to causal interval events when `interval_events` is omitted."""
    if not isinstance(raw_events, list) or len(raw_events) > 4096:
        raise ValueError('bounded events list required')
    out = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        kind = item.get('kind')
        rid = item.get('request_id')
        if kind not in KINDS or not isinstance(rid, str) or not rid:
            continue
        cleaned = {
            'kind': kind,
            'event_id': str(item.get('event_id', '')),
            'request_id': rid,
            'observed_at': item.get('observed_at', now),
        }
        if kind in {'tool_start', 'tool_end'}:
            cleaned['tool_id'] = item.get('tool_id')
        if kind == 'tool_start':
            cleaned['tool_type'] = item.get('tool_type') or 'unknown'
        if kind == 'tool_end':
            status = item.get('status', 'ok')
            cleaned['status'] = status if status in STATUSES else 'ok'
        out.append(cleaned)
    return validate_events(out, now)


def _workflow_key(req):
    meta = _metadata(req)
    family = str(meta.get('family_id', meta.get('user_id', '')) or '')
    graph = str(meta.get('graph_id', '') or '')
    session = str(meta.get('session_id', getattr(req, 'session_id', '')) or '')
    scope = str(meta.get('scope_id', meta.get('scope', '')) or '')
    if graph or session:
        return ('scope', family, graph, scope, session)
    if scope:
        return ('session', family, session, scope)
    return None


def _program_key(req):
    meta = _metadata(req)
    family = str(meta.get('family_id', meta.get('user_id', '')) or '')
    graph = str(meta.get('graph_id', '') or '')
    session = str(meta.get('session_id', getattr(req, 'session_id', '')) or '')
    if graph or session:
        return ('prog', family, graph or session)
    return None


class IntervalCacheController:
    def __init__(self, scheduler, tracking, key_factory, *, actions=None, consensus=None):
        self.s = scheduler
        self.tracking = tracking
        self.cache = scheduler.tree_cache
        self.config = IntervalConfig()
        self.now = 0.0
        self.last_tick = -1.0
        self.urgent_event = False
        self.fences = Fences()
        self.actions = actions or NativeActions(scheduler, tracking, self.fences)
        self.consensus = consensus or tracking.consensus
        self.key_factory = key_factory
        self.lifecycle = Lifecycle(self.config)
        self.capabilities = self.actions.capabilities()
        self.counts = Counter()
        self.log = deque(maxlen=256)
        self.buckets = {
            d: ByteBucket(self.config.transfer_burst_bytes, self.config.transfer_rate_bytes_s)
            for d in ('d2h', 'h2d')
        }
        self.service = OrderedDict()
        self.bypass = {}
        self.probes = {}
        self.ready_leases = {}
        self.anchor_leases = {}  # node_id -> (until_s, parent_rid, user)
        self.prefetched_anchors = set()
        self.tickets = {}
        self.output_seen = {}
        self.ended_nodes = set()
        self.demoted_closed_nodes = set()
        self.poisoned = False
        self.tracking_limit = False
        self.native_installed = False
        self.known_anchor_rids = set()
        self.completed_at = {}
        self.superseded_at = {}
        self.workflow_turns = OrderedDict()
        self.rid_turns = OrderedDict()
        self.program_service = OrderedDict()
        self.program_arrival = OrderedDict()
        self.arrival_counter = 0
        self.last_admission_action = -1.0
        self.hbm_pressure_active = False
        self.s._orvix_hbm_pressure_active = False

    @property
    def active(self):
        return (
            self.config.mode == 'active'
            and not self.poisoned
            and not self.lifecycle.overflow
            and not self.tracking_limit
        )

    def idle(self):
        native_idle = getattr(self.s, 'is_fully_idle', None)
        if callable(native_idle) and not native_idle():
            return False
        if getattr(self.s, 'result_queue', None):
            return False
        if not callable(native_idle) and getattr(self.s, 'last_batch', None) is not None:
            return False
        cc = getattr(self.cache, 'cache_controller', None)
        if cc is not None and any(
            getattr(cc, n, None) for n in ('load_queue', 'write_queue', 'ack_load_queue', 'ack_write_queue')
        ):
            return False
        return (
            not len(self.s.waiting_queue)
            and not len(self.s.running_batch.reqs)
            and getattr(self.s, 'chunked_req', None) is None
            and not self.actions.pending
            and not self.tickets
            and not getattr(self.cache, 'ongoing_load_back', {})
            and not getattr(self.cache, 'ongoing_write_through', {})
        )

    def validate_control(self, payload):
        if self.poisoned or self.actions.unsafe:
            raise ValueError('unsafe interval state requires private process restart')
        raw = payload.get('interval_cache', {})
        if not isinstance(raw, dict) or set(raw) - set(asdict(IntervalConfig())):
            raise ValueError('unknown interval configuration')
        cfg = IntervalConfig(**(asdict(self.config) | raw)).validate()
        now = payload['now']
        if isinstance(now, bool) or not isinstance(now, (float, int)) or not math.isfinite(now) or now < self.now:
            raise ValueError('invalid interval clock')
        if 'interval_events' in payload:
            events = validate_events(payload['interval_events'], now)
        elif cfg.mode != 'off' and 'events' in payload and payload['events'] is not None:
            events = _normalize_fallback_events(payload['events'], now)
        else:
            events = []
        reset = payload.get('reset', False)
        if (cfg != self.config or reset) and not self.idle():
            raise ValueError('drain engine and HiCache before interval reconfiguration/reset')
        if cfg.mode == 'active' and not self.capabilities['supported']:
            raise ValueError('interval backend unsupported: ' + ','.join(self.capabilities['reasons']))
        return cfg, float(now), events, reset

    def apply_control(self, prepared):
        cfg, now, events, reset = prepared
        if cfg != self.config or reset:
            self.lifecycle = Lifecycle(cfg)
            self.service.clear()
            self.bypass.clear()
            self.tickets.clear()
            self.output_seen.clear()
            self.ended_nodes.clear()
            self.demoted_closed_nodes.clear()
            self.ready_leases.clear()
            self.anchor_leases.clear()
            self.prefetched_anchors.clear()
            self.known_anchor_rids.clear()
            self.completed_at.clear()
            self.superseded_at.clear()
            self.workflow_turns.clear()
            self.rid_turns.clear()
            self.program_service.clear()
            self.program_arrival.clear()
            self.probes.clear()
            self.last_tick = -1.0
            self.last_admission_action = -1.0
            self.urgent_event = False
            self.tracking_limit = False
            self.hbm_pressure_active = False
            self.s._orvix_hbm_pressure_active = False
            self.s._orvix_critical_in_running = False
            self.s._orvix_active_chain_in_running = False
            self.buckets = {
                d: ByteBucket(cfg.transfer_burst_bytes, cfg.transfer_rate_bytes_s)
                for d in ('d2h', 'h2d')
            }
        self.config = cfg
        if cfg.mode != 'active':
            self.hbm_pressure_active = False
            self.s._orvix_hbm_pressure_active = False
            self.s._orvix_critical_in_running = False
            self.s._orvix_active_chain_in_running = False
        self.now = now
        self.lifecycle.config = cfg
        for event in events:
            self.lifecycle.event(event)
            if event['kind'] in {'tool_end', 'request_ready'} and self.config.queue_prefetch and self.config.tool_end_prefetch:
                rec = self.lifecycle.records.get(event['request_id'])
                if rec is not None and not rec.closed and not rec.tools:
                    self.urgent_event = True
            if event['kind'] in {'workflow_end', 'session_close'}:
                self.prefetched_anchors.discard(event['request_id'])
                anchor = self.tracking.records.get(event['request_id'])
                if anchor is not None:
                    if len(anchor.scope) == 4:
                        fam, gid, sid, scid = anchor.scope
                        if event['kind'] == 'workflow_end':
                            if gid or sid:
                                self.workflow_turns.pop(('scope', fam, gid, scid, sid), None)
                                self.workflow_turns.pop(('workflow', fam, gid, sid), None)
                                self.program_service.pop(('prog', fam, gid or sid), None)
                                self.program_arrival.pop(('prog', fam, gid or sid), None)
                            else:
                                self.workflow_turns.pop(('session', fam, sid, scid), None)
                        else:
                            for wk in list(self.workflow_turns):
                                if (wk[0] == 'scope' and wk[1] == fam and wk[4] == sid) or (
                                    wk[0] == 'workflow' and wk[1] == fam and wk[3] == sid
                                ) or (
                                    wk[0] == 'session' and wk[1] == fam and wk[2] == sid
                                ):
                                    self.workflow_turns.pop(wk, None)
                            if sid:
                                self.program_service.pop(('prog', fam, sid), None)
                                self.program_arrival.pop(('prog', fam, sid), None)
                    for rid, other in self.tracking.records.items():
                        same = other.scope == anchor.scope
                        if len(other.scope) == 4 and len(anchor.scope) == 4:
                            if event['kind'] == 'workflow_end':
                                same = other.scope[:3] == anchor.scope[:3]
                            else:
                                same = (other.scope[0], other.scope[2]) == (anchor.scope[0], anchor.scope[2])
                        if same and rid in self.lifecycle.records:
                            r = self.lifecycle.records[rid]
                            r.closed = True
                            r.ready = True
                            r.tools.clear()
                            r.tools_all_ended_at = None
                            r.anchor_ready_at = None
        # No cache/allocator operations in an HTTP/IPC setter.

    def _workflow_turns_for_req(self, req):
        wf_key = _workflow_key(req)
        if wf_key is not None and wf_key in self.workflow_turns:
            return self.workflow_turns[wf_key]
        pred_rid = _metadata(req).get('predecessor_request_id')
        if pred_rid and str(pred_rid) in self.rid_turns:
            return self.rid_turns[str(pred_rid)]
        return 0

    def _record_completed_turn(self, req):
        rid = str(req.rid)
        wf_key = _workflow_key(req)
        base = 0
        if wf_key is not None and wf_key in self.workflow_turns:
            base = self.workflow_turns[wf_key]
        else:
            pred_rid = _metadata(req).get('predecessor_request_id')
            if pred_rid and str(pred_rid) in self.rid_turns:
                base = self.rid_turns[str(pred_rid)]
        turns = base + 1
        if wf_key is not None:
            if wf_key in self.workflow_turns:
                self.workflow_turns.move_to_end(wf_key)
            elif len(self.workflow_turns) >= self.config.max_records:
                self.workflow_turns.popitem(last=False)
            self.workflow_turns[wf_key] = turns
        if rid in self.rid_turns:
            self.rid_turns.move_to_end(rid)
        elif len(self.rid_turns) >= self.config.max_records:
            self.rid_turns.popitem(last=False)
        self.rid_turns[rid] = turns

    def completed(self, req):
        if self.config.mode == 'off':
            return
        rid = str(req.rid)
        if req.finished() and not getattr(req, 'is_retracted', False):
            self._record_completed_turn(req)
        if req.finished() and rid in self.tracking.records:
            user = str(_metadata(req).get('family_id', '__unknown__'))
            self.lifecycle.register(rid, user)
            self.completed_at[rid] = self.now
            self.known_anchor_rids.add(rid)
            retracted = bool(getattr(req, 'is_retracted', False))
            cur_scope = self.tracking.records[rid].scope
            for old_rid, rec in self.tracking.records.items():
                if rec.scope == cur_scope and old_rid != rid:
                    old = self.lifecycle.records.get(old_rid)
                    if old is not None:
                        if rec.ready:
                            old.ready = True
                            old.tools.clear()
                            old.tools_all_ended_at = None
                            old.anchor_ready_at = None
                        if not retracted and not old.tools:
                            if not getattr(old, 'superseded', False):
                                self.superseded_at[old_rid] = self.now
                            old.superseded = True
                    if rec.ready:
                        self.prefetched_anchors.discard(old_rid)
        if self.config.phase_chain_turn_threshold > 0:
            self._sync_active_chain_flag()

    def on_abort(self, recv_req):
        if self.config.mode == 'off' and not self.actions.pending and not self.ready_leases and not self.anchor_leases:
            return
        self.actions.mark_cancelled(recv_req)
        matches = getattr(recv_req, 'matches', None)
        target_rid = getattr(recv_req, 'rid', None)
        for rid in list(self.ready_leases):
            matched = bool(matches(rid)) if callable(matches) else (target_rid is not None and str(target_rid) == rid)
            if matched:
                self.ready_leases.pop(rid, None)
                self.counts['prefetch_lease_expired_or_cancelled'] += 1
        for nid, (_, parent_rid, _) in list(self.anchor_leases.items()):
            matched = bool(matches(parent_rid)) if callable(matches) else (target_rid is not None and str(target_rid) == parent_rid)
            if matched:
                self.anchor_leases.pop(nid, None)
                self.counts['anchor_lease_expired_or_cancelled'] += 1

    def key(self, req):
        ids = concat_token_ids(req.origin_input_ids, req.output_ids)
        return self.key_factory(req, ids[:-1])

    def probe(self, req):
        out_ids = req.output_ids
        if not out_ids:
            cached = getattr(req, '_orvix_probe_key_cache', None)
            in_ids = req.origin_input_ids
            in_len = len(in_ids)
            if cached is not None and cached[0] == in_len:
                return probe_tiered_prefix(self.cache, cached[1], input_len=in_len, fences=self.fences)
            k = self.key_factory(req, in_ids[:-1])
            try:
                req._orvix_probe_key_cache = (in_len, k)
            except AttributeError:
                pass
            return probe_tiered_prefix(self.cache, k, input_len=in_len, fences=self.fences)
        ids = concat_token_ids(req.origin_input_ids, out_ids)
        return probe_tiered_prefix(self.cache, self.key_factory(req, ids[:-1]), input_len=len(ids), fences=self.fences)

    def _queue(self):
        queue = list(self.s.waiting_queue)
        for r in queue:
            if not hasattr(r, '_orvix_interval_ready'):
                r._orvix_interval_ready = self.now
                self.arrival_counter += 1
                r._orvix_interval_arrival = self.arrival_counter
        return sorted(queue, key=lambda r: (r._orvix_interval_ready, r._orvix_interval_arrival))

    def _record_node(self, rid, *, exclusive=False):
        rec = self.tracking.records.get(rid)
        if rec is None:
            return None
        return self.actions.get_anchor(rec, exclusive=exclusive)

    def _exclusive_cold_nodes(self, rid):
        rec = self.tracking.records.get(rid)
        if rec is None:
            return []
        leaf = self.actions.get_anchor(rec, exclusive=True)
        if leaf is None:
            return []
        out = [leaf]
        root = self.cache.tree_core.root_node
        cur = getattr(leaf, 'parent', None)
        while cur is not None and cur is not root:
            children = getattr(cur, 'children', None)
            if not children or len(children) != 1:
                break
            if self.actions.pending_node(cur) or any(
                other.node_object is cur and other.scope != rec.scope
                for other in self.tracking.records.values()
            ):
                break
            comp_data = getattr(cur, 'component_data', ())
            comp_vals = comp_data.values() if isinstance(comp_data, dict) else comp_data
            if any(
                getattr(cd, 'lock_ref', 0) > 0 or getattr(cd, 'host_lock_ref', 0) > 0
                for cd in comp_vals
            ):
                break
            out.append(cur)
            cur = getattr(cur, 'parent', None)
        out.reverse()
        return out

    def _waiter_targets_node(self, node, parent_rid=None):
        """Return True if a request already sitting in waiting_queue matches `node` or `parent_rid`."""
        for req in list(self.s.waiting_queue)[:self.config.scan_limit]:
            if parent_rid is not None:
                pred = _metadata(req).get('predecessor_request_id')
                if pred and str(pred) == parent_rid:
                    return True
            p = self.probe(req)
            if p.host_node_id == node.id:
                return True
        return False

    def _has_anchor_lease(self, req, probe_res):
        if not self.anchor_leases:
            return False
        pred_rid = _metadata(req).get('predecessor_request_id')
        if pred_rid:
            for _, (_, parent_rid, _) in self.anchor_leases.items():
                if str(pred_rid) == parent_rid:
                    return True
        if probe_res.gpu_ready_len > 0:
            for nid in self.anchor_leases:
                try:
                    anode = self.cache.tree_core.node_by_id(nid)
                    path_len = sum(len(n.key) for n in self.actions.path(anode))
                    if probe_res.gpu_ready_len >= path_len > 0:
                        return True
                except KeyError:
                    continue
        return False

    def _agree(self, value):
        try:
            return self.consensus(value)
        except BaseException:
            self.poisoned = True
            raise

    def _agree_optional_plan(self, label, plan):
        try:
            self.consensus({'phase': 'interval_' + label, 'plan': plan})
        except CachePlanMismatch:
            self.counts[label + '_tp_skipped'] += 1
            return False
        except BaseException:
            self.poisoned = True
            raise
        return True

    def _ensure_installed(self):
        if not self.native_installed:
            self._phase('install_safety_hooks', self.actions.install)
            self.native_installed = True

    def _phase(self, label, fn):
        result = None
        error = None
        try:
            result = fn()
        except Exception as exc:
            error = type(exc).__name__ + ':' + str(exc)
        self._agree({'phase': 'interval_' + label, 'error': error})
        if error is not None:
            self.poisoned = True
            raise RuntimeError('interval ' + label + ' failed: ' + error)
        return result

    def _act(self, label, description, fn):
        self._agree({'phase': 'interval_' + label, 'plan': description})
        result = self._phase(label, fn)
        self._agree({'phase': 'interval_' + label + '_receipt', 'result': result})
        self.log.append({'clock': self.now, 'kind': label, **description, 'result': result})
        self.counts[label] += 1
        return result

    def _check_safe(self):
        if self.poisoned or self.actions.unsafe:
            raise RuntimeError('unsafe interval state: no fallback to serving; restart private process')

    def after_events(self):
        self._check_safe()
        if self.actions.pending:
            completed_nodes = {rid: t['node_id'] for rid, t in self.actions.pending.items()}
            cancelled_rids = {rid for rid, t in self.actions.pending.items() if t.get('cancelled')}
            done = self._phase('poll', self.actions.poll_completed)
            self._agree({'phase': 'interval_completed', 'rids': done})
            for rid in done:
                if rid in cancelled_rids or rid in getattr(self.actions, 'cancelled_completed', ()):
                    self.counts['prefetch_cancelled_after_ack'] += 1
                elif rid.startswith('anchor:'):
                    parent_rid = rid.split(':', 1)[1]
                    nid = completed_nodes[rid]
                    rec = self.lifecycle.records.get(parent_rid)
                    user = rec.user if rec is not None else '__unknown__'
                    self.anchor_leases[nid] = (self.now + self.config.anchor_lease_s, parent_rid, user)
                    self.counts['anchor_prefetch_completed'] += 1
                else:
                    self.ready_leases[rid] = (self.now + self.config.ready_lease_s, completed_nodes[rid])
        is_urgent = self.urgent_event
        self.urgent_event = False
        if self.config.mode == 'off':
            if self.hbm_pressure_active or getattr(self.s, '_orvix_hbm_pressure_active', False):
                self.hbm_pressure_active = False
                self.s._orvix_hbm_pressure_active = False
            self.s._orvix_critical_in_running = False
            self.s._orvix_active_chain_in_running = False
            return
        if not is_urgent and self.now - self.last_tick < self.config.tick_s:
            return
        self.last_tick = self.now
        self.lifecycle.expire(self.now)
        waiting_ids = {str(r.rid) for r in self.s.waiting_queue}
        for rid, (until, _) in list(self.ready_leases.items()):
            if self.now >= until or rid not in waiting_ids:
                self.ready_leases.pop(rid, None)
                self.counts['prefetch_lease_expired_or_cancelled'] += 1
        for nid, (until, parent_rid, _) in list(self.anchor_leases.items()):
            rec = self.lifecycle.records.get(parent_rid)
            if self.now >= until or (rec is not None and rec.closed):
                self.anchor_leases.pop(nid, None)
                self.counts['anchor_lease_expired_or_cancelled'] += 1
        self.bypass = {k: v for k, v in self.bypass.items() if k in waiting_ids}
        if self.tracking.tracking_exhausted:
            self.tracking_limit = True
        self._agree({
            'phase': 'interval_gate',
            'mode': self.config.mode,
            'active': self.active,
            'tracking_limit': self.tracking_limit,
            'lifecycle_overflow': self.lifecycle.overflow,
            'pending': list(self.actions.pending),
            'now': self.now,
            'urgent': is_urgent,
        })
        if not self.active:
            if self.hbm_pressure_active or getattr(self.s, '_orvix_hbm_pressure_active', False):
                self.hbm_pressure_active = False
                self.s._orvix_hbm_pressure_active = False
            self.s._orvix_critical_in_running = False
            self.s._orvix_active_chain_in_running = False
            return
        self._ensure_installed()
        plan = self._phase('maintenance_plan', self._maintenance_plan)
        if self.config.hbm_pressure_free_fraction > 0.0:
            if self._agree_optional_plan('hbm_pressure_plan', {'pressured': plan['hbm_pressured']}):
                self.hbm_pressure_active = plan['hbm_pressured']
                self.s._orvix_hbm_pressure_active = self.hbm_pressure_active
        elif self.hbm_pressure_active or getattr(self.s, '_orvix_hbm_pressure_active', False):
            self.hbm_pressure_active = False
            self.s._orvix_hbm_pressure_active = False
        anchor_actions = [
            (kind, rid, self._prefetch_receipt(restore))
            for kind, rid, restore in plan['anchor_prefetches']
        ]
        # Isolate sub-plans across TP ranks so a sub-millisecond async backup ACK
        # timing difference on one leaf never aborts anchor prefetches or warm leases!
        if self._agree_optional_plan('lease_plan', {'warm': plan['warm'], 'cold': plan['cold']}):
            if plan['cold']:
                nodes = []
                seen_cold = set()
                for rid in plan['cold']:
                    for n in self._exclusive_cold_nodes(rid):
                        if n.id not in seen_cold:
                            seen_cold.add(n.id)
                            nodes.append(n)
                if nodes:
                    self._act(
                        'end_demote_priority',
                        {'rids': plan['cold'], 'nodes': [n.id for n in nodes]},
                        lambda: self._touch_many(nodes, True),
                    )
                self.demoted_closed_nodes.update(plan['cold'])
                if not self.config.proactive_release:
                    self.ended_nodes.update(plan['cold'])
            if plan['warm']:
                nodes = [self.cache.tree_core.node_by_id(nid) for nid in plan['warm']]
                self._act('soft_keep', {'nodes': plan['warm']}, lambda: self._touch_many(nodes, False))

        if self._agree_optional_plan('backup_plan', plan['backups']):
            for rid, b in plan['backups']:
                if not self.actions.ensure_host_room(
                    b,
                    allow_reclaim=self.config.host_swa_reclaim,
                    allow_kv_reclaim=self.config.allow_host_kv_reclaim,
                    agree=self._agree_optional_plan,
                ):
                    self.counts['backup_host_room_or_layout_declined'] += 1
                    continue
                if not self.buckets['d2h'].consume(b['bytes'], self.now):
                    continue
                self._act('async_backup', {'rid': rid, **b}, lambda b=b: self.actions.backup(b))

        if self._agree_optional_plan('proactive_release_plan', {'releases': plan['proactive_releases'], 'ended': plan['ended']}):
            self.ended_nodes.update(plan['ended'])
            for rel in plan.get('proactive_releases', ()):
                try:
                    node = self.cache.tree_core.node_by_id(rel['node'])
                except KeyError:
                    continue
                self._act('backed_leaf_release', rel, lambda n=node: self.actions.release(n))

        if plan.get('dead_drops') and self._agree_optional_plan('dead_drop_plan', plan['dead_drops']):
            for item in plan['dead_drops']:
                self._act('dead_chain_drop', item, lambda ids=item['chain_ids']: self.actions.drop_dead_chain_ids(ids))
                rec = self.tracking.records.get(item['rid'])
                node_obj = getattr(rec, 'node_object', None) if rec is not None else None
                nodes_map = getattr(self.cache.tree_core, 'nodes', None)
                dropped_or_evicted = (
                    node_obj is None
                    or getattr(node_obj, 'evicted', False)
                    or (isinstance(nodes_map, dict) and item['node'] not in nodes_map)
                )
                if dropped_or_evicted:
                    self.ended_nodes.add(item['rid'])

        if self._agree_optional_plan('anchor_prefetch_plan', anchor_actions):
            for (_, _, p_plan), (kind, parent_rid, receipt) in zip(
                plan['anchor_prefetches'], anchor_actions, strict=True
            ):
                if len(self.actions.pending) >= self.config.max_inflight:
                    break
                if not self.buckets['h2d'].allow(p_plan['bytes'], self.now):
                    self.counts['h2d_rate_limited'] += 1
                    continue
                key_rid = 'anchor:' + parent_rid
                desc = {**receipt, 'rid': key_rid, 'parent_rid': parent_rid, 'trigger': kind}
                started = self._act(kind, desc, lambda kr=key_rid, pl=p_plan: self.actions.start_prefetch(kr, pl))
                if started:
                    self.buckets['h2d'].consume(p_plan['bytes'], self.now)
                    if not p_plan.get('partial_restore'):
                        self.prefetched_anchors.add(parent_rid)

        if is_urgent:
            self.last_admission_action = -1.0
        self.before_admission()

    @staticmethod
    def _prefetch_receipt(plan):
        receipt = {k: v for k, v in plan.items() if k not in {'tokens', 'namespace'}}
        receipt['prefix_sha256'] = hashlib.sha256(
            plan['tokens'].tobytes() + repr(plan['namespace']).encode()
        ).hexdigest()
        return receipt

    def _touch_many(self, nodes, cold):
        for node in nodes:
            self.actions.touch(node, cold=cold)
        return len(nodes)

    def _touch(self, node, cold):
        self.actions.touch(node, cold=cold)
        return True

    def _can_protect(self, extra, user, global_used, user_used, pools):
        full = extra.get('full', 0)
        return (
            global_used['full'] + full <= self.config.max_keep_tokens
            and user_used[user] + full <= self.config.per_user_keep_tokens
            and all(
                name in pools
                and global_used[name] + amount <= int(pools[name]['capacity'] * self.config.max_keep_fraction)
                and pools[name]['free'] + pools[name]['evictable'] - global_used[name] - amount
                >= max(self.config.growth_reserve_tokens, int(pools[name]['capacity'] * self.config.reserve_fraction))
                for name, amount in extra.items()
                if amount
            )
        )

    def _idle_backed_release_candidates(self, exclude_rids=()):
        """Collect backed & idle exclusive leaves sorted by longest remaining wait first."""
        excl = set(exclude_rids)
        candidates = []
        for rid, r in self.lifecycle.records.items():
            if rid in excl:
                continue
            if r.closed:
                if rid in self.ended_nodes:
                    continue
            elif r.ready or not r.tools or self.now < self.lifecycle.keep_until(r, self.now):
                continue
            record = self.tracking.records.get(rid)
            if record is None:
                continue
            raw_node = getattr(record, 'node_object', None)
            if raw_node is None or getattr(raw_node, 'evicted', False) or getattr(raw_node, 'children', None):
                continue
            if raw_node.id in self.anchor_leases or rid in self.prefetched_anchors or not self.actions.backed_and_idle(raw_node):
                continue
            node = self._record_node(rid, exclusive=True)
            if node is None or self.tracking._waiting_uses(record):
                continue
            pred = self.lifecycle.predict_remaining(r, self.now) if r.tools else None
            rem_score = 1e9 if r.closed else (pred['remaining_p50_s'] if pred is not None else 0.0)
            candidates.append((rem_score, rid, node))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates

    def _true_dead_leaf_node(self, rid, *, shared_node_ids=None, allow_superseded=True):
        """Return exclusive childless D-leaf for a closed/superseded turn, never an interior ancestor."""
        if not self.config.dead_leaf_bypass:
            return None
        r = self.lifecycle.records.get(rid)
        if r is None or r.tools:
            return None
        is_sup = getattr(r, 'superseded', False)
        if not (r.closed or (allow_superseded and is_sup)):
            return None
        rec = self.tracking.records.get(rid)
        if rec is None:
            return None
        raw_node = getattr(rec, 'node_object', None)
        if (
            raw_node is None
            or getattr(raw_node, 'evicted', False)
            or getattr(raw_node, 'children', None)
            or raw_node.id in self.anchor_leases
            or self.actions.pending_node(raw_node)
        ):
            return None
        if shared_node_ids is not None:
            if raw_node.id in shared_node_ids:
                return None
            ex_node = self._record_node(rid, exclusive=False)
        else:
            ex_node = self._record_node(rid, exclusive=True)
        if ex_node is None or self.tracking._waiting_uses(rec):
            return None
        return ex_node

    def _maintenance_plan(self):
        pools = self.actions.pool_state()
        tail_state = self.actions.swa_tail_state()
        global_used = Counter()
        user_used = Counter()
        warm = []
        cold = []
        ended = []
        backups = []
        proactive_releases = []
        dead_drops = []
        released_node_ids = set()
        anchor_prefetches = []
        anchor_shortage = Counter()
        seen = set()
        reserved_locks = Counter()
        self.buckets['d2h'].allow(0, self.now)
        bleft = int(self.buckets['d2h'].tokens)
        self.buckets['h2d'].allow(0, self.now)
        h2d_left = int(self.buckets['h2d'].tokens)
        inflight_budget = max(0, self.config.max_inflight - len(self.actions.pending))

        tail_pressured = (
            tail_state['max_tails'] is not None
            and tail_state['pressure_ratio'] >= self.config.swa_tail_watermark
        )
        def _blank_free_below(p, frac):
            return p['free'] < int(p['capacity'] * frac)

        def _avail_with_evictable_below(p, frac):
            return (p['free'] + p.get('evictable', 0)) < int(p['capacity'] * frac)

        # V11 Module 1: Separate leaf-eviction pressure (`pool_pressured`) from blank-page
        # async write-behind pressure (`backup_pressured`):
        # - `pool_pressured`: checks `free + evictable < release_free_fraction` so warm backed
        #   evictable leaves in the 6.56M full pool are not needlessly evicted.
        # - `backup_pressured`: checks blank `free < backup_free_fraction` uniformly across BOTH
        #   `swa` and `full` pools so unbacked interval nodes copy to Host asynchronously
        #   (`write_back=False`) before `swa['free']` hits zero and stalls main-thread `evict_swa`.
        pool_pressured = any(
            _avail_with_evictable_below(p, self.config.release_free_fraction)
            for p in pools.values()
        )
        full_pool = pools.get('full')
        hbm_pressured = bool(
            self.config.hbm_pressure_free_fraction > 0.0
            and full_pool is not None
            and _avail_with_evictable_below(full_pool, self.config.hbm_pressure_free_fraction)
        )
        backup_pressured = (
            (not self.config.pressure_gated_backup)
            or tail_pressured
            or pool_pressured
            or any(
                _blank_free_below(p, self.config.backup_free_fraction)
                for p in pools.values()
            )
        )
        full_pres = bool(
            full_pool is not None
            and (
                _blank_free_below(full_pool, self.config.backup_free_fraction)
                or _avail_with_evictable_below(full_pool, self.config.release_free_fraction)
            )
        )
        dead_drop_pressured = backup_pressured or pool_pressured or tail_pressured
        allow_superseded_drop = (
            (not self.config.superseded_drop_requires_full_pressure) or pool_pressured or hbm_pressured
        )

        def _can_drop_superseded(rid_k):
            if allow_superseded_drop:
                return True
            if self.config.superseded_keep_s > 0.0 and dead_drop_pressured:
                sup_t = self.superseded_at.get(rid_k, self.completed_at.get(rid_k, self.now))
                return (self.now - sup_t) >= self.config.superseded_keep_s
            return False

        active_scope_nodes = set()
        shared_node_ids = set()
        scope_by_nid = {}
        for other_rid, other in self.tracking.records.items():
            other_node = getattr(other, 'node_object', None)
            if other_node is None:
                continue
            nid = other_node.id
            prev_scope = scope_by_nid.get(nid)
            if prev_scope is None:
                scope_by_nid[nid] = other.scope
            elif prev_scope != other.scope:
                shared_node_ids.add(nid)
            if self.config.dead_leaf_bypass:
                o_rec = self.lifecycle.records.get(other_rid)
                if o_rec is not None and (
                    o_rec.closed or (getattr(o_rec, 'superseded', False) and _can_drop_superseded(other_rid))
                ):
                    continue
                active_scope_nodes.add(nid)
        dead_forbidden = active_scope_nodes | set(self.anchor_leases.keys())
        release_pressured = (not self.config.pressure_gated_backup) or pool_pressured or tail_pressured
        closed_release_pressured = release_pressured or (
            'swa' in pools and _blank_free_below(pools['swa'], self.config.backup_free_fraction)
        )

        for rid, r in self.lifecycle.records.items():
            idle_inter_round = False
            if r.closed:
                if rid in self.ended_nodes:
                    continue
            else:
                anchor_grace = (
                    self.config.tool_end_prefetch
                    and r.anchor_ready_at is not None
                    and self.now - r.anchor_ready_at <= 2.0
                    and rid not in self.prefetched_anchors
                )
                if r.ready and not anchor_grace:
                    continue
                if not r.tools and r.tools_all_ended_at is None and not anchor_grace:
                    if (
                        self.config.eager_backup
                        and backup_pressured
                        and (self.now - self.completed_at.get(rid, self.now)) >= self.config.backup_after_s
                    ):
                        raw_rec = self.tracking.records.get(rid)
                        raw_obj = getattr(raw_rec, 'node_object', None) if raw_rec is not None else None
                        if raw_obj is None or getattr(raw_obj, 'evicted', False):
                            continue
                        idle_inter_round = True
                    else:
                        continue
            node = self._record_node(rid)
            if node is None:
                continue

            if idle_inter_round:
                dead_leaf = self._true_dead_leaf_node(
                    rid, shared_node_ids=shared_node_ids, allow_superseded=_can_drop_superseded(rid)
                )
                if dead_leaf is not None:
                    if (
                        self.config.proactive_release
                        and dead_drop_pressured
                        and dead_leaf.id not in released_node_ids
                        and len(dead_drops) < self.config.max_actions * 2
                    ):
                        chain_ids = self.actions.plan_dead_chain(
                            dead_leaf,
                            allow_full_only_ancestors=full_pres,
                            forbidden_nodes=dead_forbidden,
                        )
                        if chain_ids:
                            dead_drops.append({
                                'rid': rid,
                                'node': dead_leaf.id,
                                'chain_ids': chain_ids,
                                'full_pres': full_pres,
                            })
                            released_node_ids.update(chain_ids)
                    continue
                if (
                    not node.evicted
                    and not self.actions.pending_node(node)
                    and len(backups) < self.config.max_actions
                ):
                    chunk_cap = (
                        min(self.config.backup_chunk_tokens, self.config.max_transfer_tokens)
                        if self.config.streaming_backup
                        else None
                    )
                    b = self.actions.backup_plan(
                        node,
                        max_chunk_tokens=chunk_cap,
                        max_bytes=bleft if self.config.streaming_backup else None,
                    )
                    if (
                        b
                        and b['ids']
                        and b['bytes'] is not None
                        and b['full'] <= self.config.max_transfer_tokens
                        and b['bytes'] <= bleft
                        and self.actions.host_room(b)
                        and self.buckets['d2h'].allow(b['bytes'], self.now)
                        and all(
                            name in pools
                            and pools[name]['free'] + pools[name]['evictable'] - reserved_locks[name] - amount
                            >= max(self.config.growth_reserve_tokens, int(pools[name]['capacity'] * self.config.reserve_fraction))
                            for name, amount in b['lock_footprint'].items()
                            if amount
                        )
                        and not any(set(b['ids']) & set(old['ids']) for _, old in backups)
                    ):
                        backups.append((rid, b))
                        bleft -= b['bytes']
                        reserved_locks.update(b['lock_footprint'])
                continue

            # 1) Closed / finished workflows: async backup before SWA blank pages hit zero, then O(1) release / cold demote
            if r.closed:
                if node.evicted:
                    ended.append(rid)
                    continue
                ex_node = None if (getattr(node, 'children', None) or node.id in shared_node_ids) else node
                rec = self.tracking.records.get(rid)
                if ex_node is None:
                    ended.append(rid)
                    continue
                if not self.actions.pending_node(ex_node) and not self.tracking._waiting_uses(rec):
                    dead_leaf = self._true_dead_leaf_node(
                        rid, shared_node_ids=shared_node_ids, allow_superseded=allow_superseded_drop
                    )
                    if dead_leaf is not None:
                        can_drop_closed = dead_drop_pressured or self.config.closed_immediate_drop
                        allow_closed_anc = True if self.config.closed_immediate_drop else full_pres
                        if (
                            self.config.proactive_release
                            and can_drop_closed
                            and dead_leaf.id not in released_node_ids
                            and len(dead_drops) < self.config.max_actions * 2
                        ):
                            chain_ids = self.actions.plan_dead_chain(
                                dead_leaf,
                                allow_full_only_ancestors=allow_closed_anc,
                                forbidden_nodes=dead_forbidden,
                            )
                            if chain_ids:
                                dead_drops.append({
                                    'rid': rid,
                                    'node': dead_leaf.id,
                                    'chain_ids': chain_ids,
                                    'full_pres': allow_closed_anc,
                                })
                                released_node_ids.update(chain_ids)
                                continue
                        if dead_leaf.parent is not self.cache.tree_core.root_node and not self.config.backed_only_cold_demote:
                            if self.config.soft_keep and rid not in self.demoted_closed_nodes and len(cold) < self.config.max_actions:
                                cold.append(rid)
                            if not self.config.proactive_release:
                                ended.append(rid)
                            continue
                    if self.config.proactive_release and closed_release_pressured and self.actions.backed_and_idle(ex_node):
                        if self.config.soft_keep and rid not in self.demoted_closed_nodes and len(cold) < self.config.max_actions:
                            cold.append(rid)
                        if len(proactive_releases) < self.config.max_actions and ex_node.id not in released_node_ids:
                            proactive_releases.append({
                                'rid': rid,
                                'node': ex_node.id,
                                'reason': 'closed_workflow',
                            })
                            released_node_ids.add(ex_node.id)
                            ended.append(rid)
                        continue
                    queued_closed_backup = False
                    host_full_for_closed = False
                    if (
                        self.config.proactive_release
                        and self.config.eager_backup
                        and backup_pressured
                        and len(backups) < self.config.max_actions
                    ):
                        chunk_cap = (
                            min(self.config.backup_chunk_tokens, self.config.max_transfer_tokens)
                            if self.config.streaming_backup
                            else None
                        )
                        b = self.actions.backup_plan(
                            ex_node,
                            max_chunk_tokens=chunk_cap,
                            max_bytes=bleft if self.config.streaming_backup else None,
                        )
                        if b and b['ids']:
                            if not self.actions.host_room(b):
                                host_full_for_closed = True
                            elif (
                                b['bytes'] is not None
                                and b['full'] <= self.config.max_transfer_tokens
                                and b['bytes'] <= bleft
                                and self.buckets['d2h'].allow(b['bytes'], self.now)
                                and all(
                                    name in pools
                                    and pools[name]['free'] + pools[name]['evictable'] - reserved_locks[name] - amount
                                    >= max(self.config.growth_reserve_tokens, int(pools[name]['capacity'] * self.config.reserve_fraction))
                                    for name, amount in b['lock_footprint'].items()
                                    if amount
                                )
                                and not any(set(b['ids']) & set(old['ids']) for _, old in backups)
                            ):
                                backups.append((rid, b))
                                bleft -= b['bytes']
                                reserved_locks.update(b['lock_footprint'])
                                queued_closed_backup = True
                    if self.config.backed_only_cold_demote:
                        can_demote_cold = (
                            not queued_closed_backup
                            and (
                                self.actions.backed_and_idle(ex_node)
                                or host_full_for_closed
                                or not self.config.proactive_release
                            )
                        )
                    else:
                        can_demote_cold = (
                            not queued_closed_backup
                            and (
                                self.actions.backed_and_idle(ex_node)
                                or host_full_for_closed
                                or not self.config.proactive_release
                                or not self.config.eager_backup
                                or not backup_pressured
                            )
                        )
                    if self.config.soft_keep and can_demote_cold and rid not in self.demoted_closed_nodes and len(cold) < self.config.max_actions:
                        cold.append(rid)
                    if not self.config.proactive_release or host_full_for_closed:
                        ended.append(rid)
                continue

            # 2) Insurance A: All tools on parent `rid` finished (`tool_end` or pre-arrival `anchor_ready_at`)
            anchor_eligible = (
                r.tools_all_ended_at is not None
                or (r.anchor_ready_at is not None and self.now - r.anchor_ready_at <= 2.0)
            )
            if (
                self.config.queue_prefetch
                and self.config.tool_end_prefetch
                and anchor_eligible
                and not r.closed
                and not r.tools
                and rid not in self.prefetched_anchors
                and len(anchor_prefetches) < inflight_budget
                and not self._waiter_targets_node(node, parent_rid=rid)
            ):
                needs_h2d = node.evicted or any(cd.value is None and cd.host_value is not None for cd in node.component_data)
                p_plan = self.actions.restore_plan(node, self.config) if needs_h2d else None
                if p_plan is not None and p_plan['bytes'] is not None:
                    if (
                        (p_plan['full'] + p_plan['swa']) >= self.config.min_host_tokens
                        and p_plan['full'] <= self.config.max_transfer_tokens
                        and p_plan['bytes'] <= h2d_left
                    ):
                        anchor_prefetches.append(('tool_end_prefetch', rid, p_plan))
                        h2d_left -= p_plan['bytes']
                elif self.config.proactive_release:
                    raw_need = self.actions.restore_plan(node, self.config, ignore_capacity=True)
                    if raw_need is not None and raw_need['full'] >= self.config.min_host_tokens:
                        for pname in ('full', 'swa'):
                            if pname in pools and raw_need.get(pname, 0) > 0:
                                rsv = max(self.config.growth_reserve_tokens, int(pools[pname]['capacity'] * self.config.reserve_fraction))
                                target_tok = min(raw_need[pname], self.config.max_transfer_tokens) if pname == 'full' else raw_need[pname]
                                anchor_shortage[pname] = max(anchor_shortage[pname], max(0, target_tok + rsv - pools[pname]['free']))

            if r.ready or not r.tools:
                continue

            # 3) Active tool interval: online prediction + soft_keep + pressure-gated backup + countdown prefetch + proactive release
            pred = self.lifecycle.predict_remaining(r, self.now)
            until = self.lifecycle.keep_until(r, self.now)

            # Insurance B: Countdown early H2D prefetch before predictable long tool finishes!
            if (
                self.config.queue_prefetch
                and self.config.countdown_prefetch
                and pred is not None
                and pred['confident']
                and 0.0 <= pred['remaining_p50_s'] <= self.config.prefetch_lead_s
                and rid not in self.prefetched_anchors
                and len(anchor_prefetches) < inflight_budget
            ):
                needs_h2d = node.evicted or any(cd.value is None and cd.host_value is not None for cd in node.component_data)
                p_plan = self.actions.restore_plan(node, self.config) if needs_h2d else None
                if p_plan is not None and p_plan['bytes'] is not None:
                    if (
                        (p_plan['full'] + p_plan['swa']) >= self.config.min_host_tokens
                        and p_plan['full'] <= self.config.max_transfer_tokens
                        and p_plan['bytes'] <= h2d_left
                    ):
                        anchor_prefetches.append(('countdown_prefetch', rid, p_plan))
                        h2d_left -= p_plan['bytes']
                elif self.config.proactive_release:
                    raw_need = self.actions.restore_plan(node, self.config, ignore_capacity=True)
                    if raw_need is not None and raw_need['full'] >= self.config.min_host_tokens:
                        for pname in ('full', 'swa'):
                            if pname in pools and raw_need.get(pname, 0) > 0:
                                rsv = max(self.config.growth_reserve_tokens, int(pools[pname]['capacity'] * self.config.reserve_fraction))
                                target_tok = min(raw_need[pname], self.config.max_transfer_tokens) if pname == 'full' else raw_need[pname]
                                anchor_shortage[pname] = max(anchor_shortage[pname], max(0, target_tok + rsv - pools[pname]['free']))

            if self.config.soft_keep and self.now < until:
                path = self.actions.path(node)
                win_fp = self.actions.swa_window_footprint(node)
                extra = Counter()
                for n in path:
                    if n.id not in seen:
                        extra['full'] += self.actions.footprint(n).get('full', 0)
                extra['swa'] = win_fp.get('swa', 0)
                if self._can_protect(extra, r.user, global_used, user_used, pools):
                    for n in path:
                        if n.id not in seen:
                            warm.append(n.id)
                            seen.add(n.id)
                    global_used.update(extra)
                    user_used[r.user] += extra.get('full', 0)
                else:
                    self.counts['keep_budget_declined'] += 1

            elapsed = self.now - min(t.start for t in r.tools.values())
            if self.config.eager_backup and elapsed >= self.config.backup_after_s and len(backups) < self.config.max_actions:
                if not backup_pressured:
                    self.counts['backup_skipped_low_pressure'] += 1
                else:
                    chunk_cap = (
                        min(self.config.backup_chunk_tokens, self.config.max_transfer_tokens)
                        if self.config.streaming_backup
                        else None
                    )
                    b = self.actions.backup_plan(
                        node,
                        max_chunk_tokens=chunk_cap,
                        max_bytes=bleft if self.config.streaming_backup else None,
                    )
                    if b and b['ids']:
                        if b['bytes'] is None:
                            self.counts['backup_unknown_bytes'] += 1
                        elif (
                            b['full'] > self.config.max_transfer_tokens
                            or b['bytes'] > bleft
                            or not self.buckets['d2h'].allow(b['bytes'], self.now)
                        ):
                            self.counts['backup_rate_or_size_declined'] += 1
                        elif not all(
                            name in pools
                            and pools[name]['free'] + pools[name]['evictable'] - reserved_locks[name] - amount
                            >= max(self.config.growth_reserve_tokens, int(pools[name]['capacity'] * self.config.reserve_fraction))
                            for name, amount in b['lock_footprint'].items()
                            if amount
                        ):
                            self.counts['backup_growth_reserve_declined'] += 1
                        elif not any(set(b['ids']) & set(old['ids']) for _, old in backups):
                            backups.append((rid, b))
                            bleft -= b['bytes']
                            reserved_locks.update(b['lock_footprint'])

            # Proactive post-ACK release of exclusive backed leaves ONLY when under pressure (when pressure_gated_backup=True)
            if (
                self.config.proactive_release
                and self.now >= until
                and node.id not in self.anchor_leases
                and node.id not in released_node_ids
                and rid not in self.prefetched_anchors
                and len(proactive_releases) < self.config.max_actions
            ):
                long_wait = (
                    pred is not None
                    and pred['remaining_p50_s'] >= self.config.release_min_remaining_s
                    and pred['remaining_p50_s'] > self.config.prefetch_lead_s + 0.5
                    and (not self.config.pressure_gated_backup or pool_pressured or tail_pressured)
                )
                if long_wait or tail_pressured or pool_pressured:
                    ex_node = None if (getattr(node, 'children', None) or node.id in shared_node_ids) else node
                    rec = self.tracking.records.get(rid)
                    if (
                        ex_node is not None
                        and ex_node.id not in released_node_ids
                        and self.actions.backed_and_idle(ex_node)
                        and not self.tracking._waiting_uses(rec)
                    ):
                        proactive_releases.append({
                            'rid': rid,
                            'node': ex_node.id,
                            'reason': 'predicted_long_tool' if long_wait else ('swa_tail_pressure' if tail_pressured else 'pool_pressure'),
                        })
                        released_node_ids.add(ex_node.id)

        # If an urgent anchor prefetch needed pool space, proactively release backed idle leaves to make room!
        if any(anchor_shortage.values()) and len(proactive_releases) < self.config.max_actions:
            freed = Counter()
            for _, cand_rid, cand_node in self._idle_backed_release_candidates():
                if len(proactive_releases) >= self.config.max_actions:
                    break
                if cand_node.id in released_node_ids:
                    continue
                proactive_releases.append({
                    'rid': cand_rid,
                    'node': cand_node.id,
                    'reason': 'anchor_prefetch_shortage',
                })
                released_node_ids.add(cand_node.id)
                freed.update(self.actions.footprint(cand_node))
                if all(freed[k] >= v for k, v in anchor_shortage.items()):
                    break

        # Protect anchor_leases (completed early anchor prefetches awaiting child HTTP request)
        for nid, (until, parent_rid, user) in self.anchor_leases.items():
            if self.now >= until:
                continue
            try:
                node = self.cache.tree_core.node_by_id(nid)
            except KeyError:
                continue
            win_fp = self.actions.swa_window_footprint(node)
            extra = Counter()
            for n in self.actions.path(node):
                if n.id not in seen:
                    extra['full'] += self.actions.footprint(n).get('full', 0)
            extra['swa'] = win_fp.get('swa', 0)
            if self._can_protect(extra, user, global_used, user_used, pools):
                for n in self.actions.path(node):
                    if n.id not in seen:
                        warm.append(n.id)
                        seen.add(n.id)
                global_used.update(extra)
                user_used[user] += extra.get('full', 0)
            else:
                self.counts['anchor_lease_budget_declined'] += 1

        ready_users = {str(r.rid): str(_metadata(r).get('family_id', '__unknown__')) for r in self._queue()}
        for rid, (until, nid) in self.ready_leases.items():
            if self.now >= until or rid not in ready_users:
                continue
            user = ready_users[rid]
            try:
                node = self.cache.tree_core.node_by_id(nid)
            except KeyError:
                continue
            win_fp = self.actions.swa_window_footprint(node)
            extra = Counter()
            for n in self.actions.path(node):
                if n.id not in seen:
                    extra['full'] += self.actions.footprint(n).get('full', 0)
            extra['swa'] = win_fp.get('swa', 0)
            if self._can_protect(extra, user, global_used, user_used, pools):
                for n in self.actions.path(node):
                    if n.id not in seen:
                        warm.append(n.id)
                        seen.add(n.id)
                global_used.update(extra)
                user_used[user] += extra.get('full', 0)
            else:
                self.counts['ready_lease_budget_declined'] += 1

        if self.config.dead_leaf_bypass and self.config.proactive_release and dead_drop_pressured:
            for rid, r in self.lifecycle.records.items():
                if len(dead_drops) >= self.config.max_actions * 2:
                    break
                can_sup_k = _can_drop_superseded(rid)
                if rid in self.ended_nodes or not (
                    r.closed or (can_sup_k and getattr(r, 'superseded', False))
                ):
                    continue
                dead_leaf = self._true_dead_leaf_node(
                    rid, shared_node_ids=shared_node_ids, allow_superseded=can_sup_k
                )
                if dead_leaf is None or dead_leaf.id in released_node_ids:
                    continue
                chain_ids = self.actions.plan_dead_chain(
                    dead_leaf,
                    allow_full_only_ancestors=full_pres,
                    forbidden_nodes=dead_forbidden,
                )
                if chain_ids:
                    dead_drops.append({
                        'rid': rid,
                        'node': dead_leaf.id,
                        'chain_ids': chain_ids,
                        'full_pres': full_pres,
                    })
                    released_node_ids.update(chain_ids)

        if self.config.dead_leaf_bypass and backups:
            backups = [
                (b_rid, b)
                for b_rid, b in backups
                if self._true_dead_leaf_node(
                    b_rid, shared_node_ids=shared_node_ids, allow_superseded=_can_drop_superseded(b_rid)
                )
                is None
            ]

        return {
            'warm': warm,
            'cold': cold,
            'ended': ended,
            'backups': backups,
            'proactive_releases': proactive_releases,
            'dead_drops': dead_drops,
            'hbm_pressured': hbm_pressured,
            'anchor_prefetches': [
                (k, prid, {pk: pv for pk, pv in pl.items() if pk not in {'tokens', 'namespace'}} | {'tokens': pl['tokens'], 'namespace': pl['namespace']})
                for k, prid, pl in anchor_prefetches
            ],
        }

    def before_admission(self):
        self._check_safe()
        if not self.active:
            return
        self._ensure_installed()
        if not self.config.demand_release and not self.config.queue_prefetch:
            return
        if self.now - self.last_admission_action < self.config.tick_s:
            return
        self.last_admission_action = self.now
        queue = self._queue()
        self.probes = self._phase('probe_queue', lambda: {str(r.rid): self.probe(r) for r in queue[:self.config.scan_limit]})
        self._agree({
            'phase': 'interval_admission_snapshot',
            'queue': [(str(r.rid), dict(vars(self.probes[str(r.rid)]))) for r in queue[:self.config.scan_limit]],
            'slot_room': self.actions.slot_room(),
            'pools': self.actions.pool_state(),
            'pending': list(self.actions.pending),
        })
        if not queue:
            return
        if not self.config.allow_full_slot_prefetch and self.actions.slot_room() <= len(self.actions.pending):
            self.counts['no_state_slot'] += 1
            return
        if self.config.demand_release:
            releases = self._phase('release_plan', lambda: self._release_plan(queue))
            if self._agree_optional_plan('release_plan', releases):
                for item in releases:
                    node = self.cache.tree_core.node_by_id(item['node'])
                    self._act('backed_leaf_release', item, lambda n=node: self.actions.release(n))
        if not self.config.queue_prefetch or len(self.actions.pending) >= self.config.max_inflight:
            return
        partial = getattr(self.s, 'chunked_req', None)
        if partial is not None and not self.config.prefetch_during_long_continuation:
            total = len(partial.origin_input_ids) + len(partial.output_ids)
            planned_end = getattr(partial, 'extend_end', getattr(partial, 'prefix_len', 0))
            chunk = max(1, getattr(self.s, '_orvix_base_chunk_size', None) or getattr(self.s, 'chunked_prefill_size', 2048) or 2048)
            if total - planned_end > chunk * self.config.prefetch_max_remaining_chunks:
                self.counts['prefetch_deferred_long_continuation'] += 1
                return
        candidates = queue[:self.config.scan_limit]
        overdue = [r for r in candidates if self.now - r._orvix_interval_ready >= self.config.age_guard_s]
        window = (overdue[:1] + [r for r in candidates if r not in overdue[:1]])[:self.config.prefetch_window]
        for req in window:
            p = self.probes[str(req.rid)]
            if (
                p.prefetch_pending_len
                or p.host_node_id is None
                or p.host_recoverable_len - p.gpu_ready_len < self.config.min_host_tokens
            ):
                continue
            node = self.cache.tree_core.node_by_id(p.host_node_id)
            plan = self._phase('restore_plan', lambda: self.actions.restore_plan(node, self.config))
            if plan is None:
                self.counts['prefetch_capacity_or_state_declined'] += 1
                continue
            if plan['bytes'] is None or plan['full'] > self.config.max_transfer_tokens:
                self.counts['prefetch_unknown_bytes_or_size_declined'] += 1
                continue
            if not self.buckets['h2d'].allow(plan['bytes'], self.now):
                self.counts['h2d_rate_limited'] += 1
                continue
            desc = {**self._prefetch_receipt(plan), 'rid': str(req.rid)}
            started = self._act('queue_prefetch', desc, lambda: self.actions.start_prefetch(str(req.rid), plan))
            if started:
                self.buckets['h2d'].consume(plan['bytes'], self.now)
                self.probes[str(req.rid)] = self.probe(req)
                if len(self.actions.pending) >= self.config.max_inflight:
                    break

    def _release_plan(self, queue):
        pools = self.actions.pool_state()
        tail_state = self.actions.swa_tail_state()
        page = getattr(self.cache, 'page_size', 256)
        chunk = getattr(self.s, '_orvix_base_chunk_size', None) or getattr(self.s, 'chunked_prefill_size', 2048) or 2048
        if chunk <= 0:
            return []
        tail_shortage = 0
        if tail_state['max_tails'] is not None:
            threshold = int(tail_state['max_tails'] * self.config.swa_tail_watermark)
            if tail_state['active_tails'] >= threshold:
                tail_shortage = max(1, tail_state['active_tails'] - threshold + 1)

        needs_list = []
        for req in queue[:self.config.prefetch_window]:
            p = self.probes[str(req.rid)]
            if p.prefetch_pending_len:
                continue
            need_by_pool = {}
            # Align demand release with FULL H2D restore_plan size when Host prefix is recoverable!
            raw_restore = None
            if (
                self.config.queue_prefetch
                and p.host_node_id is not None
                and p.host_recoverable_len - p.gpu_ready_len >= self.config.min_host_tokens
            ):
                try:
                    hnode = self.cache.tree_core.node_by_id(p.host_node_id)
                    raw_restore = self.actions.restore_plan(hnode, self.config, ignore_capacity=True)
                except KeyError:
                    raw_restore = None

            chunk_need = ((min(chunk, p.true_missing_len) + page - 1) // page + 1) * page
            for pname, x in pools.items():
                rsv = max(self.config.growth_reserve_tokens, int(x['capacity'] * self.config.reserve_fraction))
                target_need = chunk_need + self.config.growth_reserve_tokens
                if raw_restore is not None and raw_restore.get(pname, 0) > 0:
                    cap_tok = min(raw_restore[pname], self.config.max_transfer_tokens) if pname == 'full' else raw_restore[pname]
                    target_need = max(target_need, cap_tok + rsv)
                need_by_pool[pname] = max(0, target_need - x['free'])
            if any(need_by_pool.values()) or tail_shortage > 0:
                needs_list.append((req, need_by_pool, raw_restore is not None))
        if not needs_list:
            return []

        candidates = self._idle_backed_release_candidates()
        if not candidates:
            return []

        for req, shortage, for_restore in needs_list:
            chosen = []
            freed = Counter()
            tails_freed = 0
            for _, rid, node in candidates:
                if len(chosen) >= self.config.max_actions:
                    break
                chosen.append((rid, node.id))
                fp = self.actions.footprint(node)
                freed.update(fp)
                if fp.get('swa', 0) > 0:
                    tails_freed += 1
                if all(freed[n] >= v for n, v in shortage.items()) and tails_freed >= tail_shortage:
                    break
            # Under streaming_prefetch or Host restore, partial freeing still unlocks prefix-slice H2D progress!
            enough_full = all(freed[n] >= v for n, v in shortage.items())
            partial_useful = (
                for_restore
                and self.config.streaming_prefetch
                and freed.get('full', 0) >= self.config.min_host_tokens
            )
            if chosen and (enough_full or partial_useful or tails_freed > 0):
                return [
                    {'rid': rid, 'node': nid, 'beneficiary': str(req.rid), 'shortage': shortage, 'tail_shortage': tail_shortage}
                    for rid, nid in chosen
                ]
        return []

    def order(self, native):
        self._check_safe()
        native = list(native)
        if not self.active and not self.actions.pending:
            return native
        if len(native) <= 1 and not self.actions.pending:
            for r in native:
                if not hasattr(r, '_orvix_interval_ready'):
                    r._orvix_interval_ready = self.now
                    self.arrival_counter += 1
                    r._orvix_interval_arrival = self.arrival_counter
            return native
        queue = self._queue()
        allowed = {str(r.rid): r for r in native}
        window = queue[:self.config.scan_limit]
        items = []
        for req in window:
            rid = str(req.rid)
            if rid not in allowed:
                continue
            p = self.probe(req)
            self.probes[rid] = p
            user = str(_metadata(req).get('family_id', '__unknown__'))
            is_prefetched = (rid in self.ready_leases) or self._has_anchor_lease(req, p)
            pk = _program_key(req)
            if pk is not None:
                if pk not in self.program_arrival:
                    if len(self.program_arrival) >= self.config.max_records:
                        self.program_arrival.popitem(last=False)
                    self.program_arrival[pk] = req._orvix_interval_ready
                prog_arr = self.program_arrival[pk]
                prog_srv = float(self.program_service.get(pk, 0.0))
            else:
                prog_arr = req._orvix_interval_ready
                prog_srv = 0.0
            items.append(
                QueueItem(
                    rid,
                    user,
                    req._orvix_interval_ready,
                    p.gpu_ready_len,
                    p.host_recoverable_len,
                    p.true_missing_len,
                    p.prefetch_pending_len > 0,
                    is_prefetched,
                    self._workflow_turns_for_req(req),
                    int(getattr(req, '_orvix_interval_arrival', 0)),
                    prog_srv,
                    prog_arr,
                )
            )
        chunk = max(1, getattr(self.s, '_orvix_base_chunk_size', None) or getattr(self.s, 'chunked_prefill_size', 2048) or 2048)
        running_count = len(getattr(getattr(self.s, 'running_batch', None), 'reqs', ()) or ())
        max_slots = int(getattr(self.s, 'max_running_requests', 64) or 64)
        if self.active and self.config.ready_order:
            selected = order_window(
                items,
                self.now,
                self.config,
                self.service,
                self.bypass,
                chunk,
                running_count=running_count,
                max_slots=max_slots,
            )
        else:
            pending = any(x.pending for x in items)
            selected = [
                x
                for x in items
                if not x.pending and (
                    not pending
                    or not self.config.block_unrelated_during_prefetch
                    or (x.missing <= chunk * self.config.filler_max_chunks and x.host <= x.gpu)
                )
            ]
        window_ids = {x.rid for x in items}
        pending_any = bool(self.actions.pending)
        tail = []
        for req in native:
            if str(req.rid) in window_ids:
                continue
            p = self.probe(req)
            if p.prefetch_pending_len:
                continue
            if (
                pending_any
                and self.config.block_unrelated_during_prefetch
                and (p.true_missing_len > chunk * self.config.filler_max_chunks or p.host_recoverable_len > p.gpu_ready_len)
            ):
                continue
            tail.append(req)
        ready = [allowed[x.rid] for x in selected] + tail
        chosen = {id(r) for r in ready}
        ordered = ready + [r for r in native if id(r) not in chosen]
        self._agree({'phase': 'interval_order', 'rids': [str(r.rid) for r in ordered]})
        return ordered

    def can_admit(self, req):
        self._check_safe()
        if not self.actions.pending:
            return True
        p = self.probe(req)
        if p.prefetch_pending_len:
            self.counts['deferred_pending_prefix'] += 1
            return False
        if self.config.block_unrelated_during_prefetch:
            chunk = max(1, getattr(self.s, '_orvix_base_chunk_size', None) or getattr(self.s, 'chunked_prefill_size', 2048) or 2048)
            if p.true_missing_len > chunk * self.config.filler_max_chunks or p.host_recoverable_len > p.gpu_ready_len:
                self.counts['deferred_long_filler'] += 1
                return False
        return True

    def on_admitted(self, reqs):
        if self.config.mode == 'off':
            return
        ids = {str(r.rid) for r in reqs}
        queue = self._queue()
        for r in queue:
            rid = str(r.rid)
            if rid not in ids and any(
                x._orvix_interval_ready > r._orvix_interval_ready for x in reqs if hasattr(x, '_orvix_interval_ready')
            ):
                self.bypass[rid] = min(self.config.max_bypass, self.bypass.get(rid, 0) + 1)
        for r in reqs:
            rid = str(r.rid)
            if rid in self.ready_leases:
                _, nid = self.ready_leases.pop(rid)
                p = self.probe(r)
                self.counts['prefetch_consumed_candidates'] += 1
                if p.gpu_ready_len > 0:
                    self.counts['prefetch_admitted_with_device_prefix'] += 1
            if self.anchor_leases:
                meta = _metadata(r)
                pred_rid = meta.get('predecessor_request_id')
                matched_nid = None
                for nid, (_, parent_rid, _) in list(self.anchor_leases.items()):
                    if pred_rid and str(pred_rid) == parent_rid:
                        matched_nid = nid
                        break
                if matched_nid is None:
                    p = self.probe(r)
                    if p.gpu_ready_len > 0:
                        for nid in list(self.anchor_leases):
                            try:
                                anode = self.cache.tree_core.node_by_id(nid)
                                path_len = sum(len(n.key) for n in self.actions.path(anode))
                                if p.gpu_ready_len >= path_len > 0:
                                    matched_nid = nid
                                    break
                            except KeyError:
                                continue
                if matched_nid is not None:
                    self.anchor_leases.pop(matched_nid, None)
                    self.counts['anchor_prefetch_consumed'] += 1
                    self.counts['prefetch_admitted_with_device_prefix'] += 1
            self.bypass.pop(rid, None)
        if self.config.phase_chain_turn_threshold > 0:
            self._sync_active_chain_flag()

    def _sync_active_chain_flag(self):
        running_reqs = getattr(getattr(self.s, 'running_batch', None), 'reqs', ()) or ()
        self.s._orvix_active_chain_in_running = bool(
            self.active
            and self.config.phase_chain_turn_threshold > 0
            and any(
                self._workflow_turns_for_req(r) >= self.config.phase_chain_turn_threshold
                for r in running_reqs
            )
        )

    def on_launch(self, batch):
        if self.config.mode == 'off':
            self.s._orvix_critical_in_running = False
            self.s._orvix_active_chain_in_running = False
            return
        running_reqs = getattr(getattr(self.s, 'running_batch', None), 'reqs', ()) or ()
        self.s._orvix_critical_in_running = bool(
            self.active
            and self.config.critical_in_batch_fold
            and any(
                self._workflow_turns_for_req(r) >= self.config.critical_fold_turn_threshold
                for r in running_reqs
            )
        )
        self.s._orvix_active_chain_in_running = bool(
            self.active
            and self.config.phase_chain_turn_threshold > 0
            and any(
                self._workflow_turns_for_req(r) >= self.config.phase_chain_turn_threshold
                for r in running_reqs
            )
        )
        ticket = getattr(batch, 'forward_iter', None)
        if ticket is None or ticket in self.tickets:
            return
        new_users = {str(_metadata(r).get('family_id', '__unknown__')) for r in batch.reqs} - self.service.keys()
        if len(self.service) + len(new_users) > self.config.max_records:
            self.tracking_limit = True
            self.counts['service_tracking_overflow'] += 1
            return
        rows = []
        lengths = getattr(batch, 'extend_lens', None) or []
        for i, r in enumerate(batch.reqs):
            user = str(_metadata(r).get('family_id', '__unknown__'))
            if user not in self.service:
                self.service[user] = min(self.service.values(), default=0.0)
            is_decode = not batch.forward_mode.is_extend() or any(
                r is x for x in (getattr(batch, 'decoding_reqs', None) or ())
            )
            amount = self.config.output_weight if is_decode else float(lengths[i] if i < len(lengths) else 0)
            self.service[user] += amount
            pk = _program_key(r)
            if pk is not None:
                if pk in self.program_service:
                    self.program_service.move_to_end(pk)
                elif len(self.program_service) >= self.config.max_records:
                    self.program_service.popitem(last=False)
                self.program_service[pk] = self.program_service.get(pk, 0.0) + amount
            self.output_seen.setdefault(str(r.rid), len(r.output_ids))
            rows.append((r, user, len(r.output_ids), self.config.output_weight if is_decode else 0.0))
        self.tickets[ticket] = rows

    def on_result(self, batch, result):
        if self.config.mode == 'off':
            return
        ticket = getattr(batch, 'forward_iter', None)
        rows = self.tickets.pop(ticket, None)
        if rows is None:
            return
        for req, user, start, reserved_output in rows:
            rid = str(req.rid)
            seen = self.output_seen.get(rid, start)
            delta = max(0, len(req.output_ids) - seen)
            self.output_seen[rid] = max(seen, len(req.output_ids))
            adj = delta * self.config.output_weight - reserved_output
            self.service[user] += adj
            pk = _program_key(req)
            if pk is not None and pk in self.program_service:
                self.program_service[pk] = max(0.0, self.program_service[pk] + adj)
            if req.finished() and not any(
                any(str(r.rid) == rid for r, _, _, _ in rs) for rs in self.tickets.values()
            ):
                self.output_seen.pop(rid, None)
        self.counts['settled_batches'] += 1

    def snapshot(self):
        return {
            'version': VERSION,
            'implementation': IMPLEMENTATION,
            'config': asdict(self.config),
            'effective_mode': self.config.mode if (self.config.mode != 'active' or self.active) else 'observe',
            'capabilities': self.capabilities,
            'clock': self.now,
            'counts': dict(self.counts),
            'lifecycle': dict(self.lifecycle.counts),
            'history': {
                k: {s: sum(status == s for status, _ in v) for s in ('ok', 'error', 'cancelled', 'timeout')}
                for k, v in self.lifecycle.history.items()
            },
            'pending_h2d': list(self.actions.pending),
            'anchor_leases_active': len(self.anchor_leases),
            'fenced_prefixes': len(self.fences.items),
            'active_workflows_tracked': len(self.workflow_turns),
            'critical_workflows_active': sum(
                1 for v in self.workflow_turns.values() if v >= self.config.critical_turn_threshold
            ),
            'hbm_pressure_active': self.hbm_pressure_active,
            'swa_tails': self.actions.swa_tail_state(),
            'native_actions': dict(self.actions.counts),
            'idle': self.idle(),
            'poisoned': self.poisoned or self.actions.unsafe,
            'native_writeback_instrumented': self.actions.native_writeback_instrumented,
            'native_writeback_calls': self.actions.native_writeback_calls,
            'native_writeback_wall_s': self.actions.native_writeback_wall_s,
            'service_proxy': dict(self.service),
            'receipts': list(self.log),
            'gpu_validated': False,
        }


def install_interval(scheduler, tracking, key_factory, parallel=None):
    controller = IntervalCacheController(scheduler, tracking, key_factory)
    if parallel is not None:
        for name in ('pp_size', 'attn_dp_size', 'attn_cp_size'):
            if getattr(parallel, name, 1) != 1:
                controller.capabilities['reasons'].append(name + '_unqualified')
                controller.capabilities['supported'] = False
    if getattr(scheduler, 'is_mixed_chunk', False):
        controller.capabilities['reasons'].append('mixed_chunk_not_qualified_for_v7')
        controller.capabilities['supported'] = False
    scheduler._orvix_interval_cache = controller
    controller.actions.install_writeback_timing()
    if controller.capabilities['supported']:
        controller.actions.install()
        controller.native_installed = True
    previous = scheduler._process_hicache_events

    def events(*args, **kwargs):
        result = previous(*args, **kwargs)
        controller.after_events()
        return result

    scheduler._process_hicache_events = events
    previous_abort = getattr(scheduler, 'abort_request', None)
    if callable(previous_abort):
        def abort_request(recv_req, *args, **kwargs):
            controller.on_abort(recv_req)
            return previous_abort(recv_req, *args, **kwargs)

        scheduler.abort_request = abort_request
    return controller
