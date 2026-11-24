"""Qualified native actions. No hand-written tensor copies or allocator frees.

H2D is enqueued by native load_back, then submitted through L2TransferEngine
WITHOUT using layer_done_counter. Native load ACKs own all locks and cleanup.
The private completion does share the native DMA stream (not bandwidth). A
prefix-wide fence blocks every waiting consumer until native TP ACK processing.
"""
from __future__ import annotations
from collections import Counter
from array import array
import math
import time
from .interval_probe import PrefixFence, raw_ids, namespace
from .unified_cache_control import _node_identity, _name


class NativeActions:
    def __init__(self, scheduler, tracking, fences, *, ack_factory=None):
        self.s = scheduler
        self.cache = scheduler.tree_cache
        self.core = self.cache.tree_core
        self.tracking = tracking
        self.fences = fences
        self.ack_factory = ack_factory
        self.counts = Counter()
        self.pending = {}
        self.cancelled_completed = set()
        self.stats = {}
        self.native_writeback_instrumented = False
        self.native_writeback_calls = 0
        self.native_writeback_wall_s = 0.0
        self.prefetch_receipts = []
        self.unsafe = False

    def capabilities(self):
        reasons = list(self.tracking.capabilities.get('reasons', []))
        if not self.tracking.capabilities.get('tool_demotion_supported'):
            reasons.append('base_hicache_contract_unqualified')
        if getattr(self.core, 'is_eagle', False):
            reasons.append('bigram_keys_unqualified')
        if getattr(self.cache, 'enable_storage', False):
            reasons.append('l3_storage_unqualified')
        if getattr(self.cache, 'buffer_pipeline', None) is not None:
            reasons.append('buffer_only_unqualified')
        if getattr(self.core, 'enable_session_radix_cache', False):
            reasons.append('session_lru_partition_unqualified')
        if not callable(getattr(self.core, 'match_prefix', None)):
            reasons.append('match_fence_unavailable')
        cc = getattr(self.cache, 'cache_controller', None)
        required = (
            'load_queue',
            'ack_load_queue',
            'layer_done_counter',
            'l2_transfer_engine',
            '_move_op_indices',
            '_l2_load_transfers',
            '_transfer_num_bytes',
            '_num_tokens_by_pool',
            'mem_pool_device_allocator',
            'transfer_layer_id_max',
        )
        if cc is None or any(not hasattr(cc, n) for n in required):
            reasons.append('independent_h2d_contract_missing')
        for spec in getattr(self.cache, 'sidecar_pool_specs', ()):
            if _name(getattr(spec, 'indices_from_pool', None)) not in {'KV', 'SWA'}:
                reasons.append('independent_sidecar_allocation_unqualified')
        for lru in getattr(self.core, 'lru_lists', {}).values():
            if getattr(lru, 'mid', None) is not None:
                reasons.append('partitioned_lru_unqualified')
        if not callable(getattr(self.cache, 'load_back', None)):
            reasons.append('native_load_back_missing')
        return {
            'supported': not reasons,
            'reasons': sorted(set(reasons)),
            'prefetch_channel': 'independent_completion_on_native_H2D_stream',
            'publication': 'native_early_commit_with_prefix_fence_until_tp_ack',
            'logical_zero_byte_host_anchor_supported': True,
            'streaming_incremental_backup_supported': True,
            'streaming_partial_h2d_restore_supported': True,
            'host_swa_cold_reclaim_supported': callable(getattr(self.cache, 'evict_host', None)),
            'anchor_early_prefetch_supported': True,
            'eager_shared_host_arena_supported': False,
            'independent_sidecar_allocator_supported': False,
            'gpu_validated': False,
        }

    def install_writeback_timing(self):
        original = getattr(self.cache, 'backup_node_for_write_back', None)
        if callable(original) and not self.native_writeback_instrumented:
            def timed(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    self.native_writeback_calls += 1
                    self.native_writeback_wall_s += time.perf_counter() - start

            self.cache.backup_node_for_write_back = timed
            self.native_writeback_instrumented = True

    def install(self):
        self.fences.install(self.cache)
        self.install_writeback_timing()
        # Flush/reset cannot erase native ACKs while private fences still own
        # allocations. Refuse rather than cancel DMA or pretend a flush worked.
        original_reset = self.cache.reset

        def reset(*args, **kwargs):
            if self.pending or self.unsafe:
                raise RuntimeError('drain interval H2D before cache reset; unsafe state requires process restart')
            return original_reset(*args, **kwargs)

        self.cache.reset = reset

    def path(self, node):
        out = []
        while node is not self.core.root_node:
            if node is None or len(out) >= 4096:
                raise ValueError('invalid radix path')
            out.append(node)
            node = node.parent
        return out[::-1]

    def get_anchor(self, record, *, exclusive=False):
        try:
            node = self.core.node_by_id(record.node_id)
        except KeyError:
            return None
        if node is not record.node_object:
            return None
        if exclusive and node.children:
            return None
        if _node_identity(node, self.core.root_node)[0] != record.fingerprint:
            return None
        if exclusive:
            if any(r.node_object is node and r.scope != record.scope for r in self.tracking.records.values()):
                return None
        return node

    def pending_node(self, node):
        return (
            getattr(node, 'write_through_pending_id', None) is not None
            or getattr(node, 'load_back_pending_id', None) is not None
            or getattr(node, 'write_back_queued', False)
        )

    def touch(self, node, *, cold=False):
        # Timestamp alone is insufficient: SWA eviction also traverses its LRU.
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import get_and_increase_time_counter
        stamp = -1.0 if cold else get_and_increase_time_counter()
        node.last_access_time = stamp
        joint = getattr(self.core, 'joint_lru', None)
        if joint is not None and node.id in joint.swa_access_time:
            joint.swa_access_time[node.id] = stamp
        for lru in self.core.lru_lists.values():
            if not lru.in_list(node):
                continue
            if cold:
                if getattr(lru, 'mid', None) is not None:
                    raise RuntimeError('partitioned LRU not qualified')
                lru._remove_node(node)
                lru._add_node_after(lru.tail.lru_prev[lru._pt], node)
            else:
                lru.reset_node_mru(node)

    def footprint(self, node, *, host=False):
        return {
            _name(ct).lower(): len(getattr(node.component_data[ct], 'host_value' if host else 'value'))
            if getattr(node.component_data[ct], 'host_value' if host else 'value') is not None
            else 0
            for ct in self.cache.components
        }

    def swa_window_footprint(self, node):
        """Compute active resident footprint along path(node), capping SWA to one sliding window."""
        swa_comp = next((c for c in self.core.components if _name(c.component_type) == 'SWA'), None)
        window = getattr(swa_comp, 'sliding_window_size', None)
        page = getattr(self.core, 'page_size', 256) or 256
        out = Counter()
        swa_seen = 0
        for n in reversed(self.path(node)):
            fp = self.footprint(n, host=False)
            out['full'] += fp.get('full', 0)
            if fp.get('swa', 0):
                if window is None or swa_seen < window + page:
                    out['swa'] += fp['swa']
                    swa_seen += fp['swa']
        return dict(out)

    def swa_load_need(self, target_node):
        """Exact SWA tokens allocated by SGLang `SWAComponent.build_hicache_transfers(LOAD_BACK)`.

        Walks backward from `target_node` up to `sliding_window_size`, skipping
        segments already resident on GPU (`cd.value is not None`). Returns `None`
        if an unbacked SWA hole (`value is None and host_value is None`) lies
        inside the required window, preventing unsafe interior slice endpoints.
        """
        swa_comp = next((c for c in self.core.components if _name(c.component_type) == 'SWA'), None)
        swa_ct = next((ct for ct in self.cache.components if _name(ct) == 'SWA'), None)
        if swa_ct is None:
            return 0
        window = getattr(swa_comp, 'sliding_window_size', None)
        gates = getattr(swa_comp, 'gates_prefix_match', True)
        n_swa = 0
        need = 0
        cur = target_node
        while cur is not self.core.root_node and (window is None or n_swa < window):
            cd = cur.component_data[swa_ct]
            if cd.value is not None:
                n_swa += len(cd.value)
            elif cd.host_value is not None:
                need += len(cd.host_value)
                n_swa += len(cd.host_value)
            else:
                if gates or cur is target_node:
                    return None
                break
            cur = cur.parent
        return need

    def swa_tail_state(self):
        """Inspect active resident SWA prefix-tail leaves vs configured swa_prefix_tails quota."""
        max_tails = (
            getattr(getattr(self.s, 'server_args', None), 'swa_prefix_tails', None)
            or getattr(self.cache, 'swa_prefix_tails', None)
            or getattr(self.core, 'swa_prefix_tails', None)
            or getattr(getattr(self.cache, 'token_to_kv_pool_allocator', None), 'swa_prefix_tails', None)
        )
        swa_ct = next((ct for ct in self.cache.components if _name(ct) == 'SWA'), None)
        fast_active = getattr(self.core, 'active_swa_tails_count', None)
        active = 0
        if type(fast_active) is int and fast_active >= 0:
            active = fast_active
        elif swa_ct is not None:
            nodes_map = getattr(self.core, 'nodes', None)
            if isinstance(nodes_map, dict):
                for n in nodes_map.values():
                    if n is not self.core.root_node and not n.children:
                        cd = n.component_data[swa_ct]
                        if getattr(cd, 'value', None) is not None:
                            active += 1
            else:
                stack = list(getattr(self.core.root_node, 'children', {}).values())
                visited = 0
                while stack and visited < 32768:
                    cur = stack.pop()
                    visited += 1
                    children = getattr(cur, 'children', None)
                    if children:
                        stack.extend(children.values())
                    else:
                        cd = cur.component_data[swa_ct]
                        if getattr(cd, 'value', None) is not None:
                            active += 1
        valid_cap = int(max_tails) if type(max_tails) is int and max_tails > 0 else None
        return {
            'active_tails': active,
            'max_tails': valid_cap,
            'pressure_ratio': (active / valid_cap) if valid_cap else 0.0,
        }

    def pool_state(self):
        alloc = self.cache.token_to_kv_pool_allocator
        states = {}
        for name, cap, free in (('full', 'size_full', 'full_available_size'), ('swa', 'size_swa', 'swa_available_size')):
            capacity = getattr(alloc, cap, None)
            fn = getattr(alloc, free, None)
            if name == 'full':
                capacity = capacity or getattr(self.s, 'max_total_num_tokens', None)
                fn = fn if callable(fn) else getattr(alloc, 'available_size', None)
            if type(capacity) is int and capacity > 0 and callable(fn):
                evict_fn = getattr(self.cache, name + '_evictable_size', None)
                evictable = max(0, int(evict_fn())) if callable(evict_fn) else 0
                states[name] = {'capacity': capacity, 'free': int(fn()), 'evictable': evictable}
        return states

    def slot_room(self):
        batch = self.s.running_batch
        running = len(batch.reqs)
        rows = self.s.req_to_token_pool.available_size()
        room = min(int(rows), max(0, self.s.max_running_requests - running))
        # A native partial may own a row outside running_batch; reserve its
        # running-capacity place too (not a second physical request row).
        partial = getattr(self.s, 'chunked_req', None)
        if partial is not None and all(r is not partial for r in batch.reqs):
            room = min(room, max(0, self.s.max_running_requests - running - 1))
        return max(0, room)

    def bytes_upper_bound(self, full, swa):
        """Compute transfer byte upper bound across primary and sidecar host pools.

        DeepSeek-V4.1 uses `LogicalHostPool` (`size_per_token == 0`) for the `KV`
        anchor while physical bytes reside in `SWA` and `DEEPSEEK_V4_C4*` sidecars.
        Zero-byte logical anchor pools are skipped as long as at least one pool
        carries positive byte metadata (or zero tokens are requested).
        """
        group = getattr(self.cache, 'host_pool_group', None)
        entry_map = getattr(group, 'entry_map', None)
        if not isinstance(entry_map, dict) or not entry_map:
            return None
        total = 0
        positive_pools = 0
        for name, entry in entry_map.items():
            pool = getattr(entry, 'host_pool', None)
            size = getattr(pool, 'size_per_token', None)
            if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size < 0:
                return None
            if size == 0:
                continue
            positive_pools += 1
            label = _name(name)
            count = swa if label == 'SWA' else full
            for spec in getattr(self.cache, 'sidecar_pool_specs', ()):
                if spec.pool_name == name and _name(spec.indices_from_pool) == 'SWA':
                    count = swa
            item_bytes = getattr(pool, 'item_bytes', None)
            slot_page_size = getattr(pool, 'slot_page_size', None)
            if (
                not isinstance(item_bytes, bool)
                and isinstance(item_bytes, (int, float))
                and item_bytes > 0
                and type(slot_page_size) is int
                and slot_page_size > 0
            ):
                pages = (count + slot_page_size - 1) // slot_page_size if count > 0 else 0
                layers = getattr(pool, 'num_layers', getattr(pool, 'layer_num', None))
                if type(layers) is not int or layers <= 0:
                    lm = getattr(entry, 'layer_mapping', None)
                    bufs = getattr(pool, 'device_buffers', None)
                    if lm is not None and hasattr(lm, '__len__') and len(lm) > 0:
                        layers = len(lm)
                    elif bufs is not None and hasattr(bufs, '__len__') and len(bufs) > 0:
                        layers = len(bufs)
                    else:
                        layers = 1
                total += int(pages * layers * math.ceil(item_bytes))
            else:
                total += int(math.ceil(size * count))
        if positive_pools == 0:
            return 0 if (full == 0 and swa == 0) else None
        return total

    def _async_backup_closure(self, node):
        """Top-down unbacked closure for non-evicting `write_back=False` async D2H backup.

        Unlike `CacheControl._backup_closure` (which targets demand leaf eviction),
        read-only async D2H backup (`_execute_and_commit_kv_backup(write_back=False)`)
        safely increments `lock_ref` on shared read-only ancestors (`parent is not node`,
        such as a shared 3K system prompt held by decoding siblings in `running_batch`).
        Only the target `node` itself must be unlocked, and no ancestor in the backup
        action may carry pending transfers or active `host_lock_ref`.
        """
        if node.backuped and all(
            cd.value is None or cd.host_value is not None for cd in node.component_data
        ):
            return []
        if self.pending_node(node) or any(
            cd.lock_ref or cd.host_lock_ref for cd in node.component_data
        ):
            return None
        build = getattr(self.core, 'build_backup_kv_action', None)
        if not callable(build):
            return None
        ids = list(build(node.id).node_ids)
        if not ids:
            return None
        for nid in ids:
            parent = self.core.node_by_id(nid)
            if self.pending_node(parent) or any(
                cd.host_lock_ref for cd in parent.component_data
            ):
                return None
        return ids

    def _closure_cost_for_ids(self, chosen_ids, target_node):
        full_type = next(ct for ct in self.cache.components if _name(ct) == 'FULL')
        swa_component = next((c for c in self.core.components if _name(c.component_type) == 'SWA'), None)
        full = 0
        unbacked_swa = {}
        for nid in chosen_ids:
            current = self.core.node_by_id(nid)
            data = current.component_data[full_type]
            if data.value is not None and data.host_value is None:
                full += len(data.value)
            if swa_component is not None:
                for missing in swa_component._collect_unbacked_swa_nodes(current):
                    unbacked_swa[missing.id] = len(missing.component_data[swa_component.component_type].value)
        swa = sum(unbacked_swa.values())
        locked = Counter()
        last_node = self.core.node_by_id(chosen_ids[-1]) if chosen_ids else target_node
        for n in self.path(last_node):
            for ct in self.cache.components:
                cd = n.component_data[ct]
                if cd.value is not None and not cd.lock_ref:
                    locked[_name(ct).lower()] += len(cd.value)
        return {
            'ids': list(chosen_ids),
            'full': full,
            'swa': swa,
            'bytes': self.bytes_upper_bound(full, swa),
            'lock_footprint': dict(locked),
        }

    def backup_plan(self, node, *, max_chunk_tokens=None, max_bytes=None):
        ids = self._async_backup_closure(node)
        if ids is None:
            return None
        if not ids:
            return {'ids': [], 'full': 0, 'swa': 0, 'bytes': 0, 'lock_footprint': {}}
        full_plan = self._closure_cost_for_ids(ids, node)
        if (max_chunk_tokens is None and max_bytes is None) or len(ids) <= 1:
            return full_plan
        fits_tokens = max_chunk_tokens is None or full_plan['full'] <= max_chunk_tokens
        fits_bytes = max_bytes is None or (full_plan['bytes'] is not None and full_plan['bytes'] <= max_bytes)
        if fits_tokens and fits_bytes:
            return full_plan

        best = None
        for end_idx in range(1, len(ids) + 1):
            cand = self._closure_cost_for_ids(ids[:end_idx], node)
            if cand['bytes'] is None:
                return cand
            ok_tok = max_chunk_tokens is None or cand['full'] <= max_chunk_tokens
            ok_bytes = max_bytes is None or cand['bytes'] <= max_bytes
            if ok_tok and ok_bytes:
                best = cand
            elif best is not None:
                break
            else:
                best = cand
                break
        if best is not None and len(best['ids']) < len(ids):
            best = dict(best, partial_slice=True, remaining_nodes=len(ids) - len(best['ids']))
        return best

    def host_room(self, plan):
        group = self.cache.host_pool_group
        for name, entry in group.entry_map.items():
            pool = entry.host_pool
            if getattr(pool, 'shared_allocation_domain', None) is not None:
                return False
            if _name(name) in {'KV', 'SWA'}:
                needed = plan['full' if _name(name) == 'KV' else 'swa']
                fn = getattr(pool, 'available_size', None)
                if not callable(fn) or fn() < needed:
                    return False
        return True

    def _full_host_duplicate_victims(self, component, deficit):
        """Read the victim prefix of the native write-back duplicate walk."""
        if not getattr(self.core, 'is_write_back', False):
            return []
        dups = getattr(self.core, 'full_host_duplicates', None)
        if not isinstance(dups, dict) or not dups:
            return []
        preferred, deferred = [], []
        for node in dups.values():
            if getattr(node, 'write_back_queued', False) or not self.core._can_reclaim_full_host_duplicate(node):
                continue
            cd = node.component_data[component]
            hv = getattr(cd, 'host_value', None)
            if hv is None:
                continue
            victim = (node.id, len(hv))
            tier = deferred if self.core._is_device_walk_candidate(node) else preferred
            tier.append(victim)
        selected = []
        for victim in preferred + deferred:
            selected.append(victim)
            deficit -= victim[1]
            if deficit <= 0:
                break
        return selected

    def ensure_host_room(self, plan, *, allow_reclaim=False, allow_kv_reclaim=False, agree=None):
        """Coordinate optional host reclaim before invoking the native allocator."""
        room = self.host_room(plan)
        if agree is not None and not agree('backup_host_room', room):
            return False
        if room:
            return True
        if not allow_reclaim:
            return False
        evict_host = getattr(self.cache, 'evict_host', None)
        if not callable(evict_host):
            return False
        group = self.cache.host_pool_group
        targets = []
        for name, entry in group.entry_map.items():
            pool = entry.host_pool
            if getattr(pool, 'shared_allocation_domain', None) is not None:
                targets = None
                break
            lname = _name(name)
            if lname in {'SWA', 'KV'}:
                key = 'swa' if lname == 'SWA' else 'full'
                target_comp = 'SWA' if lname == 'SWA' else 'FULL'
                need = plan.get(key, 0)
                if need > 0:
                    fn = getattr(pool, 'available_size', None)
                    component = next((ct for ct in self.cache.components if _name(ct) == target_comp), None)
                    if not callable(fn) or component is None:
                        targets = None
                        break
                    targets.append((lname, component, need, fn))
        layout = None if targets is None else [(name, need) for name, _, need, _ in targets]
        if agree is not None and not agree('backup_host_layout', layout):
            return False
        if targets is None:
            return False
        deficits = [(name, component, max(0, need - int(available()))) for name, component, need, available in targets]
        # Default allow_kv_reclaim=False keeps Host SWA reclaim active while preventing
        # self-canceling FULL Host KV duplicate-reclaim thrash and TP8 rank divergence.
        blocked = any(name == 'KV' and deficit > 0 and not allow_kv_reclaim for name, _, deficit in deficits)
        for name, component, deficit in deficits:
            eff_deficit = 0 if (blocked or (name == 'KV' and not allow_kv_reclaim)) else deficit
            if agree is not None:
                reclaim = {'pool': name, 'deficit': eff_deficit}
                if name == 'KV' and eff_deficit:
                    reclaim['duplicate_victims'] = self._full_host_duplicate_victims(component, eff_deficit)
                if not agree('backup_host_reclaim', reclaim):
                    return False
            if eff_deficit:
                reclaimed = evict_host(eff_deficit, component)
                metric_prefix = 'host_swa' if name == 'SWA' else 'host_kv'
                self.counts[metric_prefix + '_reclaimed_tokens'] += max(0, int(reclaimed or 0))
                self.counts[metric_prefix + '_reclaim_calls'] += 1
        room = self.host_room(plan)
        if agree is not None and not agree('backup_host_room_after_reclaim', room):
            return False
        return room

    def backup(self, plan):
        written = self.cache._execute_and_commit_kv_backup(
            self.tracking.backup_factory(node_ids=plan['ids']), write_back=False
        )
        self.cache.flush_pending_backups()
        if plan.get('partial_slice'):
            self.counts['streaming_backup_slices'] += 1
        return int(written)

    def backed_and_idle(self, node):
        return (
            self.core._is_device_leaf(node)
            and node.backuped
            and not self.pending_node(node)
            and all(
                not cd.lock_ref
                and not cd.host_lock_ref
                and (cd.value is None or cd.host_value is not None)
                for cd in node.component_data
            )
        )

    def release(self, node):
        if not self.backed_and_idle(node):
            return {}
        tracker = {ct: 0 for ct in self.cache.components}
        action = self.cache._evict_device_leaf(node.id, tracker)
        if action is not None:
            raise RuntimeError('unexpected synchronous backup requirement')
        return {_name(ct).lower(): int(n) for ct, n in tracker.items()}

    def _host_only_subtree_clean(self, node, *, depth=0):
        if node is None or depth > 64:
            return False
        if not getattr(node, 'evicted', False) or not getattr(node, 'backuped', False):
            return False
        if self.pending_node(node):
            return False
        if any(cd.lock_ref or cd.host_lock_ref for cd in node.component_data):
            return False
        return all(
            self._host_only_subtree_clean(ch, depth=depth + 1)
            for ch in getattr(node, 'children', {}).values()
        )

    def can_drop_or_release_dead_node(self, node, *, expected_removed_child_id=None):
        """Return True when a dead/superseded non-root-child node can be safely dropped or released."""
        if (
            node is None
            or node is self.core.root_node
            or node.parent is None
            or node.parent is self.core.root_node
            or getattr(node, 'evicted', False)
            or self.pending_node(node)
            or any(cd.lock_ref or cd.host_lock_ref for cd in node.component_data)
        ):
            return False
        for ch in getattr(node, 'children', {}).values():
            if expected_removed_child_id is not None and ch.id == expected_removed_child_id:
                continue
            if not self._host_only_subtree_clean(ch):
                return False
        if not node.backuped and not callable(getattr(self.cache, '_drop_subtree_no_host', None)):
            return False
        return True

    def plan_dead_chain(self, leaf, *, allow_full_only_ancestors=True, forbidden_nodes=()):
        """Plan an exclusive dead-branch drop/release chain bottom-up, protecting root-child shared prompts."""
        root = self.core.root_node
        forbidden = set(forbidden_nodes)
        chain_ids = []
        cur = leaf
        prev_id = None
        steps = 0
        while cur is not None and cur is not root and steps < 32:
            if cur.id in forbidden or not self.can_drop_or_release_dead_node(
                cur, expected_removed_child_id=prev_id
            ):
                break
            if steps == 0 and not self.core._is_device_leaf(cur):
                break
            fp = self.footprint(cur)
            if steps > 0 and fp.get('swa', 0) == 0 and not allow_full_only_ancestors:
                break
            chain_ids.append(cur.id)
            prev_id = cur.id
            parent = cur.parent
            if parent is None or parent is root or len(getattr(parent, 'children', ())) > 1:
                break
            cur = parent
            steps += 1
        return chain_ids

    def drop_dead_chain_ids(self, chain_ids):
        """Execute a planned dead-branch drop/release chain bottom-up using native SGLang primitives."""
        root = self.core.root_node
        freed = {'full': 0, 'swa': 0}
        drop_fn = getattr(self.cache, '_drop_subtree_no_host', None)
        for nid in chain_ids:
            try:
                cur = self.core.node_by_id(nid)
            except KeyError:
                continue
            if (
                cur is root
                or cur.parent is None
                or cur.parent is root
                or not self.core._is_device_leaf(cur)
                or self.pending_node(cur)
                or any(cd.lock_ref or cd.host_lock_ref for cd in cur.component_data)
            ):
                break
            if cur.backuped:
                rel = self.release(cur)
                if not rel:
                    break
                for k, v in rel.items():
                    freed[k] = freed.get(k, 0) + int(v)
            else:
                if not callable(drop_fn):
                    break
                tracker = {ct: 0 for ct in self.cache.components}
                if not drop_fn(cur.id, tracker):
                    break
                self.counts['dead_leaf_dropped_no_host'] += 1
                for ct, v in tracker.items():
                    k = _name(ct).lower()
                    freed[k] = freed.get(k, 0) + int(v)
        return {k: int(v) for k, v in freed.items()}

    def _fits_device_pools(self, full, swa, pools, config):
        needs = {'full': full, 'swa': swa}
        for name, need in needs.items():
            if need == 0:
                continue
            if name not in pools:
                return False
            p = pools[name]
            reserve = max(config.growth_reserve_tokens, int(p['capacity'] * config.reserve_fraction))
            avail = p['free'] + (p.get('evictable', 0) if getattr(config, 'proactive_release', False) else 0)
            if avail < need + reserve:
                return False
        return True

    def restore_plan(self, node, config, *, ignore_capacity=False):
        path = self.path(node)
        full = 0
        earliest = None
        tokens = array('q')
        candidates = []
        for n in path:
            if self.pending_node(n):
                return None
            before = len(tokens)
            tokens.extend(raw_ids(n.key))
            for ct in self.cache.components:
                cd = n.component_data[ct]
                if cd.value is None and cd.host_value is not None:
                    if earliest is None:
                        earliest = before
                    if _name(ct) == 'FULL':
                        full += len(cd.host_value)
            if earliest is not None:
                candidates.append((n, full, len(tokens)))
        if earliest is None or not candidates:
            return None

        safe = earliest
        if ignore_capacity:
            swa_need = self.swa_load_need(node)
            if swa_need is None:
                return None
            return {
                'node_id': node.id,
                'full': full,
                'swa': swa_need,
                'bytes': self.bytes_upper_bound(full, swa_need),
                'safe_len': safe,
                'tokens': tokens,
                'namespace': namespace(node.key),
            }

        pools = self.pool_state()
        allow_streaming = bool(getattr(config, 'streaming_prefetch', False))
        search_list = list(reversed(candidates)) if allow_streaming else [candidates[-1]]
        for cand_node, cand_full, cand_tok_len in search_list:
            is_partial = cand_node is not node
            if is_partial:
                if cand_full < config.min_host_tokens or cand_full > config.max_transfer_tokens:
                    continue
                if cand_node.id in self.cache.ongoing_load_back:
                    continue
                if any(t['node_id'] == cand_node.id for t in self.pending.values()):
                    continue
            swa_need = self.swa_load_need(cand_node)
            if swa_need is None:
                continue
            if cand_full == 0 and swa_need == 0:
                continue
            if not self._fits_device_pools(cand_full, swa_need, pools, config):
                continue
            cand_tokens = tokens if not is_partial else array('q', tokens[:cand_tok_len])
            out = {
                'node_id': cand_node.id,
                'full': cand_full,
                'swa': swa_need,
                'bytes': self.bytes_upper_bound(cand_full, swa_need),
                'safe_len': safe,
                'tokens': cand_tokens,
                'namespace': namespace(cand_node.key),
            }
            if is_partial:
                out['partial_restore'] = True
            return out
        return None

    def start_prefetch(self, rid, plan):
        cc = self.cache.cache_controller
        if self.unsafe:
            raise RuntimeError('unsafe interval transfer state; restart private process')
        if cc.load_queue:
            return False
        # Keep one native layer-counter slot for on-demand loads. Our private
        # completion also occupies the common ACK queue, not its producer ring.
        slots = int(getattr(cc.layer_done_counter, 'num_counters', 0))
        if slots < 2 or len(cc.ack_load_queue) >= slots - 1:
            return False
        if plan['node_id'] in self.cache.ongoing_load_back:
            return False
        if any(t['node_id'] == plan['node_id'] for t in self.pending.values()):
            return False
        fence = PrefixFence(rid, plan['node_id'], plan['tokens'], plan['namespace'], plan['safe_len'], len(plan['tokens']))
        self.fences.items[rid] = fence
        ok = False
        try:
            ok = self.cache.load_back(plan['node_id'], mem_quota=None, req=None)
            if not ok:
                self.fences.items.pop(rid, None)
                return False
            if len(cc.load_queue) != 1 or cc.load_queue[0].node_ids != [plan['node_id']]:
                raise RuntimeError('prefetch must own exactly one isolated native load op')
            op = cc.load_queue[0]
            host, device, aux = cc._move_op_indices(op)
            cc.load_queue.clear()
            if cc.load_fence_stream is not None:
                cc.l2_transfer_engine.host_to_device_stream.wait_stream(cc.load_fence_stream)
            # No layer_done_counter update, no consumer index, no current batch
            # is bound to this copy. L2 engine creates an independent start event.
            done = cc.l2_transfer_engine.submit_host_to_device(
                cc._l2_load_transfers(host, device, aux),
                transfer_layer_id_max=cc.transfer_layer_id_max,
            )
            cc.mem_pool_device_allocator.set_hicache_transfer_done_event((id(cc), 'load'), done.finish_event)
            factory = self.ack_factory
            if factory is None:
                from sglang.srt.managers.cache_controller import HiCacheAck
                factory = HiCacheAck
            ack = factory(
                start_event=done.start_event,
                finish_event=done.finish_event,
                node_ids=op.node_ids,
                num_tokens=len(op.device_indices),
                timing_enabled=done.timing_enabled,
                num_tokens_by_pool=cc._num_tokens_by_pool(op),
                num_bytes=cc._transfer_num_bytes(op),
            )
            cc.ack_load_queue.append(ack)
            self.pending[rid] = {'node_id': plan['node_id'], 'ack': ack, 'cancelled': False}
            self.counts['prefetch_started'] += 1
            if plan.get('partial_restore'):
                self.counts['streaming_prefetch_slices'] += 1
            return True
        except BaseException:
            self.unsafe = True
            self.counts['fatal_prefetch_error'] += 1
            raise

    def mark_cancelled(self, recv_req):
        """Mark matching in-flight prefetches cancelled without aborting DMA early."""
        matches = getattr(recv_req, 'matches', None)
        target_rid = getattr(recv_req, 'rid', None)
        marked = []
        for rid, ticket in self.pending.items():
            raw_rid = rid.split(':', 1)[1] if rid.startswith('anchor:') else rid
            matched = (
                bool(matches(raw_rid) or matches(rid))
                if callable(matches)
                else (target_rid is not None and str(target_rid) in {rid, raw_rid})
            )
            if matched and not ticket.get('cancelled'):
                ticket['cancelled'] = True
                self.counts['prefetch_cancelled_inflight'] += 1
                marked.append(rid)
        return marked

    def poll_completed(self):
        completed = []
        self.cancelled_completed.clear()
        # Called only AFTER native TP-consensus check_hicache_events/loading_check.
        for rid, ticket in list(self.pending.items()):
            if ticket['node_id'] in self.cache.ongoing_load_back:
                continue
            ack = ticket['ack']
            if any(a is ack for a in self.cache.cache_controller.ack_load_queue):
                raise RuntimeError('native load receipt disappeared before ACK')
            if not ack.finish_event.query():
                raise RuntimeError('native ACK before DMA completion')
            self.fences.items.pop(rid, None)
            del self.pending[rid]
            if ticket.get('cancelled'):
                self.cancelled_completed.add(rid)
                try:
                    node = self.core.node_by_id(ticket['node_id'])
                    if self.core._is_device_leaf(node) and not self.pending_node(node):
                        self.touch(node, cold=True)
                except KeyError:
                    pass
            completed.append(rid)
            self.counts['prefetch_completed'] += 1
        return completed
