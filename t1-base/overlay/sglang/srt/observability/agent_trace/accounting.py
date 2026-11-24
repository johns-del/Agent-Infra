"""Read native DSV4 pool metadata; distinguish backing capacity from occupancy.

No KV values, extra cache lookups, allocator mutations, or GPU synchronization.
The supported scope is non-unified DSV4 KV, indexer and compression-state pools;
weights, allocator metadata and HiCache copy scratch are explicitly excluded.
"""
from collections import defaultdict


def reserved_audit(layouts, inventory):
    p = layouts.pool
    unsupported = {'complete': False, 'reason': 'unsupported_native_accounting_contract'}
    if not layouts.dsv4 or getattr(p, '_unified_kv', True):
        return unsupported
    # The native PD buffer interfaces independently enumerate all three groups.
    native = {}
    for method in ('get_contiguous_buf_infos', 'get_state_buf_infos', 'get_c128_state_buf_infos'):
        fn = getattr(p, method, None)
        if not callable(fn):
            return unsupported
        ptrs, sizes, _ = fn()
        for ptr, size in zip(ptrs, sizes):
            key = int(ptr)
            if key in native and native[key] != int(size):
                return {'complete': False, 'reason': 'native_alias_size_mismatch'}
            native[key] = int(size)
    gpu_native = sum(native.values())
    gpu_inventory = sum(n for dev, n in inventory['unique_storage_bytes_by_device'].items() if dev.startswith('cuda:'))
    host_native = {}
    all_private = True
    entries = getattr(getattr(layouts.cache, 'host_pool_group', None), 'entries', [])
    for entry in entries:
        hp = entry.host_pool
        if type(hp).__name__ == 'LogicalHostPool':
            continue
        if type(hp).__name__ not in ('DeepSeekV4PagedHostPool', 'DeepSeekV4StateHostPool'):
            return unsupported
        for tensor in hp.get_hybrid_pool_buffer():
            storage = tensor.untyped_storage()
            host_native[int(storage.data_ptr())] = int(storage.nbytes())
            all_private &= not storage.is_shared()
    host_bytes = sum(host_native.values())
    matched = (gpu_native == gpu_inventory and host_bytes == inventory['unique_storage_bytes_by_device'].get('cpu', 0)
               and not inventory['enumeration_budget_exhausted'])
    return {'complete': matched, 'scope': 'DSV4_KV_indexer_compression_state_backing',
            'byte_basis': 'native_buffer_interfaces_reconciled_with_deduplicated_tensor_storages',
            'device_native_buffer_count': len(native), 'device_native_bytes': gpu_native,
            'host_native_storage_count': len(host_native), 'host_native_bytes': host_bytes,
            'host_storages_process_private': all_private,
            'excluded': ['model_weights', 'allocator_index_metadata', 'HiCache_copy_scratch', 'CUDA_allocator_reserve'],
            'includes_padding_and_unused_capacity': True,
            'is_valid_context_bytes': False}


def allocation_totals(layouts):
    """Charge occupied allocator pages, including native C4 sidecars and C128 slots."""
    p, a, s = layouts.pool, layouts.alloc, layouts.s
    if not layouts.dsv4 or getattr(p, '_unified_kv', True) or type(a).__name__ != 'SWATokenToKVPoolAllocator':
        return {'complete': False, 'reason': 'unsupported_native_allocator_contract'}
    page = layouts.full_page
    full_used = int(a.size_full) - int(a.full_available_size())
    swa_used = int(a.size_swa) - int(a.swa_available_size())
    if full_used < 0 or swa_used < 0 or full_used % page or swa_used % p.swa_page_size:
        return {'complete': False, 'reason': 'allocator_page_conservation_failed'}
    rows = []
    for name, used in (('c4', full_used), ('c128', full_used), ('c4_indexer', full_used), ('swa', swa_used)):
        spec = layouts.specs.get(('GPU', name))
        if spec is None or spec['page_bytes'] is None:
            return {'complete': False, 'reason': 'native_pool_geometry_missing'}
        rows.append({'tier': 'GPU', 'pool': name, 'occupied_pages': used // spec['page_size'],
                     'bytes': used // spec['page_size'] * spec['page_bytes'],
                     'capacity_bytes': int(a.size_swa if name == 'swa' else a.size_full) // spec['page_size'] * spec['page_bytes'],
                     'basis': 'native_allocator_occupied_pages'})
    req_pool = getattr(s, 'req_to_token_pool', None)
    if req_pool is None or type(req_pool).__name__ != 'ReqToTokenPool':
        return {'complete': False, 'reason': 'request_slot_accounting_unavailable'}
    req_used = int(req_pool.size) - int(req_pool.available_size())
    if not 0 <= req_used <= req_pool.size:
        return {'complete': False, 'reason': 'request_slot_conservation_failed'}
    for key in ('compress_state_pools', 'indexer_compress_state_pools'):
        grouped = defaultdict(int)
        capacities = defaultdict(int)
        for pool in getattr(p, key):
            if pool is None:
                continue
            ratio = int(pool.ratio)
            if ratio not in (4, 128):
                return {'complete': False, 'reason': 'unsupported_compression_ratio'}
            # Native pool configurator: one C4 ring per SWA page; C128 per request slot.
            expected = p.swa_size // p.swa_page_size * pool.ring_size
            if ratio == 4 and (p.c4_state_pool_size < expected or pool.swa_page_size != p.swa_page_size):
                return {'complete': False, 'reason': 'C4_state_mapping_contract_mismatch'}
            units = swa_used // p.swa_page_size if ratio == 4 else req_used
            grouped[ratio] += units * int(pool.ring_size) * int(pool.kv_score_buffer.kv_score[0].nbytes)
            capacity_units = int(a.size_swa) // p.swa_page_size if ratio == 4 else int(req_pool.size)
            capacities[ratio] += capacity_units * int(pool.ring_size) * int(pool.kv_score_buffer.kv_score[0].nbytes)
        for ratio, n in grouped.items():
            rows.append({'tier': 'GPU', 'pool': key + ':c' + str(ratio), 'bytes': n,
                         'capacity_bytes': capacities[ratio],
                         'basis': 'native_SWA_page_ring' if ratio == 4 else 'native_allocated_request_slot_ring'})
    entries = getattr(getattr(layouts.cache, 'host_pool_group', None), 'entries', [])
    host_by_name = {str(entry.name): entry.host_pool for entry in entries}
    sidecars = {str(x.pool_name): str(x.indices_from_pool) for x in getattr(layouts.cache, 'sidecar_pool_specs', [])}
    for entry in entries:
        hp = entry.host_pool
        spec = layouts.specs.get(('Host', str(entry.name)))
        if spec is None or spec['page_bytes'] is None:
            return {'complete': False, 'reason': 'host_pool_geometry_missing'}
        if str(entry.name) in sidecars:
            # All sidecars reuse their named anchor's indices. This includes
            # C4/C128/indexer paged buffers (LogicalHostPool anchor), not only
            # state rings (SWA anchor). Their private free lists stay unused.
            anchor = host_by_name.get(sidecars.get(str(entry.name)))
            if anchor is None or type(anchor).__name__ not in ('DeepSeekV4PagedHostPool','LogicalHostPool') or anchor.size != hp.size or anchor.page_size != hp.page_size:
                return {'complete': False, 'reason': 'host_state_anchor_contract_unavailable'}
            used = int(anchor.size) - int(anchor.available_size())
        else:
            used = int(hp.size) - int(hp.available_size())
        if used < 0 or used % spec['page_size']:
            return {'complete': False, 'reason': 'host_page_conservation_failed'}
        rows.append({'tier': 'Host', 'pool': str(entry.name), 'occupied_pages': used // spec['page_size'],
                     'bytes': used // spec['page_size'] * spec['page_bytes'],
                     'capacity_bytes': int(hp.size) // spec['page_size'] * spec['page_bytes'],
                     'basis': 'native_host_allocator_occupied_pages'})
    total = defaultdict(int)
    capacity = defaultdict(int)
    for row in rows:
        total[row['tier']] += row['bytes']
        capacity[row['tier']] += row['capacity_bytes']
    return {'complete': True, 'scope': 'DSV4_KV_indexer_compression_state_backing',
            'occupied_bytes_by_tier': dict(total), 'capacity_bytes_by_tier': dict(capacity),
            'capacity_scope': 'allocator_addressable_units_excludes_padding_and_unaddressable_state_rows',
            'pools': rows, 'allocated_request_slots': req_used,
            'full_occupied_indices': full_used, 'swa_occupied_indices': swa_used,
            'meaning': 'allocator_charged_pages_and_request_state_slots_including_padding_and_inflight',
            'is_valid_context_bytes': False, 'consistency': 'single_scheduler_metadata_cut'}
