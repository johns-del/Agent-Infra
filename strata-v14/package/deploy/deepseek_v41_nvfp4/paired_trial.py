"""Reject cross-runtime/historical pairs instead of presenting them as an A/B."""
import math
from statistics import median


VALID_CANDIDATES = {
    'AA', 'F', 'C', 'P', 'FC', 'FP', 'CP', 'FCP',
    'G', 'Q', 'FQ', 'QS',
    'MS', 'ML', 'MF',
    'O', 'K', 'E', 'H', 'KEH', 'V8', 'V9', 'V10',
    'V11_M1', 'V11_M12', 'V11_M13', 'V11',
    'V12_M1', 'V12_M12', 'V12_M13', 'V12',
    'V13_M1', 'V13_M12', 'V13_M13', 'V13', 'PACER',
    'PLAS', 'CTTL',
    'IS', 'IR', 'ID', 'IP', 'IF', 'WTS',
}


def validate_pair(baseline, candidate):
    reasons, warnings = [], []
    a, b = baseline.get('paired_context', {}), candidate.get('paired_context', {})
    for key in ('campaign_id', 'pair_id', 'runtime_python_fingerprint', 'adapter_version'):
        if not a.get(key) or a.get(key) != b.get(key):
            reasons.append('missing_or_different_' + key)
    if a.get('repeat') != b.get('repeat') or type(a.get('repeat')) is not int:
        reasons.append('invalid_repeat')
    if {a.get('position'), b.get('position')} != {0, 1}:
        reasons.append('invalid_pair_positions')
    if baseline.get('method') != 'B0' or candidate.get('method') not in VALID_CANDIDATES:
        reasons.append('invalid_arm_identity')
    if candidate.get('method') == 'AA':
        for key in ('parameters', 'knobs', 'cache_configuration', 'prefill_configuration'):
            if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
                reasons.append('aa_policy_mismatch_' + key)
    version = a.get('adapter_version')
    if version in {'orvix-dsv41-fair-chunk-v5', 'orvix-dsv41-multi-prefill-v6', 'orvix-dsv41-interval-cache-v7'}:
        expected = {'G': 'guard', 'Q': 'share', 'FQ': 'fair', 'QS': 'shadow'}
        if baseline.get('chunk_configuration', {}).get('mode') != 'off':
            reasons.append('baseline_chunk_not_explicitly_off')
        desired = expected.get(candidate.get('method'), 'off')
        if candidate.get('chunk_configuration', {}).get('mode') != desired:
            reasons.append('candidate_chunk_mode_not_acknowledged')
        if candidate.get('method') == 'AA' and baseline.get('chunk_configuration') != candidate.get('chunk_configuration'):
            reasons.append('aa_policy_mismatch_chunk_configuration')
    if version == 'orvix-dsv41-multi-prefill-v6':
        if baseline.get('multi_configuration', {}).get('mode') != 'off':
            reasons.append('baseline_multi_not_explicitly_off')
        desired = {'MS': 'shadow', 'ML': 'locality', 'MF': 'fair'}.get(candidate.get('method'), 'off')
        if candidate.get('multi_configuration', {}).get('mode') != desired:
            reasons.append('candidate_multi_mode_not_acknowledged')
        if candidate.get('method') == 'AA' and baseline.get('multi_configuration') != candidate.get('multi_configuration'):
            reasons.append('aa_policy_mismatch_multi_configuration')
    if version == 'orvix-dsv41-interval-cache-v7':
        if baseline.get('interval_configuration', {}).get('mode') not in {'off', 'observe'}:
            reasons.append('baseline_interval_not_passive')
        if candidate.get('method') == 'IS':
            if candidate.get('interval_configuration', {}).get('mode') != 'observe':
                reasons.append('candidate_interval_observe_not_acknowledged')
        elif candidate.get('method') in {
            'O', 'K', 'E', 'H', 'KEH', 'V8', 'V9', 'V10',
            'V11_M1', 'V11_M12', 'V11_M13', 'V11',
            'V12_M1', 'V12_M12', 'V12_M13', 'V12',
            'V13_M1', 'V13_M12', 'V13_M13', 'V13', 'PACER',
            'PLAS', 'CTTL',
            'IR', 'ID', 'IP', 'IF',
        }:
            if candidate.get('interval_configuration', {}).get('mode') != 'active':
                reasons.append('candidate_interval_active_not_acknowledged')
        if candidate.get('method') == 'AA' and baseline.get('interval_configuration') != candidate.get('interval_configuration'):
            reasons.append('aa_policy_mismatch_interval_configuration')
    keys_to_match = ('offered_graph_concurrency', 'replay_options')
    for key in keys_to_match:
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            reasons.append('missing_or_different_' + key)
    baseline_engine = baseline.get('engine_configuration')
    candidate_engine = candidate.get('engine_configuration')
    if baseline_engine is None or candidate_engine is None:
        reasons.append('missing_or_different_engine_configuration')
    elif candidate.get('method') == 'WTS':
        if (baseline_engine.get('hicache_write_policy') != 'write_back'
                or candidate_engine != baseline_engine | {'hicache_write_policy': 'write_through_selective'}):
            reasons.append('unexpected_engine_difference_for_write_policy')
        if candidate.get('interval_configuration') != baseline.get('interval_configuration'):
            reasons.append('unexpected_interval_difference_for_write_policy')
    elif baseline_engine != candidate_engine:
        reasons.append('missing_or_different_engine_configuration')
    if baseline.get('workload_sha256') != candidate.get('workload_sha256') or not baseline.get('workload_sha256'):
        reasons.append('missing_or_different_workload_sha256')
    if baseline.get('lifecycle_counts') != candidate.get('lifecycle_counts'):
        reasons.append('missing_or_different_lifecycle_counts')
    for row in (baseline, candidate):
        if row.get('new_build_files'):
            reasons.append('new_build_artifacts_in_timed_trial')
        if row.get('build_artifact_observation', '').startswith('not collected'):
            warnings.append('compiler_stability_not_verified')
        if row.get('model_calls') != len(row.get('calls', [])):
            reasons.append('incomplete_call_records')
        if any(c.get('queue_s') is None for c in row.get('calls', [])):
            warnings.append('missing_queue_measurements')

    def index(rows, keys):
        indexed = {tuple(r.get(k) for k in keys): r for r in rows}
        if not rows or len(indexed) != len(rows):
            reasons.append('empty_or_duplicate_measurement_ids')
        return indexed

    ca = index(baseline.get('calls', []), ('graph_id', 'node_id'))
    cb = index(candidate.get('calls', []), ('graph_id', 'node_id'))
    if ca.keys() != cb.keys():
        reasons.append('different_call_ids')
    for k in ca.keys() & cb.keys():
        if any(ca[k].get(f) != cb[k].get(f) for f in ('user_id', 'input_tokens', 'output_tokens')):
            reasons.append('different_workload_for_same_call_id')
            break
    ra = index(baseline.get('round_times', []), ('graph_id', 'round_id'))
    rb = index(candidate.get('round_times', []), ('graph_id', 'round_id'))
    if ra.keys() != rb.keys():
        reasons.append('different_round_ids')
    measured_inputs = sum(
        ca[k].get('input_sha256') is not None and cb[k].get('input_sha256') is not None
        for k in ca.keys() & cb.keys()
    )
    same_input_hash = sum(
        ca[k].get('input_sha256') is not None and ca[k].get('input_sha256') == cb[k].get('input_sha256')
        for k in ca.keys() & cb.keys()
    )
    cont_mode = (baseline.get('replay_options') or {}).get('continuity_mode')
    if cont_mode == 'forced_trace' and same_input_hash != len(ca):
        reasons.append('forced_trace_input_hashes_missing_or_different')
    elif same_input_hash < measured_inputs:
        if cont_mode == 'causal_splice':
            warnings.append('causal_splice_divergent_generated_prefix_trajectories')
        else:
            reasons.append('recorded_input_hashes_differ')
    same_hash = sum(
        ca[k].get('output_sha256') is not None and ca[k].get('output_sha256') == cb[k].get('output_sha256')
        for k in ca.keys() & cb.keys()
    )
    if cont_mode == 'forced_trace' and (same_hash != len(ca) or any(
        row.get('output_hash_basis') != 'ids' for row in (*ca.values(), *cb.values())
    )):
        reasons.append('forced_trace_output_ids_missing_or_different')
    elif same_hash != len(ca):
        warnings.append('not_token_identical_replay; DSpark_work_may_differ')
    changes = {}
    for k in (
        'wall_s', 'round_p50_s', 'round_p95_s', 'queue_p95_s',
        'client_ttft_p95_s', 'client_max_delivery_gap_s', 'native_writeback_wall_s_delta',
    ):
        av, bv = baseline.get(k), candidate.get(k)
        if type(av) in (int, float) and type(bv) in (int, float) and av > 0 and math.isfinite(av + bv):
            changes[k] = 100 * (bv / av - 1)
    paired = [rb[k]['elapsed_s'] - ra[k]['elapsed_s'] for k in ra.keys() & rb.keys()]
    return {
        'valid_paired_comparison': not reasons,
        'reasons': sorted(set(reasons)),
        'warnings': sorted(set(warnings)),
        'delta_percent': changes,
        'rounds_slower_at_least_1s': sum(x >= 1 for x in paired),
        'rounds_faster_at_least_1s': sum(x <= -1 for x in paired),
        'median_paired_round_delta_s': median(paired) if paired else None,
        'input_hash_matches': same_input_hash,
        'input_hash_compared_calls': measured_inputs,
        'output_hash_matches': same_hash,
        'paired_calls': len(ca.keys() & cb.keys()),
        'promotion': 'manual_only; neither one pair nor length-MAE establishes a speedup',
    }
