"""Compatibility replay entrypoint for maintenance scripts importing `recorded_v7_keh_replay`."""
from compare_dev_replay import (
    ENDPOINT,
    KEY,
    MODEL,
    cache_flush_acknowledged,
    engine_configuration,
    engine_state,
    nearest_models,
    numeric_counter_delta,
    overlap_metrics,
    p95,
    payload,
    post_control as _post_control,
    read_inputs,
    real_tool,
    replay_graph,
    scale_workload_copies,
    trial as _trial,
    verify_configuration,
)


async def post_control(client, enabled, params, knobs=None, *, events=None, cache=None, prefill=None,
                       chunk=None, multi=None, interval=None, reset=False):
    return await _post_control(
        client, enabled, params, knobs, events=events, cache=cache, prefill=prefill,
        chunk=chunk, multi=multi, interval=interval or {"mode": "off"}, reset=reset,
    )


async def trial(client, enabled, params, graphs, prompts, concurrency, knobs=None, *,
                recorded_gap_scale=0.0, tool_scale=1.0, lifecycle="local", cache=None, prefill=None,
                stream_metrics=False, chunk=None, multi=None, interval=None,
                continuity_mode="recorded"):
    return await _trial(
        client, enabled, params, graphs, prompts, concurrency, knobs,
        recorded_gap_scale=recorded_gap_scale, tool_scale=tool_scale, lifecycle=lifecycle,
        cache=cache, prefill=prefill, stream_metrics=stream_metrics, chunk=chunk, multi=multi,
        interval=interval or {"mode": "off"}, continuity_mode=continuity_mode,
    )
