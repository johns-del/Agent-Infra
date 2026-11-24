"""Optional client-delivery measurements for native /generate SSE.

TTFT includes network, queue and prefill. Chunk gaps are NOT per-token ITL,
particularly with DSpark or server-side stream coalescing. Missing observations
remain None; this module never synthesizes GPU execution times.
"""
import json
import math
import time


class StreamMeasurements:
    def __init__(self, sent_at):
        self.sent_at = sent_at
        self.first_at = self.last_at = None
        self.first_count = self.last_count = 0
        self.positive_frames = 0
        self.max_gap = None
        self.final = None
        self.ids = None

    def accept(self, frame, now):
        if not isinstance(frame, dict) or frame.get("error"):
            raise ValueError("invalid or failed /generate stream frame")
        meta = frame.get("meta_info", {})
        count = meta.get("completion_tokens")
        if type(count) is not int or count < self.last_count:
            raise ValueError("stream must report monotone cumulative completion_tokens")
        if not math.isfinite(now) or now < self.sent_at or (self.last_at is not None and now < self.last_at):
            raise ValueError("invalid stream observation clock")
        previous = self.last_count
        if count > previous:
            if self.first_at is None:
                self.first_at, self.first_count = now, count
            else:
                gap = now - self.last_at
                self.max_gap = gap if self.max_gap is None else max(self.max_gap, gap)
            self.last_at = now
            self.positive_frames += 1
        ids = frame.get("output_ids")
        if ids is not None:
            if not isinstance(ids, list):
                raise ValueError("output_ids must be a list")
            if len(ids) == count:
                self.ids = list(ids)  # native cumulative stream
            elif len(ids) == count - previous and self.ids is not None and len(self.ids) == previous:
                self.ids.extend(ids)  # explicit token-count-consistent delta stream
            else:
                raise ValueError("stream output_ids do not match cumulative or delta token counts")
        self.last_count = count
        self.final = dict(frame)
        if self.ids is not None:
            self.final["output_ids"] = list(self.ids)

    def metrics(self):
        emitted_after_first = self.last_count - self.first_count
        return {
            "client_ttft_s": None if self.first_at is None else self.first_at - self.sent_at,
            "client_post_first_chunk_s_per_token": (
                (self.last_at - self.first_at) / emitted_after_first
                if self.positive_frames > 1 and emitted_after_first > 0 else None),
            "client_max_delivery_gap_s": self.max_gap,
            "client_positive_stream_frames": self.positive_frames,
            "client_first_chunk_tokens": self.first_count,
        }


async def generate_with_measurements(client, body, sent_at, *, clock=time.monotonic):
    measurements = StreamMeasurements(sent_at)
    data_lines, completed = [], False

    def accept_event():
        nonlocal completed
        if not data_lines:
            return
        data = "\n".join(data_lines)
        data_lines.clear()
        if data.strip() == "[DONE]":
            completed = True
        else:
            measurements.accept(json.loads(data), clock())

    async with client.stream("POST", "/generate", json={**body, "stream": True}) as response:
        response.raise_for_status()
        async for raw_line in response.aiter_lines():
            line = raw_line.rstrip("\r\n")
            if line == "":
                accept_event()
                if completed:
                    break
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            # SSE comments / event / id / retry fields are not JSON payloads.
        if not completed:
            accept_event()
    result = measurements.final
    if result is None:
        raise ValueError("empty generation stream")
    if not completed and result.get("meta_info", {}).get("finish_reason") is None:
        raise ValueError("generation stream closed without a completion marker")
    return result, measurements.metrics()
