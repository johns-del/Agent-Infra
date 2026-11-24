"""Schema and consistency checks for Orvix workload graphs."""

from __future__ import annotations

import json
import re
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


_SCHEMA_PATH = Path(__file__).with_name("schema-v1.json")
_ORIGINS = {"measured", "reconstructed", "imputed"}
_POINTER_ESCAPE = re.compile(r"~(?:[^01]|$)")
_SAFE_PATH_KEYS = {
    "schema_version",
    "graph_id",
    "user_id",
    "session_id",
    "source",
    "split",
    "block_size",
    "time_origin",
    "rounds",
    "nodes",
    "links",
    "quality",
    "provenance",
    "id",
    "submit_s",
    "end_s",
    "terminal",
    "entry_nodes",
    "exit_nodes",
    "source_message_ids",
    "kind",
    "round_id",
    "scope",
    "deps",
    "purpose",
    "role",
    "start_s",
    "duration_s",
    "attrs",
    "evidence",
    "input_tokens",
    "visible_output_tokens",
    "reasoning_tokens",
    "output_tokens",
    "input_blocks",
    "output_blocks",
    "reuse_spans",
    "cached_tokens",
    "observations",
    "model_id",
    "tool_type",
    "censored",
    "hash_id",
    "content_id",
    "length",
    "offset",
    "parent_scope",
    "child_scope",
    "dispatch_node",
    "join_node",
    "validated",
    "replay_ready",
    "blockers",
    "calibration_ids",
    "origin",
    "method",
    "calibration",
    "conversation_alignment",
    "main_eligible",
    "human_reviewed",
    "status",
}


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator.check_schema(schema)
    return schema


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    return Draft202012Validator(_schema())


@lru_cache(maxsize=1)
def _fast_schema_validator():
    try:
        from jsonschema_rs import Draft202012Validator as RustValidator
    except ImportError:
        return None
    return RustValidator(_schema())


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    """Decode an RFC 6901 JSON Pointer; the empty string points at the root."""
    if not isinstance(pointer, str):
        raise ValueError("JSON Pointer must be a string")
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must be empty or begin with slash")

    result: list[str] = []
    for raw_token in pointer[1:].split("/"):
        if _POINTER_ESCAPE.search(raw_token):
            raise ValueError("JSON Pointer contains an invalid escape")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        result.append(token)
    return tuple(result)


def _safe_path(tokens: Any) -> str:
    parts: list[str] = []
    for token in tokens:
        if isinstance(token, int):
            parts.append(str(token))
        elif isinstance(token, str) and token in _SAFE_PATH_KEYS:
            parts.append(token.replace("~", "~0").replace("/", "~1"))
        else:
            parts.append("*")
    return "" if not parts else "/" + "/".join(parts)


def _resolve_pointer(value: Any, tokens: tuple[str, ...]) -> tuple[bool, Any]:
    current = value
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                return False, None
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                return False, None
            index = int(token)
            if index >= len(current):
                return False, None
            current = current[index]
        else:
            return False, None
    return True, current


def _data_leaves(value: Any, path: tuple[str, ...] = ()):
    """Yield data leaves while omitting the provenance map itself."""
    if isinstance(value, dict) and value:
        for key, child in value.items():
            if not path and key == "provenance":
                continue
            yield from _data_leaves(child, path + (str(key),))
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            yield from _data_leaves(child, path + (str(index),))
    else:
        yield path, value


def _effective_annotation(
    path: tuple[str, ...], annotations: list[tuple[tuple[str, ...], dict[str, Any]]]
) -> dict[str, Any] | None:
    matching = [
        (pointer, entry)
        for pointer, entry in annotations
        if len(pointer) <= len(path) and path[: len(pointer)] == pointer
    ]
    if not matching:
        return None
    return max(matching, key=lambda item: len(item[0]))[1]


def _index_by_id(items: list[Any], label: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str):
            continue
        if item_id in result:
            errors.append(f"duplicate {label} ids")
        else:
            result[item_id] = item
    return result


def _check_block_coverage(
    blocks: Any,
    token_count: Any,
    node_index: int,
    side: str,
    errors: list[str],
) -> None:
    if not isinstance(blocks, list):
        return
    expected_offset = 0
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        offset = block.get("offset")
        length = block.get("length")
        if isinstance(offset, int) and not isinstance(offset, bool) and offset != expected_offset:
            errors.append(f"node[{node_index}] {side} block offsets are not continuous")
        if isinstance(length, int) and not isinstance(length, bool):
            if length < 1 or length > 64:
                errors.append(f"node[{node_index}] {side} block length is outside [1, 64]")
            if length < 64 and index != len(blocks) - 1:
                errors.append(f"node[{node_index}] {side} has a partial non-final block")
            expected_offset += length
    if (
        token_count is not None
        and isinstance(token_count, int)
        and not isinstance(token_count, bool)
        and expected_offset != token_count
    ):
        errors.append(f"node[{node_index}] {side} block lengths do not match token count")


def _check_provenance(graph: dict[str, Any], errors: list[str], ready: bool) -> None:
    raw_provenance = graph.get("provenance")
    if not isinstance(raw_provenance, dict):
        raw_provenance = {}

    annotations: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    for pointer, entry in raw_provenance.items():
        try:
            tokens = _pointer_tokens(pointer)
        except ValueError:
            errors.append("provenance contains an invalid JSON Pointer")
            continue
        if tokens and tokens[0] == "provenance":
            errors.append("provenance pointer cannot target provenance")
            continue
        exists, _ = _resolve_pointer(graph, tokens)
        if not exists:
            errors.append("provenance pointer has no corresponding graph value")
            continue
        if not isinstance(entry, dict):
            continue
        annotations.append((tokens, entry))

    # An inherited annotation proves coverage for its entire subtree unless
    # a more-specific annotation overrides it. Avoid visiting millions of hash
    # block leaves just to rediscover the same parent annotation.
    annotation_by_path = dict(annotations)
    overridden = {path[:i] for path in annotation_by_path for i in range(len(path))}

    def regions(value, path=(), inherited=None):
        annotation = annotation_by_path.get(path, inherited)
        if annotation is not None and path not in overridden:
            yield path, value, annotation
        elif isinstance(value, dict) and value:
            for key, child in value.items():
                if not path and key == 'provenance':
                    continue
                yield from regions(child, path + (str(key),), annotation)
        elif isinstance(value, list) and value:
            for i, child in enumerate(value):
                yield from regions(child, path + (str(i),), annotation)
        else:
            yield path, value, annotation

    def has_nonnull_leaf(value):
        if isinstance(value, dict) and value:
            return any(has_nonnull_leaf(v) for v in value.values())
        if isinstance(value, list) and value:
            return any(has_nonnull_leaf(v) for v in value)
        return value is not None

    for path, value, annotation in regions(graph):
        if annotation is None:
            errors.append(f"missing provenance for {_safe_path(path) or '<root>'}")
            continue
        evidence = annotation.get("evidence")
        if not isinstance(evidence, list) or not evidence or any(
            not isinstance(item, str) or not item for item in evidence
        ):
            errors.append(f"empty or invalid provenance evidence for {_safe_path(path) or '<root>'}")
        if ready and annotation.get("origin") == "imputed" and has_nonnull_leaf(value):
            calibration = annotation.get("calibration")
            if not isinstance(calibration, str) or not calibration:
                errors.append(f"imputed value lacks calibration for {_safe_path(path) or '<root>'}")


def _check_graph_consistency(graph: dict[str, Any], errors: list[str], ready: bool) -> None:
    nodes = graph.get("nodes")
    rounds = graph.get("rounds")
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(rounds, list):
        rounds = []

    node_by_id = _index_by_id(nodes, "node", errors)
    round_by_id = _index_by_id(rounds, "round", errors)
    for round_index, round_ in enumerate(rounds):
        if not isinstance(round_, dict):
            continue
        submit = round_.get("submit_s")
        end = round_.get("end_s")
        if isinstance(submit, (int, float)) and not isinstance(submit, bool) and isinstance(end, (int, float)) and not isinstance(end, bool) and end < submit:
            errors.append(f"round[{round_index}] ends before it is submitted")
        round_id = round_.get("id")
        for field in ("entry_nodes", "exit_nodes"):
            references = round_.get(field)
            if not isinstance(references, list):
                continue
            for node_id in references:
                target = node_by_id.get(node_id) if isinstance(node_id, str) else None
                if target is None:
                    errors.append(f"round[{round_index}] {field} references a missing node")
                elif target.get("round_id") != round_id:
                    errors.append(f"round[{round_index}] {field} references a node in another round")

    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        round_id = node.get("round_id")
        if round_id is not None and (not isinstance(round_id, str) or round_id not in round_by_id):
            errors.append(f"node[{node_index}] references a missing round")

        start = node.get("start_s")
        end = node.get("end_s")
        duration = node.get("duration_s")
        numeric_start = isinstance(start, (int, float)) and not isinstance(start, bool)
        numeric_end = isinstance(end, (int, float)) and not isinstance(end, bool)
        numeric_duration = isinstance(duration, (int, float)) and not isinstance(duration, bool)
        if numeric_start and numeric_end and end < start:
            errors.append(f"node[{node_index}] ends before it starts")
        if numeric_duration and duration < 0:
            errors.append(f"node[{node_index}] has a negative duration")
        if numeric_start and numeric_end and numeric_duration and abs(duration - (end - start)) > 1e-5:
            errors.append(f"node[{node_index}] duration does not match its timestamps")

    references_complete = True
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            continue
        raw_deps = node.get("deps")
        deps = raw_deps if isinstance(raw_deps, list) else []
        for dep_id in deps:
            if not isinstance(dep_id, str) or dep_id not in node_by_id:
                errors.append(f"node[{node_index}] dependency refers to a missing node")
                references_complete = False
                continue
            predecessor = node_by_id[dep_id]
            predecessor_end = predecessor.get("end_s")
            node_start = node.get("start_s")
            if (
                isinstance(predecessor_end, (int, float))
                and not isinstance(predecessor_end, bool)
                and isinstance(node_start, (int, float))
                and not isinstance(node_start, bool)
                and predecessor_end > node_start + 1e-6
            ):
                errors.append(f"node[{node_index}] starts before a dependency ends")

    if len(node_by_id) == len(nodes) and references_complete:
        try:
            _topological_ids(nodes, node_by_id)
        except ValueError as exc:
            if "cycle" in str(exc):
                errors.append("node dependency graph contains a cycle")

    for link_index, link in enumerate(graph.get("links", []) if isinstance(graph.get("links"), list) else []):
        if not isinstance(link, dict):
            continue
        for field in ("dispatch_node", "join_node"):
            node_id = link.get(field)
            if node_id is not None and (not isinstance(node_id, str) or node_id not in node_by_id):
                errors.append(f"link[{link_index}] {field} references a missing node")

    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict) or node.get("kind") != "model":
            continue
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            continue
        input_tokens = attrs.get("input_tokens")
        visible = attrs.get("visible_output_tokens")
        reasoning = attrs.get("reasoning_tokens")
        output = attrs.get("output_tokens")
        cached = attrs.get("cached_tokens")

        for field, value in (
            ("input_tokens", input_tokens),
            ("visible_output_tokens", visible),
            ("reasoning_tokens", reasoning),
            ("output_tokens", output),
            ("cached_tokens", cached),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                errors.append(f"node[{node_index}] {field} must be a nonnegative integer or null")

        if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and isinstance(cached, int) and not isinstance(cached, bool) and cached > input_tokens:
            errors.append(f"node[{node_index}] cached tokens exceed input tokens")
        if all(isinstance(value, int) and not isinstance(value, bool) for value in (visible, reasoning, output)):
            if output != visible + reasoning:
                errors.append(f"node[{node_index}] output token counts do not add up")

        _check_block_coverage(attrs.get("input_blocks"), input_tokens, node_index, "input", errors)
        _check_block_coverage(attrs.get("output_blocks"), output, node_index, "output", errors)

    if not ready:
        return

    quality = graph.get("quality")
    if isinstance(quality, dict) and quality.get("blockers"):
        errors.append("ready graph has quality blockers")

    for round_index, round_ in enumerate(rounds):
        if isinstance(round_, dict) and round_.get("terminal") not in {"finish", "ask"}:
            errors.append(f"ready graph round[{round_index}] has a nonterminal outcome")
        if isinstance(round_, dict) and (not round_.get('entry_nodes') or not round_.get('exit_nodes')):
            errors.append(f"ready graph round[{round_index}] lacks a terminal dependency frontier")

    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        kind = node.get("kind")
        if (kind == "tool" or kind == "think") and node.get("duration_s") is None:
            errors.append(f"ready graph node[{node_index}] lacks duration")
        if kind == "model":
            attrs = node.get("attrs")
            if not isinstance(attrs, dict):
                continue
            for field in (
                "input_tokens",
                "visible_output_tokens",
                "reasoning_tokens",
                "output_tokens",
                "cached_tokens",
            ):
                # An independently labelled historical reconstruction can be
                # replayed without knowing the old engine's physical cache hits.
                # Keep the value unknown; do not invent zero hits. R/native
                # captures retain their existing complete-count requirement.
                if (field == "cached_tokens" and graph.get("source") == "H"
                        and attrs.get("token_sequence_basis") == "retokenized_historical_text"):
                    continue
                if attrs.get(field) is None:
                    errors.append(f"ready graph node[{node_index}] lacks model token counts")
                    break
            for field, token_field in (("input_blocks", "input_tokens"), ("output_blocks", "output_tokens")):
                blocks = attrs.get(field)
                token_count = attrs.get(token_field)
                if not isinstance(blocks, list) or (
                    isinstance(token_count, int) and not isinstance(token_count, bool) and token_count > 0 and not blocks
                ):
                    errors.append(f"ready graph node[{node_index}] lacks {field}")


def _topological_ids(
    nodes: list[Any], node_by_id: dict[str, dict[str, Any]] | None = None
) -> list[str]:
    if node_by_id is None:
        node_by_id = {}
        for node in nodes:
            if not isinstance(node, dict) or not isinstance(node.get("id"), str):
                raise ValueError("nodes require string ids")
            node_id = node["id"]
            if node_id in node_by_id:
                raise ValueError("node ids must be unique")
            node_by_id[node_id] = node
    if len(node_by_id) != len(nodes):
        raise ValueError("node ids must be unique and present")

    original_order = {node["id"]: index for index, node in enumerate(nodes)}
    indegree = {node_id: 0 for node_id in node_by_id}
    children: dict[str, list[str]] = {node_id: [] for node_id in node_by_id}
    for node_id, node in node_by_id.items():
        raw_deps = node.get("deps")
        if not isinstance(raw_deps, list):
            raise ValueError("node dependencies must be lists")
        if any(not isinstance(dep_id, str) for dep_id in raw_deps):
            raise ValueError("node dependencies must contain string ids")
        for dep_id in set(raw_deps):
            if dep_id not in node_by_id:
                raise ValueError("node dependency refers to a missing node")
            indegree[node_id] += 1
            children[dep_id].append(node_id)

    for child_list in children.values():
        child_list.sort(key=original_order.__getitem__)
    queue = deque(node_id for node_id in node_by_id if indegree[node_id] == 0)
    result: list[str] = []
    while queue:
        node_id = queue.popleft()
        result.append(node_id)
        for child_id in children[node_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)
    if len(result) != len(node_by_id):
        raise ValueError("node dependency graph contains a cycle")
    return result


def topological_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return graph nodes in stable dependency order, preserving input order on ties."""
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError("graph must contain a node list")
    nodes = graph["nodes"]
    order = _topological_ids(nodes)
    by_id = {node["id"]: node for node in nodes}
    return [by_id[node_id] for node_id in order]


def validate_graph(graph: Any, ready_only: bool = False) -> list[str]:
    """Return safe, non-content-bearing schema and graph consistency errors."""
    errors: list[str] = []
    fast = _fast_schema_validator()
    schema_errors = sorted(
        [] if fast is not None and fast.is_valid(graph) else _schema_validator().iter_errors(graph),
        key=lambda error: (_safe_path(error.absolute_path), str(error.validator)),
    )
    for error in schema_errors:
        errors.append(
            f"schema violation ({error.validator}) at {_safe_path(error.absolute_path) or '<root>'}"
        )

    if not isinstance(graph, dict):
        return errors

    quality = graph.get("quality")
    declared_ready = isinstance(quality, dict) and quality.get("replay_ready") is True
    if ready_only and not declared_ready:
        errors.append('graph is not admitted for replay')
    ready = bool(ready_only or declared_ready)
    _check_graph_consistency(graph, errors, ready)
    _check_provenance(graph, errors, ready)
    return errors


def annotate(
    graph: dict[str, Any],
    pointer: str,
    origin: str,
    evidence: list[str],
    method: str | None = None,
    calibration: str | None = None,
) -> dict[str, Any]:
    """Set provenance for a JSON Pointer; return the same graph object."""
    if not isinstance(graph, dict):
        raise TypeError("graph must be an object")
    tokens = _pointer_tokens(pointer)
    if tokens and tokens[0] == "provenance":
        raise ValueError("provenance cannot annotate itself")
    if not isinstance(origin, str) or origin not in _ORIGINS:
        raise ValueError("origin must be measured, reconstructed, or imputed")
    if not isinstance(evidence, list) or not evidence or any(
        not isinstance(item, str) or not item for item in evidence
    ):
        raise ValueError("evidence must be a nonempty list of nonempty strings")
    if method is not None and (not isinstance(method, str) or not method):
        raise ValueError("method must be a nonempty string or null")
    if calibration is not None and (not isinstance(calibration, str) or not calibration):
        raise ValueError("calibration must be a nonempty string or null")

    provenance = graph.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise TypeError("graph provenance must be an object")
    provenance[pointer] = {
        "origin": origin,
        "evidence": list(evidence),
        "method": method,
        "calibration": calibration,
    }
    return graph
