"""Fail-closed, reviewable hooks for the supplied SGLang scheduler source.

No dynamic exec of rewritten production methods. Patch only a NEW private tree.
Every edit preserves the surrounding native allocation and checkpoint code.
"""
import ast

MARKER = "# ORVIX_FAIR_CHUNK_V5"


def edits():
    old = "        self.init_schedule_policy()\n"
    new = old + '''        # ORVIX_FAIR_CHUNK_V5
        self._orvix_chunk_patch_level = 5
        self.orvix_encoder_replay_enabled = get_exec().features.enable_encoder_swa_bounded_replay
'''
    yield old, new
    old = '''        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            adder.chunked_req_limit = self.policy.shortest_prefill_chunk_limit(
                self.chunked_req,
                self.waiting_queue,
                adder.rem_chunk_tokens or 0,
                self.page_size,
            )
            self.chunked_req = adder.add_chunked_req(self.chunked_req)
'''
    new = '''        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            adder.chunked_req_limit = self.policy.shortest_prefill_chunk_limit(
                self.chunked_req,
                self.waiting_queue,
                adder.rem_chunk_tokens or 0,
                self.page_size,
            )
        # ORVIX_FAIR_CHUNK_V5
        chunk_control = getattr(self, "_orvix_chunk_control", None)
        if chunk_control is not None:
            chunk_control.prepare_pass(adder, running_batch, self.chunked_req)
        if self.chunked_req is not None:
            self.chunked_req = adder.add_chunked_req(self.chunked_req)
'''
    yield old, new
    old = '''            if self.enable_lora and not self.can_schedule_lora_req(req, running_loras):
                continue
'''
    new = '''            # ORVIX_FAIR_CHUNK_V5
            if chunk_control is not None and not chunk_control.before_candidate(adder, req):
                continue
''' + old
    yield old, new
    old = '''            and self.min_free_slots_delayer.should_delay(
                running_bs=running_bs,
'''
    new = '''            # ORVIX_FAIR_CHUNK_V5
            and getattr(self, "_orvix_min_slots_should_delay",
                        self.min_free_slots_delayer.should_delay)(
                running_bs=running_bs,
'''
    yield old, new
    old = '''        self._on_step_launch(batch)

        if self.forward_sleep_time is not None:
'''
    new = '''        self._on_step_launch(batch)
        # ORVIX_FAIR_CHUNK_V5
        chunk_control = getattr(self, "_orvix_chunk_control", None)
        if chunk_control is not None:
            chunk_control.on_launch(batch)

        if self.forward_sleep_time is not None:
'''
    yield old, new
    old = '''        self._on_step_processed(batch)

        # Submit this batch's queued host backups before the next scheduler step.
'''
    new = '''        self._on_step_processed(batch)
        # ORVIX_FAIR_CHUNK_V5
        chunk_control = getattr(self, "_orvix_chunk_control", None)
        if chunk_control is not None:
            chunk_control.on_result(batch, result)

        # Submit this batch's queued host backups before the next scheduler step.
'''
    yield old, new


def instrument_chunk_scheduler(source: str) -> str:
    pairs = list(edits())
    if MARKER in source:
        if source.count(MARKER) != len(pairs) or any(source.count(new) != 1 for _, new in pairs):
            raise ValueError("partial or foreign v5 chunk hooks; inspect native source")
        return source
    out = source
    for old, new in pairs:
        if out.count(old) != 1:
            raise ValueError("v5 scheduler anchor changed: " + old.splitlines()[0])
        out = out.replace(old, new)
    ast.parse(out)
    return out


def canonical_admission_ast(node):
    # Python 3.12 adds the empty FunctionDef.type_params field. Normalize that
    # field so identical non-generic methods qualify under Python 3.11–3.13.
    if isinstance(node, ast.AST):
        return [type(node).__name__, {key: canonical_admission_ast(value)
                for key, value in ast.iter_fields(node)
                if not (key == "type_params" and not value)}]
    if isinstance(node, list):
        return [canonical_admission_ast(item) for item in node]
    return node


def admission_digest(node):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(canonical_admission_ast(node), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


NATIVE_ADMISSION_AST_SHA256 = {'budget_state': '3ecdd031a40268a2f8a295980c9f9bbbfd514862b0e68ef7494ed11b08d249d6', '_update_prefill_budget': '96fb35767506386b853f35a06f6aacee9edcf0e53207cd5907c158101e1ae8cb', 'add_chunked_req': '35b7505e072ef9ffcae27f76157d331cf6d8d38e12cc236d2d50eac288311ab3', 'add_one_req': '6f9080c495d4d07bb30715797f909e8fefb7cf3a5824f560fe7058c1d97c35ac', '_select_prefill_admission': 'c62a0469f1a5c761db3033f7f2aceea634112b399d92504f53bbb2ce55f0169b', '_commit_prefill_admission': '61f301bb34b9644c1db22581b90f3119f9d63e49f284341bf9222d93b5d73151'}

def qualify_admission_source(source: str):
    classes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "PrefillAdder"]
    if len(classes) != 1:
        raise ValueError("qualified PrefillAdder source required")
    methods = {n.name: n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    for name, expected in NATIVE_ADMISSION_AST_SHA256.items():
        node = methods.get(name)
        actual = admission_digest(node) if node else None
        if actual != expected:
            raise ValueError("native admission source changed: " + name)
    return dict(NATIVE_ADMISSION_AST_SHA256)
