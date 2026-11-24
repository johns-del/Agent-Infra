"""Instrument exact native early exits without replacing any decision.

This patch is intentionally tied to the supplied Python scheduler shape. Unknown
or partially patched sources fail closed during runtime preparation, not at run
 time. Counters are observations of branch occurrences, NOT time fractions.
"""
import ast

MARKER = "# ORVIX_GATE_AUDIT_V4"


def instrument_scheduler(source):
    if MARKER in source:
        if source.count(MARKER) != 5:
            raise ValueError("partial or foreign v4 admission-gate instrumentation")
        # Check every expected instrumented anchor, not just the marker count.
        for old, new in _edits():
            if source.count(new) != 1:
                raise ValueError("unqualified existing v4 gate patch")
        return source
    patched = source
    for old, new in _edits():
        if patched.count(old) != 1:
            raise ValueError("native admission gate anchor changed; inspect scheduler version")
        patched = patched.replace(old, new)
    ast.parse(patched)
    return patched


def _edits():
    old = """        if (
            running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None, running_batch
"""
    new = old.replace("            return None, running_batch", """            # ORVIX_GATE_AUDIT_V4
            audit = getattr(self, "_orvix_admission_feedback", None)
            if audit is not None:
                audit.record_gate("empty_waiting" if len(self.waiting_queue) == 0
                                  else "native_batch_full_latch", running_batch)
            return None, running_batch""")
    yield old, new
    old = """                num_allocatable_reqs=self.get_num_allocatable_reqs(
                    running_bs, running_batch=running_batch
                ),
            )
        ):
            return None, running_batch
"""
    new = old.replace("            return None, running_batch", """            # ORVIX_GATE_AUDIT_V4
            audit = getattr(self, "_orvix_admission_feedback", None)
            if audit is not None:
                audit.record_gate("min_free_slots_delayer", running_batch)
            return None, running_batch""")
    yield old, new
    old = """            running_batch.batch_is_full = True
            return None, running_batch

        # Get priority queue
"""
    new = old.replace("            return None, running_batch", """            # ORVIX_GATE_AUDIT_V4
            audit = getattr(self, "_orvix_admission_feedback", None)
            if audit is not None:
                audit.record_gate("no_allocatable_request_slot", running_batch,
                                  allocatable_slots=num_allocatable_reqs)
            return None, running_batch""")
    yield old, new
    old = """            if adder.budget_state() == AddReqResult.OTHER:
                break
"""
    new = old.replace("                break", """                # ORVIX_GATE_AUDIT_V4
                audit = getattr(self, "_orvix_admission_feedback", None)
                if audit is not None:
                    audit.record_gate("pass_budget_other", running_batch,
                                      selected_in_pass=len(adder.can_run_list),
                                      remaining_input_budget=getattr(adder, "rem_input_tokens", None),
                                      remaining_chunk_budget=getattr(adder, "rem_chunk_tokens", None))
                break""")
    yield old, new
    old = """            if running_batch.batch_is_full:
                if not self.enable_priority_preemption or not adder.preempt_to_schedule(
                    req
                ):
                    break
"""
    new = old.replace("                    break", """                    # ORVIX_GATE_AUDIT_V4
                    audit = getattr(self, "_orvix_admission_feedback", None)
                    if audit is not None:
                        audit.record_gate("batch_slot_limit", running_batch,
                                          selected_in_pass=len(adder.can_run_list),
                                      remaining_input_budget=getattr(adder, "rem_input_tokens", None),
                                      remaining_chunk_budget=getattr(adder, "rem_chunk_tokens", None))
                    break""")
    yield old, new
