#!/usr/bin/env python3
"""Comprehensive unit & regression tests for SystemPrefixPatcher (Option 2) and fixed CausalPrefixStabilizer."""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "package/deploy/deepseek_v41_nvfp4"))

from system_prefix_patcher import (  # noqa: E402
    PATCH_CLOSE_TAG,
    PATCH_OPEN_TAG,
    SystemPrefixPatcher,
    diff_semantic_blocks,
    split_semantic_blocks,
)
from causal_prefix_stabilizer import CausalPrefixStabilizer, canonicalize_against_spine  # noqa: E402


BASE_SYSTEM = (
    "You are an expert software engineering assistant.\n\n"
    "## Core Rules\n"
    + ("Follow repository coding standards and preserve existing tests.\n" * 40)
    + "\n## Bash\n"
    + ("Always run commands safely inside the workspace.\n" * 20)
    + "\n<session>\n<user>engineer_a</user>\n<mode>standard</mode>\n</session>\n\n"
    + "<task>\n<task_id>T-100</task_id>\n<status>in_progress</status>\n<goal>Analyze cache hit rate</goal>\n</task>\n"
)


class SystemPrefixPatcherTests(unittest.TestCase):
    def test_split_semantic_blocks_exact_reversibility(self):
        blocks = split_semantic_blocks(BASE_SYSTEM)
        self.assertEqual("".join(blocks), BASE_SYSTEM)
        # Ensure XML blocks stay intact as single atomic blocks
        session_blocks = [b for b in blocks if "<session>" in b]
        task_blocks = [b for b in blocks if "<task>" in b]
        self.assertEqual(len(session_blocks), 1)
        self.assertIn("</session>", session_blocks[0])
        self.assertEqual(len(task_blocks), 1)
        self.assertIn("</task>", task_blocks[0])

    def test_passthrough_when_unchanged(self):
        p = SystemPrefixPatcher(enabled=True)
        body1 = {"messages": [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": "hi"}]}
        out1, d1 = p.apply("s1", body1)
        self.assertEqual(d1.action, "init")
        self.assertIs(out1, body1)

        body2 = {
            "messages": [
                {"role": "system", "content": BASE_SYSTEM},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "next"},
            ]
        }
        out2, d2 = p.apply("s1", body2)
        self.assertEqual(d2.action, "passthrough")
        self.assertIs(out2, body2)

    def test_per_turn_counter_mutation_no_stale_accumulation(self):
        """Simulates the real production 213-byte tail step counter that increments every turn."""
        p = SystemPrefixPatcher(enabled=True)
        msgs = [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": "Start task."}]
        out0, d0 = p.apply("s-counter", {"messages": copy.deepcopy(msgs)})
        self.assertEqual(d0.action, "init")

        for step in range(1, 11):
            msgs.append({"role": "assistant", "content": f"Step {step-1} done."})
            msgs.append({"role": "tool", "tool_call_id": f"tc_{step}", "content": f"tool_output_{step}"})
            mutated_sys = BASE_SYSTEM + f"\n[Reminder] You are currently at step {step} of 50. Keep concise.\n"
            req_msgs = copy.deepcopy(msgs)
            req_msgs[0]["content"] = mutated_sys

            out, dec = p.apply("s-counter", {"messages": req_msgs})
            self.assertEqual(dec.action, "patched_new_delta")
            # System message must remain 100% frozen to BASE_SYSTEM
            self.assertEqual(out["messages"][0]["content"], BASE_SYSTEM)
            # No new user message was inserted (role sequence strictly preserved!)
            self.assertEqual([m["role"] for m in out["messages"]], [m["role"] for m in req_msgs])
            # Earlier messages MUST NOT carry stale step counters (zero accumulation!)
            for earlier in out["messages"][1:-1]:
                self.assertNotIn(PATCH_OPEN_TAG, str(earlier.get("content") or ""))
            # Latest tail message carries ONLY the latest step counter
            tail_txt = out["messages"][-1]["content"]
            self.assertIn(PATCH_OPEN_TAG, tail_txt)
            self.assertIn(f"step {step} of 50", tail_txt)
            self.assertNotIn(f"step {step-1} of 50", tail_txt)

    def test_xml_block_replace_then_steady_state_replay(self):
        """Simulates mutating <task> once and keeping it steady for 5 subsequent turns."""
        p = SystemPrefixPatcher(enabled=True)
        msgs = [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": "Run audit."}]
        p.apply("s-steady", {"messages": copy.deepcopy(msgs)})

        updated_sys = BASE_SYSTEM.replace(
            "<task>\n<task_id>T-100</task_id>\n<status>in_progress</status>\n<goal>Analyze cache hit rate</goal>\n</task>",
            "<task>\n<task_id>T-202</task_id>\n<status>verifying</status>\n<goal>Validate patcher latency</goal>\n</task>",
        )

        # Turn 2: mutation occurs -> anchored at messages[3]
        msgs.append({"role": "assistant", "content": "Checking."})
        msgs.append({"role": "tool", "tool_call_id": "c1", "content": "result_1"})
        req2 = copy.deepcopy(msgs)
        req2[0]["content"] = updated_sys
        out2, d2 = p.apply("s-steady", {"messages": req2})
        self.assertEqual(d2.action, "patched_new_delta")
        self.assertEqual(d2.anchor_index, 3)
        self.assertEqual(out2["messages"][0]["content"], BASE_SYSTEM)
        self.assertIn("T-202", out2["messages"][3]["content"])
        self.assertIn("Validate patcher latency", out2["messages"][3]["content"])

        # Turns 3..6: same updated_sys -> steady-state replay at anchor index 3 -> 100% prefix match!
        prev_effective = out2["messages"]
        for t in range(3, 7):
            msgs.append({"role": "assistant", "content": f"Turn {t-1} ok"})
            msgs.append({"role": "tool", "tool_call_id": f"c{t}", "content": f"result_{t}"})
            req_t = copy.deepcopy(msgs)
            req_t[0]["content"] = updated_sys
            out_t, d_t = p.apply("s-steady", {"messages": req_t})
            self.assertEqual(d_t.action, "patched_steady_replay")
            self.assertEqual(d_t.anchor_index, 3)
            # Every prior message [0..len(prev)-1] is 100% byte-identical to previous turn!
            self.assertEqual(out_t["messages"][: len(prev_effective)], prev_effective)
            # And tail message does NOT duplicate the patch!
            self.assertNotIn(PATCH_OPEN_TAG, out_t["messages"][-1]["content"])
            prev_effective = out_t["messages"]

    def test_revert_to_frozen_baseline_clears_patch(self):
        p = SystemPrefixPatcher(enabled=True)
        msgs = [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": "hi"}]
        p.apply("s-rev", {"messages": copy.deepcopy(msgs)})

        # Turn 2: add temp reminder
        msgs.extend([{"role": "assistant", "content": "ok"}, {"role": "tool", "tool_call_id": "t1", "content": "r1"}])
        req2 = copy.deepcopy(msgs)
        req2[0]["content"] = BASE_SYSTEM + "\n[Temp Warning] High memory usage.\n"
        _, d2 = p.apply("s-rev", {"messages": req2})
        self.assertEqual(d2.action, "patched_new_delta")

        # Turn 3: temp warning disappears -> back to exact BASE_SYSTEM
        msgs.extend([{"role": "assistant", "content": "ok2"}, {"role": "tool", "tool_call_id": "t2", "content": "r2"}])
        req3 = copy.deepcopy(msgs)
        out3, d3 = p.apply("s-rev", {"messages": req3})
        self.assertEqual(d3.action, "reverted_to_frozen")
        self.assertEqual(out3["messages"][0]["content"], BASE_SYSTEM)
        self.assertNotIn(PATCH_OPEN_TAG, out3["messages"][-1]["content"])

    def test_removed_block_from_initial_baseline(self):
        p = SystemPrefixPatcher(enabled=True)
        msgs = [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": "hi"}]
        p.apply("s-rem", {"messages": copy.deepcopy(msgs)})

        # Remove the <task>...</task> block entirely
        stripped_sys = BASE_SYSTEM.split("<task>")[0]
        msgs.extend([{"role": "assistant", "content": "ok"}, {"role": "user", "content": "continue"}])
        req2 = copy.deepcopy(msgs)
        req2[0]["content"] = stripped_sys
        out2, d2 = p.apply("s-rem", {"messages": req2})
        self.assertEqual(d2.action, "patched_new_delta")
        self.assertEqual(out2["messages"][0]["content"], BASE_SYSTEM)
        self.assertIn("Removed system context", out2["messages"][-1]["content"])
        self.assertIn("<task> section", out2["messages"][-1]["content"])

    def test_guardrail_refreshes_on_core_prefix_rewrite(self):
        p = SystemPrefixPatcher(enabled=True, min_static_prefix_ratio=0.65)
        msgs = [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": "hi"}]
        p.apply("s-guard", {"messages": copy.deepcopy(msgs)})

        # Rewrite from the very beginning of system prompt -> violates static_prefix_ratio
        completely_new_sys = "COMPLETELY DIFFERENT PERSONA.\n\n" + BASE_SYSTEM[60:]
        msgs.extend([{"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}])
        req2 = copy.deepcopy(msgs)
        req2[0]["content"] = completely_new_sys
        out2, d2 = p.apply("s-guard", {"messages": req2})
        self.assertEqual(d2.action, "guardrail_refresh")
        self.assertEqual(out2["messages"][0]["content"], completely_new_sys)
        self.assertNotIn(PATCH_OPEN_TAG, out2["messages"][-1]["content"])

        # Subsequent turn with completely_new_sys must now passthrough cleanly
        msgs.extend([{"role": "assistant", "content": "ok2"}, {"role": "user", "content": "next2"}])
        req3 = copy.deepcopy(msgs)
        req3[0]["content"] = completely_new_sys
        _, d3 = p.apply("s-guard", {"messages": req3})
        self.assertEqual(d3.action, "passthrough")

    def test_list_content_and_assistant_prefill_handling(self):
        p = SystemPrefixPatcher(enabled=True)
        body1 = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": BASE_SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ]
        }
        p.apply("s-list", body1)

        body2 = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": BASE_SYSTEM + "\n[Note] v2\n"}]},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"type": "text", "text": "turn2"}]},
            ]
        }
        out2, d2 = p.apply("s-list", body2)
        self.assertEqual(d2.action, "patched_new_delta")
        self.assertEqual(out2["messages"][0]["content"][0]["text"], BASE_SYSTEM)
        self.assertIn(PATCH_CLOSE_TAG, out2["messages"][-1]["content"][0]["text"])

        # Assistant tail should be safely skipped
        body3 = copy.deepcopy(body2)
        body3["messages"].append({"role": "assistant", "content": "prefill..."})
        _, d3 = p.apply("s-list", body3)
        self.assertEqual(d3.action, "skip_unsupported")

    def test_causal_prefix_stabilizer_pacman_bug_fixed(self):
        """Regression test reproducing the exact 14:54 Pac-Man 1-token drift bug."""
        # Prefix A (12431), Middle M (591, NOT multiple of stride=8!), old_tok (1), Suffix C (10000)
        a = list(range(1000, 1000 + 12431))
        m = list(range(20000, 20000 + 591))
        old_tok = [999999]
        c = list(range(30000, 40000))
        t0 = a + m + old_tok + c + [1, 2, 3, 4]

        # Turn 1: insert 294 toks after A, and replace old_tok with 108 toks after M
        ins1 = list(range(500000, 500294))
        rep2 = list(range(600000, 600108))
        d1 = list(range(70000, 70500))
        t1 = a + ins1 + m + rep2 + c + d1 + [1, 2, 3, 4]

        stab = CausalPrefixStabilizer(enabled=True)
        out0, rel0 = stab.transform("g-pac", "g-pac-s0", t0)
        self.assertEqual(rel0, 0)

        out1, rel1 = stab.transform("g-pac", "g-pac-s0", t1)
        # Because old_tok was replaced, Turn 1 must NOT poison spine with a 1-token shift;
        # Turn 2..10 pure appends MUST achieve full prefix match!
        prev_out = out1
        cur_tail = d1
        for k in range(2, 12):
            cur_tail = cur_tail + list(range(80000 + k * 200, 80000 + k * 200 + 150))
            tk = a + ins1 + m + rep2 + c + cur_tail + [1, 2, 3, 4]
            out_k, rel_k = stab.transform("g-pac", "g-pac-s0", tk)
            lcp = 0
            for x, y in zip(prev_out, out_k):
                if x != y:
                    break
                lcp += 1
            self.assertGreaterEqual(lcp, len(prev_out) - 20, f"Turn {k} suffered prefix drift! lcp={lcp} vs {len(prev_out)}")
            prev_out = out_k

    def test_causal_prefix_stabilizer_pure_insertion_still_recovers(self):
        a = list(range(100, 9100))
        c0 = list(range(20000, 22000))
        t0 = a + c0 + [1001, 1002, 1003, 1004]
        b1 = [90000 + i for i in range(160)]
        d0 = list(range(30000, 31500))
        t1 = a + b1 + c0 + d0 + [1001, 1002, 1003, 1004]
        b2 = [91000 + i for i in range(160)]
        d1 = list(range(40000, 41000))
        t2 = a + b2 + c0 + d0 + d1 + [1001, 1002, 1003, 1004]

        stab = CausalPrefixStabilizer(enabled=True)
        stab.transform("g2", "g2-s0", t0)
        out1, rel1 = stab.transform("g2", "g2-s0", t1)
        out2, rel2 = stab.transform("g2", "g2-s0", t2)
        self.assertEqual(rel1, 160)
        self.assertEqual(rel2, 160)
        self.assertEqual(Counter(out1), Counter(t1))
        self.assertEqual(Counter(out2), Counter(t2))
        lcp_12 = 0
        for x, y in zip(out1, out2):
            if x != y:
                break
            lcp_12 += 1
        self.assertGreaterEqual(lcp_12, len(a) + len(c0) + len(d0) - 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
