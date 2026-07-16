"""Unit tests for MCTS Phase-1 helpers (no API calls)."""

from __future__ import annotations

import unittest
from collections import Counter
from random import Random

from eval.mcts.assignment import AssignmentRow, hamilton_quotas, snake_draft, validate_assignment
from eval.mcts.ids import persona_id_from_row
from eval.mcts.schema import build_core_record, make_run_id, validate_core_record
from eval.mcts.schwartz import (
    choose_strata_mode,
    dominant_value,
    higher_order_stratum,
    parse_schwartz_vector,
)
from eval.mcts.terminal import map_statebench_terminal, map_tau2_terminal


class TestIds(unittest.TestCase):
    def test_user_id_preferred(self):
        row = {"user_id": "bitparity", "persona": {"id": "other"}}
        self.assertEqual(persona_id_from_row(row), "bitparity")

    def test_nemotron_style(self):
        row = {
            "person": {"uuid": "12aa0864-bd69-4d03-8172-39ed7885507f"},
            "persona": {"id": "12aa0864-bd69-4d03-8172-39ed7885507f"},
        }
        self.assertEqual(persona_id_from_row(row), "12aa0864-bd69-4d03-8172-39ed7885507f")


class TestSchwartz(unittest.TestCase):
    def test_fallback_when_dominated(self):
        counts = Counter({"SELF_DIRECTION": 123, "ACHIEVEMENT": 77})
        s, mode = choose_strata_mode(counts, 200)
        self.assertEqual(s, 4)
        self.assertIn("S4", mode)

    def test_s10_when_balanced(self):
        counts = Counter({f"V{i}": 20 for i in range(10)})
        s, mode = choose_strata_mode(counts, 200)
        self.assertEqual(s, 10)

    def test_tie_break_lex(self):
        vec = {"ACHIEVEMENT": 0.5, "BENEVOLENCE": 0.5, "POWER": 0.1}
        self.assertEqual(dominant_value(vec), "ACHIEVEMENT")

    def test_higher_order(self):
        vec = parse_schwartz_vector(
            {
                "SELF_DIRECTION": 1.0,
                "STIMULATION": 1.0,
                "HEDONISM": 1.0,
                "POWER": 0.0,
                "ACHIEVEMENT": 0.0,
                "UNIVERSALISM": 0.0,
                "BENEVOLENCE": 0.0,
                "TRADITION": 0.0,
                "CONFORMITY": 0.0,
                "SECURITY": 0.0,
            }
        )
        self.assertEqual(higher_order_stratum(vec), "OPENNESS_TO_CHANGE")


class TestTerminal(unittest.TestCase):
    def test_tau2_success(self):
        state, ok, xfer = map_tau2_terminal(reward=1.0, termination_reason="user_stop")
        self.assertEqual(state, "success")
        self.assertTrue(ok)
        self.assertFalse(xfer)

    def test_tau2_max_turns(self):
        state, ok, xfer = map_tau2_terminal(reward=0.0, termination_reason="max_steps")
        self.assertEqual(state, "max_turns")

    def test_tau2_transfer_tool(self):
        msgs = [{"role": "assistant", "tool_calls": [{"name": "transfer_to_human_agents"}]}]
        state, ok, xfer = map_tau2_terminal(reward=1.0, termination_reason="user_stop", messages=msgs)
        self.assertEqual(state, "transfer")
        self.assertTrue(xfer)
        self.assertFalse(ok)

    def test_tau2_sim_error(self):
        state, _, _ = map_tau2_terminal(reward=None, termination_reason="agent_error")
        self.assertEqual(state, "sim_error")

    def test_statebench_map(self):
        state, ok, xfer = map_statebench_terminal(raw_terminal="success", state_requirements_met=True)
        self.assertEqual(state, "success")
        self.assertTrue(ok)
        state, ok, xfer = map_statebench_terminal(raw_terminal="incomplete")
        self.assertEqual(state, "max_turns")
        state, ok, xfer = map_statebench_terminal(raw_terminal="transfer")
        self.assertEqual(state, "transfer")
        self.assertTrue(xfer)


class TestAssignmentQuotas(unittest.TestCase):
    def test_hamilton_and_draft(self):
        # Synthetic: 4 strata, T=4, K=4, P=16
        sizes = {"A": 7, "B": 5, "C": 3, "D": 1}
        tasks = [f"t{i}" for i in range(4)]
        quotas = hamilton_quotas(sizes, tasks, k=4)
        for tid in tasks:
            self.assertEqual(sum(quotas[tid].values()), 4)
        for lab, sz in sizes.items():
            self.assertEqual(sum(quotas[tid][lab] for tid in tasks), sz)

        groups = {lab: [f"{lab}_{i}" for i in range(sz)] for lab, sz in sizes.items()}
        drafted = snake_draft(groups, tasks, 4, Random(0))
        self.assertEqual(len(drafted), 16)
        rows = [
            AssignmentRow("d", tid, slot, pid, 0, lab, 1)
            for tid, slot, pid, lab in drafted
        ]
        errs = validate_assignment(rows, k=4)
        self.assertEqual(errs, [])


class TestSchema(unittest.TestCase):
    def test_roundtrip(self):
        rec = build_core_record(
            bench="tau2",
            config_hash="abc",
            block=1,
            domain="retail",
            task_id="0",
            persona_id="bitparity",
            arm="fixed_prompt",
            model="gpt-oss",
            sim_model="openai/gemma",
            sim_seed=123,
            judge_model=None,
            judge_prompt_hash=None,
            terminal_state="success",
            task_success=True,
            transfer=False,
            n_turns=3,
            full_transcript=[{"role": "user", "content": "hi"}],
            persona_variant_hash=None,
        )
        self.assertEqual(validate_core_record(rec), [])
        self.assertEqual(
            rec["run_id"],
            make_run_id(
                bench="tau2",
                block=1,
                domain="retail",
                task_id="0",
                persona_id="bitparity",
                arm="fixed_prompt",
                model="gpt-oss",
                sim_seed=123,
            ),
        )


class TestSimContamination(unittest.TestCase):
    def test_hard_fail(self):
        from eval.mcts.run_mcts import assert_sim_not_in_models

        with self.assertRaises(SystemExit):
            assert_sim_not_in_models("gemma", ["gpt-oss", "gemma"])
        with self.assertRaises(SystemExit):
            assert_sim_not_in_models("openai/gemma", ["gemma"])
        assert_sim_not_in_models("gemma", ["gpt-oss"])  # no raise


if __name__ == "__main__":
    unittest.main()
