"""Tests for tau2-style internal monologue strip + Eval transcript parity.

WHY these matter:
- If strip breaks, private Thinker text leaks into the agent LLM and contaminates the
  agent-under-test (the original reason STATE-Bench skipped monologue).
- If Trajectory.to_dict / aggregate serialization strips monologue, Eval JSON loses
  tau2 parity (`simulations[].messages` keep full Thinker+Talker text).
- If the satisfaction critic ignores inner monologue when tags exist, emotion deltas
  are scored on spoken text only and diverge from the calibrated monologue prompt.
"""

from __future__ import annotations

from state_bench.schemas import EfficiencyMetrics, Trajectory
from state_bench.scripts.satisfaction_critic import extract_monologue_for_critic
from state_bench.simulator import conversation_without_monologue, strip_internal_monologue

FULL_USER = (
    "<internal_monologue>\n"
    "**Trigger Mapping**: Agent was vague.\n"
    "**Strategic Planning**: Demand specifics; short tone.\n"
    "</internal_monologue>\n"
    "I need the exact departure time, not a vague delay notice."
)


def test_strip_internal_monologue_removes_thinker_keeps_talker():
    stripped = strip_internal_monologue(FULL_USER)
    assert "<internal_monologue>" not in stripped
    assert "I need the exact departure time" in stripped
    # No-op when tags are absent (baseline runs).
    assert strip_internal_monologue("plain reply") == "plain reply"


def test_conversation_without_monologue_does_not_mutate_canonical():
    canonical = [
        {"role": "user", "content": "opening"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": FULL_USER},
    ]
    agent_view = conversation_without_monologue(canonical)
    assert "<internal_monologue>" not in agent_view[2]["content"]
    # Canonical transcript must retain full monologue for Eval save / sim history.
    assert "<internal_monologue>" in canonical[2]["content"]
    assert agent_view[2] is not canonical[2]


def test_trajectory_to_dict_preserves_monologue_in_conversation():
    # run_all_personas does sim = traj.to_dict(); monologue must survive that path.
    traj = Trajectory(
        task_id="1-example",
        user_id="user_001",
        task_summary="example",
        conversation=[
            {"role": "user", "content": "Hi, cancel my flight."},
            {"role": "assistant", "content": "I can help with that."},
            {"role": "user", "content": FULL_USER},
        ],
        efficiency=EfficiencyMetrics(turns=2, tool_calls=0, tool_errors=0, redundant_calls=0),
        metadata={"terminal_state": "incomplete"},
    )
    payload = traj.to_dict()
    aggregate = {"simulations": [payload]}
    user_turns = [
        m["content"]
        for m in aggregate["simulations"][0]["conversation"]
        if m.get("role") == "user"
    ]
    assert any("<internal_monologue>" in c for c in user_turns)
    assert "I need the exact departure time" in user_turns[-1]


def test_extract_monologue_for_critic_prefers_inner_block():
    inner = extract_monologue_for_critic(FULL_USER)
    assert "Trigger Mapping" in inner
    assert "I need the exact departure time" not in inner
    # Back-compat: no tags => full spoken content.
    assert extract_monologue_for_critic("just spoken") == "just spoken"
