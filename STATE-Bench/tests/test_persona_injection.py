"""Tests for the unofficial persona-injection layer.

These encode WHY the behavior matters, not just what it does:
- The wrapper must PRESERVE task facts (the scored ground truth) while adding the persona
  layer; if a future change drops facts, the persona could no longer be held to the budget
  and the comparison to the locked baseline would be invalid.
- A non-success terminal (abandon/transfer) MUST carry an attributable trigger; the whole
  point of the design is that "abandoned" counts are defensible (R4 Table 1). A test that
  let abandonment be logged without a cause would not catch the failure that matters.
"""

from __future__ import annotations

from state_bench.scripts.persona_injection import (
    classify_terminal,
    persona_id_from_yaml,
    wrap_build_simulator_prompt,
)

PERSONA_YAML = """\
persona_profile:
  id: knowledge_toolbox_seeker
  demographics: null
  interaction_policy:
    escalation_trigger: I demand a human specialist when the agent replies with vague info.
  termination_abandonment: The user disengages after repeated vague replies.
  termination_success: The user receives a concise, well-referenced answer.
"""


def _fake_original_builder(task, env_data, user_id):  # noqa: ANN001 - test stub
    return "## User Identity\n- Budget: $900 — you will NOT accept anything above this amount\n- Meal: vegetarian"


def test_wrapper_preserves_task_facts_and_injects_persona():
    wrapped = wrap_build_simulator_prompt(_fake_original_builder, PERSONA_YAML)
    prompt = wrapped(None, None, "u1")

    # Facts bind: the original factual lines survive verbatim.
    assert "Budget: $900 — you will NOT accept anything above this amount" in prompt
    assert "Meal: vegetarian" in prompt
    # Persona spec is injected verbatim inside the tag.
    assert "<PERSONA_BEHAVIORAL_SPEC>" in prompt and "</PERSONA_BEHAVIORAL_SPEC>" in prompt
    assert "knowledge_toolbox_seeker" in prompt
    # The override + attribution + terminal contract is present.
    assert "FACTS BIND" in prompt
    assert "[TASK_DONE]" in prompt
    assert "TERMINAL: abandoned" in prompt


def test_persona_id_extracted():
    assert persona_id_from_yaml(PERSONA_YAML, fallback="x") == "knowledge_toolbox_seeker"
    assert persona_id_from_yaml("not: yaml: profile", fallback="fallback_id") == "fallback_id"


def _conv(last_user_text: str):
    return [
        {"role": "user", "content": "opening"},
        {"role": "assistant", "content": "agent reply"},
        {"role": "user", "content": last_user_text},
    ]


def test_classify_success_tag():
    out = classify_terminal(_conv("Thanks, that works. [TASK_DONE] [TERMINAL: success]"))
    assert out["terminal_state"] == "success"


def test_classify_abandoned_requires_cause():
    out = classify_terminal(
        _conv(
            'forget it. [TASK_DONE] [TERMINAL: abandoned | trigger="termination_abandonment" '
            '| agent_behavior="gave repeated vague answers"]'
        )
    )
    assert out["terminal_state"] == "abandoned"
    # The defensibility guarantee: abandonment carries its persona-policy cause.
    assert out["terminal_trigger"] == "termination_abandonment"
    assert out["terminal_agent_behavior"] == "gave repeated vague answers"


def test_classify_transfer_tag():
    out = classify_terminal(
        _conv('get me a human. [TASK_DONE] [TERMINAL: transfer | trigger="escalation_trigger" | agent_behavior="vague"]')
    )
    assert out["terminal_state"] == "transfer"
    assert out["terminal_trigger"] == "escalation_trigger"


def test_classify_bare_task_done_is_success():
    out = classify_terminal(_conv("ok done. [TASK_DONE]"))
    assert out["terminal_state"] == "success"
    assert out["terminal_trigger"] is None


def test_classify_no_terminal_is_incomplete():
    # Last user message lacks [TASK_DONE] => the run exhausted its turns.
    out = classify_terminal(_conv("still waiting for the refund details"))
    assert out["terminal_state"] == "incomplete"
