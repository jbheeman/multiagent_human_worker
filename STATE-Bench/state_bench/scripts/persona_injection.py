"""Persona-YAML injection for the STATE-Bench user simulator (unofficial, tau2-style).

This mirrors the tau2-bench persona workflow: the user simulator is driven by a YAML
persona behavioral spec injected into its system prompt. We do this without touching
STATE-Bench core by *wrapping* a domain's ``build_simulator_prompt`` callable — the
original output (task facts: identity, budget, preferences, known/unknown info, task
rules) is preserved, and the persona behavioral layer is appended.

Design (locked decisions):
- **Facts bind.** The persona may NOT override the task's factual constraints; it governs
  tone, linguistics, patience, and whether/when to escalate or give up.
- **Attributable termination.** The persona may end via abandon/transfer ONLY when the
  agent's *conduct* trips a condition defined in its YAML (interaction_policy.
  escalation_trigger, termination_abandonment, or a state_transition_rules entry) — never
  because a task fact clashes with its expectation. Every non-success terminal must carry
  the tripped trigger + the agent behavior that tripped it, emitted in a structured tag.
- **Terminal taxonomy** = success / transfer / abandoned / incomplete (distinct states,
  not complements — answers reviewer R4's Table 1 question).
- **Internal monologue (tau2-parity).** Every reactive reply must open with a Thinker
  layer ``<internal_monologue>...</internal_monologue>`` then a Talker (visible) message.
  The full string is kept in the saved transcript; the orchestrator strips the monologue
  only on the agent LLM input path so private reasoning does not leak to the agent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

# --- Persona behavioral guidelines (persona-mode; tau2-style internal monologue) ---------

_PERSONA_GUIDELINES = """\
## Persona Behavioral Spec (governs ALL of your behavior)

You are the simulated customer. Your personality, tone, vocabulary, punctuation, patience,
and decision to continue / escalate / end are governed 100% by the
<PERSONA_BEHAVIORAL_SPEC> below. Ignore any personality or tone hints in the identity or
task-context text above — those sections give you FACTS, not a personality.

Behavioral rules:
- Stay in character. Adopt the persona's formality, sentence structure, vocabulary, and
  punctuation from its communication_style. Sound like a real human typing, not an AI.
- Never use AI customer-service platitudes ("I apologize for the inconvenience", "I
  understand how frustrating this is") unless the persona is explicitly that polite.
- Disclose information progressively: give the agent a fact only when it asks for it.
- Generate one message at a time; keep natural conversation flow.

FACTS BIND (non-negotiable):
- The identity facts above (your name, budget, preferences, what you know / don't know,
  and the task rules) are ground truth. You may dislike, complain about, negotiate, or
  grudgingly accept a fact, but you may NOT end the conversation merely because a fact
  contradicts your expectation. A fact-clash is NEVER a valid reason to abandon or transfer.

WHEN YOU MAY END (attribution rule):
- You may end via abandonment or transfer ONLY when the AGENT'S CONDUCT trips a condition
  defined in your persona spec: interaction_policy.escalation_trigger,
  termination_abandonment, or a state_transition_rules entry. The trigger must be the
  agent's behavior (vague / unverified answers, dismissing your requests, stalling,
  repeated mistakes, excessive policy friction), NOT a task fact you happen to dislike.
- If no policy condition is tripped and the task is not yet resolved, keep going.

TERMINAL MARKERS (the harness only stops on the literal token [TASK_DONE]):
- When the conversation ends, your FINAL message must contain exactly one terminal tag.
  Always include [TASK_DONE] so the harness stops.
- Success (your termination_success is met):
  [TASK_DONE] [TERMINAL: success]
- Transfer (your policy makes you demand a human / supervisor / different channel):
  [TASK_DONE] [TERMINAL: transfer | trigger="<the persona condition that fired>" | agent_behavior="<what the agent did>"]
- Abandon (your policy makes you disengage / walk away):
  [TASK_DONE] [TERMINAL: abandoned | trigger="<the persona condition that fired>" | agent_behavior="<what the agent did>"]
- Do NOT emit a terminal tag until you are actually ending. Do not end right after the
  agent merely proposes an action — wait until it confirms the action is done.
- Terminal tags belong in the Talker (visible) layer ONLY — never only inside the
  <internal_monologue> block.

## Response Generation Process

### Step 1: The Thinker Layer (<internal_monologue>)
Every time you receive a message from the agent, you must open an <internal_monologue>
block. Inside this block, you must explicitly and analytically complete the following:

- **Trigger Mapping**: Cross-reference the agent's latest message against your persona
  state_transition_rules and interaction_policy. Did the agent trigger a specific rule
  (e.g., asking you to wait, denying a request, providing ambiguous data)?

- **Cognitive Appraisal**: Evaluate the agent's message through your communication_style
  and interaction_policy tolerances (authority_challenge, escalation_trigger,
  gratification_delay_tolerance, policy_friction_tolerance, verification_patience). How
  do your traits dictate your internal reaction?

- **Termination Check**: Evaluate against termination_success, termination_abandonment,
  and escalation_trigger. Have success, abandon, or transfer conditions been met?

- **Strategic Planning**: Based on the analysis above, state your immediate conversational
  goal and the exact tone for the next visible message, aligned with the persona.

Once your internal analysis is fully formulated, close the </internal_monologue> block.

### Step 2: The Talker Layer
After closing </internal_monologue>, generate your visible chat message. This message is
the "tip of the iceberg." It must execute the strategy from your monologue while strictly
adhering to the persona and Anti-LLM Formatting rules. Do not explain your reasoning in
the visible output. Put any [TASK_DONE] / [TERMINAL: ...] tags in this visible layer.

**Execution Example:**

<internal_monologue>
**Trigger Mapping**: The agent stated my flight is delayed but did not provide an updated
departure time. This triggers my state_transition_rule: "IF the agent provides a delay
without a timeline, THEN demand the exact cause and an estimated time."

**Cognitive Appraisal**: My gratification_delay_tolerance is low and authority_challenge
is high. I view the lack of specifics as stalling. I am irritated.

**Termination Check**: Abandonment condition not met; this is turn 2.

**Strategic Planning**: Demand the exact reason and a timeline. Tone: abrupt, annoyed,
short syntax, no pleasantries.
</internal_monologue>
I don't need a generic apology, I need a time. Why exactly is it delayed and when is the
plane getting here?
"""


def load_persona_yaml(path: str | Path) -> str:
    """Read raw persona YAML text (tau2 _load_persona_yaml analog)."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Persona YAML file is empty: {path}")
    return text


def persona_id_from_yaml(persona_yaml: str, *, fallback: str) -> str:
    """Extract persona_profile.id. Uses yaml if available, else a regex, else fallback."""
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(persona_yaml)
        if isinstance(data, dict):
            profile = data.get("persona_profile")
            if isinstance(profile, dict) and profile.get("id"):
                return str(profile["id"])
    except Exception:
        pass
    match = re.search(r"^\s*id:\s*[\"']?([^\"'\n]+)", persona_yaml, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback


def wrap_build_simulator_prompt(
    original: Callable[..., str],
    persona_yaml: str,
) -> Callable[..., str]:
    """Return a build_simulator_prompt wrapper that appends the persona behavioral layer.

    The wrapper keeps the original prompt verbatim (task facts) and appends the persona
    guidelines + the verbatim YAML inside <PERSONA_BEHAVIORAL_SPEC> tags.
    """

    def _wrapped(task: Any, env_data: Any, user_id: str) -> str:
        base = original(task, env_data, user_id)
        persona_block = (
            f"{_PERSONA_GUIDELINES}\n\n"
            f"<PERSONA_BEHAVIORAL_SPEC>\n{persona_yaml.strip()}\n</PERSONA_BEHAVIORAL_SPEC>"
        )
        return f"{base}\n\n---\n\n{persona_block}"

    return _wrapped


# --- Terminal-state classification -------------------------------------------------------

_TERMINAL_RE = re.compile(
    r"\[TERMINAL:\s*(?P<state>success|transfer|abandoned)(?P<rest>[^\]]*)\]",
    re.IGNORECASE,
)
_TRIGGER_RE = re.compile(r'trigger\s*=\s*"([^"]*)"', re.IGNORECASE)
_BEHAVIOR_RE = re.compile(r'agent_behavior\s*=\s*"([^"]*)"', re.IGNORECASE)


def _last_user_content(conversation: list[dict[str, Any]]) -> str:
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            return str(msg.get("content", "") or "")
    return ""


def classify_terminal(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the terminal state from the final user turn.

    Returns metadata keys: terminal_state in {success, transfer, abandoned, incomplete},
    terminal_trigger, terminal_agent_behavior (None when not applicable / not provided).

    Rules:
    - An explicit [TERMINAL: <state> | trigger=... | agent_behavior=...] tag wins.
    - A bare [TASK_DONE] with no tag is treated as success (back-compat).
    - No [TASK_DONE] in the last user message => the run exhausted its turns => incomplete.
    """
    last_user = _last_user_content(conversation)
    match = _TERMINAL_RE.search(last_user)
    if match:
        state = match.group("state").lower()
        rest = match.group("rest") or ""
        trigger_match = _TRIGGER_RE.search(rest)
        behavior_match = _BEHAVIOR_RE.search(rest)
        return {
            "terminal_state": state,
            "terminal_trigger": trigger_match.group(1).strip() if trigger_match else None,
            "terminal_agent_behavior": behavior_match.group(1).strip() if behavior_match else None,
        }
    if "[TASK_DONE]" in last_user:
        return {"terminal_state": "success", "terminal_trigger": None, "terminal_agent_behavior": None}
    return {"terminal_state": "incomplete", "terminal_trigger": None, "terminal_agent_behavior": None}
