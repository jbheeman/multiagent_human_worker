"""Turn-level user-emotion critic (partner's calibrated emotion-delta prompt).

After each reactive user turn, a critic model estimates the CHANGE in the user's emotion
(emotion_delta) caused by the agent's immediately preceding response, on an integer scale
from -10 to +10 (0 = no change). Per-turn deltas are aggregated into cumulative (sum of
deltas = net emotion change), worst-case (min = biggest single-step drop), mean, and final.
It needs nothing but the transcript, so it is independent of STATE-Bench's native judge.

Scoring is ONLINE: each step sees only the assistant's previous response and the user's
reaction at that turn — never the future — matching "estimate the change at this step."

When the transcript includes a tau2-style ``<internal_monologue>`` block (persona runs),
the critic is fed the **inner monologue text**. If tags are absent (baseline / old
transcripts), it falls back to the full spoken user content.
"""

from __future__ import annotations

import json
import re
from typing import Any

# --- Critic prompt (partner's calibrated emotion-delta prompt, verbatim system) ----------

SATISFACTION_SYSTEM = """\
You are evaluating how a user's emotion changes after a single interaction step.
The user's INTERNAL MONOLOGUE reflects their true thoughts and feelings.
---
# INPUT
You are given:
- the user's goal
- the assistant's previous response
- the user's INTERNAL MONOLOGUE after that response
---
# TASK
Determine how the user's emotion changed as a result of the assistant's response.
---
# OUTPUT
Return:
emotion_delta ∈ [-10, +10]
---
# GUIDELINES
Base your judgment primarily on the INTERNAL MONOLOGUE.
Interpret the monologue as the user's reaction to the assistant's response.
Use the following scale consistently:
- No emotional change → 0
- Mild reaction (slight concern, mild satisfaction) → ±1 to ±3
- Moderate reaction (noticeable concern, annoyance, appreciation) → ±4 to ±6
- Strong emotion (frustration, relief, disappointment) → ±7 to ±9
- Very strong emotion (e.g., "unacceptable", escalation, demanding supervisor) → ±9 to ±10
CRITICAL:
- Strong emotional language MUST produce large magnitude deltas (≥ |8|)
- Do NOT underestimate frustration or escalation signals
- If the monologue expresses increasing frustration, assign a negative delta
- If it expresses relief or satisfaction, assign a positive delta
- If the monologue contains only planning or neutral reasoning, assign 0 or a very small delta
---
# IMPORTANT
- Do NOT consider any future conversation
- Do NOT try to maintain or infer an absolute score
- Only estimate the CHANGE in emotion at this step
---
# RESPONSE FORMAT
{
  "emotion_delta": <number>,
  "feeling": "<brief explanation grounded in the internal monologue>"
}
"""

SATISFACTION_USER = """\
# USER GOAL
{goal}

# ASSISTANT'S PREVIOUS RESPONSE
{assistant_response}

# USER'S INTERNAL MONOLOGUE (their reaction to that response)
{monologue}

Estimate the emotion_delta for this step as strict JSON.
"""

_INNER_MONOLOGUE_RE = re.compile(
    r"<internal_monologue>(.*?)</internal_monologue>",
    re.DOTALL,
)


def extract_monologue_for_critic(content: str) -> str:
    """Prefer inner ``<internal_monologue>`` text; else return full content (back-compat)."""
    if content and "<internal_monologue>" in content:
        match = _INNER_MONOLOGUE_RE.search(content)
        if match:
            return match.group(1).strip()
    return content or ""


def _render_message(msg: dict[str, Any]) -> str:
    """Render a single message's content, inlining any assistant tool calls."""
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls")
    if str(msg.get("role", "")).lower() == "assistant" and tool_calls:
        tc = "\n".join(
            f"[Called {c.get('name')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)[:200]})]"
            for c in tool_calls
            if isinstance(c, dict)
        )
        content = f"{tc}\n{content}" if content else tc
    return content


def _goal_text(conversation: list[dict[str, Any]]) -> str:
    """The task/goal is the opening user message (before any agent behavior)."""
    for msg in conversation:
        if msg.get("role") == "user" and (msg.get("content") or "").strip():
            return str(msg.get("content"))
    return ""


def _prev_assistant_response(conversation: list[dict[str, Any]], before_index: int) -> str:
    """Rendered content of the most recent assistant message before `before_index`."""
    for msg in reversed(conversation[:before_index]):
        if msg.get("role") == "assistant":
            rendered = _render_message(msg)
            if rendered.strip():
                return rendered
    return ""


def _reactive_user_turn_indices(conversation: list[dict[str, Any]]) -> list[int]:
    """Indices of user turns that react to the agent (i.e., preceded by an assistant turn).

    The opening user message (task setup, before any agent behavior) is not scored.
    """
    seen_assistant = False
    indices: list[int] = []
    for i, msg in enumerate(conversation):
        role = msg.get("role")
        if role == "assistant":
            seen_assistant = True
        elif role == "user" and seen_assistant and (msg.get("content") or "").strip():
            indices.append(i)
    return indices


def _clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(-10, min(10, score))


def score_transcript(client: Any, conversation: list[dict[str, Any]], *, max_tokens: int = 8192) -> dict[str, Any]:
    """Score each reactive user turn and aggregate. Returns a satisfaction summary dict.

    client must implement complete_json(prompt=, system_prompt=, max_tokens=) -> dict.
    """
    turn_indices = _reactive_user_turn_indices(conversation)
    goal = _goal_text(conversation)
    per_turn: list[dict[str, Any]] = []
    for idx in turn_indices:
        user_content = _render_message(conversation[idx])
        prompt = SATISFACTION_USER.format(
            goal=goal,
            assistant_response=_prev_assistant_response(conversation, idx),
            monologue=extract_monologue_for_critic(user_content),
        )
        try:
            resp = client.complete_json(prompt=prompt, system_prompt=SATISFACTION_SYSTEM, max_tokens=max_tokens)
            score = _clamp_score(resp.get("emotion_delta"))
            reason = str(resp.get("feeling", ""))[:300]
        except Exception as exc:  # noqa: BLE001 - one bad turn shouldn't void the transcript
            score = 0
            reason = f"critic error: {type(exc).__name__}: {exc}"
        per_turn.append({"conversation_index": idx, "score": score, "reason": reason})

    scores = [t["score"] for t in per_turn]
    if scores:
        aggregates = {
            "satisfaction_cumulative": sum(scores),
            "satisfaction_worst_case": min(scores),
            "satisfaction_best_case": max(scores),
            "satisfaction_mean": round(sum(scores) / len(scores), 2),
            "satisfaction_final": scores[-1],
            "satisfaction_turns_scored": len(scores),
        }
    else:
        aggregates = {
            "satisfaction_cumulative": None,
            "satisfaction_worst_case": None,
            "satisfaction_best_case": None,
            "satisfaction_mean": None,
            "satisfaction_final": None,
            "satisfaction_turns_scored": 0,
        }
    return {"satisfaction_per_turn": per_turn, **aggregates}
