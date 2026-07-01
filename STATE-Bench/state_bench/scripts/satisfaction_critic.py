"""Turn-level user-satisfaction critic (tau2-style), operating on a transcript only.

After each user turn (a customer message that reacts to the agent), a critic model rates
the customer's satisfaction so far on an integer scale from -10 (furious / walking away) to
+10 (delighted), 0 = neutral. Per-turn scores are aggregated into cumulative (sum),
worst-case (min), mean, and final. This is the satisfaction signal used in the tau2
pipeline (cumulative + worst-case satisfaction); it needs nothing but the transcript, so it
is independent of STATE-Bench's native task/UX judge.

Scoring is ONLINE: each user turn is scored from the conversation prefix up to and
including that turn — the critic never sees the future, matching "score after each step
where the user responds."

NOTE: the prompt below is a placeholder — replace SATISFACTION_SYSTEM / SATISFACTION_USER
with the first author's calibrated critic prompt to make the numbers comparable to tau2.
"""

from __future__ import annotations

import json
from typing import Any

# --- Critic prompt (placeholder — swap with the first author's calibrated prompt) --------

SATISFACTION_SYSTEM = """\
You are a strict evaluator of CUSTOMER SATISFACTION in a customer-service conversation.
You are given the conversation up to and including the customer's latest message. Rate how
satisfied the customer is with the agent's service SO FAR, based only on what has happened
up to this point (do not assume anything about the future).

Scale (integer): -10 = furious / abandoning, -5 = clearly frustrated, 0 = neutral,
+5 = pleased, +10 = delighted. Reward the agent for being accurate, efficient, and
responsive to the customer's actual needs; penalize stalling, wrong or unverified actions,
repetition, and ignoring the customer. Judge the customer's experience, not task success.

Output STRICT JSON only: {"score": <integer -10..10>, "reason": "<one short sentence>"}.
"""

SATISFACTION_USER = """\
CONVERSATION SO FAR:
{conversation}

Rate the customer's satisfaction at this point as strict JSON.
"""


def _render_conversation_prefix(conversation: list[dict[str, Any]], up_to_index: int) -> str:
    """Render conversation[0..up_to_index] as readable ROLE: content lines for the critic."""
    lines: list[str] = []
    for msg in conversation[: up_to_index + 1]:
        role = str(msg.get("role", "")).upper()
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")
        if role == "ASSISTANT" and tool_calls:
            tc = "\n".join(
                f"[Called {c.get('name')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)[:200]})]"
                for c in tool_calls
                if isinstance(c, dict)
            )
            content = f"{tc}\n{content}" if content else tc
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


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
    per_turn: list[dict[str, Any]] = []
    for idx in turn_indices:
        prefix = _render_conversation_prefix(conversation, idx)
        prompt = SATISFACTION_USER.format(conversation=prefix)
        try:
            resp = client.complete_json(prompt=prompt, system_prompt=SATISFACTION_SYSTEM, max_tokens=max_tokens)
            score = _clamp_score(resp.get("score"))
            reason = str(resp.get("reason", ""))[:300]
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
