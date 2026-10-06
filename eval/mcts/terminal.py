"""Map bench-specific outcomes onto the exclusive terminal_state enum."""

from __future__ import annotations

from typing import Any

TERMINAL_SUCCESS = "success"
TERMINAL_TRANSFER = "transfer"
TERMINAL_FAILURE = "failure_no_transfer"
TERMINAL_MAX_TURNS = "max_turns"
TERMINAL_SIM_ERROR = "sim_error"


def _messages_have_transfer(messages: list[Any] | None) -> bool:
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            name = None
            if isinstance(tc, dict):
                name = tc.get("name")
                if not name and isinstance(tc.get("function"), dict):
                    name = tc["function"].get("name")
            if name and "transfer" in str(name).lower():
                return True
    return False


def map_tau2_terminal(
    *,
    reward: float | int | None,
    termination_reason: str | None,
    messages: list[Any] | None = None,
    error: str | None = None,
) -> tuple[str, bool, bool]:
    """Return (terminal_state, task_success, transfer).

    Priority (exclusive):
      sim_error > max_turns > transfer(handoff) > success(reward==1) > failure_no_transfer
    """
    reason = (termination_reason or "").lower()
    if error or reason in {"agent_error", "user_error", "too_many_errors"}:
        return TERMINAL_SIM_ERROR, False, False
    if reason in {"max_steps", "max_turns"}:
        return TERMINAL_MAX_TURNS, False, False

    transferred = _messages_have_transfer(messages)
    if transferred or "handoff" in reason or "transfer" in reason:
        return TERMINAL_TRANSFER, False, True

    if reward is not None and float(reward) >= 1.0 - 1e-9:
        return TERMINAL_SUCCESS, True, False

    return TERMINAL_FAILURE, False, False


def map_statebench_terminal(
    *,
    raw_terminal: str | None,
    error: str | None = None,
    state_requirements_met: bool | float | None = None,
) -> tuple[str, bool, bool]:
    """Map STATE-Bench classify_terminal / error paths to the exclusive enum.

    STATE-Bench native: success | transfer | abandoned | incomplete | error
    """
    if error or (raw_terminal or "").lower() == "error":
        return TERMINAL_SIM_ERROR, False, False

    state = (raw_terminal or "").lower()
    if state == "success":
        success = True
        if state_requirements_met is not None:
            success = bool(state_requirements_met) or float(state_requirements_met) >= 1.0 - 1e-9
        # Prefer explicit TERMINAL:success; still mark task_success from requirements when present.
        if state_requirements_met is None:
            return TERMINAL_SUCCESS, True, False
        if success:
            return TERMINAL_SUCCESS, True, False
        return TERMINAL_FAILURE, False, False
    if state == "transfer":
        return TERMINAL_TRANSFER, False, True
    if state in {"incomplete", "abandoned"}:
        # incomplete ≈ turn budget / no TASK_DONE; abandoned ≈ user left.
        if state == "incomplete":
            return TERMINAL_MAX_TURNS, False, False
        return TERMINAL_FAILURE, False, False
    if not state:
        return TERMINAL_SIM_ERROR, False, False
    return TERMINAL_FAILURE, False, False
