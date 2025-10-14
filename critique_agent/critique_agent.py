from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from planning_agent.plan_models import PlanState

from .prompts import CRITIQUE_SYSTEM_PROMPT


class CritiqueResult(BaseModel):
    decision: str  # "approve" | "revise"
    rationale: str
    revised_prompt: str = ""
    plan_adjustments: str = ""


@tool
def submit_critique(decision: str, rationale: str, revised_prompt: str = "", plan_adjustments: str = "") -> str:
    """
    Submit a critique decision and optional revision details for the current step.

    Args:
        decision: Either "approve" (output is sufficient) or "revise" (needs improvement).
        rationale: A short explanation justifying the decision.
        revised_prompt: If decision=="revise", a concise prompt/context to rerun the step.
        plan_adjustments: Optional short note proposing updates to upcoming steps.

    Returns:
        A simple acknowledgment string.
    """
    return "OK"


class CritiqueAgent:
    def __init__(self, model: OpenAIServerModel):
        self._agent = ToolCallingAgent(
            tools=[submit_critique],
            model=model,
            max_steps=1,
        )
        self._system_prompt = CRITIQUE_SYSTEM_PROMPT

    def run(
        self,
        overall_goal: str,
        step_task: str,
        step_result: str,
        plan_state: PlanState,
        execution_logs: str = "",
    ) -> CritiqueResult:
        # Only pass essential information to avoid redundancy
        context = (
            f"{self._system_prompt}\n\n"
            f"overall_goal: {overall_goal}\n"
            f"step_task: {step_task}\n"
            f"step_result: {step_result}\n"
            f"prior_results: {plan_state.results}\n"
            f"execution_logs: {execution_logs}\n"
            f"Return your decision by calling submit_critique with the appropriate fields."
        )
        self._agent.run(context)

        # Extract the tool call from memory
        for step in self._agent.memory.steps:
            if hasattr(step, "tool_calls") and step.tool_calls:
                call = step.tool_calls[0]
                if call.name == "submit_critique":
                    args: Dict[str, Any] = call.arguments
                    return CritiqueResult(
                        decision=args.get("decision", "approve"),
                        rationale=args.get("rationale", ""),
                        revised_prompt=args.get("revised_prompt", ""),
                        plan_adjustments=args.get("plan_adjustments", ""),
                    )

        # Fallback approve
        return CritiqueResult(decision="approve", rationale="No critique returned; proceeding.")


