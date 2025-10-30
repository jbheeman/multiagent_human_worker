from __future__ import annotations
from typing import Dict, Any
from pydantic import BaseModel
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from planning_agent.plan_models import PlanState, PlanStep
from file_surfer_agent import file_tools
from file_surfer_agent.markdown_file_browser import MarkdownFileBrowser

from .prompts import CRITIQUE_SYSTEM_PROMPT
from .tools import execute_file


class CritiqueResult(BaseModel):
    decision: str  # "approve" | "revise"
    rationale: str
    revised_prompt: str = ""
    next_step_suggestion: str = ""


@tool
def submit_critique(decision: str, rationale: str, revised_prompt: str = "", next_step_suggestion: str = "") -> str:
    """
    Submit a critique decision and optional revision details for the current step.

    Args:
        decision: Either "approve" (output is sufficient) or "revise" (needs improvement).
        rationale: A short explanation justifying the decision.
        revised_prompt: If decision=="revise", a concise prompt/context to rerun the step.
        next_step_suggestion: Optional short note suggesting an adjustment to the next step.

    Returns:
        A simple acknowledgment string.
    """
    return "OK"


class PostExecutionCritiqueAgent:
    def __init__(self, model: OpenAIServerModel):
        self._agent = ToolCallingAgent(
            tools=[
                submit_critique, 
                execute_file,
                file_tools.open_path,
                file_tools.page_up,
                file_tools.page_down,
                file_tools.find_on_page,
                file_tools.find_next,
            ],
            model=model,
            max_steps=10, # Increased to allow for tool use before critique
            instructions="""You have access to a file browser to inspect the project structure and file contents. Your current working directory is the root of the current run's log directory.

- Use `open_path` to view files or directories.
- Output files from previous steps are located in the `workspace` subdirectory.
- If the result of a step is a filename, you should look for it in `workspace/<filename>`.
- Execution logs are in the current directory. You can access them using `./<log_file_name>`.

If the result is a file path, you can use the `execute_file` tool to run the file and inspect its output.

The naming convention for the logs is `step_<step_number>[b,c,...]_<agent_name>.log`.
For a revised step, the log of the latest run is the one with the highest letter suffix (e.g., `step_1c_...` is later than `step_1b_...`).
Redundant logs have a `_redundant` suffix."""
        )
        self._system_prompt = CRITIQUE_SYSTEM_PROMPT

    def run(
        self,
        overall_goal: str,
        step_task: str,
        step_result: str,
        plan_state: PlanState,
        next_step_task: str | None = None,
    ) -> CritiqueResult:
        context = (
            f"{self._system_prompt}\n\n"
            f"Your current working directory is the root of the current run's log directory. You can find output files in the `workspace` subdirectory and execution logs in the current directory.\n\n"
            f"overall_goal: {overall_goal}\n"
            f"step_task: {step_task}\n"
            f"step_result: {step_result}\n"
            f"prior_results: {plan_state.results}\n"
            f"next_step_task: {next_step_task}\n\n"
            f"Return your decision by calling submit_critique with the appropriate fields."
        )
        self._agent.run(context)

        for step in self._agent.memory.steps:
            if hasattr(step, "tool_calls") and step.tool_calls:
                for call in step.tool_calls:
                    if call.name == "submit_critique":
                        args: Dict[str, Any] = call.arguments
                        return CritiqueResult(
                            decision=args.get("decision", "approve"),
                            rationale=args.get("rationale", ""),
                            revised_prompt=args.get("revised_prompt", ""),
                            next_step_suggestion=args.get("next_step_suggestion", ""),
                        )

        return CritiqueResult(decision="approve", rationale="No critique returned; proceeding.")

class PreExecutionCritiqueResult(BaseModel):
    decision: str  # "proceed" | "skip" | "revise"
    revised_task: str = ""
    revised_output_key: str = ""

@tool
def submit_pre_execution_critique(decision: str, revised_task: str = "", revised_output_key: str = "") -> str:
    """
    Submit a critique of the current plan step.

    Args:
        decision: Your decision, one of "proceed", "skip", or "revise".
        revised_task: If you chose "revise", this is the updated task for the current step.
        revised_output_key: If you chose "revise", you can optionally provide a new output key.

    Returns:
        A simple acknowledgment string.
    """
    return "OK"

class PreExecutionCritiqueAgent:
    def __init__(self, model: OpenAIServerModel):
        self._agent = ToolCallingAgent(
            tools=[
                submit_pre_execution_critique,
                execute_file,
                file_tools.open_path,
                file_tools.page_up,
                file_tools.page_down,
                file_tools.find_on_page,
                file_tools.find_next,
            ],
            model=model,
            max_steps=10,
        )

    def run(
        self, overall_goal: str, plan_state: PlanState, current_step: PlanStep, next_step_suggestion: str = ""
    ) -> PreExecutionCritiqueResult:
        file_tools.browser = MarkdownFileBrowser(base_path=".", viewport_size=4096)
        plan_steps_str = "\n".join([f"{i+1}. {step.task}" for i, step in enumerate(plan_state.plan.steps)])
        context = (
            f"""You are a proactive pre-execution critic. Your job is to review a plan step *before* it is executed to ensure it is still valid and necessary.

**Information you will be given:**
- `overall_goal`: The main objective.
- `full_plan`: The complete list of all plan steps.
- `prior_results`: The results from all previous steps.
- `suggestion_for_current_step`: A suggestion from the previous post-execution critic.
- `current_step_task`: The task for the step you are about to critique.

**Your process is as follows:**

**1. Analysis (Optional):**
   - You have access to file browsing and code execution tools. You can use them to analyze the current state of the workspace if you need more context to make your decision.

**2. Critique and Finalize (a two-step process):**
   - **Step 1: Submit Critique:** After your analysis (if any), you MUST call the `submit_pre_execution_critique` tool ONCE to provide your decision (`proceed`, `skip`, or `revise`).
   - **Step 2: Final Answer:** In your next step, you MUST call the `final_answer` tool with the message 'Critique submitted.' to complete your work.

---

**Data for your analysis:**

**Overall Goal:** {overall_goal}
**Full Plan:**\n{plan_steps_str}

**Prior Results:** {plan_state.results}
**Suggestion for current step:** {next_step_suggestion}

**Current Step to Critique:**
- Task: {current_step.task}
- Output Key: {current_step.output_key}
"""
        )

        self._agent.run(context)

        for step in self._agent.memory.steps:
            if hasattr(step, "tool_calls") and step.tool_calls:
                for call in step.tool_calls:
                    if call.name == "submit_pre_execution_critique":
                        args = call.arguments
                        return PreExecutionCritiqueResult(
                            decision=args.get("decision", "proceed"),
                            revised_task=args.get("revised_task", ""),
                            revised_output_key=args.get("revised_output_key", ""),
                        )
        
        return PreExecutionCritiqueResult(decision="proceed")
    