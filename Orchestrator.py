import os
import json
from pathlib import Path

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    WebSearchTool,
)

# Tools
from websurfer_agent.web_surfer_tool import WebSurferTool
from file_surfer_agent.file_surfer_tool import FileSurferTool
from coder_agent.coder_tool import CoderTool
# Correctly import the factory function instead of the old class name
from common_tools.llm_chat_tool import create_llm_chat_tool

# Planner and state manager
from planning_agent.planning_agent import PlanningAgent
from planning_agent.plan_state_manager import PlanStateManager
from critique_agent.critique_agent import CritiqueAgent


def main():
    """
    The main orchestration logic.
    """
    print("--- 🚀 Initializing Orchestrator ---")

    # 1. Initialize the shared model
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ.get("NAUT_API_KEY", "your_default_api_key"),
    )

    # 2. Initialize the Planning Agent
    planner = PlanningAgent(model=model)
    critique = CritiqueAgent(model=model)

    # 3. Initialize the Worker/Manager Agent with its tools
    web_surfer_tool = WebSurferTool(model=model)
    file_surfer_tool = FileSurferTool(
        model=model,
        base_path=str(Path.cwd()),
        viewport_size=2048,
    )
    coder_tool = CoderTool(
        model=model,
        max_debug_rounds=3,
        use_local_executor=True,
        work_dir=Path.cwd(),
    )
    

    # --- Start of the Workflow ---

    # 4. Get the user's high-level goal
    user_goal = (
        "Write a report on nature's design of leaves. Don't surf the web. Save the report to results.txt"
    )
    print(f"\n🎯 User Goal: {user_goal}")

    # 5. Generate the plan
    plan = planner.run(user_goal)

    print("\n--- 🏁 Starting Plan Execution ---")
    # Initialize the Plan State manager first
    state_manager = PlanStateManager(user_goal, plan)

    # 7. Main execution loop, driven by the orchestrator
    while not state_manager.is_finished():
        current_state = state_manager.get_current_state()
        current_task = state_manager.get_current_step_task()

        # Recreate llm_tool with updated plan state
        llm_tool = create_llm_chat_tool(model=model, plan_state=current_state)
        
        # Recreate manager_agent with updated tools
        manager_agent = ToolCallingAgent(
            tools=[web_surfer_tool, file_surfer_tool, coder_tool, llm_tool, WebSearchTool()],
            model=model,
        )

        print("\n" + "=" * 50)
        print(f"▶️ Executing Step {current_state.current_step_index + 1}: {current_task}")
        print("=" * 50)

        # Provide overall goal and full results JSON for context
        prior_results = current_state.results or {}
        results_json = json.dumps(prior_results, indent=2)

        prompt = f"""
Your current, specific subtask is: "{current_task}"

This subtask is a part of the larger goal: "{user_goal}". You do not need to complete this larger goal, only the subtask.

Results so far, formatted as [prior_step_id]_[prior_result] (JSON):
{results_json}



Analyze the subtask and the available results, choose the best tool, execute it, and return the result of your action.
The result should be a clear, self-contained piece of information that can be added to the project state.

**IMPORTANT**: Your final answer should ONLY answer the subtask. The overall goal is only there to provide context.
"""

        result = manager_agent.run(prompt)
        result_str = str(result)

        print(f"📝 Result for Step {current_state.current_step_index + 1}: {result_str}")
        # Critique the step result and optionally revise once
        critique_result = critique.run(
            overall_goal=user_goal,
            step_task=current_task or "",
            step_result=result_str,
            execution_logs=prompt,
        )

        if critique_result.decision == "revise" and critique_result.revised_prompt:
            revised = manager_agent.run(critique_result.revised_prompt)
            result_str = str(revised)

        # Update the state with the final (possibly revised) result
        state_manager.update_state(result_str)

    print("\n--- ✅ Plan Execution Finished ---")

    # 8. Display the final state
    final_state = state_manager.get_current_state()
    print("\nFinal Project State:")
    print(final_state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

