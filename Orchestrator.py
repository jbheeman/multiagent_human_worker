import os
from pathlib import Path

import json
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
from simulated_humans.human_tools import create_human_tools

# Planner and state manager
from planning_agent.planning_agent import PlanningAgent
from planning_agent.plan_state_manager import PlanStateManager


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
    # Instantiate the tool using the new factory function
    llm_tool = create_llm_chat_tool(model=model)

    USE_SIMULATED_HUMANS = True
    if USE_SIMULATED_HUMANS:
        human_tools = create_human_tools(model=model, use_simulated=True)
    else:
        human_tools = []


    manager_agent = ToolCallingAgent(
        tools=[web_surfer_tool, file_surfer_tool, coder_tool, llm_tool, WebSearchTool(), *human_tools],
        model=model,
    )
# Create simulated human tools for human-in-the-loop decisions
# Set USE_SIMULATED_HUMANS to False to disable human tools (not yet implemented for real humans)

    # --- Start of the Workflow ---

    # 4. Get the user's high-level goal
    user_goal = (
        "Perform a detailed analysis of web_surfer_agent.py"
    )
    print(f"\n🎯 User Goal: {user_goal}")

    # 5. Generate the plan
    plan = planner.run(user_goal)

    # 6. Initialize the Plan executor/state manager
    executor = PlanStateManager(user_goal, plan)

    print("\n--- 🏁 Starting Plan Execution ---")

    # 7. Main execution loop, driven by the orchestrator
    while not executor.is_finished():
        current_state = executor.get_current_state()
        current_task = executor.get_current_step_task()

        print("\n" + "=" * 50)
        print(f"▶️ Executing Step {current_state.current_step_index + 1}: {current_task}")
        print("=" * 50)

        prompt = f"""
Here is the current state of the project, including the overall goal and results from previous steps:
{current_state.model_dump_json(indent=2)}

Your current, specific task is: "{current_task}"

Analyze the state and the task, choose the best tool, execute it, and return the result of your action.
The result should be a clear, self-contained piece of information that can be added to the project state.
"""

        result = manager_agent.run(prompt)
        result_str = str(result)

        print(f"📝 Result for Step {current_state.current_step_index + 1}: {result_str}")

        # Update the state with the result via the executor
        executor.update_state(result_str)

    print("\n--- ✅ Plan Execution Finished ---")

    # 8. Display the final state
    final_state = executor.get_current_state()
    print("\nFinal Project State:")
    print(final_state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

