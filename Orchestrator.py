import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
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
from common_tools.sideinformation import get_side_info
from common_tools.sideinformation import load_side_info_from_metadata

# Planner and state manager
from planning_agent.planning_agent import PlanningAgent
from planning_agent.plan_state_manager import PlanStateManager


#Expert agent
from simulated_humans.simulated_human import ExpertAgent
def main():
    """
    The main orchestration logic.
    """
    print("--- 🚀 Initializing Orchestrator ---")

    # 1. Web surfer needs Gemma because its the only multimodal model that supports web surfing

    WebSurferModel = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )


    #qwen3 model for orchestrator and planner since its better at reasoning
    
    model = OpenAIServerModel(
        model_id="qwen3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

    #can initalize human coworkers here with their own models




    # 2. Initialize the Planning Agent
    planner = PlanningAgent(model=model)

    # 3. Initialize the Worker/Manager Agent with its tools
    web_surfer_tool = WebSurferTool(model=WebSurferModel)
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

    side_info = load_side_info_from_metadata()
    print(f"Side info: {side_info}")
    expert_agent = ExpertAgent(name="Expert", model=model, side_info=side_info)

   

    manager_agent = ToolCallingAgent(
        tools=[web_surfer_tool, file_surfer_tool, coder_tool, llm_tool, WebSearchTool()],
        model=model,
    )
# Create simulated human tools for human-in-the-loop decisions
# Set USE_SIMULATED_HUMANS to False to disable human tools (not yet implemented for real humans)

    # --- Start of the Workflow ---

    # 4. Get the user's high-level goal
    user_goal = (

        """
        A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?
        """
    )
    print(f"\n🎯 User Goal: {user_goal}")

    # 5. Generate the plan
    plan = planner.run(user_goal)

    plan_summary = llm_tool(query=f"Summarize the following plan in a concise manner: {plan}")



    expert_review = expert_agent.review_plan(user_goal, plan, side_info)
    
    #modify the plan based on the expert review, if there are edits, add them to the plan
    print(f"\n🔍 Expert Review: {expert_review}")

    print(f"\n🔍 Plan Summary: {plan_summary}")

    replanner_agent = PlanningAgent(model=model)

    replan_plan = replanner_agent.refine(user_goal, plan_summary, expert_review)

    print(f"\n🔍 Re-planned Plan: {replan_plan}")

    # 6. Initialize the Plan executor/state manager
    executor = PlanStateManager(user_goal, replan_plan)

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



# "Annotator Metadata": {"Steps": "1. Go to arxiv.org and navigate to the Advanced Search page.\n2. Enter \"AI regulation\" in the search box and select \"All fields\" from the dropdown.\n3. Enter 2022-06-01 and 2022-07-01 into the date inputs, select \"Submission date (original)\", and submit the search.\n4. Go through the search results to find the article that has a figure with three axes and labels on each end of the axes, titled \"Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation\".\n5. Note the six words used as labels: deontological, egalitarian, localized, standardized, utilitarian, and consequential.\n6. Go back to arxiv.org\n7. Find \"Physics and Society\" and go to the page for the \"Physics and Society\" category.\n8. Note that the tag for this category is \"physics.soc-ph\".\n9. Go to the Advanced Search page.\n10. Enter \"physics.soc-ph\" in the search box and select \"All fields\" from the dropdown.\n11. Enter 2016-08-11 and 2016-08-12 into the date inputs, select \"Submission date (original)\", and submit the search.\n12. Search for instances of the six words in the results to find the paper titled \"Phase transition from egalitarian to hierarchical societies driven by competition between cognitive and social constraints\", indicating that \"egalitarian\" is the correct answer."