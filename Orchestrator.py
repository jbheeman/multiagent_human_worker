import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import json
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    WebSearchTool,
    WikipediaSearchTool,
)

import yaml


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
from simulated_humans.simulated_human import ExpertAgent, ask_human_expert_for_help, set_expert_agent

def run_orchestrator_task(user_goal, side_info=None):
    """
    Run the orchestrator task with a given user goal and optional side information.
    
    Args:
        user_goal (str): The user's goal/question to solve
        side_info (dict, optional): Side information to help with the task
    
    Returns:
        The final state of the execution
    """
    print("--- 🚀 Initializing Orchestrator ---")

    orchestrator_prompt = yaml.safe_load(open("orchestrator_agent.yaml").read())
    coder_prompt_template = yaml.safe_load(open("coder_coworker.yaml").read())
    websurfer_prompt_template = yaml.safe_load(open("websurfer_coworker.yaml").read())
    file_surfer_prompt_template = yaml.safe_load(open("file_coworker.yaml").read())
    
    # Use full prompt templates for proper smolagents integration
    coder_prompt_templates = coder_prompt_template
    websurfer_prompt_templates = websurfer_prompt_template
    file_surfer_prompt_templates = file_surfer_prompt_template

    # print("Loaded prompt templates:")
    # print(f"Coder prompt templates: {coder_prompt_templates}")
    # print(f"Websurfer prompt templates: {websurfer_prompt_templates}")
    # print(f"File surfer prompt templates: {file_surfer_prompt_templates}")

    # 1. Web surfer needs Gemma because its the only multimodal model that supports web surfing
    WebSurferModel = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

    # print(f"WebSurferModel: {WebSurferModel}")

    # qwen3 model for orchestrator and planner since its better at reasoning
    model = OpenAIServerModel(
        model_id="qwen3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

    # can initialize human coworkers here with their own models
    human_websurfer_coworker = WebSurferTool(model=WebSurferModel, prompt_templates=websurfer_prompt_templates)
    human_file_coworker = FileSurferTool(model=model, prompt_templates=file_surfer_prompt_templates)
    human_coder_coworker = CoderTool(model=model, prompt_templates=coder_prompt_templates)

    # 2. Initialize the Planning Agent
    print("Initializing Planning Agent")
    planner = PlanningAgent(model=model)
    print("Planning Agent initialized")

    # 3. Initialize the Worker/Manager Agent with its tools
    print("Initializing Web Surfer Tool")
    web_surfer_tool = WebSurferTool(model=WebSurferModel)
    print("Web Surfer Tool initialized")

    print("Initializing File Surfer Tool")
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

    # Use provided side_info or load from metadata
    if side_info is None:
        side_info = load_side_info_from_metadata()
    print(f"Side info: {side_info}")
    
    expert_agent = ExpertAgent(name="Expert", model=model, side_info=side_info)

    # Set the expert agent for the co-execution tool
    set_expert_agent(expert_agent)

    manager_agent = ToolCallingAgent(
        tools=[web_surfer_tool, file_surfer_tool, coder_tool, llm_tool, WebSearchTool(), WikipediaSearchTool(), ask_human_expert_for_help],
        model=model,
        prompt_templates=orchestrator_prompt,
    )

    # --- Start of the Workflow ---
    print(f"\n🎯 User Goal: {user_goal}")

    # 5. Generate the plan
    plan = planner.run(user_goal)

    simhuman_review = expert_agent.review_plan(user_goal, plan, side_info)
    
    # modify the plan based on the expert review, if there are edits, add them to the plan
    print(f"\n🔍 Expert Review: {simhuman_review}")

    replanner_agent = PlanningAgent(model=model)
    replan_plan = replanner_agent.refine(user_goal, plan, simhuman_review)

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

If you get stuck or need guidance on how to proceed with the current task, you can use the ask_human_expert_for_help tool to get assistance from a simulated human expert who has access to side information about this task.
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
    
    return final_state


def main():
    """
    The main orchestration logic for interactive use.
    """
    # Default task for interactive use
    # user_goal = """I'm researching species that became invasive after people who kept them as pets released them. There's a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. According to the USGS, where was this fish found as a nonnative species, before the year 2020? I need the answer formatted as the five-digit zip codes of the places the species was found, separated by commas if there is more than one place."""
    # user_goal = """If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation."""
    user_goal = """In Unlambda, what exact charcter or text needs to be added to correct the following code to output \"For penguins\"? If what is needed is a character, answer with the name of the character. If there are different names for the character, use the shortest. The text location is not needed. Code:\n\n`r```````````.F.o.r. .p.e.n.g.u.i.n.si"""
    
    return run_orchestrator_task(user_goal)


if __name__ == "__main__":
    main()



# "Annotator Metadata": {"Steps": "1. Go to arxiv.org and navigate to the Advanced Search page.\n2. Enter \"AI regulation\" in the search box and select \"All fields\" from the dropdown.\n3. Enter 2022-06-01 and 2022-07-01 into the date inputs, select \"Submission date (original)\", and submit the search.\n4. Go through the search results to find the article that has a figure with three axes and labels on each end of the axes, titled \"Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation\".\n5. Note the six words used as labels: deontological, egalitarian, localized, standardized, utilitarian, and consequential.\n6. Go back to arxiv.org\n7. Find \"Physics and Society\" and go to the page for the \"Physics and Society\" category.\n8. Note that the tag for this category is \"physics.soc-ph\".\n9. Go to the Advanced Search page.\n10. Enter \"physics.soc-ph\" in the search box and select \"All fields\" from the dropdown.\n11. Enter 2016-08-11 and 2016-08-12 into the date inputs, select \"Submission date (original)\", and submit the search.\n12. Search for instances of the six words in the results to find the paper titled \"Phase transition from egalitarian to hierarchical societies driven by competition between cognitive and social constraints\", indicating that \"egalitarian\" is the correct answer."