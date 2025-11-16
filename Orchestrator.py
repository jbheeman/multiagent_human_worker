import os
import json
from pathlib import Path
import sys
import tempfile
import shutil

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
from common_tools.llm_chat_tool import create_llm_chat_tool
from common_tools.logger import Logger


# Planner and state manager
from planning_agent.planning_agent import PlanningAgent
from planning_agent.plan_state_manager import PlanStateManager
from critique_agent.critique_agent import PostExecutionCritiqueAgent, PreExecutionCritiqueAgent

#Expert agent
from simulated_humans.simulated_human import SimulatedHumanAgent, ask_human_expert_for_help, set_simulated_human_agent
from simulated_humans.sideinformation import load_behavior_info

def run_orchestrator_task(user_goal, user_id, side_info=None):
    """
    Run the orchestrator task with a given user goal and optional side information.
    
    Args:
        user_goal (str): The user's goal/question to solve
        side_info (dict, optional): Side information to help with the task
    
    Returns:
        The final state of the execution
    """
    logger = Logger()
    logger.log_overview(f"--- 🚀 Initializing Orchestrator ---")
    logger.log_overview(f"Logs for this run will be saved in: {logger.run_dir}")
    logger.log_overview("Log file naming convention: step_<step_number>[b,c,...]_<agent_name>.log")
    logger.log_overview("Redundant logs will have a `_redundant` suffix.")

    # Create a workspace directory for this run inside the logs directory
    work_dir = os.path.abspath(os.path.join(logger.run_dir, "workspace"))
    os.makedirs(work_dir, exist_ok=True)
    workspace_source_dir = "workspace"
    if not os.path.exists(workspace_source_dir):
        os.makedirs(workspace_source_dir)
    if os.path.exists(workspace_source_dir):
        shutil.copytree(workspace_source_dir, work_dir, dirs_exist_ok=True)
    logger.log_overview(f"Working directory for this run: {work_dir}")

    orchestrator_prompt = yaml.safe_load(open("orchestrator_agent.yaml").read())
   
    simulated_human_prompt_template = yaml.safe_load(open("simulated_human_coworker.yaml").read())
    simulated_human_prompt_templates = simulated_human_prompt_template



    # 1. Define models for different roles
    # gemma3 model for most agents
    gemma_model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

    # qwen3 model for planner and expert agents, as it's better at reasoning
    planning_model = OpenAIServerModel(
        model_id="qwen3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

 
    # 2. Initialize the Planning Agent
    print("Initializing Planning Agent")
    planner = PlanningAgent(model=planning_model)
    post_execution_critique = PostExecutionCritiqueAgent(model=gemma_model)
    pre_execution_critique = PreExecutionCritiqueAgent(model=gemma_model)
    web_surfer_tool = WebSurferTool(model=gemma_model)
   
    llm_tool = create_llm_chat_tool(model=gemma_model)
    
    user_products = load_behavior_info(user_id)
    simulated_human_agent = SimulatedHumanAgent(name="SimulatedHuman", model=planning_model, user_products=user_products)
    set_simulated_human_agent(simulated_human_agent)

    manager_agent = ToolCallingAgent(
        tools=[web_surfer_tool, llm_tool, WebSearchTool(), WikipediaSearchTool(), ask_human_expert_for_help],
        model=gemma_model,
        prompt_templates=orchestrator_prompt,
    )
    

    # --- Start of the Workflow ---
    # 4. Use the user's high-level goal provided as an argument
    logger.log_overview(f"\n🎯 User Goal: {user_goal}")

    # 5. Generate the plan
    logger.log_overview("\n--- 📝 Generating Plan ---")
    log_file = logger.get_log_file("PlanningAgent")
    old_stdout = sys.stdout
    sys.stdout = log_file
    plan = planner.run(user_goal)
    sys.stdout = old_stdout
    log_file.close()
    logger.log_overview("✅ Plan Generated")
    logger.log_overview("\n--- Generated Plan ---")
    for i, step in enumerate(plan.steps):
        logger.log_overview(f"Step {i+1}: {step.task}")
    logger.log_overview("--------------------")

    # Expert review of the plan - do we still do this? or does the orchestrator just ask the simulated human agent for help? ifl now the plan doesnt really need to be reivewd? 
    logger.log_overview("\n--- 🧑‍🏫 Expert Review of the Plan ---")
    log_file = logger.get_log_file("SimulatedHumanAgent")
    old_stdout = sys.stdout
    sys.stdout = log_file
    simhuman_review = simulated_human_agent.review_plan(user_goal, plan.model_dump_json(indent=2), side_info)
    sys.stdout = old_stdout
    log_file.close()
    logger.log_overview(f"\n🔍 Expert Review: {simhuman_review}")

    # If the expert accepts the plan, use it directly. Otherwise, refine it.
    if isinstance(simhuman_review, dict) and simhuman_review.get("feedback", "").lower() == "accept":
        logger.log_overview("✅ Expert accepted the plan. Proceeding with original plan.")
        replan_plan = plan
    else:
        logger.log_overview("\n--- 🔄 Refining Plan based on Expert Review ---")
        # The simhuman_review is now a structured dictionary, which is what 'refine' expects.
        replanner_agent = PlanningAgent(model=planning_model)
        log_file = logger.get_log_file("PlanningAgent_Refine")
        old_stdout = sys.stdout
        sys.stdout = log_file
        replan_plan = replanner_agent.refine(user_goal, plan.model_dump_json(indent=2), simhuman_review)
        sys.stdout = old_stdout
        log_file.close()
        logger.log_overview("✅ Plan Refined")

    logger.log_overview("\n--- Refined Plan ---")
    for i, step in enumerate(replan_plan.steps):
        logger.log_overview(f"Step {i+1}: {step.task} (output: {step.output_key})")
    logger.log_overview("--------------------")

    logger.log_overview("\n--- 🏁 Starting Plan Execution ---")
    # Initialize the Plan State manager first
    state_manager = PlanStateManager(user_goal, replan_plan)

    # 7. Main execution loop, driven by the orchestrator
    next_step_suggestion = ""
    while not state_manager.is_finished():
        current_state = state_manager.get_current_state()
        current_step = current_state.plan.steps[current_state.current_step_index]
        step_number = current_state.current_step_index + 1
        
        # Pre-execution critique
        logger.log_overview(f"\n--- 🤔 Pre-execution critique for step {step_number} ---")
        log_file = logger.get_log_file("PreExecutionCritiqueAgent", step_number=step_number)
        old_stdout = sys.stdout
        sys.stdout = log_file
        pre_execution_critique_result = pre_execution_critique.run(
            overall_goal=user_goal,
            plan_state=current_state,
            current_step=current_step,
            next_step_suggestion=next_step_suggestion,
        )
        next_step_suggestion = "" # Reset after use
        sys.stdout = old_stdout
        log_file.close()
        logger.log_overview(f"🧠 Pre-Execution Critique Result:\n{json.dumps(pre_execution_critique_result.model_dump(), indent=2)}")

        if pre_execution_critique_result.decision == "skip":
            logger.log_overview(f"⏭️ Skipping Step {step_number}: {current_step.task}")
            state_manager.update_state("Step skipped by pre-execution critic.")
            continue

        if pre_execution_critique_result.decision == "revise":
            current_step.task = pre_execution_critique_result.revised_task
            logger.log_overview(f"🔄 Revising Step {step_number} to: {current_step.task}")
            if pre_execution_critique_result.revised_output_key:
                current_step.output_key = pre_execution_critique_result.revised_output_key
                logger.log_overview(f"🔄 Revising Step {step_number} output key to: {current_step.output_key}")

        current_task = current_step.task

        # Recreate llm_tool with updated plan state
        llm_tool = create_llm_chat_tool(model=gemma_model, plan_state=current_state)
        
        # Recreate manager_agent with updated tools
        manager_agent = ToolCallingAgent(
            tools=[web_surfer_tool, file_surfer_tool, coder_tool, llm_tool, WebSearchTool(), ask_human_expert_for_help],
            model=gemma_model,
        )

        logger.log_overview("\n" + "=" * 50)
        logger.log_overview(f"▶️ Executing Step {step_number}: {current_task}")
        logger.log_overview("=" * 50)

        # Provide overall goal and full results JSON for context
        prior_results = current_state.results or {}
        results_json = json.dumps(prior_results, indent=2)

        prompt = f"""
Your current, specific subtask is: "{current_task}"

This subtask is a part of the larger goal: "{user_goal}". You do not need to complete this larger goal, only the subtask.

Results so far, formatted as [prior_step_id]_[prior_result] (JSON):
{results_json}

Analyze the subtask and the available results, choose the best tools, execute it, and return the result of your action.
The result should be a clear, self-contained piece of information that can be added to the project state.

**IMPORTANT**: Your final answer should ONLY answer the subtask. The overall goal is only there to provide context.
"""

        log_file = logger.get_log_file("ManagerAgent", step_number=step_number)
        old_stdout = sys.stdout
        sys.stdout = log_file
        result = manager_agent.run(prompt)
        sys.stdout = old_stdout
        log_file.close()

        result_str = str(result)

        logger.log_overview(f"📝 Result for Step {step_number}: {result_str}")
        
        # Critique the step result and optionally revise once
        logger.log_overview(f"\n--- 🤔 Post-execution critique for step {step_number} ---")
        original_browser = file_tools.browser
        file_tools.browser = MarkdownFileBrowser(base_path=logger.run_dir)
        try:
            log_file = logger.get_log_file("PostExecutionCritiqueAgent", step_number=step_number)
            old_stdout = sys.stdout
            sys.stdout = log_file
            next_step_task = state_manager.get_next_step_task()
            critique_result = post_execution_critique.run(
                overall_goal=user_goal,
                step_task=current_task or "",
                step_result=result_str,
                plan_state=state_manager.get_current_state(),
                next_step_task=next_step_task,
            )
            sys.stdout = old_stdout
            log_file.close()
        finally:
            file_tools.browser = original_browser

        logger.log_overview(f"🧠 Post-Execution Critique Result:\n{json.dumps(critique_result.model_dump(), indent=2)}")

        if critique_result.next_step_suggestion:
            next_step_suggestion = critique_result.next_step_suggestion

        if critique_result.decision == "revise" and critique_result.revised_prompt:
            logger.log_overview(f"🔄 Revising Step {step_number} based on critique")
            logger.mark_as_redundant("ManagerAgent", step_number=step_number)
            log_file = logger.get_log_file("ManagerAgent", step_number=step_number)
            old_stdout = sys.stdout
            sys.stdout = log_file
            revised = manager_agent.run(critique_result.revised_prompt)
            sys.stdout = old_stdout
            log_file.close()
            result_str = str(revised)
            logger.log_overview(f"📝 Revised Result for Step {step_number}: {result_str}")

        # Update the state with the final (possibly revised) result
        state_manager.update_state(result_str)
        result_key = f"{current_step.step_id}_{current_step.output_key}"
        logger.log_overview(f"✅ Result stored under key '{result_key}'")

    logger.log_overview("\n--- ✅ Plan Execution Finished ---")

    # 8. Display the final state
    final_state = state_manager.get_current_state()
    logger.log_overview("\nFinal Project State:")
    logger.log_overview(final_state.model_dump_json(indent=2))
