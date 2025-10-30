import os
import json
from pathlib import Path
import sys
import tempfile
import shutil

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    WebSearchTool,
)

# Tools
from websurfer_agent.web_surfer_tool import WebSurferTool
from file_surfer_agent.file_surfer_tool import FileSurferTool
from file_surfer_agent import file_tools
from file_surfer_agent.markdown_file_browser import MarkdownFileBrowser
from coder_agent.coder_tool import CoderTool
from common_tools.llm_chat_tool import create_llm_chat_tool
from common_tools.logger import Logger
from common_tools.sideinformation import get_side_info, load_side_info_from_metadata

# Planner and state manager
from planning_agent.planning_agent import PlanningAgent
from planning_agent.plan_state_manager import PlanStateManager
from critique_agent.critique_agent import PostExecutionCritiqueAgent, PreExecutionCritiqueAgent

#Expert agent
from simulated_humans.simulated_human import ExpertAgent, ask_human_expert_for_help, set_expert_agent

def main():
    """
    The main orchestration logic.
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

    # 1. Initialize the shared model
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ.get("NAUT_API_KEY", "your_default_api_key"),
    )
    
    WebSurferModel = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

    # 2. Initialize Agents and Tools
    planner = PlanningAgent(model=model)
    post_execution_critique = PostExecutionCritiqueAgent(model=model)
    pre_execution_critique = PreExecutionCritiqueAgent(model=model)
    web_surfer_tool = WebSurferTool(model=WebSurferModel)
    file_tools.browser = MarkdownFileBrowser(base_path=work_dir, viewport_size=4096)
    file_surfer_tool = FileSurferTool(
        model=model,
        base_path=work_dir,
        viewport_size=4096,
    )
    coder_tool = CoderTool(
        model=model,
        max_debug_rounds=3,
        use_local_executor=True,
        work_dir=Path(work_dir),
    )
    llm_tool = create_llm_chat_tool(model=model)
    
    side_info = load_side_info_from_metadata()
    expert_agent = ExpertAgent(name="Expert", model=model, side_info=side_info)
    set_expert_agent(expert_agent)

    manager_agent = ToolCallingAgent(
        tools=[web_surfer_tool, file_surfer_tool, coder_tool, llm_tool, WebSearchTool(), ask_human_expert_for_help],
        model=model,
    )
    

    # --- Start of the Workflow ---

    # 4. Get the user's high-level goal
    user_goal = (
        "Write a multi chapter fantasy tale about leaves on a tree. Save each chapter to the file chapter_name.txt"
    )
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

    # Expert review of the plan
    logger.log_overview("\n--- 🧑‍🏫 Expert Review of the Plan ---")
    log_file = logger.get_log_file("ExpertAgent")
    old_stdout = sys.stdout
    sys.stdout = log_file
    simhuman_review = expert_agent.review_plan(user_goal, plan.model_dump_json(indent=2), side_info)
    sys.stdout = old_stdout
    log_file.close()
    logger.log_overview(f"\n🔍 Expert Review: {simhuman_review}")

    # Refine the plan based on the expert's review
    replanner_agent = PlanningAgent(model=model)
    replan_plan = replanner_agent.refine(user_goal, plan.model_dump_json(indent=2), simhuman_review)
    logger.log_overview(f"\n🔍 Re-planned Plan: {replan_plan}")


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
        llm_tool = create_llm_chat_tool(model=model, plan_state=current_state)
        
        # Recreate manager_agent with updated tools
        manager_agent = ToolCallingAgent(
            tools=[web_surfer_tool, file_surfer_tool, coder_tool, llm_tool, WebSearchTool(), ask_human_expert_for_help],
            model=model,
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

--- File Writing Guidelines ---
- Do not use generic filenames like `results.txt` for intermediate files. Use descriptive names that reflect the content (e.g., `leaf_research_notes.txt`).
- Only write to a file when the task explicitly requires it.
- For intermediate results, it is often better to return the text directly instead of writing to a file.

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


if __name__ == "__main__":
    main()