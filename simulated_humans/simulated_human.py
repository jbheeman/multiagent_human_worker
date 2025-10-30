from smolagents import tool
from smolagents.models import OpenAIServerModel
import os
from smolagents import ToolCallingAgent
import json
from typing import Optional, List, Any
from pydantic import BaseModel, Field

from .simulated_prompts import (
    SYSTEM_MESSAGE_PLANNING_PHASE_SOFT,
    SYSTEM_MESSAGE_EXECUTION_PHASE_SOFT,
    SYSTEM_MESSAGE_PLANNING_PHASE_STRICT,
    SYSTEM_MESSAGE_EXECUTION_PHASE_STRICT,
    SYSTEM_MESSAGE_PLANNING_PHASE_NO_HINTS,
    SYSTEM_MESSAGE_EXECUTION_PHASE_NO_HINTS,
)

# Pydantic models for structured plan feedback
class PlanEdit(BaseModel):
    """A specific edit for a single step in the plan."""
    step_id: str = Field(..., description="The step_id of the plan step to be edited (e.g., 'step_1').")
    task: Optional[str] = Field(None, description="The revised task description for the step.")
    output_key: Optional[str] = Field(None, description="The revised output_key for the step.")

class PlanFeedback(BaseModel):
    """A container for plan feedback, including specific edits and overall comments."""
    edits: List[PlanEdit] = Field(default_factory=list, description="A list of specific edits to apply to the plan steps.")
    feedback: str = Field(..., description="Natural language feedback explaining the reasoning for the edits and providing overall guidance.")

@tool
def provide_plan_feedback(feedback_data: PlanFeedback) -> str:
    """
    Use this tool to provide structured feedback on a plan.
    Args:
        feedback_data: A JSON object containing a list of 'edits' and a 'feedback' string.
    Returns:
        A confirmation that feedback has been provided.
    """
    # This function's body is not critical as we just want the structured arguments.
    # The agent's output will be the tool call itself.
    return "Feedback provided."

class ExpertAgent:
    def __init__(self, name: str, model: OpenAIServerModel, side_info: str):
        description = """
        The expert agent is an LLM without any tools, instructed to interact with the orchestrator the way we expect a human would act.
        This expert has access to side information about each task, which includes a human-written plan to solve the task.
        It is not to reveal the ground-truth answer directly as the answer is usually found inside the human written plan. 
        Instead, it is prompted to guide Orchestrator to find the answer indirectly. 
        """
        self.name = name
        # The agent's only purpose is to call the provide_plan_feedback tool
        self.agent = ToolCallingAgent(name=name, model=model, description=description, tools=[provide_plan_feedback], max_steps=1)
        self.side_info = side_info
        self.description = description

    def review_plan(self, task: str, plan: str, side_info: str) -> dict[str, Any]:
        prompt = SYSTEM_MESSAGE_PLANNING_PHASE_SOFT.format(task=task, plan=plan, helpful_task_hints=side_info, answer="") # Assuming soft has answer
        print(f"Prompt: {prompt}")
        result = self.agent.run(prompt)
        
        if isinstance(result, str) and result.strip().lower() == "accept":
            return {"edits": [], "feedback": "accept"}

        # The result of a ToolCallingAgent run is a dictionary representing the tool call
        if isinstance(result, dict) and result.get("name") == "provide_plan_feedback":
            # The arguments are parsed into a dictionary; we need the nested 'feedback_data' object.
            return result["arguments"]["feedback_data"]

        # Fallback for unexpected responses
        print(f"Warning: ExpertAgent returned an unexpected format: {result}")
        return {"edits": [], "feedback": str(result) if result else "No feedback provided."}

    def help_with_execution(self, task: str, current_step: str, current_state: str) -> str:
        """
        Get help from the simulated human expert during task execution.
        
        Args:
            task: The overall task being worked on
            current_step: The current step that needs help
            current_state: The current state/results so far
            
        Returns:
            Helpful guidance to proceed with the current step
        """
        prompt = SYSTEM_MESSAGE_EXECUTION_PHASE_SOFT.format(
            task=task, 
            current_step=current_step, 
            current_state=current_state, 
            side_info=self.side_info
        )
        print(f"Co-execution prompt: {prompt}")
        result = self.agent.run(prompt)
        return str(result)



# Global variable to store the expert agent instance for the tool
_expert_agent_instance = None

def set_expert_agent(expert_agent: ExpertAgent):
    """Set the global expert agent instance for the tool"""
    global _expert_agent_instance
    _expert_agent_instance = expert_agent

@tool
def ask_human_expert_for_help(task: str, current_step: str, current_state: str) -> str:
    """
    Ask a simulated human expert for help during task execution when stuck or needing guidance.
    
    Args:
        task: The overall task being worked on
        current_step: The current step that needs help  
        current_state: The current state/results so far
        
    Returns:
        Helpful guidance from the simulated human expert to proceed with the current step
    """
    if _expert_agent_instance is None:
        return "Error: Expert agent not initialized. Please call set_expert_agent() first."
    
    return _expert_agent_instance.help_with_execution(task, current_step, current_state)


if __name__ == "__main__":
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ.get("NAUT_API_KEY", "your_default_api_key"),
    )

    expert = ExpertAgent(name="Expert", model=model, side_info="The capital of France is Paris.")
    expert.review_plan(task="What is the capital of France?", plan="Search the internet for the capital of France.", side_info=expert.side_info)
