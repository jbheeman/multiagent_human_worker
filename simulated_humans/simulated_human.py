from smolagents import tool, CodeAgent
from smolagents.models import OpenAIServerModel
import os
from smolagents import ToolCallingAgent
import json
from typing import Optional

SYSTEM_MESSAGE_PLANNING_PHASE_SOFT = """
**Context:**
You are tasked with role playing as a human user who is interacting with an AI to solve a task for you. 

The task is: {task} 

The AI will provide a plan for the task in the past messages.

The side information is: {side_info}

The plan is: {plan}

**INSTRUCTIONS:**

Review the plan against your side information and provide structured feedback in this EXACT format:

# critique = {{"edits": [...], "risks": [...], "hints": [...], "missing_preconditions": [...]}}

Where:
- edits: Specific changes needed to improve the plan
- risks: Potential issues or problems with the current plan
- hints: Helpful suggestions that don't reveal the answer directly
- missing_preconditions: Important steps or information the plan is missing

Do NOT reveal the ground truth answer directly. Instead, provide guidance through hints and edits.
"""

SYSTEM_MESSAGE_EXECUTION_PHASE = """
**Context:**
You are a knowledgeable human user helping an AI agent execute a task. The AI is currently stuck or needs guidance on a specific step.

The overall task is: {task}

The current step being executed is: {current_step}

The current state/results so far: {current_state}

The side information available to you: {side_info}

**INSTRUCTIONS:**

Provide helpful guidance to help the AI agent proceed with the current step. Your response should:

1. **Analyze the current situation** - What has been done so far and what's the current challenge?
2. **Provide specific guidance** - Give concrete suggestions for how to proceed
3. **Offer hints** - Share relevant information that helps without revealing the final answer
4. **Suggest next actions** - Recommend specific tools or approaches to try

**IMPORTANT:** 
- Do NOT reveal the ground truth answer directly
- Focus on helping the AI learn and discover the answer through guidance
- Be encouraging and supportive in your tone
- Provide actionable advice that moves the task forward

Format your response as helpful guidance that an AI agent can follow.
"""



@tool
def review_plan(plan: str) -> str:
    """
    Use this function to review the plan and provide feedback on the plan.
    Args:
        plan: The plan to review.
    Returns:
        The feedback on the plan.
    """
    return "I have reviewed the plan and provided feedback on the plan."

@tool
def help_with_execution(task: str, current_step: str, current_state: str, side_info: str) -> str:
    """
    Use this function to get help from a simulated human expert during task execution.
    Args:
        task: The overall task being worked on.
        current_step: The current step that needs help.
        current_state: The current state/results so far.
        side_info: Side information available to the expert.
    Returns:
        Helpful guidance to proceed with the current step.
    """
    return "I have provided guidance for the current execution step."

class ExpertAgent:
    def __init__(self, name: str, model: OpenAIServerModel, side_info: str):
        description = """
        The expert agent is an LLM without any tools, instructed to interact with the orchestrator the way we expect a human would act.
        This expert has access to side information about each task, which includes a human-written plan to solve the task.
        It is not to reveal the ground-truth answer directly as the answer is usually found inside the human written plan. 
        Instead, it is prompted to guide Orchestrator to find the answer indirectly. 
        """
        self.name = name
        self.agent = ToolCallingAgent(name=name, model=model, description=description, tools=[review_plan, help_with_execution], max_steps=1)
        self.side_info = side_info
        self.description = description

    def review_plan(self, task: str, plan: str, side_info: str) -> str:
        prompt = SYSTEM_MESSAGE_PLANNING_PHASE_SOFT.format(task=task, plan=plan, side_info=side_info)
        print(f"Prompt: {prompt}")
        result = self.agent.run(prompt)
      
        try:
            if "# critique = " in str(result):
                critique_line = [line for line in str(result).split('\n') if "# critique = " in line][0]
                critique_json = critique_line.replace("# critique = ", "").strip()
                critique_dict = json.loads(critique_json)
                return critique_dict
        except (json.JSONDecodeError, IndexError):
            pass


        
        # Fallback: return the raw result
        return str(result)

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
        prompt = SYSTEM_MESSAGE_EXECUTION_PHASE.format(
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
