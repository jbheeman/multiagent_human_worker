from smolagents import tool, CodeAgent
from smolagents.models import OpenAIServerModel
import os
from smolagents import ToolCallingAgent
import json

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

class ExpertAgent:
    def __init__(self, name: str, model: OpenAIServerModel, side_info: str):
        description = """
        The expert agent is an LLM without any tools, instructed to interact with the orchestrator the way we expect a human would act.
        This expert has access to side information about each task, which includes a human-written plan to solve the task.
        It is not to reveal the ground-truth answer directly as the answer is usually found inside the human written plan. 
        Instead, it is prompted to guide Orchestrator to find the answer indirectly. 
        """
        self.name = name
        self.agent = ToolCallingAgent(name=name, model=model, description=description, tools=[review_plan], max_steps=1)
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



if __name__ == "__main__":
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ.get("NAUT_API_KEY", "your_default_api_key"),
    )

    expert = ExpertAgent(name="Expert", model=model, side_info="The capital of France is Paris.")
    expert.review_plan(task="What is the capital of France?", plan="Search the internet for the capital of France.", side_info=expert.side_info)
