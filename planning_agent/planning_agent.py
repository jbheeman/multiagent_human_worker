from __future__ import annotations # Keep this for good practice with Pydantic
import os
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
    ActionStep, # Import the correct step type
)
from typing import List

# Import our structured data models and prompts
from .plan_models import Plan, PlanStep
from .prompts import PLANNER_SYSTEM_PROMPT

@tool
def create_plan(steps: List[PlanStep]) -> str:
    """
    Use this function to create and finalize the execution plan.
    
    Args:
        steps: The full, sequential list of steps for the plan.
    """
    return "Plan created successfully."



class PlanningAgent:
    """
    A self-contained agent that takes a user's goal and decomposes it into a structured,
    executable plan by calling a specific tool.
    """
    def __init__(self, model: OpenAIServerModel):
        """
        Initializes the PlanningAgent.

        Args:
            model: The language model to use for planning.
        """
        self._system_prompt = PLANNER_SYSTEM_PROMPT
            
        self._agent = ToolCallingAgent(
            tools=[create_plan],
            model=model,
            max_steps=1, # Force the agent to stop after its one required action.
            instructions=self._system_prompt,  # Pass system prompt as instructions
        )

    def run(self, user_goal: str) -> Plan:
        """
        Runs the agent to generate a plan for a given user goal.

        Args:
            user_goal: The high-level objective from the user.

        Returns:
            A Plan object containing the structured steps.
        """
        
        print(f"🤖 PlanningAgent: Devising a plan for goal: '{user_goal}'")
        
        initial_context = (
            f"{self._system_prompt}\n\n"
            f"--- CURRENT TASK ---\n"
            f"USER_GOAL: {user_goal}\n\n"
            f"Begin your work. You must call the `create_plan` function with the generated plan."
        )
        
        # We run the agent and ignore its direct return value.
        self._agent.run(initial_context)

        # We inspect the agent's memory to find the tool call.
        tool_calls = None
        # Iterate through the steps recorded in the agent's memory
        for step in self._agent.memory.steps:
            # Find the step that contains the tool call action
            if isinstance(step, ActionStep) and step.tool_calls:
                tool_calls = step.tool_calls
                break

        # This parsing logic will now succeed because we are using attribute access (dot notation).
        if not tool_calls or tool_calls[0].name != 'create_plan':
            raise ValueError("Planner failed to find a valid `create_plan` tool call in the agent's history.")

        # FINAL FIX: Access arguments using .arguments, which is the correct attribute name.
        plan_data = tool_calls[0].arguments
        
        plan = Plan(**plan_data)
        
        print(f"✅ PlanningAgent: Plan created successfully with {len(plan.steps)} steps.")
        return plan

    def refine(self, user_goal: str, prior_plan_summary: str, reviewed_plan: str) -> Plan:
        """
        Re-generate a plan that incorporates any edits/critique from an expert review.
        The model must call `create_plan` with a complete, consistent plan JSON.
        """
      
        prompt = (
            f"{self._system_prompt}\n\n"
            "You will refine a prior plan using edits/critique from an expert review. "
            "Return a *complete* plan via a single call to `create_plan`.\n\n"
            "--- USER GOAL ---\n"
            f"{user_goal}\n\n"
            "--- PRIOR PLAN SUMMARY ---\n"
            f"{prior_plan_summary}\n\n"
            "--- EDITS/CRITIQUE (must apply) ---\n"
            f"{reviewed_plan}\n\n"
            "Call `create_plan` with the full plan JSON."
        )

        
        self._agent.run(prompt)

        # Extract tool call as in your `run()`
        tool_calls = None
        for step in self._agent.memory.steps:
            if isinstance(step, ActionStep) and step.tool_calls:
                tool_calls = step.tool_calls
                break
        if not tool_calls or tool_calls[0].name != 'create_plan':
            raise ValueError("Refine: no `create_plan` call found.")

        plan_data = tool_calls[0].arguments
        return Plan(**plan_data)

   

# Example usage to test the planner independently
if __name__ == "__main__":
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ.get("NAUT_API_KEY", "your_default_api_key"),
    )

    planner = PlanningAgent(model=model)

    sample_goal = (
        "Find information on the newest trends in AI by visiting the website of NVIDIA, "
        "and write a file in the current directory called trends.txt that is a report of your findings."
    )

    try:
        generated_plan = planner.run(sample_goal)

        print("\n" + "="*50)
        print("GENERATED PLAN")
        print("="*50)
        for step in generated_plan.steps:
            print(f"Step {step.step_id}: {step.task}")
        print("="*50)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

