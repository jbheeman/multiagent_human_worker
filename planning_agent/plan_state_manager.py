from __future__ import annotations
from .plan_models import PlanState, Plan

class PlanStateManager:
    """
    Manages the state of a plan's execution. It holds the AgentState
    and updates it based on the results of each step. It does not
    contain any execution logic itself.
    """
    def __init__(self, user_goal: str, plan: Plan):
        """
        Initializes the executor and creates the initial agent state.

        Args:
            user_goal: The original high-level goal from the user.
            plan: The structured Plan object from the PlanningAgent.
        """
        self.state = PlanState(user_goal=user_goal, plan=plan)
        print("✅ PlanExecutor: State initialized.")

    def get_current_state(self) -> PlanState:
        """Returns the current state of the execution."""
        return self.state

    def get_current_step_task(self) -> str | None:
        """Returns the task description for the current step, or None if finished."""
        if self.is_finished():
            return None
        return self.state.plan.steps[self.state.current_step_index].task

    def get_next_step_task(self) -> str | None:
        """Returns the task description for the next step, or None if it's the last step."""
        if self.state.current_step_index + 1 >= len(self.state.plan.steps):
            return None
        return self.state.plan.steps[self.state.current_step_index + 1].task
    
    def is_finished(self) -> bool:
        """Checks if all steps in the plan have been completed."""
        return self.state.current_step_index >= len(self.state.plan.steps)

    def update_state(self, step_result: str):
        """
        Updates the state with the result of a completed step and increments the step counter.

        Args:
            step_result: The string output from the manager_agent for the completed step.
        """
        current_index = self.state.current_step_index
        current_step = self.state.plan.steps[current_index]
        print(f"▶️ PlanExecutor: Updating state for completed step {current_index + 1} ('{current_step.task}')...")

        # Programmatically create the key by combining the step ID and the semantic key from the plan.
        result_key = f"{current_step.step_id}_{current_step.output_key}"
        self.state.results[result_key] = step_result
        
        # Also update the result directly in the plan step object for better tracking
        self.state.plan.steps[current_index].result = step_result

        # Increment the step index to move to the next step
        self.state.current_step_index += 1
        print(f"✅ PlanExecutor: State updated. Result stored under key '{result_key}'. Moved to step {self.state.current_step_index + 1}.")
        
