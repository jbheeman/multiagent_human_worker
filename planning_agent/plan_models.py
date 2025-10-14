from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class PlanStep(BaseModel):
    """A single step in an execution plan."""
    step_id: int = Field(..., description="A sequential, 1-indexed identifier for the step.")
    task: str = Field(..., description="A clear and concise description of the step's objective.")
    # NEW: The key under which the step's result will be stored.
    output_key: str = Field(..., description="A descriptive snake_case key for storing the step's output artifact.")
    result: Optional[str] = None

class Plan(BaseModel):
    """Represents a structured, multi-step plan to achieve a goal."""
    steps: List[PlanStep] = Field(default_factory=list)

class PlanState(BaseModel):
    """The central state object that holds all information for a given task."""
    user_goal: str
    plan: Optional[Plan] = None
    results: Dict[str, str] = Field(default_factory=dict)
    current_step_index: int = 0

