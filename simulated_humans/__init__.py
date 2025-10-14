"""Simulated Human Agents for Human-in-the-Loop Decision Making.

This module provides simulated human agents that can make decisions, approvals,
and provide feedback just like a real human would. These agents are designed to
work seamlessly with the SmolAgents orchestrator.

Usage:
    from simulated_humans import create_human_tools
    
    # Create human tools for the orchestrator
    human_tools = create_human_tools(model=your_model, use_simulated=True)
    
    # Add to orchestrator
    manager_agent = ToolCallingAgent(
        tools=[*other_tools, *human_tools],
        model=model,
    )

The orchestrator can then call these tools when it needs human decisions:
    - web_human: For web navigation and source selection decisions
    - code_human: For code review and development decisions  
    - file_human: For file organization and documentation decisions
"""

from .human_decision import HumanDecision
from .simulated_human_agent import SimulatedHumanAgent
from .human_tools import (
    HumanApprovalTool,
    WebHumanTool,
    CodeHumanTool,
    FileHumanTool,
    create_human_tools,
)
from .personas import (
    get_persona_prompt,
    get_temperature,
    get_default_preferences,
)

__all__ = [
    # Core types
    "HumanDecision",
    
    # Agents
    "SimulatedHumanAgent",
    
    # Tools
    "HumanApprovalTool",
    "WebHumanTool",
    "CodeHumanTool",
    "FileHumanTool",
    "create_human_tools",
    
    # Persona utilities
    "get_persona_prompt",
    "get_temperature",
    "get_default_preferences",
]

