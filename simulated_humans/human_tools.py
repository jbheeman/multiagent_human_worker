"""SmolAgents Tool wrappers for simulated human agents.

These tools allow the orchestrator to call simulated humans for approvals,
preferences, critiques, and other human-in-the-loop decisions.
"""

import json
from typing import Optional, List, Dict, Any, Literal

from smolagents import Tool

try:
    from .simulated_human_agent import SimulatedHumanAgent
except ImportError:
    from simulated_human_agent import SimulatedHumanAgent


class HumanApprovalTool(Tool):
    """A tool that delegates decisions to a simulated human agent.
    
    This tool can be used for approvals, preferences, critiques, and other
    human-in-the-loop decision points. The simulated human thinks through
    the decision and returns a structured response.
    """
    
    # These will be set by subclasses or instances
    name = "human_approval"
    description = "Delegate a decision to a simulated human for approval, preference, or critique."
    
    inputs = {
        "phase": {
            "type": "string",
            "description": "Gate being triggered: 'plan' (review strategy before execution), 'guard' (approve risky actions like delete/overwrite/credentials), 'help' (stuck after failures), or 'verify' (check final result quality). Default: 'guard'",
            "nullable": True,
        },
        "task": {
            "type": "string",
            "description": "One-line goal: what needs a decision (e.g., 'choose best source for X', 'review this file deletion')",
        },
        "context": {
            "type": "string",
            "description": "What was tried/observed so far: plan state, code snippets, diffs, file paths, errors encountered, etc.",
            "nullable": True,
        },
        "needs": {
            "type": "string",
            "description": "Decision type needed: 'approval', 'clarification', 'critique', 'preference', or 'takeover'. Default: 'approval'",
            "nullable": True,
        },
        "options": {
            "type": "string",
            "description": "JSON list of candidate choices as strings (e.g., links, files, strategies). Pass as JSON string.",
            "nullable": True,
        },
        "constraints": {
            "type": "string",
            "description": "JSON object with constraints (e.g., time limits, risk levels, cost limits). Pass as JSON string. Example: '{\"max_cost\": 100, \"risk_level\": \"low\"}'",
            "nullable": True,
        },
        "hints": {
            "type": "string",
            "description": "JSON object with additional hints/information (e.g., credentials, user preferences, session state). Pass as JSON string. Example: '{\"username\": \"john\", \"password\": \"secret\", \"preferences\": {\"language\": \"en\"}}'",
            "nullable": True,
        },
    }
    output_type = "string"
    
    def __init__(
        self,
        model,
        role: Literal["web", "code", "file"],
        name: Optional[str] = None,
        **kwargs
    ):
        """Initialize the HumanApprovalTool.
        
        Args:
            model: The language model to use for the simulated human
            role: The role of the simulated human (web, code, or file)
            name: Optional custom name for the tool
            **kwargs: Additional arguments
        """
        super().__init__()
        self.role = role
        
        # Set the tool name and description based on role
        if name:
            self.name = name
        else:
            self.name = f"{role}_human"
        
        self.description = self._get_role_description(role)
        
        # Initialize the simulated human agent
        self._agent = SimulatedHumanAgent(model=model, role=role)
    
    def _get_role_description(self, role: str) -> str:
        """Get role-specific tool description with cost-aware gates."""
        descriptions = {
            "web": (
                "Consult the web specialist for web navigation and research decisions. "
                "\n\n**COST-AWARE POLICY:**"
                "\n- **HARD RULE (phase='guard')**: Call BEFORE any action involving credentials, "
                "payment, account creation, form submission with personal data, or potential TOS violations "
                "(aggressive scraping, bypassing CAPTCHAs, etc.)."
                "\n- **SOFT RULE (phase='help')**: Call after ≥2 failures on the same task (blocked sites, "
                "CAPTCHAs, can't find info, etc.)."
                "\n- **SOFT RULE (phase='plan')**: Call if confidence < 0.6 or task is novel/unfamiliar."
                "\n\n**Usage:** Prefer direct tool use first. Escalate to human only when policy requires it."
            ),
            "code": (
                "Consult the code reviewer for code changes and development decisions. "
                "\n\n**COST-AWARE POLICY:**"
                "\n- **HARD RULE (phase='guard')**: Call BEFORE destructive operations (delete files, "
                "overwrite without backup, drop tables, force push, reset --hard), executing untrusted code, "
                "or making security-sensitive changes (auth, encryption, credentials)."
                "\n- **SOFT RULE (phase='help')**: Call after ≥2 test failures or build errors on same code."
                "\n- **SOFT RULE (phase='verify')**: Call if aggregate test coverage < 0.7 or confidence < 0.6."
                "\n\n**Usage:** Write/edit code directly. Only consult human when policy requires approval."
            ),
            "file": (
                "Consult the documentation owner for file organization and structure decisions. "
                "\n\n**COST-AWARE POLICY:**"
                "\n- **HARD RULE (phase='guard')**: Call BEFORE deleting/moving critical files (configs, docs, "
                "data), bulk file operations (>10 files), or distributing files externally."
                "\n- **SOFT RULE (phase='plan')**: Call when setting up new directory structures or major reorganizations."
                "\n- **SOFT RULE (phase='verify')**: Call to review documentation completeness before release."
                "\n\n**Usage:** Read/navigate files directly. Only consult human when policy requires approval."
            ),
        }
        return descriptions.get(role, "Consult a simulated human for decisions.")
    
    def forward(
        self,
        task: str,
        phase: Optional[str] = None,
        context: Optional[str] = None,
        needs: Optional[str] = None,
        options: Optional[str] = None,
        constraints: Optional[str] = None,
        hints: Optional[str] = None,
    ) -> str:
        """Execute the human decision request.
        
        Args:
            task: What needs a decision
            phase: Gate being triggered (plan/guard/help/verify)
            context: What was tried/observed so far
            needs: Type of decision needed
            options: JSON string of candidate choices (will be parsed)
            constraints: JSON string of constraints (will be parsed)
            hints: JSON string of hints/additional info (credentials, preferences, etc.)
            
        Returns:
            JSON string of the HumanDecision
        """
        # Parse options if provided
        parsed_options = None
        if options:
            try:
                parsed_options = json.loads(options)
                if not isinstance(parsed_options, list):
                    parsed_options = [str(parsed_options)]
            except json.JSONDecodeError:
                # If not valid JSON, treat as a single option
                parsed_options = [options]
        
        # Parse constraints if provided
        parsed_constraints = None
        if constraints:
            try:
                parsed_constraints = json.loads(constraints)
                if not isinstance(parsed_constraints, dict):
                    parsed_constraints = {"info": str(parsed_constraints)}
            except json.JSONDecodeError:
                # If not valid JSON, treat as a single constraint
                parsed_constraints = {"info": constraints}
        
        # Parse hints if provided
        parsed_hints = None
        if hints:
            try:
                parsed_hints = json.loads(hints)
                if not isinstance(parsed_hints, dict):
                    parsed_hints = {"info": str(parsed_hints)}
            except json.JSONDecodeError:
                # If not valid JSON, treat as a single hint
                parsed_hints = {"info": hints}
        
        # Validate phase
        valid_phases = ["plan", "guard", "help", "verify"]
        if phase and phase not in valid_phases:
            phase = "guard"
        
        # Validate needs
        valid_needs = ["approval", "clarification", "critique", "preference", "takeover"]
        if needs and needs not in valid_needs:
            needs = "approval"
        
        # Call the simulated human agent
        try:
            decision = self._agent.decide(
                task=task,
                phase=phase or "guard",  # type: ignore
                context=context,
                needs=needs or "approval",  # type: ignore
                options=parsed_options,
                constraints=parsed_constraints,
                hints=parsed_hints,
            )
            
            # Return as JSON string
            return json.dumps(decision, indent=2)
            
        except Exception as e:
            # Return error as a denial
            error_decision = {
                "decision": "deny",
                "message": f"Error in human decision process: {str(e)}",
                "revisions": None,
            }
            return json.dumps(error_decision, indent=2)


class WebHumanTool(HumanApprovalTool):
    """Specialized tool for web-related human decisions."""
    
    def __init__(self, model, **kwargs):
        super().__init__(model=model, role="web", name="web_human", **kwargs)


class CodeHumanTool(HumanApprovalTool):
    """Specialized tool for code-related human decisions."""
    
    def __init__(self, model, **kwargs):
        super().__init__(model=model, role="code", name="code_human", **kwargs)


class FileHumanTool(HumanApprovalTool):
    """Specialized tool for file-related human decisions."""
    
    def __init__(self, model, **kwargs):
        super().__init__(model=model, role="file", name="file_human", **kwargs)


def create_human_tools(
    model,
    use_simulated: bool = True,
    roles: Optional[List[Literal["web", "code", "file"]]] = None,
) -> List[Tool]:
    """Factory function to create human tools for the orchestrator.
    
    Args:
        model: The language model to use for simulated humans
        use_simulated: If True, use simulated humans; if False, use real humans (not yet implemented)
        roles: Which roles to create tools for (default: all three)
        
    Returns:
        List of Tool instances for the orchestrator
    """
    if not use_simulated:
        raise NotImplementedError(
            "Real human tools are not yet implemented. Set use_simulated=True."
        )
    
    if roles is None:
        roles = ["web", "code", "file"]
    
    tools = []
    for role in roles:
        if role == "web":
            tools.append(WebHumanTool(model=model))
        elif role == "code":
            tools.append(CodeHumanTool(model=model))
        elif role == "file":
            tools.append(FileHumanTool(model=model))
    
    return tools

