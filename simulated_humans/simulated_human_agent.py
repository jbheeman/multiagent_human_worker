"""Simulated Human Agent - An LLM-powered agent that simulates human decision-making.

This agent thinks through decisions and returns structured HumanDecision outputs.
It can be used as a multi-step reasoning agent internally while providing a
single-round response to the orchestrator.
"""

import json
from typing import Literal, Optional, List, Dict, Any

from smolagents import ToolCallingAgent

try:
    from .personas import get_persona_prompt, get_temperature, get_default_preferences
    from .human_decision import HumanDecision
except ImportError:
    from personas import get_persona_prompt, get_temperature, get_default_preferences
    from human_decision import HumanDecision


class SimulatedHumanAgent:
    """A simulated human that thinks through decisions and returns structured decisions.
    
    This agent is itself a multi-step reasoning agent that can think through
    problems internally, but returns a single HumanDecision to the caller.
    It maintains preferences in memory during the session.
    """
    
    def __init__(
        self,
        model,
        role: Literal["web", "code", "file"],
        name: Optional[str] = None,
        max_thinking_steps: int = 3,
    ):
        """Initialize the simulated human agent.
        
        Args:
            model: The language model to use (e.g., OpenAIServerModel)
            role: The role of this human (web, code, or file)
            name: Optional custom name for the agent
            max_thinking_steps: Maximum internal reasoning steps
        """
        self.role = role
        self.name = name or f"{role.capitalize()}Human"
        self.model = model
        self.max_thinking_steps = max_thinking_steps
        
        # Get persona configuration
        self.system_prompt = get_persona_prompt(role)
        self.temperature = get_temperature(role)
        self.preferences = get_default_preferences(role)
        
        # Session memory (decisions made this session)
        self.decision_history: List[Dict[str, Any]] = []
    
    def decide(
        self,
        task: str,
        phase: Literal["plan", "guard", "help", "verify"] = "guard",
        context: Optional[str] = None,
        needs: Literal["approval", "clarification", "critique", "preference", "takeover"] = "approval",
        options: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        hints: Optional[Dict[str, Any]] = None,
    ) -> HumanDecision:
        """Make a decision on the given task.
        
        The agent thinks through the decision internally (multi-step reasoning)
        and returns a structured HumanDecision.
        
        Args:
            task: One-line goal: what needs a decision
            phase: Gate being triggered (plan/guard/help/verify)
            context: What was tried/observed so far
            needs: Type of decision needed
            options: Candidate choices (links, files, strategies)
            constraints: Optional constraints (time, risk, cost limits)
            hints: Additional hints/info (credentials, user prefs, session state, etc.)
            
        Returns:
            HumanDecision dict with decision, message, and optional revisions
        """
        # Build the request message
        request_parts = [
            f"**Task:** {task}",
            f"**Phase:** {phase}",
            f"**Decision Type:** {needs}",
        ]
        
        if context:
            request_parts.append(f"**Context:**\n{context}")
        
        if options:
            options_str = "\n".join([f"  {i+1}. {opt}" for i, opt in enumerate(options)])
            request_parts.append(f"**Options:**\n{options_str}")
        
        if constraints:
            constraints_str = json.dumps(constraints, indent=2)
            request_parts.append(f"**Constraints:**\n{constraints_str}")
        
        if hints:
            hints_str = json.dumps(hints, indent=2)
            request_parts.append(f"**Available Hints/Info:**\n{hints_str}")
        
        if self.preferences:
            prefs_str = json.dumps(self.preferences, indent=2)
            request_parts.append(f"**Your Preferences:**\n{prefs_str}")
        
        request_parts.append(
            "\n**Instructions:**\n"
            "Think through this decision carefully. Consider the implications, "
            "risks, and benefits. Then provide your decision as a JSON object.\n\n"
            "Your response MUST be a valid JSON object with this structure:\n"
            "{\n"
            '  "decision": "approve|deny|revise|suggest|verify_ok|verify_fail",\n'
            '  "message": "your brief rationale",\n'
            '  "revisions": {"key": "value"} or null  // optional corrections/suggestions\n'
            "}\n\n"
            "**Decision types by phase:**\n"
            "- plan/guard: approve (go ahead), deny (block it), revise (change the approach)\n"
            "- help: suggest (with revisions containing next steps or hints)\n"
            "- verify: verify_ok (looks good), verify_fail (needs work, include issues in message)\n"
        )
        
        request_message = "\n\n".join(request_parts)
        
        # Prepare messages for the model
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": request_message},
        ]
        
        # Call the model to get the decision
        try:
            # Use the model directly (smolagents model interface)
            response = self.model(messages, temperature=self.temperature)
            
            # Extract response content
            if isinstance(response, str):
                response_text = response
            elif isinstance(response, dict):
                response_text = response.get("content", str(response))
            elif hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse the JSON response
            decision = self._parse_decision(response_text)
            
            # Store in decision history
            self.decision_history.append({
                "task": task,
                "needs": needs,
                "decision": decision,
            })
            
            return decision
            
        except Exception as e:
            # If something goes wrong, return a denial with error info
            return HumanDecision(
                decision="deny",
                message=f"Error processing decision: {str(e)}",
                revisions=None,
            )
    
    def _parse_decision(self, response_text: str) -> HumanDecision:
        """Parse the LLM response into a HumanDecision.
        
        Attempts to extract JSON from the response, handling various formats.
        
        Args:
            response_text: The raw response from the LLM
            
        Returns:
            Parsed HumanDecision dict
        """
        # Try to find JSON in the response
        # The LLM might wrap it in markdown code blocks or add extra text
        
        # First, try to extract from code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            # Try to find JSON object directly
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
            else:
                json_str = response_text
        
        # Parse the JSON
        try:
            data = json.loads(json_str)
            
            # Validate and construct HumanDecision
            valid_decisions = ["approve", "deny", "revise", "suggest", "verify_ok", "verify_fail"]
            decision_value = data.get("decision")
            
            # Fallback: if decision is missing or invalid, default to deny
            if not decision_value or decision_value not in valid_decisions:
                return HumanDecision(
                    decision="deny",
                    message=data.get("message", "Invalid or missing decision field - defaulting to deny"),
                    revisions=None,
                )
            
            return HumanDecision(
                decision=decision_value,  # type: ignore
                message=data.get("message", "No message provided"),
                revisions=data.get("revisions"),
            )
        
        except json.JSONDecodeError as e:
            # If we can't parse JSON, return a denial
            return HumanDecision(
                decision="deny",
                message=f"Failed to parse decision (invalid JSON): {str(e)}",
                revisions=None,
            )
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get the history of decisions made this session.
        
        Returns:
            List of decision records with task, needs, and decision
        """
        return self.decision_history.copy()
    
    def update_preferences(self, new_preferences: Dict[str, Any]) -> None:
        """Update the agent's preferences.
        
        Args:
            new_preferences: New preferences to merge with existing ones
        """
        self.preferences.update(new_preferences)
    
    def clear_history(self) -> None:
        """Clear the decision history."""
        self.decision_history.clear()
    
    def __repr__(self) -> str:
        return f"SimulatedHumanAgent(role={self.role}, name={self.name})"

