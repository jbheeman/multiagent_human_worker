from smolagents import tool
from smolagents.models import OpenAIServerModel
import os
import json
from typing import Optional, Dict, Any
from simulated_prompts import SIMULATED_HUMAN_PROMPT
from sideinformation import load_behavior_info
class SimulatedHumanAgent:
    def __init__(self, name: str, model: OpenAIServerModel, user_products: Dict[str, Any]):
        """
        Initialize the simulated human agent.
        
        Args:
            name: Name of the agent
            model: The LLM model to use
            user_products: Dictionary of products the user bought/clicked on (the "golden truth")
        """
        description = """
        The simulated human agent represents a real user with specific preferences and purchase history.
        It can answer questions about what the user likes, what products they would be interested in,
        and provide guidance based on the user's past behavior and preferences.
        It will not reveal any items from the Golden Truth to the orchestrator.
        """
        self.name = name
        self.model = model
        self.user_products = user_products  # The "golden truth" - products user actually interacted with
        
        # Create a simple agent that can answer questions (no tools needed, just direct responses)
        # We'll use a simple prompt-based approach instead of tool calling
        self.description = description

    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """
        Answer a question about the user's preferences based on their product history and persona.
        
        Args:
            question: The question being asked (e.g., "What does the human like?", "Would this item be liked by the human?")
            context: Optional context about what the orchestrator is currently looking at/considering
            
        Returns:
            An answer based on the user's product history and persona
        """
        # Format the user products for the prompt
        # products_summary = json.dumps(self.user_products, indent=2)
        
        prompt = SIMULATED_HUMAN_PROMPT.format(user_products=self.user_products, question=question)
        messages = [{"role": "user", "content": prompt}]
        response = self.model(messages)
        
        return response.content

# Global variable to store the simulated human agent instance
_simulated_human_agent_instance = None

def set_simulated_human_agent(simulated_human_agent: SimulatedHumanAgent):
    """Set the global simulated human agent instance for the tool"""
    global _simulated_human_agent_instance
    _simulated_human_agent_instance = simulated_human_agent

@tool
def ask_human_expert_for_help(question: str, context: Optional[str] = None) -> str:
    """
    Ask the simulated human about user preferences or whether they would like a product.
    
    Args:
        question: The question to ask (e.g., "What does the human like?", 
                  "Would this item be liked by the human?", 
                  "What categories of products does the user prefer?")
        context: Optional context about what product/item is being considered
        
    Returns:
        An answer from the simulated human based on their product history and persona
    """
    if _simulated_human_agent_instance is None:
        return "Error: Simulated human agent not initialized. Please call set_simulated_human_agent() first."
    
    return _simulated_human_agent_instance.answer_question(question, context)

if __name__ == "__main__":
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ.get("NAUT_API_KEY"),
    )
    user_id = "7f0c8207-6a6f-49cd-9d7e-17987cfafcb9"
    user_products = load_behavior_info(user_id)
    simulated_human_agent = SimulatedHumanAgent(name="Simulated Human", model=model, user_products=user_products)
    set_simulated_human_agent(simulated_human_agent)
    print(ask_human_expert_for_help(question="What does the human like?", context=""))