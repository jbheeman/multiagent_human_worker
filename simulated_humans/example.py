"""Example usage of simulated human agents.

This script demonstrates how to use the simulated human tools
both directly and through the orchestrator.
"""

import os
import sys
import json

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smolagents import OpenAIServerModel, ToolCallingAgent
from simulated_humans import (
    create_human_tools,
    SimulatedHumanAgent,
    WebHumanTool,
    CodeHumanTool,
    FileHumanTool,
)


def example_direct_agent_usage():
    """Example: Using SimulatedHumanAgent directly."""
    print("=" * 80)
    print("EXAMPLE 1: Using SimulatedHumanAgent Directly")
    print("=" * 80)
    
    # Create model
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ["NAUT_API_KEY"],
    )
    
    # Create a web human agent
    web_human = SimulatedHumanAgent(model=model, role="web")
    
    # Ask for a decision
    decision = web_human.decide(
        task="Choose the best source for Python asyncio documentation",
        context="I need to learn about Python's asyncio library for concurrent programming.",
        options=[
            "https://docs.python.org/3/library/asyncio.html",
            "https://medium.com/some-blog/asyncio-tutorial",
            "https://stackoverflow.com/questions/tagged/asyncio",
        ],
        needs="preference",
    )
    
    print(f"\nTask: Choose Python asyncio documentation source")
    print(f"Decision: {decision['decision']}")
    print(f"Message: {decision['message']}")
    if decision.get('revisions'):
        print(f"Revisions: {json.dumps(decision['revisions'], indent=2)}")
    print()


def example_tool_usage():
    """Example: Using human tools directly."""
    print("=" * 80)
    print("EXAMPLE 2: Using Human Tools Directly")
    print("=" * 80)
    
    # Create model
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ["NAUT_API_KEY"],
    )
    
    # Create a code human tool
    code_human_tool = CodeHumanTool(model=model)
    
    # Ask for approval
    result = code_human_tool.forward(
        task="Review file deletion operation",
        context="About to delete all files in /tmp/cache/* directory. This is a cleanup operation.",
        needs="approval",
    )
    
    print(f"\nTask: Review file deletion")
    print(f"Result:\n{result}")
    print()


def example_with_orchestrator():
    """Example: Using human tools with the orchestrator."""
    print("=" * 80)
    print("EXAMPLE 3: Human Tools in Orchestrator")
    print("=" * 80)
    
    # Create model
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ["NAUT_API_KEY"],
    )
    
    # Create human tools
    human_tools = create_human_tools(model=model, use_simulated=True)
    
    # Create a simple orchestrator with only human tools
    agent = ToolCallingAgent(
        tools=human_tools,
        model=model,
    )
    
    # Ask the orchestrator to get human approval for something
    result = agent.run(
        "I need to decide whether to use SQLite or PostgreSQL for a small web app. "
        "The app will have ~1000 users initially but might grow. "
        "Ask the code human for their preference."
    )
    
    print(f"\nOrchestrator's task: Choose database for web app")
    print(f"Result:\n{result}")
    print()


def example_code_review():
    """Example: Code review scenario."""
    print("=" * 80)
    print("EXAMPLE 4: Code Review Scenario")
    print("=" * 80)
    
    # Create model
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ["NAUT_API_KEY"],
    )
    
    # Create code human
    code_human = SimulatedHumanAgent(model=model, role="code")
    
    # Present code for review
    code_snippet = '''
def process_user_input(user_input):
    # Process user input
    result = eval(user_input)
    return result
'''
    
    decision = code_human.decide(
        task="Review this code for security issues",
        context=f"Here's a function that processes user input:\n{code_snippet}",
        needs="critique",
    )
    
    print(f"\nTask: Security review of code")
    print(f"Decision: {decision['decision']}")
    print(f"Message: {decision['message']}")
    if decision.get('revisions'):
        print(f"Suggested revisions: {json.dumps(decision['revisions'], indent=2)}")
    print()


def example_file_naming():
    """Example: File naming and organization."""
    print("=" * 80)
    print("EXAMPLE 5: File Naming Scenario")
    print("=" * 80)
    
    # Create model
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ["NAUT_API_KEY"],
    )
    
    # Create file human
    file_human = SimulatedHumanAgent(model=model, role="file")
    
    decision = file_human.decide(
        task="Review file naming for a new Python package",
        context="Creating a new package with these files: MainHelper.py, utils_FINAL.py, test.py",
        needs="critique",
    )
    
    print(f"\nTask: Review file naming")
    print(f"Decision: {decision['decision']}")
    print(f"Message: {decision['message']}")
    if decision.get('revisions'):
        print(f"Suggested revisions: {json.dumps(decision['revisions'], indent=2)}")
    print()


if __name__ == "__main__":
    print("\n🤖 Simulated Human Agents - Examples\n")
    
    try:
        # Run examples
        example_direct_agent_usage()
        example_tool_usage()
        example_code_review()
        example_file_naming()
        
        # Note: Orchestrator example requires actual agent execution
        # which may take longer, so it's commented out by default
        # Uncomment to run:
        # example_with_orchestrator()
        
        print("=" * 80)
        print("✅ All examples completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

