"""
Simple demo script showing CoderAgent usage.

This is a minimal example to quickly test the CoderAgent.

Usage:
    export NAUT_API_KEY="your_api_key_here"
    python3 demo.py
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import LiteLLMModel
from coder_agent.coder_agent import CoderAgent


def main():
    # Configure API
    api_key = os.environ.get("NAUT_API_KEY")
    if not api_key:
        print("ERROR: NAUT_API_KEY environment variable not set.")
        print("Please set it: export NAUT_API_KEY='your_key'")
        return
    
    os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize
    print("Initializing CoderAgent...")
    model = LiteLLMModel(model_id="openai/llama3")
    
    # Use context manager for automatic cleanup
    with CoderAgent(model=model, max_debug_rounds=3) as coder:
        print("CoderAgent ready!\n")
        
        # Simple task
        print("="*60)
        print("Task: Calculate fibonacci number at position 10")
        print("="*60)
        
        result = coder.run("Calculate the 10th Fibonacci number using Python")
        
        print(f"\nResult:\n{result}")
        print("\n" + "="*60)


if __name__ == "__main__":
    main()

