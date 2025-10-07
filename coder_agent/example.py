"""
An example script demonstrating how to use the CoderAgent.

This script shows how to use the CoderAgent to solve tasks using code execution.

Usage:
    export NAUT_API_KEY="your_api_key_here"
    python3 example.py
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import LiteLLMModel
from coder_agent.coder_agent import CoderAgent


def main():
    """Sets up and runs the CoderAgent example."""
    
    # --- 1. CONFIGURATION & INITIALIZATION ---
    print("Configuring API and initializing model...")
    api_key = os.environ.get("NAUT_API_KEY")
    if not api_key:
        print("\nERROR: NAUT_API_KEY environment variable not set.")
        print("Please set it before running the example: export NAUT_API_KEY='your_key'")
        return
    
    os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
    os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        model = LiteLLMModel(model_id="openai/llama3")
        coder = CoderAgent(
            model=model,
            max_debug_rounds=3
        )
        print("Initialization complete.")
    except Exception as e:
        print(f"\nError during initialization: {e}")
        print("Please check your API key and network connection.")
        return
    
    # --- 2. EXAMPLE TASKS ---
    
    # Example 1: Simple calculation
    print("\n" + "="*60)
    print("Example 1: Calculate the sum of numbers from 1 to 100")
    print("="*60)
    task1 = "Calculate the sum of all numbers from 1 to 100 using Python code."
    result1 = coder.run(task1)
    print(f"\nResult:\n{result1}\n")
    
    # Example 2: Data analysis
    print("\n" + "="*60)
    print("Example 2: Create and analyze data")
    print("="*60)
    task2 = """Create a list of 10 random numbers between 1 and 100, 
    then calculate their mean, median, and standard deviation."""
    result2 = coder.run(task2)
    print(f"\nResult:\n{result2}\n")
    
    # Example 3: File operations
    print("\n" + "="*60)
    print("Example 3: Write and read a file")
    print("="*60)
    task3 = """Write a Python script that creates a file called 'test.txt' with the text 
    'Hello from CoderAgent!', then read it back and print the contents."""
    result3 = coder.run(task3)
    print(f"\nResult:\n{result3}\n")
    
    # --- 3. CLEANUP ---
    print("\nCleaning up...")
    coder.close()
    print("Done!")


if __name__ == "__main__":
    main()

