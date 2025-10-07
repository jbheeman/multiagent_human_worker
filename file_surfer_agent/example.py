# example.py
"""
An example script demonstrating how to use the FileSurfer agent.

This script automatically sets up a temporary project directory, runs the 
agent to analyze a file within it, prints the result, and then cleans up 
all temporary files and directories.

Usage:
    export NAUTILUS_API_KEY="your_api_key_here"
    python3 example.py
"""
import os
import shutil
from smolagents import LiteLLMModel
from file_surfer import FileSurfer

def main():
    """Sets up, runs, and tears down the FileSurfer agent example."""
    project_dir = "example_project"

    try:
        # --- 1. SETUP PHASE ---
        print(f"Setting up temporary project directory: '{project_dir}'...")
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, "main_script.py"), "w") as f:
            f.write("# This is the main script for our project.\n")
            f.write("import antigravity\n\n")
            f.write("def main():\n")
            f.write("    # A simple function that prints a message.\n")
            f.write("    print('Hello from the main function!')\n\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    main()\n")
        print("Setup complete.")

        # --- 2. CONFIGURATION & INITIALIZATION ---
        print("\nConfiguring API and initializing model...")
        api_key = os.environ.get("NAUTILUS_API_KEY")
        if not api_key:
            print("\nERROR: NAUTILUS_API_KEY environment variable not set.")
            print("Please set it before running the example: export NAUTILUS_API_KEY='your_key'")
            return # Exit gracefully

        os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
        os.environ["OPENAI_API_KEY"] = api_key

        try:
            model = LiteLLMModel(model_id="openai/llama3")
            file_surfer = FileSurfer(
                model=model, 
                base_path=project_dir,
                viewport_size=2048 # Example of setting a custom page size
            )
            print("Initialization complete.")
        except Exception as e:
            print(f"\nError during initialization: {e}")
            print("Please check your API key and network connection.")
            return

        # --- 3. EXECUTION ---
        task_to_run = "First, list all files in the current directory. Then, open 'main_script.py' and tell me what the main function does."
        print(f"\nRunning task: '{task_to_run}'")
        
        final_answer = file_surfer.run(task_to_run)

        print("\n" + "="*20 + " FINAL ANSWER " + "="*20)
        print(final_answer)
        print("="*54)

    finally:
        # --- 4. TEARDOWN PHASE ---
        print(f"\nTearing down temporary project directory: '{project_dir}'...")
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        print("Cleanup complete.")


if __name__ == "__main__":
    main()