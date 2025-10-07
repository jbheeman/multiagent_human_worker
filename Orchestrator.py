import os
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    OpenAIServerModel,
    WebSearchTool,
)

from websurfer_agent import WebSurferTool
from file_surfer_agent import FileSurferTool
from coder_agent import CoderTool



# Create the model that will be used by both the orchestrator and the web surfer
model = OpenAIServerModel(
    model_id="gemma3",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key=os.environ["NAUT_API_KEY"],
)

# Create the WebSurfer tool with the same model
web_surfer_tool = WebSurferTool(model=model)

# Create the FileSurfer tool with the same model
file_surfer_tool = FileSurferTool(
    model=model,
    base_path=".",  # Current directory - change this to restrict access to a specific folder
    viewport_size=8192
)

# Create the Coder tool with the same model
# Set work_dir to current directory so files are created here, not in temp sandbox
from pathlib import Path
coder_tool = CoderTool(
    model=model,
    max_debug_rounds=3,
    use_local_executor=True,
    work_dir=Path.cwd()  # Use current working directory instead of temp sandbox
)

# Create the manager agent with all tools
# Use ToolCallingAgent instead of CodeAgent - it delegates to tools instead of executing code directly
# This prevents "forbidden function" errors when tools need to use open(), requests, etc.
manager_agent = ToolCallingAgent(
    tools=[web_surfer_tool, file_surfer_tool, coder_tool, WebSearchTool()],
    model=model,
)

# Run the agent with the query
answer = manager_agent.run(
    "Good stocks to buy, and write a file in the current directory called good_stocks.txt that is a report of your findings"
)

# answer = manager_agent.run("What is the weather in Tokyo?")

print("="*80)
print("ANSWER:")
print("="*80)
print(answer)
