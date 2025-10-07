import os
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    OpenAIServerModel,
    WebSearchTool,
)

from websurfer_agent import WebSurferTool
from file_surfer_agent import FileSurferTool



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

# Create the manager agent with both tools
manager_agent = CodeAgent(
    tools=[web_surfer_tool, file_surfer_tool, WebSearchTool()],
    model=model,
    additional_authorized_imports=["time", "numpy", "pandas"],
)

# Run the agent with the query
answer = manager_agent.run(
    "Good stocks to buy, and write a file in the current directory called good_stocks.txt"
)

# answer = manager_agent.run("What is the weather in Tokyo?")

print("="*80)
print("ANSWER:")
print("="*80)
print(answer)
