import os
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    OpenAIServerModel,
    WebSearchTool,
)

from websurfer_agent import WebSurferTool

# Create the model that will be used by both the orchestrator and the web surfer
model = OpenAIServerModel(
    model_id="gemma3",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key=os.environ["NAUT_API_KEY"],
)

# Create the WebSurfer tool with the same model
web_surfer_tool = WebSurferTool(model=model)

# Create the manager agent with the web surfer as a tool
manager_agent = CodeAgent(
    tools=[web_surfer_tool, WebSearchTool()],
    model=model,
    additional_authorized_imports=["time", "numpy", "pandas"],
)

# Run the agent with the query
answer = manager_agent.run(
    "Find when Professor Mohsen Lesani's office hours are at UCSC for spring 2025."
)

# answer = manager_agent.run("What is the weather in Tokyo?")

print("="*80)
print("ANSWER:")
print("="*80)
print(answer)
