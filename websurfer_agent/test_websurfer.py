import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import json
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    WebSearchTool,
)

# Add parent directory to path to fix relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from websurfer_agent.web_surfer_agent import WebSurferAgent

async def main():
    WebSurferModel = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

    user_goal = """Visit arXiv.org and search for papers with 'AI regulation' in title/abstract submitted between 2022-06-01 and 2022-06-30 using original submission date filter. Identify the specific paper containing a three-axis figure and extract its arXiv ID."""

    websurfer_agent = WebSurferAgent(
        name="WebSurferAgent",
        model=WebSurferModel,
    )

    result = await websurfer_agent.run(user_goal)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())