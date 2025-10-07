# file_surfer.py
from smolagents import CodeAgent, LiteLLMModel
from .markdown_file_browser import MarkdownFileBrowser

from .prompts import DETAILED_SYSTEM_PROMPT
from . import file_tools

class FileSurfer:
    """A self-contained agent that uses function-based tools."""
    def __init__(self, model: LiteLLMModel, base_path: str = ".", viewport_size: int = 8192):
        """
        Initializes the FileSurfer agent.

        Args:
            model: The language model to use.
            base_path: The root directory the agent is allowed to access.
            viewport_size: The approximate number of characters per page.
        """
        # Initialize the global browser instance located in the tools module
        file_tools.browser = MarkdownFileBrowser(
            base_path=base_path, 
            viewport_size=viewport_size
        )
        
        self._system_prompt = DETAILED_SYSTEM_PROMPT
            
        self._agent = CodeAgent(
            tools=[
                file_tools.open_path,
                file_tools.page_up,
                file_tools.page_down,
                file_tools.find_on_page,
                file_tools.find_next,
            ],
            model=model,
        )

    def run(self, task: str) -> str:
        """Runs the agent to complete a task."""
        initial_context = (
            f"{self._system_prompt}\n\n"
            f"--- CURRENT TASK ---\n"
            f"USER_TASK: {task}\n\n"
            f"Here is your starting view. Begin your work.\n\n"
            f"{file_tools.get_browser_state()}"
        )
        result = self._agent.run(initial_context)
        return result

# ... (your __main__ section for running the agent can go here) ...