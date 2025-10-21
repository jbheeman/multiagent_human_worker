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
        if file_tools.browser is None:
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
            instructions="""You have access to the following tools:

- open_path: Opens a file or a directory to view its contents. Paths are relative.
- page_up: Scrolls the viewport up to the previous page in a large file.
- page_down: Scrolls the viewport down to the next page in a large file.
- find_on_page: Searches for text in the currently open file. The search starts from the current page and is case-insensitive, matching whole words (e.g., 'log' will not find 'logging'). It jumps the viewport to the first match it finds.
- find_next: Jumps the viewport to the next match for the last search. The search will wrap around to the beginning of the file if it reaches the end."""
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


