# file_surfer.py
from typing import Dict
from smolagents import ToolCallingAgent, LiteLLMModel, PromptTemplates
from .markdown_file_browser import MarkdownFileBrowser

from .prompts import DETAILED_SYSTEM_PROMPT
from . import file_tools

class FileSurfer:
    """A self-contained agent that uses function-based tools."""
    def __init__(self, model: LiteLLMModel, base_path: str = ".", viewport_size: int = 8192, system_prompt: str = None, prompt_templates: Dict = None):
        """
        Initializes the FileSurfer agent.

        Args:
            model: The language model to use.
            base_path: The root directory the agent is allowed to access.
            viewport_size: The approximate number of characters per page.
            system_prompt: Custom system prompt to use instead of the default one.
            prompt_templates: Full prompt templates dictionary to use instead of default.
        """
        # Initialize the global browser instance located in the tools module
        file_tools.browser = MarkdownFileBrowser(
            base_path=base_path, 
            viewport_size=viewport_size
        )
        
        self._system_prompt = system_prompt or DETAILED_SYSTEM_PROMPT
        self._prompt_templates = prompt_templates
        
        # Create prompt templates if provided, otherwise use system prompt
        if prompt_templates:
            prompt_templates_obj = PromptTemplates(**prompt_templates)
        else:
            prompt_templates_obj = PromptTemplates(system_prompt=self._system_prompt)
            
        self._agent = ToolCallingAgent(
            tools=[
                file_tools.open_path,
                file_tools.page_up,
                file_tools.page_down,
                file_tools.find_on_page,
                file_tools.find_next,
            ],
            model=model,
            prompt_templates=prompt_templates_obj,
        )

    def run(self, task: str) -> str:
        """Runs the agent to complete a task."""
        initial_context = (
            f"--- CURRENT TASK ---\n"
            f"USER_TASK: {task}\n\n"
            f"Here is your starting view. Begin your work.\n\n"
            f"{file_tools.get_browser_state()}"
        )
        result = self._agent.run(initial_context)
        return str(result)
