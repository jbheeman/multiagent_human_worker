"""WebSurfer Tool wrapper for smolagents orchestrator."""
import asyncio
from typing import Optional
from smolagents import Tool
from smolagents.models import OpenAIServerModel

from .web_surfer_agent import WebSurferAgent


class WebSurferTool(Tool):
    """A tool that wraps the WebSurferAgent to make it usable by the smolagents orchestrator."""
    
    name = "web_surfer"
    description = """Use this tool to browse the web and interact with web pages.
    
    The websurfer can:
    - Visit web pages and URLs
    - Perform web searches using search engines
    - Click buttons, links, and other elements on pages
    - Type text into input fields and forms
    - Scroll up and down pages
    - Navigate browser history (back/forward)
    - Answer questions about page content
    - Interact with dropdowns, tabs, and other UI elements
    - Download and upload files
    
    This tool performs multiple browser actions sequentially to complete a task in a single call.
    For example, it can search for something, click on a result, read the content, and answer a question all in one execution.
    
    Use this tool when you need to:
    - Find information on the web
    - Interact with websites
    - Extract data from web pages
    - Perform tasks that require web browsing
    """
    
    inputs = {
        "task": {
            "type": "string",
            "description": "The task to perform using the web browser. Be specific about what you want to accomplish.",
        }
    }
    output_type = "string"
    
    def __init__(self, model: Optional[OpenAIServerModel] = None, personality_config: Optional[Dict] = None, **kwargs):
        """Initialize the WebSurferTool.
        
        Args:
            model: The OpenAIServerModel to use for the WebSurferAgent
            personality_config: The personality configuration dictionary
            **kwargs: Additional arguments to pass to WebSurferAgent
        """
        super().__init__()
        self.model = model
        self.personality_config = personality_config
        self.agent_kwargs = kwargs
        self._agent = None
        self._loop = None
        
    def _get_or_create_loop(self):
        """Get or create an event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    def _get_agent(self):
        """Lazy initialization of the WebSurferAgent."""
        if self._agent is None:
            self._agent = WebSurferAgent(
                model=self.model,
                personality_config=self.personality_config,
                **self.agent_kwargs
            )
        return self._agent
    
    def forward(self, task: str) -> str:
        """Execute the web browsing task.
        
        Args:
            task: The task to perform using the web browser
            
        Returns:
            The result of the web browsing task
        """
        loop = self._get_or_create_loop()
        agent = self._get_agent()
        
        # Run the async task
        try:
            result = loop.run_until_complete(agent.run(task))
            return result
        except Exception as e:
            return f"Error executing web surfer task: {str(e)}"
    
    def __del__(self):
        """Cleanup when the tool is destroyed."""
        if self._agent is not None:
            loop = self._get_or_create_loop()
            try:
                loop.run_until_complete(self._agent.close())
            except:
                pass

