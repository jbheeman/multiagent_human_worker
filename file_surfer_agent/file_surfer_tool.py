"""FileSurfer Tool wrapper for smolagents orchestrator."""
from typing import Optional
from smolagents import Tool
from smolagents.models import OpenAIServerModel

from .file_surfer import FileSurfer


class FileSurferTool(Tool):
    """A tool that wraps the FileSurfer agent to make it usable by the smolagents orchestrator."""
    
    name = "file_surfer"
    description = (
        "Delegates a complex, multi-step file system task to a specialist agent. This tool is for any task that "
        "requires finding information within the project's local files. It is strictly read-only.\n"
        "The specialist agent has the following capabilities:\n"
        "- Navigate the file system by listing directory contents and opening files.\n"
        "- Read the full contents of files, paging through large files when necessary.\n"
        "- Search for specific text, keywords, or code snippets within any file.\n"
        "- Analyze file contents to answer questions or summarize information.\n"
        "- Operate within a secure, sandboxed directory."
    )
    
    inputs = {
        "task": {
            "type": "string",
            "description": "A clear and specific instruction for the file browsing task to be performed.",
        }
    }
    output_type = "string"
    
    def __init__(self, model: Optional[OpenAIServerModel] = None, base_path: str = ".", viewport_size: int = 8192, **kwargs):
        """Initialize the FileSurferTool.
        
        Args:
            model: The model to use for the FileSurfer agent
            base_path: The root directory the agent is allowed to access (default: current directory)
            viewport_size: The approximate number of characters per page (default: 8192)
            **kwargs: Additional arguments to pass to FileSurfer
        """
        super().__init__()
        self.model = model
        self.base_path = base_path
        self.viewport_size = viewport_size
        self.agent_kwargs = kwargs
        self._agent = None
        
    def _get_agent(self):
        """Lazy initialization of the FileSurfer agent."""
        if self._agent is None:
            self._agent = FileSurfer(
                model=self.model,
                base_path=self.base_path,
                viewport_size=self.viewport_size,
                **self.agent_kwargs
            )
        return self._agent
    
    def forward(self, task: str) -> str:
        """Execute the file browsing task.
        
        Args:
            task: The task to perform using the file browser
            
        Returns:
            The result of the file browsing task
        """
        agent = self._get_agent()
        
        try:
            result = agent.run(task)
            return result
        except Exception as e:
            return f"Error executing file surfer task: {str(e)}"