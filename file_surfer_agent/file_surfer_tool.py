"""FileSurfer Tool wrapper for smolagents orchestrator."""
from typing import Optional
from smolagents import Tool
from smolagents.models import OpenAIServerModel

from .file_surfer import FileSurfer


class FileSurferTool(Tool):
    """A tool that wraps the FileSurfer agent to make it usable by the smolagents orchestrator."""
    
    name = "file_surfer"
    description = """Use this tool to browse and analyze files and directories in the local filesystem.
    
    The file surfer can:
    - List files and directories
    - Open and read file contents
    - Navigate through file structure
    - Search for specific text within files
    - Page through large files
    - Answer questions about file contents
    
    This tool performs multiple file operations sequentially to complete a task in a single call.
    For example, it can list a directory, open a file, search for specific content, and answer a question all in one execution.
    
    Use this tool when you need to:
    - Explore directory structures
    - Read and analyze file contents
    - Find specific information in files
    - Get an overview of a codebase or project
    """
    
    inputs = {
        "task": {
            "type": "string",
            "description": "The task to perform using the file browser. Be specific about what files or information you want to access.",
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

