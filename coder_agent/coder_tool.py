"""CoderTool wrapper for smolagents orchestrator."""
import asyncio
from typing import Optional
from pathlib import Path
from smolagents import Tool
from smolagents.models import OpenAIServerModel

try:
    from .coder_agent import CoderAgent
except ImportError:
    from coder_agent import CoderAgent


class CoderTool(Tool):
    """A tool that wraps the CoderAgent to make it usable by the smolagents orchestrator."""
    
    name = "coder"
    description = """Use this tool to write and execute Python or shell code in an isolated environment.

IMPORTANT: This tool executes code in its own sandbox where ALL Python operations are allowed, including:
- File I/O operations (open, read, write, etc.)
- System operations
- Installing libraries with pip
- Any standard Python functionality

The coder agent will:
1. Write the necessary Python/shell code to complete your task
2. Execute the code in its isolated environment
3. Debug and retry if errors occur (up to max_debug_rounds times)
4. Return the results or created files

Use this tool when you need to:
- Create, read, or modify files (use this instead of trying to use 'open()' directly)
- Perform calculations or data processing
- Install and use Python libraries
- Generate data, reports, or visualizations
- Any task requiring code execution with full Python capabilities

Example usage:
- coder(task="Create a file called output.txt with the content 'Hello World'")
- coder(task="Read data.csv, calculate the mean of column 'values', and save to results.txt")
"""
    
    inputs = {
        "task": {
            "type": "string",
            "description": "The task to perform using code. Be specific about what you want to accomplish.",
        }
    }
    output_type = "string"
    
    def __init__(
        self, 
        model: Optional[OpenAIServerModel] = None,
        max_debug_rounds: int = 3,
        summarize_output: bool = False,
        work_dir: Optional[Path] = None,
        use_local_executor: bool = True,
        **kwargs
    ):
        """Initialize the CoderTool.
        
        Args:
            model: The model to use for the CoderAgent
            max_debug_rounds: Maximum number of debugging iterations (default: 3)
            summarize_output: Whether to summarize code execution results (default: False)
            work_dir: Working directory for code execution (default: temp directory)
            use_local_executor: Whether to use local executor vs Docker (default: True)
            **kwargs: Additional arguments to pass to CoderAgent
        """
        super().__init__()
        self.model = model
        self.max_debug_rounds = max_debug_rounds
        self.summarize_output = summarize_output
        self.work_dir = work_dir
        self.use_local_executor = use_local_executor
        self.agent_kwargs = kwargs
        self._agent = None
        
    def _get_agent(self):
        """Lazy initialization of the CoderAgent."""
        if self._agent is None:
            self._agent = CoderAgent(
                model=self.model,
                max_debug_rounds=self.max_debug_rounds,
                summarize_output=self.summarize_output,
                work_dir=self.work_dir,
                use_local_executor=self.use_local_executor,
                **self.agent_kwargs
            )
        return self._agent
    
    def forward(self, task: str) -> str:
        """Execute the coding task.
        
        Args:
            task: The task to perform using code
            
        Returns:
            The result of the coding task
        """
        agent = self._get_agent()
        
        try:
            # Use synchronous run method (which wraps async internally)
            result = agent.run(task)
            return result
        except Exception as e:
            return f"Error executing coder task: {str(e)}"
    
    def __del__(self):
        """Cleanup when the tool is destroyed."""
        if self._agent is not None:
            try:
                self._agent.close_sync()
            except:
                pass
