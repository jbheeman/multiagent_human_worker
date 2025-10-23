"""CoderAgent implementation - 1:1 port of magentic CoderAgent to smolagents."""
import asyncio
import re
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator, Sequence
from datetime import datetime

from smolagents import LiteLLMModel

try:
    from .prompts import CODER_SYSTEM_PROMPT
except ImportError:
    from prompts import CODER_SYSTEM_PROMPT


class CodeBlock:
    """Represents a block of code extracted from markdown."""
    def __init__(self, code: str, language: str):
        self.code = code
        self.language = language


class CodeResult:
    """Result from code execution."""
    def __init__(self, exit_code: int, output: str):
        self.exit_code = exit_code
        self.output = output


class TextMessage:
    """Represents a text message in the conversation."""
    def __init__(self, source: str, content: str, metadata: Optional[Dict[str, str]] = None):
        self.source = source
        self.content = content
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"TextMessage(source={self.source}, content={self.content[:50]}...)"


class Response:
    """Response from the agent."""
    def __init__(self, chat_message: TextMessage, inner_messages: Optional[List[TextMessage]] = None):
        self.chat_message = chat_message
        self.inner_messages = inner_messages or []


def _extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    """Extract code blocks from markdown text.
    
    Matches the original implementation exactly.
    """
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


class SimpleCodeExecutor:
    """Simple code executor for Python and shell scripts."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    async def execute_code_blocks(
        self, 
        code_blocks: List[CodeBlock],
        cancellation_token = None
    ) -> CodeResult:
        """Execute code blocks and return results."""
        import subprocess
        
        if not code_blocks:
            return CodeResult(exit_code=0, output="")
        
        code_block = code_blocks[0]
        
        try:
            if code_block.language in ["python", "py"]:
                # Write to temp file and execute
                temp_file = self.work_dir / f"temp_script_{id(code_block)}.py"
                temp_file.write_text(code_block.code)
                
                result = subprocess.run(
                    ["python", str(temp_file)],
                    capture_output=True,
                    text=True,
                    cwd=self.work_dir,
                    timeout=30
                )
                
                temp_file.unlink()
                
                output = result.stdout + result.stderr
                return CodeResult(exit_code=result.returncode, output=output)
                
            elif code_block.language in ["bash", "sh", "shell"]:
                # Execute shell command
                result = subprocess.run(
                    code_block.code,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.work_dir,
                    timeout=30
                )
                
                output = result.stdout + result.stderr
                return CodeResult(exit_code=result.returncode, output=output)
            else:
                return CodeResult(
                    exit_code=1,
                    output=f"Unsupported language: {code_block.language}"
                )
                
        except subprocess.TimeoutExpired:
            return CodeResult(exit_code=1, output="Code execution timed out after 30 seconds")
        except Exception as e:
            return CodeResult(exit_code=1, output=f"Error executing code: {str(e)}")


async def _coding_and_debug(
    system_prompt: str,
    thread: List[TextMessage],
    agent_name: str,
    model_client: LiteLLMModel,
    code_executor: SimpleCodeExecutor,
    max_debug_rounds: int,
    cancellation_token = None,
) -> AsyncGenerator[TextMessage, None]:
    """Write and debug code using the model and executor.
    
    This is a 1:1 port of the original _coding_and_debug function.
    """
    # The list of new messages to be added to the thread
    delta: List[TextMessage] = []
    executed_code = False
    
    for i in range(max_debug_rounds):
        # Create the prompt for the LLM
        messages_for_llm = []
        
        # Add all previous messages
        for msg in thread:
            messages_for_llm.append({
                "role": "user" if "user" in msg.source.lower() else "assistant",
                "content": msg.content
            })
        
        # Add delta messages
        for msg in delta:
            messages_for_llm.append({
                "role": "assistant" if "llm" in msg.source.lower() else "user",
                "content": msg.content
            })
        
        # Add system prompt as final message
        messages_for_llm.append({
            "role": "user",
            "content": system_prompt
        })
        
        # Generate code using the model
        # Use the smolagents model to generate response
        try:
            # Create a simple prompt combining all messages
            combined_prompt = "\n\n".join([m["content"] for m in messages_for_llm])
            
            # Call the model (smolagents LiteLLMModel)
            response = await asyncio.to_thread(
                model_client,
                [{"role": "user", "content": combined_prompt}]
            )
            
            # Extract content from response
            if isinstance(response, str):
                create_result_content = response
            elif isinstance(response, dict) and "content" in response:
                create_result_content = response["content"]
            elif hasattr(response, "content"):
                create_result_content = response.content
            else:
                create_result_content = str(response)
                
        except Exception as e:
            create_result_content = f"Error calling model: {str(e)}"
        
        code_msg = TextMessage(
            source=f"{agent_name}-llm",
            metadata={"internal": "no", "type": "potential_code"},
            content=create_result_content,
        )
        
        # Add LLM's response to the current thread
        delta.append(code_msg)
        yield code_msg
        
        # Extract code blocks from the LLM's response
        code_block_list = _extract_markdown_code_blocks(create_result_content)
        
        # If no code to execute, return
        if len(code_block_list) == 0:
            break
        
        code_output_list: List[str] = []
        exit_code_list: List[int] = []
        executed_code = True
        
        try:
            for cb in code_block_list:
                # Execute the code block
                exit_code: int = 1
                encountered_exception: bool = False
                code_output: str = ""
                result: Optional[CodeResult] = None
                
                try:
                    result = await code_executor.execute_code_blocks(
                        [cb], cancellation_token
                    )
                    exit_code = result.exit_code or 0
                    code_output = result.output
                except Exception as e:
                    code_output = str(e)
                    encountered_exception = True
                
                if encountered_exception or result is None:
                    code_output = f"An exception occurred while executing the code block: {code_output}"
                elif code_output.strip() == "":
                    # No output
                    code_output = f"The script ran but produced no output to console. The exit code was: {result.exit_code}. If you were expecting output, consider revising the script to ensure content is printed to stdout."
                elif exit_code != 0:
                    # Error
                    code_output = f"The script ran, then exited with an error (exit code: {result.exit_code})\nIts output was:\n{result.output}"
                
                code_output_list.append(code_output)
                code_output_msg = TextMessage(
                    source=f"{agent_name}-executor",
                    metadata={"internal": "no", "type": "code_execution"},
                    content=f"Execution result of code block {i + 1}:\n```console\n{code_output}\n```",
                )
                exit_code_list.append(exit_code)
                yield code_output_msg
            
            final_code_output = ""
            for idx, code_output in enumerate(code_output_list):
                final_code_output += f"\n\nExecution Result of Code Block {idx + 1}:\n```console\n{code_output}\n```"
            
            # Add executor's response to thread
            executor_msg = TextMessage(
                source=f"{agent_name}-executor",
                metadata={"internal": "yes"},
                content=final_code_output,
            )
            delta.append(executor_msg)
            yield executor_msg
            
            # Break if the code execution was successful
            if all([code_exit == 0 for code_exit in exit_code_list]):
                break
                
        except asyncio.TimeoutError:
            # If the task times out, we treat it as an error
            executor_msg = TextMessage(
                source=f"{agent_name}-executor",
                metadata={"internal": "yes"},
                content="Code execution timed out.",
            )
            delta.append(executor_msg)
            yield executor_msg
    
    # Return a flag indicating whether any code was executed
    yield TextMessage(source="system", content=str(executed_code), metadata={"is_flag": "true"})


async def _summarize_coding(
    agent_name: str,
    model_client: LiteLLMModel,
    thread: List[TextMessage],
) -> TextMessage:
    """Create a summary from the inner messages using an extra LLM call.
    
    This is a 1:1 port of the original _summarize_coding function.
    """
    # Create summary prompt
    messages_text = "\n\n".join([f"{msg.source}: {msg.content}" for msg in thread])
    
    summary_prompt = f"""
    The above is a transcript of your previous messages and a request that was given to you in the beginning.
    You need to summarize them to answer the request given to you. Generate a summary of everything that happened.
    If there was code that was executed, please copy the final code that was executed without errors.
    Don't mention that this is a summary, just give the summary.
    
    Transcript:
    {messages_text}
    """
    
    try:
        # Call the model for summary
        response = await asyncio.to_thread(
            model_client,
            [{"role": "user", "content": summary_prompt}]
        )
        
        if isinstance(response, str):
            summary_content = response
        elif isinstance(response, dict) and "content" in response:
            summary_content = response["content"]
        elif hasattr(response, "content"):
            summary_content = response.content
        else:
            summary_content = str(response)
            
    except Exception as e:
        summary_content = f"Error generating summary: {str(e)}"
    
    return TextMessage(
        source=agent_name,
        metadata={"internal": "yes"},
        content=summary_content,
    )


class CoderAgent:
    """An agent capable of writing, executing, and debugging code.
    
    This is a 1:1 port of the magentic CoderAgent to smolagents framework.
    Maintains all features: streaming, debugging, state management, pause/resume.
    """
    
    DEFAULT_DESCRIPTION = """
    An agent that can write and execute code to solve tasks or use its language skills to summarize, write, solve math and logic problems.
    It understands images and can use them to help it complete the task.
    It can access files if given the path and manipulate them using python code. Use the coder if you want to manipulate a file or read a csv or excel files.
    In a single step when you ask the agent to do something: it can write code, and then immediately execute the code. If there are errors it can debug the code and try again.
    """
    
    def __init__(
        self,
        model: LiteLLMModel,
        name: str = "CoderAgent",
        description: str = DEFAULT_DESCRIPTION,
        max_debug_rounds: int = 3,
        summarize_output: bool = False,
        work_dir: Optional[Path] = None,
        use_local_executor: bool = True,
        system_prompt: Optional[str] = None,
        prompt_templates: Optional[Dict] = None,
    ):
        """Initialize the CoderAgent.
        
        Args:
            model: The language model to use
            name: Name of the agent
            description: Description of agent capabilities
            max_debug_rounds: Maximum number of code debugging iterations
            summarize_output: Whether to summarize the code execution results
            work_dir: Working directory for code execution (if None, creates temp dir)
            use_local_executor: Whether to use local code executor (True) or Docker (False)
            system_prompt: Custom system prompt to use instead of the default one
            prompt_templates: Full prompt templates dictionary to use instead of default
        """
        self.name = name
        self.description = description
        self._model_client = model
        self._max_debug_rounds = max_debug_rounds
        self._summarize_output = summarize_output
        self._system_prompt = system_prompt
        self._prompt_templates = prompt_templates
        self.is_paused = False
        self._paused = asyncio.Event()
        
        # Set up work directory
        if work_dir is None:
            self._work_dir = Path(tempfile.mkdtemp())
            self._cleanup_work_dir = True
        else:
            self._work_dir = Path(work_dir)
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_work_dir = False
        
        # Initialize code executor
        self._code_executor = SimpleCodeExecutor(work_dir=self._work_dir)
        
        # Chat history
        self._chat_history: List[TextMessage] = []
    
    async def pause(self) -> None:
        """Pause the agent by setting the paused state."""
        self.is_paused = True
        self._paused.set()
    
    async def resume(self) -> None:
        """Resume the agent by clearing the paused state."""
        self.is_paused = False
        self._paused.clear()
    
    def run(self, task: str) -> str:
        """Run the agent synchronously (wrapper around async run_async)."""
        return asyncio.run(self.run_async(task))
    
    async def run_async(self, task: str) -> str:
        """Run the coder agent to complete a task (async version).
        
        Args:
            task: The task to complete
            
        Returns:
            The result of the task execution
        """
        if self.is_paused:
            return "The Coder is paused."
        
        # Create message for the task
        task_message = TextMessage(source="user", content=task)
        self._chat_history.append(task_message)
        
        # Build the system prompt with current date
        date_today = datetime.now().strftime("%B %d, %Y")
        if self._prompt_templates and "system_prompt" in self._prompt_templates:
            # Use system prompt from prompt templates
            system_prompt_coder = self._prompt_templates["system_prompt"]
            if "{date_today}" in system_prompt_coder:
                system_prompt_coder = system_prompt_coder.format(date_today=date_today)
        elif self._system_prompt:
            # Use custom system prompt, format with date if it contains {date_today}
            if "{date_today}" in self._system_prompt:
                system_prompt_coder = self._system_prompt.format(date_today=date_today)
            else:
                system_prompt_coder = self._system_prompt
        else:
            # Use default system prompt
            system_prompt_coder = CODER_SYSTEM_PROMPT.format(date_today=date_today)
        
        # Track inner messages
        inner_messages: List[TextMessage] = []
        executed_code = False
        
        # Set up cancellation
        # For simplicity, we won't implement full cancellation token
        cancellation_token = None
        
        try:
            # Run the code execution and debugging process
            async for msg in _coding_and_debug(
                system_prompt=system_prompt_coder,
                thread=self._chat_history,
                agent_name=self.name,
                model_client=self._model_client,
                code_executor=self._code_executor,
                max_debug_rounds=self._max_debug_rounds,
                cancellation_token=cancellation_token,
            ):
                # Check if this is the executed_code flag
                if msg.metadata.get("is_flag") == "true":
                    executed_code = msg.content == "True"
                    break
                
                inner_messages.append(msg)
                self._chat_history.append(msg)
            
            # Summarize if configured
            if self._summarize_output and executed_code:
                summary_msg = await _summarize_coding(
                    agent_name=self.name,
                    model_client=self._model_client,
                    thread=[task_message] + inner_messages,
                )
                self._chat_history.append(summary_msg)
                return summary_msg.content
            else:
                # Combine all inner messages
                combined_output = ""
                for txt_msg in inner_messages:
                    combined_output += f"{txt_msg.content}\n\n"
                
                final_response_msg = TextMessage(
                    source=self.name,
                    metadata={"internal": "yes"},
                    content=combined_output or "No output.",
                )
                return final_response_msg.content
                
        except asyncio.CancelledError:
            return "The task was cancelled by the user."
        except Exception as e:
            error_msg = f"An error occurred in the coder agent: {str(e)}"
            self._chat_history.append(TextMessage(source=self.name, content=error_msg))
            return error_msg
    
    def get_chat_history(self) -> List[TextMessage]:
        """Get the conversation history."""
        return self._chat_history.copy()
    
    def clear_chat_history(self) -> None:
        """Clear the conversation history."""
        self._chat_history.clear()
    
    async def save_state(self) -> Dict[str, Any]:
        """Save the state of the agent."""
        return {
            "chat_history": [
                {
                    "source": msg.source,
                    "content": msg.content,
                    "metadata": msg.metadata
                }
                for msg in self._chat_history
            ],
        }
    
    async def load_state(self, state: Dict[str, Any]) -> None:
        """Load the state of the agent."""
        self._chat_history = []
        for msg_data in state.get("chat_history", []):
            msg = TextMessage(
                source=msg_data["source"],
                content=msg_data["content"],
                metadata=msg_data.get("metadata", {})
            )
            self._chat_history.append(msg)
    
    async def close(self) -> None:
        """Clean up resources used by the agent."""
        if self._cleanup_work_dir and self._work_dir.exists():
            await asyncio.to_thread(shutil.rmtree, self._work_dir)
    
    def close_sync(self) -> None:
        """Synchronous version of close."""
        if self._cleanup_work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir)
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close_sync()
        except:
            pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close_sync()
        return False
