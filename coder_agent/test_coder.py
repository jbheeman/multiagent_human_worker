"""
Comprehensive test suite for the CoderAgent.

This test suite is based on the magentic-ui Coder implementation's expected behavior.
It includes unit tests, integration tests, and end-to-end tests with a live LLM.
"""
import unittest
import os
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import Optional

# ==============================================================================
# PART 1: IMPORTS
# ==============================================================================

import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from coder_agent.coder_agent import CoderAgent, TextMessage, Response
    from coder_agent.coder_tool import CoderTool
    from smolagents import LiteLLMModel
except ImportError as e:
    print("="*80 + f"\nFATAL ERROR: Could not import a required module: {e}.\n"
          "Please ensure the coder_agent package is properly installed.\n" + "="*80)
    exit()


# ==============================================================================
# PART 2: HELPER FUNCTIONS
# ==============================================================================

def async_test(coro):
    """Decorator to run async tests."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper


# ==============================================================================
# PART 3: UNIT TESTS - INITIALIZATION
# ==============================================================================

class TestCoderAgentInitialization(unittest.TestCase):
    """
    Suite 1: Tests for CoderAgent initialization and configuration.
    Based on magentic-ui Coder's initialization patterns.
    """
    
    def test_initialization_with_temp_dir(self):
        """Test that CoderAgent creates a temporary directory when none is provided."""
        coder = CoderAgent(
            model=None,
            name="TestCoder",
            max_debug_rounds=2
        )
        
        # Check that work_dir was created
        self.assertTrue(coder._work_dir.exists())
        self.assertTrue(coder._cleanup_work_dir)
        self.assertEqual(coder._max_debug_rounds, 2)
        self.assertEqual(coder.name, "TestCoder")
        
        # Cleanup
        coder.close_sync()
        self.assertFalse(coder._work_dir.exists())
    
    def test_initialization_with_custom_dir(self):
        """Test that CoderAgent respects a custom work directory."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            coder = CoderAgent(
                model=None,
                work_dir=temp_dir,
                max_debug_rounds=5,
                name="CustomCoder"
            )
            
            # Check that the custom directory is used
            self.assertEqual(coder._work_dir, temp_dir)
            self.assertFalse(coder._cleanup_work_dir)
            self.assertEqual(coder._max_debug_rounds, 5)
            
            # Cleanup should not remove the custom directory
            coder.close_sync()
            self.assertTrue(temp_dir.exists())
            
        finally:
            # Manual cleanup of custom directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def test_initialization_default_values(self):
        """Test that CoderAgent has correct default values."""
        coder = CoderAgent(model=None)
        
        # Check defaults
        self.assertEqual(coder.name, "CoderAgent")
        self.assertEqual(coder._max_debug_rounds, 3)
        self.assertTrue(coder._work_dir.exists())
        self.assertTrue(coder._cleanup_work_dir)
        
        # Cleanup
        coder.close_sync()
    
    def test_multiple_coder_instances(self):
        """Test that multiple CoderAgent instances can coexist."""
        coder1 = CoderAgent(model=None, name="Coder1")
        coder2 = CoderAgent(model=None, name="Coder2")
        
        # Both should have different work directories
        self.assertNotEqual(coder1._work_dir, coder2._work_dir)
        self.assertTrue(coder1._work_dir.exists())
        self.assertTrue(coder2._work_dir.exists())
        
        # Cleanup
        coder1.close_sync()
        coder2.close_sync()
        
        # Both directories should be cleaned up
        self.assertFalse(coder1._work_dir.exists())
        self.assertFalse(coder2._work_dir.exists())


# ==============================================================================
# PART 4: UNIT TESTS - CODE EXECUTION
# ==============================================================================

class TestCoderAgentCodeExecution(unittest.TestCase):
    """
    Suite 2: Tests for code extraction and execution functionality.
    Tests the core code execution mechanisms without requiring LLM.
    """
    
    def setUp(self):
        """Set up test environment before each test."""
        self.work_dir = Path(tempfile.mkdtemp())
        self.coder = CoderAgent(
            model=None,
            work_dir=self.work_dir,
            max_debug_rounds=2
        )
    
    def tearDown(self):
        """Clean up test environment after each test."""
        self.coder.close_sync()
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
    
    @async_test
    async def test_python_code_execution(self):
        """Test that Python code can be extracted and executed."""
        from coder_agent.coder_agent import _extract_markdown_code_blocks, SimpleCodeExecutor
        
        # Code block with Python
        markdown = """
Here's the solution:
```python
x = 42
y = 17
result = x * y
print(f"The result is {result}")
```
"""
        
        # Extract code blocks
        code_blocks = _extract_markdown_code_blocks(markdown)
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(code_blocks[0].language, "python")
        self.assertIn("x = 42", code_blocks[0].code)
        
        # Execute the code
        executor = SimpleCodeExecutor(work_dir=self.work_dir)
        result = await executor.execute_code_blocks(code_blocks)
        
        # Check execution result
        self.assertEqual(result.exit_code, 0)
        self.assertIn("714", result.output)
    
    @async_test
    async def test_shell_code_execution(self):
        """Test that shell commands can be extracted and executed."""
        from coder_agent.coder_agent import _extract_markdown_code_blocks, SimpleCodeExecutor
        
        # Code block with shell command
        markdown = """
Here's how to list files:
```sh
echo "Hello from shell"
```
"""
        
        # Extract code blocks
        code_blocks = _extract_markdown_code_blocks(markdown)
        self.assertEqual(len(code_blocks), 1)
        self.assertIn(code_blocks[0].language.lower(), ["sh", "bash", "shell"])
        
        # Execute the code
        executor = SimpleCodeExecutor(work_dir=self.work_dir)
        result = await executor.execute_code_blocks(code_blocks)
        
        # Check execution result
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hello from shell", result.output)
    
    @async_test
    async def test_file_creation_in_work_dir(self):
        """Test that code can create files in the work directory."""
        from coder_agent.coder_agent import _extract_markdown_code_blocks, SimpleCodeExecutor
        
        # Code to create a file
        markdown = """
```python
with open('test_file.txt', 'w') as f:
    f.write('Hello from CoderAgent!')
print('File created successfully')
```
"""
        
        # Extract and execute
        code_blocks = _extract_markdown_code_blocks(markdown)
        executor = SimpleCodeExecutor(work_dir=self.work_dir)
        result = await executor.execute_code_blocks(code_blocks)
        
        # Check that file was created
        test_file = self.work_dir / "test_file.txt"
        self.assertTrue(test_file.exists())
        self.assertEqual(test_file.read_text(), "Hello from CoderAgent!")
        self.assertIn("File created successfully", result.output)
    
    @async_test
    async def test_error_handling_in_code(self):
        """Test that code execution errors are properly captured."""
        from coder_agent.coder_agent import _extract_markdown_code_blocks, SimpleCodeExecutor
        
        # Code with an error
        markdown = """
```python
# This will cause a division by zero error
result = 1 / 0
```
"""
        
        # Extract and execute
        code_blocks = _extract_markdown_code_blocks(markdown)
        executor = SimpleCodeExecutor(work_dir=self.work_dir)
        result = await executor.execute_code_blocks(code_blocks)
        
        # Check that error was captured
        self.assertNotEqual(result.exit_code, 0)
        self.assertTrue(
            "ZeroDivisionError" in result.output or "division" in result.output.lower()
        )
    
    def test_multiple_code_blocks_extraction(self):
        """Test that multiple code blocks can be extracted from markdown."""
        from coder_agent.coder_agent import _extract_markdown_code_blocks
        
        markdown = """
First, let's define a function:
```python
def add(a, b):
    return a + b
```

Then, let's test it:
```python
result = add(5, 3)
print(result)
```

And here's a shell command:
```sh
echo "Done"
```
"""
        
        code_blocks = _extract_markdown_code_blocks(markdown)
        self.assertEqual(len(code_blocks), 3)
        self.assertEqual(code_blocks[0].language, "python")
        self.assertEqual(code_blocks[1].language, "python")
        self.assertIn(code_blocks[2].language.lower(), ["sh", "bash", "shell"])


# ==============================================================================
# PART 5: UNIT TESTS - CODER TOOL
# ==============================================================================

class TestCoderToolBasics(unittest.TestCase):
    """
    Suite 3: Tests for the CoderTool wrapper.
    Tests the tool interface for use in multi-agent systems.
    """
    
    def test_coder_tool_initialization(self):
        """Test that CoderTool can be initialized properly."""
        tool = CoderTool(
            model=None,
            max_debug_rounds=3
        )
        
        # Check tool attributes
        self.assertEqual(tool.name, "coder")
        self.assertEqual(tool.max_debug_rounds, 3)
        self.assertIsNotNone(tool.description)
        self.assertIn("task", tool.inputs)
        self.assertEqual(tool.output_type, "string")
        self.assertIn("code", tool.description.lower())
    
    def test_coder_tool_with_custom_work_dir(self):
        """Test that CoderTool can use a custom work directory."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            tool = CoderTool(
                model=None,
                work_dir=temp_dir,
                max_debug_rounds=5
            )
            
            self.assertEqual(tool.max_debug_rounds, 5)
            # Work dir management is handled by the underlying CoderAgent
            
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def test_multiple_coder_tools(self):
        """Test that multiple CoderTool instances can coexist."""
        tool1 = CoderTool(model=None, max_debug_rounds=2)
        tool2 = CoderTool(model=None, max_debug_rounds=4)
        
        # Both should be independent
        self.assertEqual(tool1.max_debug_rounds, 2)
        self.assertEqual(tool2.max_debug_rounds, 4)
        self.assertEqual(tool1.name, "coder")
        self.assertEqual(tool2.name, "coder")


# ==============================================================================
# PART 6: END-TO-END TESTS WITH LIVE LLM
# ==============================================================================

@unittest.skipIf(not all([CoderAgent, LiteLLMModel]), "Skipping E2E tests due to missing dependencies.")
class TestCoderAgent_E2E(unittest.TestCase):
    """
    Suite 4: End-to-end tests for the complete CoderAgent with a live LLM.
    Based on magentic-ui Coder's expected behaviors and capabilities.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment with a live API connection."""
        cls.work_dir = Path(tempfile.mkdtemp())
        
        # Check for API key
        api_key = os.environ.get("NAUTILUS_API_KEY")
        if not api_key:
            raise unittest.SkipTest("NAUTILUS_API_KEY not set. Skipping E2E tests.")
        
        # Configure API
        os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        
        try:
            # Initialize model
            model = LiteLLMModel(model_id="openai/llama3")
            
            # Create the coder agent
            cls.coder = CoderAgent(
                model=model,
                work_dir=cls.work_dir,
                max_debug_rounds=3
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to initialize model, skipping E2E tests: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if hasattr(cls, 'coder'):
            # Use sync cleanup for teardown
            cls.coder.close_sync()
        if cls.work_dir.exists():
            shutil.rmtree(cls.work_dir)
    
    @async_test
    async def test_1_simple_calculation(self):
        """Test that the agent can perform a simple calculation."""
        print("\n--- Running E2E Test 1: Simple Calculation ---")
        result = await self.coder.run_async("Calculate 42 * 17 using Python and show me the result")
        
        # The result should contain the answer (714)
        result_str = str(result.chat_message.content if hasattr(result, 'chat_message') else result)
        self.assertTrue(
            "714" in result_str,
            f"Expected '714' in result, got: {result_str}"
        )
    
    @async_test
    async def test_2_file_creation(self):
        """Test that the agent can create and write to a file."""
        print("\n--- Running E2E Test 2: File Creation ---")
        result = await self.coder.run_async(
            "Create a file called 'test_output.txt' with the text 'Hello from CoderAgent!'"
        )
        
        # Check that the file was created
        test_file = self.work_dir / "test_output.txt"
        self.assertTrue(
            test_file.exists(),
            f"File {test_file} should have been created"
        )
        
        # Check file contents
        if test_file.exists():
            content = test_file.read_text()
            self.assertIn("Hello from CoderAgent", content)
    
    @async_test
    async def test_3_data_processing(self):
        """Test that the agent can process data using Python."""
        print("\n--- Running E2E Test 3: Data Processing ---")
        result = await self.coder.run_async(
            "Create a list of numbers [1, 2, 3, 4, 5] and calculate their sum and average"
        )
        
        # Should contain sum (15) and average (3.0)
        result_str = str(result.chat_message.content if hasattr(result, 'chat_message') else result).lower()
        has_sum = "15" in result_str
        has_avg = "3.0" in result_str or "3" in result_str
        
        self.assertTrue(
            has_sum and has_avg,
            f"Expected sum (15) and average (3) in result: {result_str}"
        )
    
    @async_test
    async def test_4_list_comprehension(self):
        """Test that the agent can use advanced Python features."""
        print("\n--- Running E2E Test 4: List Comprehension ---")
        result = await self.coder.run_async(
            "Use a list comprehension to create a list of squares from 1 to 10 and print them"
        )
        
        # Should mention squares or contain square numbers
        result_str = str(result.chat_message.content if hasattr(result, 'chat_message') else result)
        
        # At least some square numbers should appear
        squares = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
        squares_found = sum(1 for sq in squares if str(sq) in result_str)
        
        self.assertGreater(
            squares_found, 3,
            f"Expected to find several square numbers in result: {result_str}"
        )
    
    @async_test
    async def test_5_error_recovery(self):
        """Test that the agent can recover from errors through debugging."""
        print("\n--- Running E2E Test 5: Error Recovery ---")
        # Ask for something that might initially fail but can be recovered
        result = await self.coder.run_async(
            "Calculate the factorial of 5 using recursion and print the result"
        )
        
        # Should successfully complete and show 120
        result_str = str(result.chat_message.content if hasattr(result, 'chat_message') else result)
        self.assertTrue(
            "120" in result_str,
            f"Expected '120' (factorial of 5) in result: {result_str}"
        )
    
    @async_test
    async def test_6_file_reading_and_processing(self):
        """Test that the agent can read and process files."""
        print("\n--- Running E2E Test 6: File Reading and Processing ---")
        
        # First, create a data file
        data_file = self.work_dir / "data.txt"
        data_file.write_text("10\n20\n30\n40\n50\n")
        
        result = await self.coder.run_async(
            "Read the numbers from 'data.txt', calculate their sum, and print the result"
        )
        
        # Should contain sum (150)
        result_str = str(result.chat_message.content if hasattr(result, 'chat_message') else result)
        self.assertTrue(
            "150" in result_str,
            f"Expected '150' (sum of numbers) in result: {result_str}"
        )
    
    @async_test
    async def test_7_json_data_manipulation(self):
        """Test that the agent can work with JSON data."""
        print("\n--- Running E2E Test 7: JSON Data Manipulation ---")
        result = await self.coder.run_async(
            "Create a JSON object with keys 'name', 'age', and 'city', write it to 'person.json', "
            "then read it back and print the name"
        )
        
        # Check that JSON file was created
        json_file = self.work_dir / "person.json"
        self.assertTrue(
            json_file.exists(),
            f"JSON file {json_file} should have been created"
        )
        
        # Result should mention the operation
        result_str = str(result.chat_message.content if hasattr(result, 'chat_message') else result)
        self.assertTrue(
            len(result_str) > 10,
            f"Expected meaningful result: {result_str}"
        )
    
    @async_test
    async def test_8_multi_file_operation(self):
        """Test that the agent can work with multiple files."""
        print("\n--- Running E2E Test 8: Multi-File Operation ---")
        result = await self.coder.run_async(
            "Create two files: 'file1.txt' with content 'Hello' and 'file2.txt' with content 'World', "
            "then read both files and combine their contents into 'combined.txt'"
        )
        
        # Check that all files were created
        file1 = self.work_dir / "file1.txt"
        file2 = self.work_dir / "file2.txt"
        combined = self.work_dir / "combined.txt"
        
        self.assertTrue(file1.exists(), "file1.txt should exist")
        self.assertTrue(file2.exists(), "file2.txt should exist")
        self.assertTrue(combined.exists(), "combined.txt should exist")
        
        # Check combined content
        if combined.exists():
            content = combined.read_text()
            self.assertIn("Hello", content)
            self.assertIn("World", content)


# ==============================================================================
# PART 7: CODER TOOL E2E TESTS
# ==============================================================================

@unittest.skipIf(not all([CoderTool, LiteLLMModel]), "Skipping tool tests due to missing dependencies.")
class TestCoderTool_E2E(unittest.TestCase):
    """
    Suite 5: Tests for the CoderTool wrapper with a live LLM.
    Tests the tool interface for use in multi-agent orchestration.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment with a live API connection."""
        cls.work_dir = Path(tempfile.mkdtemp())
        
        api_key = os.environ.get("NAUTILUS_API_KEY")
        if not api_key:
            raise unittest.SkipTest("NAUTILUS_API_KEY not set. Skipping tool tests.")
        
        os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        
        try:
            model = LiteLLMModel(model_id="openai/llama3")
            cls.tool = CoderTool(
                model=model,
                work_dir=cls.work_dir,
                max_debug_rounds=2
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to initialize model, skipping tool tests: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if cls.work_dir.exists():
            shutil.rmtree(cls.work_dir)
    
    def test_tool_forward_calculation(self):
        """Test that the CoderTool.forward() method works for calculations."""
        print("\n--- Running Tool Test: Forward Method (Calculation) ---")
        result = self.tool.forward("Calculate the factorial of 6 and print the result")
        
        # Factorial of 6 is 720
        self.assertTrue(
            "720" in str(result),
            f"Expected '720' in result: {result}"
        )
    
    def test_tool_forward_file_operation(self):
        """Test that the CoderTool.forward() method works for file operations."""
        print("\n--- Running Tool Test: Forward Method (File Operation) ---")
        result = self.tool.forward("Create a file 'tool_test.txt' with the content 'Testing CoderTool'")
        
        # Check that file was created
        test_file = self.work_dir / "tool_test.txt"
        self.assertTrue(
            test_file.exists(),
            f"File {test_file} should have been created"
        )
        
        if test_file.exists():
            content = test_file.read_text()
            self.assertIn("Testing CoderTool", content)
    
    def test_tool_forward_data_analysis(self):
        """Test that the CoderTool.forward() method works for data analysis."""
        print("\n--- Running Tool Test: Forward Method (Data Analysis) ---")
        result = self.tool.forward(
            "Create a list of even numbers from 2 to 20 and calculate their average"
        )
        
        # Average of even numbers 2-20 is 11
        self.assertTrue(
            "11" in str(result),
            f"Expected '11' (average) in result: {result}"
        )


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Starting Comprehensive CoderAgent Test Suite")
    print("Based on magentic-ui Coder implementation standards")
    print("="*80)
    print("This suite contains:")
    print("  - Unit tests for initialization")
    print("  - Unit tests for code execution")
    print("  - Unit tests for CoderTool")
    print("  - Live E2E tests (require NAUTILUS_API_KEY)")
    print("="*80)
    unittest.main(verbosity=2)

