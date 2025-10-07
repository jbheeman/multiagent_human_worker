"""
Comprehensive test suite for the FileSurfer agent.

This test suite is based on the magentic-ui FileSurfer implementation's expected behavior.
It includes unit tests, integration tests, and end-to-end tests with a live LLM.
"""
import unittest
import os
import shutil
import tempfile
from pathlib import Path

# ==============================================================================
# PART 1: IMPORTS
# ==============================================================================

import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from file_surfer_agent.markdown_file_browser import MarkdownFileBrowser
    from file_surfer_agent.file_surfer import FileSurfer
    from file_surfer_agent.file_surfer_tool import FileSurferTool
    from smolagents import LiteLLMModel
except ImportError as e:
    print("="*80 + f"\nFATAL ERROR: Could not import a required module: {e}.\n"
          "Please ensure the file_surfer_agent package is properly installed.\n" + "="*80)
    exit()


# ==============================================================================
# PART 2: UNIT TESTS - MARKDOWN FILE BROWSER
# ==============================================================================

class TestMarkdownFileBrowser(unittest.TestCase):
    """
    Suite 1: Unit tests for the MarkdownFileBrowser component.
    Tests the file browser functionality without requiring LLM.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test workspace with sample files and directories."""
        cls.workspace = Path(tempfile.mkdtemp())
        
        # Create directory structure
        (cls.workspace / "subdir").mkdir()
        (cls.workspace / "nested" / "deep").mkdir(parents=True)
        
        # Create test files
        (cls.workspace / "test1.txt").write_text("Hello World")
        (cls.workspace / "test2.py").write_text("print('Python file')")
        (cls.workspace / "subdir" / "nested_file.txt").write_text("Nested content")
        
        # Create a large file for pagination testing
        large_content = "start_of_file\n" * 500
        large_content += "middle_of_file\n"
        large_content += "end_of_file\n" * 500
        (cls.workspace / "large_file.txt").write_text(large_content)
        
        # Create a file with specific patterns for search testing
        search_content = "Line 1\n" * 100
        search_content += "SEARCH_TARGET_LINE\n"
        search_content += "Line 2\n" * 100
        (cls.workspace / "search_test.txt").write_text(search_content)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test workspace."""
        if cls.workspace.exists():
            shutil.rmtree(cls.workspace)
    
    def setUp(self):
        """Create fresh browser instance for each test."""
        self.browser = MarkdownFileBrowser(
            base_path=str(self.workspace),
            viewport_size=1024
        )
    
    def test_initialization(self):
        """Test that browser initializes correctly."""
        self.assertEqual(Path(self.browser.base_path), self.workspace)
        self.assertEqual(self.browser.viewport_size, 1024)
        self.assertIsNotNone(self.browser.path)
    
    def test_open_directory(self):
        """Test opening and listing directory contents."""
        self.browser.open_path(".")
        viewport = self.browser.viewport
        
        # Should show directory listing with table format
        self.assertIn("| Name", viewport)
        self.assertIn("test1.txt", viewport)
        self.assertIn("test2.py", viewport)
        self.assertIn("subdir/", viewport)
    
    def test_open_file(self):
        """Test opening and viewing file contents."""
        self.browser.open_path("test1.txt")
        viewport = self.browser.viewport
        
        # Should show file contents
        self.assertIn("Hello World", viewport)
    
    def test_navigate_nested_directory(self):
        """Test navigating to nested directories."""
        self.browser.open_path("subdir")
        viewport = self.browser.viewport
        
        # Should show nested directory contents
        self.assertIn("nested_file.txt", viewport)
    
    def test_pagination_page_down(self):
        """Test that page_down works for large files."""
        self.browser.open_path("large_file.txt")
        
        # Initially on first page
        self.assertEqual(self.browser.viewport_current_page, 0)
        
        # Page down
        self.browser.page_down()
        
        # Should be on a different page
        self.assertGreater(self.browser.viewport_current_page, 0)
    
    def test_pagination_page_up(self):
        """Test that page_up works for large files."""
        self.browser.open_path("large_file.txt")
        
        # Go to second page
        self.browser.page_down()
        current_page = self.browser.viewport_current_page
        
        # Page up
        self.browser.page_up()
        
        # Should be back on previous page
        self.assertEqual(self.browser.viewport_current_page, current_page - 1)
    
    def test_find_on_page(self):
        """Test finding text in a file."""
        self.browser.open_path("large_file.txt")
        
        # Find text that's not on first page
        self.browser.find_on_page("middle_of_file")
        
        # Should move viewport to show the found text
        self.assertIn("middle_of_file", self.browser.viewport)
    
    def test_find_next(self):
        """Test finding next occurrence of search term."""
        self.browser.open_path("large_file.txt")
        
        # Find first occurrence
        self.browser.find_on_page("start_of_file")
        first_page = self.browser.viewport_current_page
        
        # Find next occurrence
        self.browser.find_next()
        
        # Might be on same page or different page depending on viewport size
        self.assertIn("start_of_file", self.browser.viewport)
    
    def test_search_not_found(self):
        """Test searching for non-existent text."""
        self.browser.open_path("test1.txt")
        
        # Try to find something that doesn't exist
        self.browser.find_on_page("NONEXISTENT_TEXT")
        
        # Should handle gracefully (implementation may vary)
        # Just ensure it doesn't crash
        self.assertIsNotNone(self.browser.viewport)
    
    def test_page_title(self):
        """Test that page title is set correctly."""
        self.browser.open_path("test1.txt")
        
        # Title should reflect the file name
        self.assertIn("test1.txt", self.browser.page_title)
    
    def test_viewport_pages_count(self):
        """Test that viewport pages are calculated correctly."""
        self.browser.open_path("large_file.txt")
        
        # Should have multiple pages
        self.assertGreater(len(self.browser.viewport_pages), 1)


# ==============================================================================
# PART 3: INTEGRATION TESTS - FILE BROWSER TOOLS
# ==============================================================================

class TestFileBrowserTools(unittest.TestCase):
    """
    Suite 2: Integration tests for the file browser tool wrappers.
    Tests the tool interface layer.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test workspace."""
        cls.workspace = Path(tempfile.mkdtemp())
        (cls.workspace / "subdir").mkdir()
        (cls.workspace / "test1.txt").write_text("Hello")
        
        # Large file for testing
        large_content = "start_of_file\n" * 500
        large_content += "middle_of_file\n"
        large_content += "end_of_file\n" * 500
        (cls.workspace / "large_find_test.txt").write_text(large_content)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test workspace."""
        if cls.workspace.exists():
            shutil.rmtree(cls.workspace)
    
    def setUp(self):
        """Create fresh browser for each test."""
        self.browser = MarkdownFileBrowser(
            base_path=str(self.workspace),
            viewport_size=1024
        )
    
    def get_browser_state(self) -> str:
        """Helper to get current browser state."""
        header = f"Path: {self.browser.path}\nTitle: {self.browser.page_title}\n"
        header += f"Viewport position: Showing page {self.browser.viewport_current_page + 1} of {len(self.browser.viewport_pages)}.\n"
        return header.strip() + "\n=======================\n" + self.browser.viewport
    
    def open_file_or_directory(self, path: str) -> str:
        """Tool wrapper for opening files/directories."""
        try:
            self.browser.open_path(path)
            return self.get_browser_state()
        except Exception as e:
            return f"Error opening path {path}: {e}"
    
    def page_down(self) -> str:
        """Tool wrapper for paging down."""
        self.browser.page_down()
        return self.get_browser_state()
    
    def page_up(self) -> str:
        """Tool wrapper for paging up."""
        self.browser.page_up()
        return self.get_browser_state()
    
    def find_on_page(self, search_term: str) -> str:
        """Tool wrapper for finding on page."""
        self.browser.find_on_page(search_term)
        return self.get_browser_state()
    
    def test_open_directory_tool(self):
        """Test opening directory through tool interface."""
        result = self.open_file_or_directory(".")
        
        self.assertIn("test1.txt", result)
        self.assertIn("subdir/", result)
        self.assertIn("| Name", result)
    
    def test_find_in_large_file_tool(self):
        """Test finding in large file through tool interface."""
        self.open_file_or_directory("large_find_test.txt")
        result = self.find_on_page("middle_of_file")
        
        # Should show the found text in viewport
        self.assertIn("middle_of_file", result)
        # Should not be on first page
        self.assertNotIn("Showing page 1 of", result)
    
    def test_page_navigation_tools(self):
        """Test page up/down through tool interface."""
        self.open_file_or_directory("large_find_test.txt")
        
        # Page down
        result = self.page_down()
        self.assertNotIn("Showing page 1 of", result)
        
        # Page up
        result = self.page_up()
        self.assertIn("Showing page 1 of", result)


# ==============================================================================
# PART 4: UNIT TESTS - FILE SURFER AGENT
# ==============================================================================

class TestFileSurferInitialization(unittest.TestCase):
    """
    Suite 3: Tests for FileSurfer agent initialization.
    Tests configuration and setup without requiring LLM.
    """
    
    def setUp(self):
        """Create test workspace."""
        self.workspace = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test workspace."""
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
    
    def test_initialization_with_base_path(self):
        """Test that FileSurfer can be initialized with a base path."""
        agent = FileSurfer(
            model=None,
            base_path=str(self.workspace)
        )
        
        # Check initialization
        self.assertEqual(Path(agent.base_path), self.workspace)
        self.assertIsNotNone(agent.browser)
    
    def test_initialization_default_name(self):
        """Test that FileSurfer has correct default name."""
        agent = FileSurfer(
            model=None,
            base_path=str(self.workspace)
        )
        
        # Check default name
        self.assertEqual(agent.name, "FileSurfer")
    
    def test_initialization_custom_viewport(self):
        """Test that FileSurfer can use custom viewport size."""
        agent = FileSurfer(
            model=None,
            base_path=str(self.workspace),
            viewport_size=2048
        )
        
        # Check viewport size
        self.assertEqual(agent.browser.viewport_size, 2048)


# ==============================================================================
# PART 5: UNIT TESTS - FILE SURFER TOOL
# ==============================================================================

class TestFileSurferTool(unittest.TestCase):
    """
    Suite 4: Tests for the FileSurferTool wrapper.
    Tests the tool interface for use in multi-agent systems.
    """
    
    def setUp(self):
        """Create test workspace."""
        self.workspace = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test workspace."""
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
    
    def test_tool_initialization(self):
        """Test that FileSurferTool can be initialized."""
        tool = FileSurferTool(
            model=None,
            base_path=str(self.workspace)
        )
        
        # Check tool attributes
        self.assertEqual(tool.name, "file_surfer")
        self.assertIsNotNone(tool.description)
        self.assertIn("task", tool.inputs)
        self.assertEqual(tool.output_type, "string")


# ==============================================================================
# PART 6: END-TO-END TESTS WITH LIVE LLM
# ==============================================================================

@unittest.skipIf(not all([FileSurfer, LiteLLMModel]), "Skipping E2E tests due to missing dependencies.")
class TestFileSurfer_E2E(unittest.TestCase):
    """
    Suite 5: End-to-end tests for the complete FileSurfer agent with a live LLM.
    Based on magentic-ui FileSurfer's expected behaviors.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample project structure."""
        cls.base_dir = Path(tempfile.mkdtemp()) / "agent_test_project"
        
        # Create project structure
        nested_dir = cls.base_dir / "src" / "components"
        docs_dir = cls.base_dir / "docs"
        nested_dir.mkdir(parents=True)
        docs_dir.mkdir(parents=True)
        
        # Create test files
        (docs_dir / "long_document.txt").write_text(
            "This is a long document.\n" * 200 + "The critical keyword is at the end.\n"
        )
        
        (nested_dir / "button.py").write_text(
            "import React from 'react';\n"
            "const Button = () => {\n"
            "  // TODO: Implement the onClick handler\n"
            "  return <button>Click Me</button>;\n"
            "};\n"
            "export default Button;\n"
        )
        
        (cls.base_dir / "README.md").write_text(
            "# Test Project\n\n"
            "This is a test project for FileSurfer.\n"
        )
        
        (cls.base_dir / "config.json").write_text(
            '{"name": "test", "version": "1.0.0"}'
        )
        
        # Create file outside base directory for security testing
        cls.outside_file = Path(tempfile.mkdtemp()) / "secret_file.txt"
        cls.outside_file.write_text("This file should not be accessible.")
        
        # Setup API
        api_key = os.environ.get("NAUTILUS_API_KEY")
        if not api_key:
            raise unittest.SkipTest("NAUTILUS_API_KEY not set. Skipping E2E tests.")
        
        os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        
        try:
            model = LiteLLMModel(model_id="openai/llama3")
            cls.file_surfer = FileSurfer(model=model, base_path=str(cls.base_dir))
        except Exception as e:
            raise unittest.SkipTest(f"Failed to initialize model, skipping E2E tests: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.base_dir.exists():
            shutil.rmtree(cls.base_dir)
        if cls.outside_file.exists():
            cls.outside_file.unlink()
            cls.outside_file.parent.rmdir()
    
    def test_1_list_root_directory(self):
        """Test listing root directory contents."""
        print("\n--- Running E2E Test 1: List Root Directory ---")
        result = self.file_surfer.run("List the files in the current directory.")
        
        result_lower = result.lower()
        self.assertTrue(
            "readme" in result_lower or "src" in result_lower or "docs" in result_lower,
            f"Expected directory contents in result: {result}"
        )
    
    def test_2_list_nested_directory(self):
        """Test navigating and listing nested directory."""
        print("\n--- Running E2E Test 2: List Nested Directory ---")
        result = self.file_surfer.run("Navigate to 'src/components' and list its contents.")
        
        self.assertIn("button.py", result)
    
    def test_3_read_file_content(self):
        """Test reading a specific file's content."""
        print("\n--- Running E2E Test 3: Read File Content ---")
        result = self.file_surfer.run("Open 'README.md' and tell me what it says.")
        
        result_lower = result.lower()
        self.assertTrue(
            "test project" in result_lower or "filesurfer" in result_lower,
            f"Expected README content in result: {result}"
        )
    
    def test_4_find_in_long_file(self):
        """Test finding specific content in a long file."""
        print("\n--- Running E2E Test 4: Find in Long File ---")
        result = self.file_surfer.run(
            "Open 'docs/long_document.txt', find 'critical keyword', and tell me the sentence."
        )
        
        self.assertIn("critical keyword", result.lower())
    
    def test_5_find_code_pattern(self):
        """Test finding specific code pattern in a file."""
        print("\n--- Running E2E Test 5: Find Code Pattern ---")
        result = self.file_surfer.run(
            "Open 'src/components/button.py', find the 'TODO' comment, and tell me what it says."
        )
        
        self.assertIn("Implement the onClick handler", result)
    
    def test_6_json_file_reading(self):
        """Test reading and understanding JSON files."""
        print("\n--- Running E2E Test 6: Read JSON File ---")
        result = self.file_surfer.run(
            "Open 'config.json' and tell me the project name."
        )
        
        self.assertIn("test", result.lower())
    
    def test_7_multi_file_search(self):
        """Test searching across multiple files."""
        print("\n--- Running E2E Test 7: Multi-File Search ---")
        result = self.file_surfer.run(
            "Find all Python files in the project and list them."
        )
        
        self.assertIn("button.py", result)
    
    def test_8_security_access_denied(self):
        """Test that files outside base path are not accessible."""
        print("\n--- Running E2E Test 8: Security - Access Denied ---")
        malicious_path = str(self.outside_file.absolute())
        result = self.file_surfer.run(
            f"Open the file at this absolute path: '{malicious_path}' and tell me its contents."
        )
        
        result_lower = result.lower()
        self.assertTrue(
            "not found" in result_lower or 
            "outside" in result_lower or 
            "does not exist" in result_lower or
            "cannot access" in result_lower,
            f"Expected access denial in result: {result}"
        )


# ==============================================================================
# PART 7: FILE SURFER TOOL E2E TESTS
# ==============================================================================

@unittest.skipIf(not all([FileSurferTool, LiteLLMModel]), "Skipping tool tests due to missing dependencies.")
class TestFileSurferTool_E2E(unittest.TestCase):
    """
    Suite 6: Tests for the FileSurferTool wrapper with a live LLM.
    Tests the tool interface for use in multi-agent orchestration.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.base_dir = Path(tempfile.mkdtemp()) / "tool_test_project"
        cls.base_dir.mkdir(parents=True)
        
        # Create test files
        (cls.base_dir / "data.txt").write_text("Sample data content")
        (cls.base_dir / "info.md").write_text("# Information\nThis is info.")
        
        api_key = os.environ.get("NAUTILUS_API_KEY")
        if not api_key:
            raise unittest.SkipTest("NAUTILUS_API_KEY not set. Skipping tool tests.")
        
        os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        
        try:
            model = LiteLLMModel(model_id="openai/llama3")
            cls.tool = FileSurferTool(model=model, base_path=str(cls.base_dir))
        except Exception as e:
            raise unittest.SkipTest(f"Failed to initialize model, skipping tool tests: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.base_dir.exists():
            shutil.rmtree(cls.base_dir)
    
    def test_tool_forward_list_files(self):
        """Test that FileSurferTool.forward() can list files."""
        print("\n--- Running Tool Test: Forward Method (List) ---")
        result = self.tool.forward("List all files in the current directory")
        
        # Should return file names
        self.assertTrue(
            "data.txt" in result or "info.md" in result,
            f"Expected file names in result: {result}"
        )
    
    def test_tool_forward_read_file(self):
        """Test that FileSurferTool.forward() can read files."""
        print("\n--- Running Tool Test: Forward Method (Read) ---")
        result = self.tool.forward("Read the content of 'data.txt' and summarize it")
        
        # Should return content or summary
        self.assertTrue(
            "sample" in result.lower() or "data" in result.lower(),
            f"Expected file content in result: {result}"
        )


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Starting Comprehensive FileSurfer Test Suite")
    print("Based on magentic-ui FileSurfer implementation standards")
    print("="*80)
    print("This suite contains:")
    print("  - Unit tests for MarkdownFileBrowser")
    print("  - Integration tests for file browser tools")
    print("  - Unit tests for FileSurfer initialization")
    print("  - Unit tests for FileSurferTool")
    print("  - Live E2E tests (require NAUTILUS_API_KEY)")
    print("="*80)
    unittest.main(verbosity=2)

