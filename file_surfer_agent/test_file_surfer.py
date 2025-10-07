import unittest
import os
import shutil

# ==============================================================================
# PART 1: COMPONENT AND TOOL DEFINITIONS
# ==============================================================================

try:
    from markdown_file_browser import MarkdownFileBrowser
    from file_surfer import FileSurfer
    from smolagents import LiteLLMModel
except ImportError as e:
    print("="*80 + f"\nFATAL ERROR: Could not import a required module: {e}.\n"
          "Please ensure `markdown_file_browser.py` and `file_surfer.py` are in the same directory.\n" + "="*80)
    exit()

# --- Tool Definitions ---
browser = None 

def get_browser_state() -> str:
    """Helper function to get the current view of the browser for testing."""
    header = f"Path: {browser.path}\nTitle: {browser.page_title}\n"
    header += f"Viewport position: Showing page {browser.viewport_current_page + 1} of {len(browser.viewport_pages)}.\n"
    return header.strip() + "\n=======================\n" + browser.viewport

def open_file_or_directory(path: str) -> str:
    try: browser.open_path(path); return get_browser_state()
    except Exception as e: return f"Error opening path {path}: {e}"
def page_down() -> str: browser.page_down(); return get_browser_state()
def page_up() -> str: browser.page_up(); return get_browser_state()
def find_on_page(s: str) -> str: browser.find_on_page(s); return get_browser_state()
def find_next() -> str: browser.find_next(); return get_browser_state()


# ==============================================================================
# PART 2: THE TEST SUITES
# ==============================================================================

class TestFileBrowserTools(unittest.TestCase):
    """
    Suite 1: Integration tests for the tool wrappers against the real 
    MarkdownFileBrowser class.
    """
    @classmethod
    def setUpClass(cls):
        cls.workspace = "tool_test_workspace"
        if os.path.exists(cls.workspace): shutil.rmtree(cls.workspace)
        os.makedirs(os.path.join(cls.workspace, "subdir"))
        
        with open(os.path.join(cls.workspace, "test1.txt"), "w") as f: f.write("Hello")

        # Create a file that is unambiguously large to test finding content.
        with open(os.path.join(cls.workspace, "large_find_test.txt"), "w") as f:
            f.write("start_of_file\n" * 500)
            f.write("middle_of_file\n")
            f.write("end_of_file\n" * 500)
    
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace)

    def setUp(self):
        """This runs before each test in this class."""
        global browser
        browser = MarkdownFileBrowser(base_path=self.workspace, viewport_size=1024)
    
    def test_open_real_directory(self):
        """Tests if the browser can correctly list directory contents in a table."""
        result = open_file_or_directory(".")
        self.assertIn("test1.txt", result)
        self.assertIn("subdir/", result)
        self.assertIn("| Name", result) # Checks for the Markdown table header

    def test_find_in_large_file(self):
        """
        Tests that the `find_on_page` tool can locate content anywhere in a 
        large file and updates the viewport to show it.
        """
        open_file_or_directory("large_find_test.txt")
        
        # Find a keyword that is definitely not on the first page.
        result = find_on_page("middle_of_file")
        
        # THE FIX: The tool's job is to move the viewport. The correct test is to
        # check if the search term is now visible in the returned viewport content.
        self.assertIn("middle_of_file", result)
        # We can also confirm that paging is active and we are no longer on page 1.
        self.assertNotIn("Showing page 1 of", result)


@unittest.skipIf(not all([FileSurfer, LiteLLMModel]), "Skipping E2E tests due to missing dependencies.")
class TestFileSurfer_AgentE2E(unittest.TestCase):
    """
    Suite 2: End-to-end tests for the complete FileSurfer agent with a live LLM.
    """
    @classmethod
    def setUpClass(cls):
        cls.base_dir = "agent_test_project"
        nested_dir = os.path.join(cls.base_dir, "src", "components")
        docs_dir = os.path.join(cls.base_dir, "docs")
        if os.path.exists(cls.base_dir): shutil.rmtree(cls.base_dir)
        os.makedirs(nested_dir); os.makedirs(docs_dir)

        with open(os.path.join(docs_dir, "long_document.txt"), "w") as f:
            f.write("This is a long document.\n" * 200 + "The critical keyword is at the end.\n")
        with open(os.path.join(nested_dir, "button.py"), "w") as f:
            f.write("import React from 'react';\nconst Button = () => {\n  // TODO: Implement the onClick handler\n  return <button>Click Me</button>;\n};\nexport default Button;\n")
        
        cls.outside_file = "secret_file.txt"
        with open(cls.outside_file, "w") as f: f.write("This file should not be accessible.")

        api_key = os.environ.get("NAUTILUS_API_KEY")
        if not api_key: raise unittest.SkipTest("NAUTILUS_API_KEY not set. Skipping E2E tests.")
        
        os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        
        try:
            model = LiteLLMModel(model_id="openai/llama3")
            cls.file_surfer = FileSurfer(model=model, base_path=cls.base_dir)
        except Exception as e:
            raise unittest.SkipTest(f"Failed to initialize model, skipping E2E tests: {e}")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.base_dir): shutil.rmtree(cls.base_dir)
        if os.path.exists(cls.outside_file): os.remove(cls.outside_file)

    def test_1_list_nested_directory(self):
        print("\n--- Running E2E Test 1: List Nested Directory ---")
        result = self.file_surfer.run("Navigate into 'src/components' and list its contents.")
        self.assertIn("button.py", result)

    def test_2_read_end_of_long_file(self):
        print("\n--- Running E2E Test 2: Read End of Long File ---")
        result = self.file_surfer.run("Open 'docs/long_document.txt', find 'critical keyword', and tell me the sentence.")
        self.assertIn("critical keyword", result.lower())

    def test_3_find_in_file(self):
        print("\n--- Running E2E Test 3: Find in File ---")
        result = self.file_surfer.run("Open 'src/components/button.py', find the 'TODO' comment, and tell me what it says.")
        self.assertIn("Implement the onClick handler", result)

    def test_4_security_access_denied(self):
        print("\n--- Running E2E Test 4: Security - Access Denied ---")
        malicious_path = os.path.abspath(self.outside_file)
        result = self.file_surfer.run(f"Open the file at this absolute path: '{malicious_path}' and tell me its contents.")
        self.assertTrue("not found" in result.lower() or "outside the allowed path" in result.lower() or "does not exist" in result.lower())

if __name__ == "__main__":
    print("="*80)
    print("Starting Comprehensive FileSurfer Test Suite")
    print("This suite contains both fast integration tests and slow, live-API E2E tests.")
    print("E2E tests will be skipped if NAUTILUS_API_KEY is not set.")
    print("="*80)
    unittest.main(verbosity=2)