"""
Comprehensive test suite for the WebSurferAgent.

This test suite is based on the magentic-ui WebSurfer implementation's expected behavior.
It includes both unit tests and end-to-end tests with a live LLM.
"""
import unittest
import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional

# ==============================================================================
# PART 1: IMPORTS
# ==============================================================================

import sys
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from websurfer_agent.web_surfer_agent import WebSurferAgent
    from websurfer_agent.web_surfer_tool import WebSurferTool
    from smolagents import LiteLLMModel
    from smolagents.models import OpenAIServerModel
except ImportError as e:
    print("="*80 + f"\nFATAL ERROR: Could not import a required module: {e}.\n"
          "Please ensure the websurfer_agent package is properly installed.\n" + "="*80)
    exit()


# ==============================================================================
# PART 2: HELPER FUNCTIONS
# ==============================================================================

def get_mock_model():
    """Get a mock model for unit tests that don't actually use the LLM."""
    # Create a simple model with dummy credentials for initialization only
    # The unit tests won't actually call the model
    try:
        model = OpenAIServerModel(
            model_id="mock-model",
            api_base="https://example.com",
            api_key="mock-key",
        )
        return model
    except Exception:
        # If model creation fails, return None and let tests skip
        return None

def async_test(coro):
    """Decorator to run async tests."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper


# ==============================================================================
# PART 3: UNIT TESTS
# ==============================================================================

class TestWebSurferAgentBasics(unittest.TestCase):
    """
    Suite 1: Basic tests for WebSurferAgent initialization and configuration.
    Tests core functionality without requiring a live LLM.
    """
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.downloads_folder = self.temp_dir / "downloads"
        self.downloads_folder.mkdir(exist_ok=True)
        self.debug_dir = self.temp_dir / "debug"
        self.debug_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment after each test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_with_defaults(self):
        """Test that WebSurferAgent can be initialized with default parameters."""
        # Create agent with mock model (won't actually use it for init test)
        agent = WebSurferAgent(
            name="TestWebSurfer",
            model=get_mock_model(),
            downloads_folder=str(self.downloads_folder),
            debug_dir=str(self.debug_dir),
        )
        
        # Check basic attributes
        self.assertEqual(agent.name, "TestWebSurfer")
        self.assertEqual(agent.start_page, WebSurferAgent.DEFAULT_START_PAGE)
        self.assertEqual(agent.max_actions_per_step, 5)
        self.assertFalse(agent.did_lazy_init)
        self.assertIsNotNone(agent._browser)
        self.assertIsNotNone(agent._playwright_controller)
    
    def test_initialization_with_custom_config(self):
        """Test that WebSurferAgent can be initialized with custom configuration."""
        custom_start = "https://www.example.com"
        
        agent = WebSurferAgent(
            name="CustomWebSurfer",
            model=get_mock_model(),
            start_page=custom_start,
            downloads_folder=str(self.downloads_folder),
            debug_dir=str(self.debug_dir),
            max_actions_per_step=10,
            viewport_height=1080,
            viewport_width=1920,
            animate_actions=True,
            to_save_screenshots=True,
            single_tab_mode=True,
        )
        
        # Check custom attributes
        self.assertEqual(agent.name, "CustomWebSurfer")
        # Note: start_page might be changed to about:blank if not in allow list
        self.assertIn(agent.start_page, [custom_start, "about:blank"])
        self.assertEqual(agent.max_actions_per_step, 10)
        self.assertEqual(agent.viewport_height, 1080)
        self.assertEqual(agent.viewport_width, 1920)
        self.assertTrue(agent.animate_actions)
        self.assertTrue(agent.to_save_screenshots)
        self.assertTrue(agent.single_tab_mode)
    
    def test_initialization_url_blocking(self):
        """Test that URL blocking configuration works."""
        url_block_list = ["malicious.com", "spam.com"]
        
        agent = WebSurferAgent(
            name="SecureWebSurfer",
            model=get_mock_model(),
            url_block_list=url_block_list,
            downloads_folder=str(self.downloads_folder),
        )
        
        # Check that URLs are blocked
        self.assertTrue(agent._url_status_manager.is_url_blocked("https://malicious.com"))
        self.assertTrue(agent._url_status_manager.is_url_blocked("https://spam.com/page"))
        self.assertFalse(agent._url_status_manager.is_url_blocked("https://google.com"))
    
    def test_initialization_url_allow_list(self):
        """Test that URL allow list configuration works."""
        url_statuses = {
            "example.com": "allowed",
            "test.com": "allowed",
        }
        
        agent = WebSurferAgent(
            name="RestrictedWebSurfer",
            model=get_mock_model(),
            url_statuses=url_statuses,
            downloads_folder=str(self.downloads_folder),
        )
        
        # Check that specific URLs are allowed
        self.assertTrue(agent._url_status_manager.is_url_allowed("https://example.com"))
        self.assertTrue(agent._url_status_manager.is_url_allowed("https://test.com/page"))
        # about:blank should always be allowed
        self.assertTrue(agent._url_status_manager.is_url_allowed("about:blank"))
    
    def test_search_engine_configuration(self):
        """Test that different search engines can be configured."""
        for search_engine in ["google", "bing", "yahoo", "duckduckgo"]:
            agent = WebSurferAgent(
                name="TestWebSurfer",
                model=get_mock_model(),
                search_engine=search_engine,
                downloads_folder=str(self.downloads_folder),
            )
            self.assertEqual(agent.search_engine, search_engine)
            
            # Test URL generation
            url, domain = agent._get_search_url("test query")
            self.assertIn(search_engine, url.lower())
            self.assertIn("test", url)
    
    def test_screenshots_require_debug_dir(self):
        """Test that saving screenshots requires a debug directory."""
        with self.assertRaises(ValueError):
            agent = WebSurferAgent(
                name="TestWebSurfer",
                model=get_mock_model(),
                to_save_screenshots=True,
                debug_dir=None,  # This should cause an error
            )
    
    def test_web_surfer_tool_initialization(self):
        """Test that WebSurferTool can be initialized properly."""
        tool = WebSurferTool(
            model=get_mock_model(),
            downloads_folder=str(self.downloads_folder),
        )
        
        self.assertEqual(tool.name, "web_surfer")
        self.assertIsNotNone(tool.description)
        self.assertIn("task", tool.inputs)
        self.assertEqual(tool.output_type, "string")


# ==============================================================================
# PART 4: ASYNC UNIT TESTS
# ==============================================================================

class TestWebSurferAgentAsync(unittest.TestCase):
    """
    Suite 2: Async tests for WebSurferAgent methods.
    Tests async functionality without requiring a live LLM.
    """
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.downloads_folder = self.temp_dir / "downloads"
        self.downloads_folder.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment after each test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @async_test
    async def test_lazy_initialization(self):
        """Test that lazy initialization works correctly."""
        agent = WebSurferAgent(
            name="TestWebSurfer",
            model=get_mock_model(),
            downloads_folder=str(self.downloads_folder),
        )
        
        # Should not be initialized yet
        self.assertFalse(agent.did_lazy_init)
        
        # Trigger lazy init
        await agent.lazy_init()
        
        # Should now be initialized
        self.assertTrue(agent.did_lazy_init)
        self.assertTrue(agent._browser_just_initialized)
        
        # Cleanup
        await agent.close()
    
    @async_test
    async def test_state_management(self):
        """Test that state can be saved and loaded."""
        agent = WebSurferAgent(
            name="TestWebSurfer",
            model=get_mock_model(),
            downloads_folder=str(self.downloads_folder),
        )
        
        # Add some state
        agent._chat_history = [
            {"role": "user", "content": "test message"},
            {"role": "assistant", "content": "test response"},
        ]
        agent._last_outside_message = "test message"
        
        # Save state
        state = await agent.save_state(save_browser=False)
        
        # Verify saved state
        self.assertEqual(len(state["chat_history"]), 2)
        self.assertEqual(state["last_outside_message"], "test message")
        
        # Create new agent and load state
        new_agent = WebSurferAgent(
            name="TestWebSurfer2",
            model=get_mock_model(),
            downloads_folder=str(self.downloads_folder),
        )
        
        await new_agent.load_state(state, load_browser=False)
        
        # Verify loaded state
        self.assertEqual(len(new_agent._chat_history), 2)
        self.assertEqual(new_agent._last_outside_message, "test message")
        
        # Cleanup
        await agent.close()
        await new_agent.close()
    
    @async_test
    async def test_pause_and_resume(self):
        """Test that agent can be paused and resumed."""
        agent = WebSurferAgent(
            name="TestWebSurfer",
            model=get_mock_model(),
            downloads_folder=str(self.downloads_folder),
        )
        
        # Initially not paused
        self.assertFalse(agent.is_paused)
        
        # Pause
        await agent.pause()
        self.assertTrue(agent.is_paused)
        
        # Resume
        await agent.resume()
        self.assertFalse(agent.is_paused)
        
        # Cleanup
        await agent.close()
    
    @async_test
    async def test_chat_history_management(self):
        """Test that chat history can be managed."""
        agent = WebSurferAgent(
            name="TestWebSurfer",
            model=get_mock_model(),
            downloads_folder=str(self.downloads_folder),
        )
        
        # Initially empty
        self.assertEqual(len(agent.get_chat_history()), 0)
        
        # Add messages
        agent._chat_history.append({"role": "user", "content": "message 1"})
        agent._chat_history.append({"role": "assistant", "content": "response 1"})
        
        # Verify history
        history = agent.get_chat_history()
        self.assertEqual(len(history), 2)
        
        # Clear history
        agent.clear_chat_history()
        self.assertEqual(len(agent.get_chat_history()), 0)
        
        # Cleanup
        await agent.close()
    
    @async_test
    async def test_context_manager(self):
        """Test that agent works as an async context manager."""
        temp_dir = Path(tempfile.mkdtemp())
        downloads = temp_dir / "downloads"
        downloads.mkdir(exist_ok=True)
        
        async with WebSurferAgent(
            name="TestWebSurfer",
            model=get_mock_model(),
            downloads_folder=str(downloads),
        ) as agent:
            # Should be initialized
            self.assertTrue(agent.did_lazy_init)
        
        # Cleanup temp dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# ==============================================================================
# PART 5: END-TO-END TESTS WITH LIVE LLM
# ==============================================================================

@unittest.skipIf(not all([WebSurferAgent, LiteLLMModel]), "Skipping E2E tests due to missing dependencies.")
class TestWebSurferAgent_E2E(unittest.TestCase):
    """
    Suite 3: End-to-end tests for the complete WebSurferAgent with a live LLM.
    These tests validate the actual browsing and interaction capabilities.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment with a live API connection."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.downloads_folder = cls.temp_dir / "downloads"
        cls.downloads_folder.mkdir(exist_ok=True)
        cls.debug_dir = cls.temp_dir / "debug"
        cls.debug_dir.mkdir(exist_ok=True)
        
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
            
            # Create the web surfer agent
            cls.agent = WebSurferAgent(
                name="TestWebSurfer",
                model=model,
                downloads_folder=str(cls.downloads_folder),
                debug_dir=str(cls.debug_dir),
                to_save_screenshots=True,
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to initialize model, skipping E2E tests: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if hasattr(cls, 'agent'):
            # Run async cleanup
            loop = asyncio.get_event_loop()
            loop.run_until_complete(cls.agent.close())
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    @async_test
    async def test_1_simple_web_search(self):
        """Test that the agent can perform a simple web search."""
        print("\n--- Running E2E Test 1: Simple Web Search ---")
        result = await self.agent.run("Search for 'Python programming language' and tell me what you find")
        
        # The result should mention Python
        self.assertTrue(
            "python" in result.lower(),
            f"Expected 'python' in result, got: {result}"
        )
    
    @async_test
    async def test_2_visit_specific_url(self):
        """Test that the agent can visit a specific URL."""
        print("\n--- Running E2E Test 2: Visit Specific URL ---")
        result = await self.agent.run("Go to https://www.example.com and tell me what you see")
        
        # Should mention example.com or domain
        result_lower = result.lower()
        self.assertTrue(
            "example" in result_lower or "domain" in result_lower or "illustrative" in result_lower,
            f"Expected to find content from example.com in result: {result}"
        )
    
    @async_test
    async def test_3_multi_step_interaction(self):
        """Test that the agent can perform multiple steps in sequence."""
        print("\n--- Running E2E Test 3: Multi-Step Interaction ---")
        result = await self.agent.run(
            "Search for 'Wikipedia', click on the first result, and tell me what the page is about"
        )
        
        # Should complete multiple actions
        self.assertTrue(
            len(result) > 50,
            f"Expected a substantial response from multi-step interaction: {result}"
        )
    
    @async_test
    async def test_4_question_answering_from_page(self):
        """Test that the agent can answer questions about page content."""
        print("\n--- Running E2E Test 4: Question Answering ---")
        result = await self.agent.run(
            "Go to https://www.example.com and answer this question: What is the main purpose of this page?"
        )
        
        # Should provide an answer
        self.assertTrue(
            len(result) > 20,
            f"Expected a detailed answer about the page: {result}"
        )
    
    @async_test
    async def test_5_blocked_url_handling(self):
        """Test that the agent properly handles blocked URLs."""
        print("\n--- Running E2E Test 5: Blocked URL Handling ---")
        
        # Create agent with blocked URL
        temp_dir = Path(tempfile.mkdtemp())
        downloads = temp_dir / "downloads"
        downloads.mkdir(exist_ok=True)
        
        api_key = os.environ.get("NAUTILUS_API_KEY")
        os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        
        model = LiteLLMModel(model_id="openai/llama3")
        
        agent = WebSurferAgent(
            name="RestrictedWebSurfer",
            model=model,
            downloads_folder=str(downloads),
            url_block_list=["malicious.com"],
        )
        
        result = await agent.run("Go to https://malicious.com and tell me what you see")
        
        # Should indicate that the URL is blocked
        self.assertTrue(
            "not allowed" in result.lower() or "blocked" in result.lower(),
            f"Expected indication that URL is blocked: {result}"
        )
        
        # Cleanup
        await agent.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# ==============================================================================
# PART 6: WEB SURFER TOOL TESTS
# ==============================================================================

@unittest.skipIf(not all([WebSurferTool, LiteLLMModel]), "Skipping tool tests due to missing dependencies.")
class TestWebSurferTool_E2E(unittest.TestCase):
    """
    Suite 4: Tests for the WebSurferTool wrapper.
    Tests the tool interface for use in multi-agent systems.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment with a live API connection."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.downloads_folder = cls.temp_dir / "downloads"
        cls.downloads_folder.mkdir(exist_ok=True)
        
        api_key = os.environ.get("NAUTILUS_API_KEY")
        if not api_key:
            raise unittest.SkipTest("NAUTILUS_API_KEY not set. Skipping tool tests.")
        
        os.environ["OPENAI_API_BASE"] = "https://ellm.nrp-nautilus.io/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        
        try:
            model = LiteLLMModel(model_id="openai/llama3")
            cls.tool = WebSurferTool(
                model=model,
                downloads_folder=str(cls.downloads_folder),
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to initialize model, skipping tool tests: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def test_tool_forward_simple_search(self):
        """Test that the WebSurferTool.forward() method works for simple searches."""
        print("\n--- Running Tool Test: Forward Method (Search) ---")
        result = self.tool.forward("Search for 'artificial intelligence' and summarize what you find")
        
        # Should return a string result
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 20, f"Expected substantial result: {result}")
    
    def test_tool_forward_url_visit(self):
        """Test that the WebSurferTool.forward() method works for URL visits."""
        print("\n--- Running Tool Test: Forward Method (URL Visit) ---")
        result = self.tool.forward("Visit https://www.example.com and describe the page")
        
        # Should return a string result
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 10, f"Expected result from URL visit: {result}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Starting Comprehensive WebSurferAgent Test Suite")
    print("Based on magentic-ui WebSurfer implementation standards")
    print("="*80)
    print("This suite contains:")
    print("  - Fast unit tests (no LLM required)")
    print("  - Async unit tests (no LLM required)")
    print("  - Live E2E tests (require NAUTILUS_API_KEY)")
    print("="*80)
    unittest.main(verbosity=2)

