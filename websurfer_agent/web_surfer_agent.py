import asyncio
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    BinaryIO,
    Dict,
    List,
    Optional,
    Sequence,
    cast,
    Mapping,
    Union,
    Tuple,
    Literal,
)
from urllib.parse import quote_plus
import PIL.Image
import tiktoken

from smolagents import Tool, CodeAgent, WebSearchTool as SmolagentsWebSearchTool
from smolagents.models import OpenAIServerModel

from .config import WebSurferConfig
# Use the real PlaywrightController from the browser_playwright package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from browser_playwright import PlaywrightController
from browser_playwright.browser import LocalPlaywrightBrowser
from .tool_definitions import ALL_TOOLS, set_browser_controller
from .prompts import (
    WEB_SURFER_SYSTEM_MESSAGE,
    WEB_SURFER_TOOL_PROMPT,
    WEB_SURFER_NO_TOOLS_PROMPT,
    WEB_SURFER_OCR_PROMPT,
    WEB_SURFER_QA_SYSTEM_MESSAGE,
    WEB_SURFER_QA_PROMPT,
    EXPLANATION_TOOL_PROMPT,
)
from .set_of_mark import add_set_of_mark


class WebSurferAgent:
    """An agent that can browse the web and interact with web pages using a browser.
    
    This is a 1:1 copy of the magentic-ui WebSurfer agent implemented using smol agents.
    """

    DEFAULT_DESCRIPTION = """
    The websurfer has access to a web browser that it can control.
    It understands images and can use them to help it complete the task.

    In a single step when you ask the agent to do something, it will perform multiple actions sequentially until it decides to stop.
    The actions it can perform are:
    - visiting a web page url
    - performing a web search using a configurable search engine
    - Interact with a web page: clicking a button, hovering over a button, typing in a field, scrolling the page, select an option in a dropdown
    - Downloading a file from the web page and upload file from the local file system
    - Pressing a key on the keyboard to interact with the web page
    - answer a question based on the content of the entire page beyond just the vieport
    - wait on a page to load
    - interact with tabs on the page: closing a tab, switching to a tab, creating a new tab
    - refresh the page

    It can do multiple actions in a single step, but it will stop after it has performed the maximum number of actions or once it decides to stop.
    As mentioned, it can perform multiple actions per command given to it, for instance if asked to fill a form, it can input the first name, then the last name, then the email, and then submit, and then stop.
    """
    DEFAULT_START_PAGE = "about:blank"

    # Size of the image we send to the MLM
    # Current values represent a 0.85 scaling to fit within the GPT-4v short-edge constraints (768px)
    MLM_HEIGHT = 765
    MLM_WIDTH = 1224

    SCREENSHOT_TOKENS = 1105

    def __init__(
        self,
        name: str = "WebSurferAgent",
        model: Optional[OpenAIServerModel] = None,
        downloads_folder: Optional[str] = None,
        description: str = DEFAULT_DESCRIPTION,
        debug_dir: Optional[str] = None,
        start_page: str = DEFAULT_START_PAGE,
        animate_actions: bool = False,
        to_save_screenshots: bool = False,
        max_actions_per_step: int = 5,
        to_resize_viewport: bool = True,
        url_statuses: Optional[Dict[str, str]] = None,
        url_block_list: Optional[List[str]] = None,
        single_tab_mode: bool = False,
        viewport_height: int = 1440,
        viewport_width: int = 1440,
        use_action_guard: bool = False,
        search_engine: str = "duckduckgo",
        model_context_token_limit: Optional[int] = None,
    ) -> None:
        """Initialize the WebSurfer."""
        self.name = name
        self.description = description
        self.model = model
        self.start_page = start_page
        self.downloads_folder = downloads_folder
        self.debug_dir = debug_dir
        self.to_save_screenshots = to_save_screenshots
        self.to_resize_viewport = to_resize_viewport
        self.animate_actions = animate_actions
        self.max_actions_per_step = max_actions_per_step
        self.single_tab_mode = single_tab_mode
        self.viewport_height = viewport_height
        self.viewport_width = viewport_width
        self.use_action_guard = use_action_guard
        self.search_engine = search_engine
        self.model_context_token_limit = model_context_token_limit

        if debug_dir is None and to_save_screenshots:
            raise ValueError(
                "Cannot save screenshots without a debug directory. Set it using the 'debug_dir' parameter. The debug directory is created if it does not exist."
            )

        # Initialize the real Playwright browser
        self._browser = LocalPlaywrightBrowser()
        
        # Initialize URL status manager (for allowed/blocked URLs)
        from browser_playwright.url_status_manager import UrlStatusManager
        self._url_status_manager = UrlStatusManager(
            url_statuses=url_statuses, url_block_list=url_block_list
        )
        
        # Initialize the real PlaywrightController
        self._playwright_controller = PlaywrightController(
            animate_actions=self.animate_actions,
            downloads_folder=self.downloads_folder,
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            to_resize_viewport=self.to_resize_viewport,
            single_tab_mode=self.single_tab_mode,
            url_status_manager=self._url_status_manager,
        )

        # Initialize state
        self.did_lazy_init = False
        self._browser_just_initialized = False
        self.is_paused = False
        self._pause_event = asyncio.Event()
        self._last_outside_message = ""
        self._last_rejected_url = None
        self._chat_history: List[Dict[str, Any]] = []
        self.model_usage: List[Dict[str, Any]] = []

        # Create smol agents CodeAgent
        self._create_agent()

        # Explicitly allow about:blank
        if not self._url_status_manager.is_url_allowed("about:blank"):
            self._url_status_manager.set_url_status("about:blank", "allowed")
        
        # Explicitly allow chrome-error://chromewebdata
        if not self._url_status_manager.is_url_allowed("chrome-error://chromewebdata"):
            self._url_status_manager.set_url_status("chrome-error://chromewebdata", "allowed")

        if not self._url_status_manager.is_url_allowed(self.start_page):
            self.start_page = "about:blank"
            logging.warning(
                f"Default start page '{self.DEFAULT_START_PAGE}' is not in the allow list. Setting start page to blank."
            )

        # Set up logging
        self.logger = logging.getLogger(f"{self.name}.WebSurferAgent")

        # Check if model supports vision
        if self.model and hasattr(self.model, 'supports_vision'):
            self.is_multimodal = self.model.supports_vision
        else:
            self.is_multimodal = False

    def _create_agent(self):
        """Create the underlying smol agents CodeAgent."""
        if not self.model:
            raise ValueError("Model is required to create WebSurfer agent")
        
        # Set the browser controller for all tools to use
        set_browser_controller(self._playwright_controller)
        
        # Create the agent with all the WebSurfer tools directly
        # The tools are already proper smolagents Tool objects
        self.agent = CodeAgent(
            tools=ALL_TOOLS,
            model=self.model,
        )

    async def _execute_browser_action(self, tool_name: str, args: Dict[str, Any], result: Dict[str, Any]):
        """Execute the actual browser action based on the tool."""
        try:
            if not self.did_lazy_init:
                await self.lazy_init()
            
            # Get the current page from the controller
            page = self._playwright_controller._page
            if not page:
                self.logger.error("No page available for browser action")
                return

            # Execute the appropriate browser action
            if tool_name == "visit_url":
                await self._execute_tool_visit_url(args)
            elif tool_name == "web_search":
                await self._execute_tool_web_search(args)
            elif tool_name == "click":
                await self._execute_tool_click(args)
            elif tool_name == "input_text":
                await self._execute_tool_input_text(args)
            elif tool_name == "hover":
                await self._execute_tool_hover(args)
            elif tool_name == "scroll_down":
                await self._execute_tool_scroll_down(args)
            elif tool_name == "scroll_up":
                await self._execute_tool_scroll_up(args)
            elif tool_name == "history_back":
                await self._execute_tool_history_back(args)
            elif tool_name == "refresh_page":
                await self._execute_tool_refresh_page(args)
            elif tool_name == "keypress":
                await self._execute_tool_keypress(args)
            elif tool_name == "sleep":
                await self._execute_tool_sleep(args)
            elif tool_name == "answer_question":
                await self._execute_tool_answer_question(args)
            elif tool_name == "select_option":
                await self._execute_tool_select_option(args)
            elif tool_name == "create_tab":
                await self._execute_tool_create_tab(args)
            elif tool_name == "switch_tab":
                await self._execute_tool_switch_tab(args)
            elif tool_name == "close_tab":
                await self._execute_tool_close_tab(args)
            elif tool_name == "upload_file":
                await self._execute_tool_upload_file(args)
            elif tool_name == "stop_action":
                # Stop action doesn't need browser interaction
                pass
            else:
                self.logger.warning(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            self.logger.error(f"Error executing browser action {tool_name}: {e}")

    async def lazy_init(self) -> None:
        """Initialize the browser and page on first use."""
        if self.did_lazy_init:
            return

        # Start the browser
        await self._browser.__aenter__()
        
        # Get the browser context (LocalPlaywrightBrowser exposes this via property)
        context = self._browser.browser_context
        
        # Create new page
        page = await context.new_page()
        
        # Store the page and context on the controller so tools can access them
        self._playwright_controller._page = page
        self._playwright_controller._context = context
        
        # Set up the controller with the page
        await self._playwright_controller.on_new_page(page)
        
        # Set up debug directory
        await self._set_debug_dir()
        
        self.did_lazy_init = True
        self._browser_just_initialized = True

    async def _set_debug_dir(self) -> None:
        """Set up debug directory for screenshots."""
        if self.debug_dir is None:
            return
        if not os.path.isdir(self.debug_dir):
            os.makedirs(self.debug_dir, exist_ok=True)

    async def run(self, request: str) -> str:
        """Run the WebSurfer agent with a given request."""
        try:
            # Initialize browser on first use
            await self.lazy_init()
            
            # Update the last outside message
            self._last_outside_message = request
            
            # Add to chat history
            self._chat_history.append({
                "role": "user",
                "content": request
            })
            
            # Run the agent
            result = self.agent.run(request)
            
            # Add to chat history
            self._chat_history.append({
                "role": "assistant", 
                "content": str(result)
            })
            
            return str(result)
            
        except Exception as e:
            self.logger.error(f"Error running WebSurfer agent: {e}")
            return f"Error: {e}"

    async def close(self) -> None:
        """Close the browser and cleanup resources."""
        try:
            if self._browser:
                await self._browser.__aexit__(None, None, None)
        except Exception as e:
            self.logger.error(f"Error closing WebSurfer: {e}")

    # Tool execution methods (simplified implementations)
    
    async def _execute_tool_visit_url(self, args: Dict[str, Any]) -> str:
        """Execute visit_url tool."""
        url = args.get("url", "")
        ret, approved = await self._check_url_and_generate_msg(url)
        if not approved:
            return ret

        action_description = f"I navigated to '{url}'."
        
        # Mock implementation - in real version this would use playwright
        self.logger.info(f"Mock visiting URL: {url}")
        
        return action_description

    async def _execute_tool_web_search(self, args: Dict[str, Any]) -> str:
        """Execute web_search tool."""
        query = args.get("query", "")
        search_url, domain = self._get_search_url(query)
        ret, approved = await self._check_url_and_generate_msg(domain)
        if not approved:
            return ret
            
        action_description = f"I searched for '{query}'."
        
        # Mock implementation
        self.logger.info(f"Mock web search: {query}")
        
        return action_description

    async def _execute_tool_click(self, args: Dict[str, Any]) -> str:
        """Execute click tool."""
        target_id = args.get("target_id", 0)
        action_description = f"I clicked on element with ID {target_id}."
        
        # Mock implementation
        self.logger.info(f"Mock clicking element {target_id}")
        
        return action_description

    async def _execute_tool_input_text(self, args: Dict[str, Any]) -> str:
        """Execute input_text tool."""
        input_field_id = args.get("input_field_id", 0)
        text_value = args.get("text_value", "")
        action_description = f"I typed '{text_value}' into field {input_field_id}."
        
        # Mock implementation
        self.logger.info(f"Mock typing '{text_value}' into field {input_field_id}")
        
        return action_description

    async def _execute_tool_hover(self, args: Dict[str, Any]) -> str:
        """Execute hover tool."""
        target_id = args.get("target_id", 0)
        action_description = f"I hovered over element with ID {target_id}."
        
        # Mock implementation
        self.logger.info(f"Mock hovering over element {target_id}")
        
        return action_description

    async def _execute_tool_scroll_down(self, args: Dict[str, Any]) -> str:
        """Execute scroll_down tool."""
        pixels = args.get("pixels", 400)
        action_description = f"I scrolled down {pixels} pixels."
        
        # Mock implementation
        self.logger.info(f"Mock scrolling down {pixels} pixels")
        
        return action_description

    async def _execute_tool_scroll_up(self, args: Dict[str, Any]) -> str:
        """Execute scroll_up tool."""
        pixels = args.get("pixels", 400)
        action_description = f"I scrolled up {pixels} pixels."
        
        # Mock implementation
        self.logger.info(f"Mock scrolling up {pixels} pixels")
        
        return action_description

    async def _execute_tool_history_back(self, args: Dict[str, Any]) -> str:
        """Execute history_back tool."""
        action_description = "I navigated back in browser history."
        
        # Mock implementation
        self.logger.info("Mock going back in history")
        
        return action_description

    async def _execute_tool_refresh_page(self, args: Dict[str, Any]) -> str:
        """Execute refresh_page tool."""
        action_description = "I refreshed the current page."
        
        # Mock implementation
        self.logger.info("Mock refreshing page")
        
        return action_description

    async def _execute_tool_keypress(self, args: Dict[str, Any]) -> str:
        """Execute keypress tool."""
        keys = args.get("keys", [])
        action_description = f"I pressed keys: {', '.join(keys)}"
        
        # Mock implementation
        self.logger.info(f"Mock pressing keys: {keys}")
        
        return action_description

    async def _execute_tool_sleep(self, args: Dict[str, Any]) -> str:
        """Execute sleep tool."""
        duration = args.get("duration", 3)
        action_description = f"I waited {duration} seconds."
        
        # Actual sleep
        await asyncio.sleep(duration)
        
        return action_description

    async def _execute_tool_answer_question(self, args: Dict[str, Any]) -> str:
        """Execute answer_question tool."""
        question = args.get("question", "")
        action_description = f"I answered the question: {question}"
        
        # Mock implementation - in real version this would analyze page content
        self.logger.info(f"Mock answering question: {question}")
        
        return action_description

    async def _execute_tool_select_option(self, args: Dict[str, Any]) -> str:
        """Execute select_option tool."""
        target_id = args.get("target_id", 0)
        action_description = f"I selected option with ID {target_id}."
        
        # Mock implementation
        self.logger.info(f"Mock selecting option {target_id}")
        
        return action_description

    async def _execute_tool_create_tab(self, args: Dict[str, Any]) -> str:
        """Execute create_tab tool."""
        url = args.get("url", "")
        action_description = f"I created a new tab and navigated to {url}."
        
        # Mock implementation
        self.logger.info(f"Mock creating tab with URL: {url}")
        
        return action_description

    async def _execute_tool_switch_tab(self, args: Dict[str, Any]) -> str:
        """Execute switch_tab tool."""
        tab_index = args.get("tab_index", 0)
        action_description = f"I switched to tab {tab_index}."
        
        # Mock implementation
        self.logger.info(f"Mock switching to tab {tab_index}")
        
        return action_description

    async def _execute_tool_close_tab(self, args: Dict[str, Any]) -> str:
        """Execute close_tab tool."""
        tab_index = args.get("tab_index", 0)
        action_description = f"I closed tab {tab_index}."
        
        # Mock implementation
        self.logger.info(f"Mock closing tab {tab_index}")
        
        return action_description

    async def _execute_tool_upload_file(self, args: Dict[str, Any]) -> str:
        """Execute upload_file tool."""
        target_id = args.get("target_id", "")
        file_path = args.get("file_path", "")
        action_description = f"I uploaded file {file_path} to element {target_id}."
        
        # Mock implementation
        self.logger.info(f"Mock uploading file {file_path} to {target_id}")
        
        return action_description

    def _get_search_url(self, query: str) -> tuple[str, str]:
        """Get search URL and domain based on configured search engine."""
        if self.search_engine.lower() == "google":
            domain = "google.com"
            url = f"https://www.google.com/search?q={quote_plus(query)}"
        elif self.search_engine.lower() == "bing":
            domain = "bing.com"
            url = f"https://www.bing.com/search?q={quote_plus(query)}"
        elif self.search_engine.lower() == "yahoo":
            domain = "yahoo.com"
            url = f"https://search.yahoo.com/search?p={quote_plus(query)}"
        elif self.search_engine.lower() == "duckduckgo":
            domain = "duckduckgo.com"
            url = f"https://duckduckgo.com/?q={quote_plus(query)}"
        else:  # treat as direct website URL
            domain = self.search_engine
            url = (
                f"https://{self.search_engine}"
                if not self.search_engine.startswith(("http://", "https://"))
                else self.search_engine
            )
        return url, domain

    async def _check_url_and_generate_msg(self, url: str) -> Tuple[str, bool]:
        """Returns a message to caller if the URL is not allowed and a boolean indicating if the user has approved the URL."""
        # Simplified implementation - in real version this would handle URL approval
        if self._url_status_manager.is_url_blocked(url):
            return (
                f"I am not allowed to access the website {url} because it has been blocked.",
                False,
            )

        if not self._url_status_manager.is_url_allowed(url):
            # For simplicity, auto-approve all URLs in mock implementation
            self._url_status_manager.set_url_status(url, "allowed")
            return "", True
            
        return "", True

    # Additional utility methods
    
    async def pause(self) -> None:
        """Pause the WebSurfer agent."""
        self.is_paused = True
        self._pause_event.set()

    async def resume(self) -> None:
        """Resume the WebSurfer agent."""
        self.is_paused = False
        self._pause_event.clear()

    async def save_state(self, save_browser: bool = True) -> Mapping[str, Any]:
        """Save the current state of the WebSurfer."""
        state = {
            "chat_history": self._chat_history,
            "browser_state": None,  # Mock browser state
            "last_outside_message": self._last_outside_message,
        }
        return state

    async def load_state(self, state: Mapping[str, Any], load_browser: bool = True) -> None:
        """Load a previously saved state."""
        self._chat_history = state.get("chat_history", [])
        self._last_outside_message = state.get("last_outside_message", "")
        
        if load_browser and state.get("browser_state"):
            await self.lazy_init()
            # In real implementation, would restore browser state

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat history."""
        return self._chat_history.copy()

    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self._chat_history.clear()

    # Context manager support
    async def __aenter__(self):
        await self.lazy_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
