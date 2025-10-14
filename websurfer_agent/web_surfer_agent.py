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
from .tool_definitions import ALL_TOOLS, set_browser_controller, set_id_mapping
from .prompts import SIMPLE_WEB_SURFER_SYSTEM_MESSAGE
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

    def _screenshot_callback(self, memory_step, agent):
        """Callback to capture annotated screenshots after each step."""
        import asyncio
        
        # Get the current event loop or create one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Get annotated screenshot
        try:
            annotated_img, visible_ids, ids_above, ids_below, id_mapping = loop.run_until_complete(
                self._get_annotated_screenshot()
            )
            
            # Store ID mapping for tool execution
            self._current_id_mapping = id_mapping
            # Also set it globally so tools can access it
            set_id_mapping(id_mapping)
            
            # ID mapping updated in screenshot callback
            
            # Resize for optimal tokens (matches magentic-ui sizing)
            resized_img = annotated_img.resize((self.MLM_WIDTH, self.MLM_HEIGHT))

            # Persist the screenshot for debugging/inspection
            try:
                import os
                from datetime import datetime
                save_dir = os.path.join(os.getcwd(), "websurfer_agent", "screenshots")
                os.makedirs(save_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # Save both original annotated and model-sized images
                annotated_path = os.path.join(save_dir, f"annotated_{ts}.png")
                resized_path = os.path.join(save_dir, f"mlm_{ts}.png")
                annotated_img.save(annotated_path)
                resized_img.save(resized_path)
            except Exception as save_err:
                self.logger.debug(f"Failed to save screenshots: {save_err}")

            
            # Attach to memory step for VLM to see
            memory_step.observations_images = [resized_img.copy()]
            
            # Add marker information to observations text
            marker_info = f"""
[Screenshot with numbered markers attached]
Interactive elements on current page:
- Visible elements with markers: {', '.join(visible_ids) if visible_ids else 'none visible'}
- Elements above viewport (scroll up to see): {', '.join(ids_above) if ids_above else 'none'}
- Elements below viewport (scroll down to see): {', '.join(ids_below) if ids_below else 'none'}

IMPORTANT: Use ONLY the numbered markers visible in the screenshot to interact with elements.
Example: To click the element with red box and number "5", use click(target_id=5)
The page content text is for context only - always use the visual markers for interaction.
"""
            
            # Add to observations
            if memory_step.observations:
                memory_step.observations += "\n" + marker_info
            else:
                memory_step.observations = marker_info
                
        except Exception as e:
            self.logger.error(f"Error in screenshot callback: {e}")
    
    def _create_agent(self):
        """Create the underlying smol agents CodeAgent with vision support."""
        if not self.model:
            raise ValueError("Model is required to create WebSurfer agent")
        
        # Set the browser controller for all tools to use
        set_browser_controller(self._playwright_controller)
        
        # Create the agent with vision support via step callbacks
        # The callback will capture annotated screenshots after each action
        self.agent = CodeAgent(
            tools=ALL_TOOLS,
            model=self.model,
            step_callbacks=[self._screenshot_callback],  # Enable vision!
            max_steps=self.max_actions_per_step,
        )
        
        # Initialize ID mapping storage
        self._current_id_mapping = {}

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

    async def _get_annotated_screenshot(
        self
    ) -> Tuple[PIL.Image.Image, List[str], List[str], List[str], Dict[str, str]]:
        """
        Get an annotated screenshot with numbered markers for interactive elements.
        
        Returns:
            Tuple containing:
            - PIL.Image.Image: Annotated image with numbered markers
            - List[str]: IDs of visible elements  
            - List[str]: IDs of elements above viewport (scroll up to see)
            - List[str]: IDs of elements below viewport (scroll down to see)
            - Dict[str, str]: Mapping from display IDs to real element IDs
        """
        page = self._playwright_controller._page
        if not page:
            # Return empty/default values if no page
            blank_img = PIL.Image.new('RGB', (self.MLM_WIDTH, self.MLM_HEIGHT), color='white')
            return blank_img, [], [], [], {}
        
        # 1. Get interactive elements from the page
        interactive_regions = await self._playwright_controller.get_interactive_rects(page)
        
        # 2. Take screenshot
        screenshot_bytes = await self._playwright_controller.get_screenshot(page)
        
        # 3. Annotate with numbered markers
        annotated_image, visible_ids, ids_above, ids_below, id_mapping = add_set_of_mark(
            screenshot=screenshot_bytes,
            ROIs=interactive_regions,
            use_sequential_ids=True  # Use 1, 2, 3... instead of element IDs
        )
        
        # 4. Save debug screenshot if enabled
        if self.to_save_screenshots and self.debug_dir:
            timestamp = int(time.time())
            debug_path = os.path.join(self.debug_dir, f"screenshot_annotated_{timestamp}.png")
            annotated_image.save(debug_path)
            self.logger.debug(f"Saved annotated screenshot to {debug_path}")
        
        return annotated_image, visible_ids, ids_above, ids_below, id_mapping

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
            
            # Get current page information
            page = self._playwright_controller._page
            current_url = page.url if page else "about:blank"
            
            # No initial screenshot needed since we start at about:blank
            marker_info = ""
            
            # Build enhanced context with system prompt and current state
            # Similar to how FileSurfer does it
            date_today = datetime.now().strftime("%B %d, %Y")
            system_prompt = SIMPLE_WEB_SURFER_SYSTEM_MESSAGE.format(date_today=date_today)
            
            initial_context = (
                f"{system_prompt}\n\n"
                f"--- CURRENT TASK ---\n"
                f"USER_REQUEST: {request}\n\n"
                f"You are currently viewing: {current_url}\n\n"
                f"{marker_info}\n\n"
                f"Begin your work. Use the available tools to complete the task.\n"
            )
            
            result = self.agent.run(initial_context)
            
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
