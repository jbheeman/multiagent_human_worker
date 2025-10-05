import asyncio
import io
import json
import logging
import os
import time
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    BinaryIO,
)
from urllib.parse import quote_plus
import PIL.Image
import tiktoken
# Mock playwright classes for standalone implementation
class BrowserContext:
    def __init__(self):
        self.pages = []

class Download:
    def __init__(self):
        self.suggested_filename = "download.pdf"

class Page:
    def __init__(self):
        self.url = "about:blank"
        self.viewport_size = {"width": 1440, "height": 1440}
    
    async def title(self):
        return "Mock Page"
    
    async def goto(self, url, wait_until=None):
        self.url = url
    
    async def screenshot(self, full_page=False):
        return b"mock_screenshot_data"
    
    async def evaluate(self, script):
        return "mock_evaluation_result"
    
    async def wait_for_load_state(self, state):
        pass
    
    async def set_viewport_size(self, size):
        self.viewport_size = size
    
    async def set_download_behavior(self, behavior, download_path):
        pass
    
    async def on(self, event, handler):
        pass
    
    async def go_back(self):
        pass
    
    async def reload(self):
        pass
    
    async def keyboard(self):
        return MockKeyboard()
    
    async def new_page(self):
        return Page()
    
    async def bring_to_front(self):
        pass
    
    async def close(self):
        pass

class MockKeyboard:
    async def press(self, key):
        pass

# Mock implementations for missing dependencies
class MockPlaywrightBrowser:
    """Mock browser class for standalone implementation."""
    
    def __init__(self, **kwargs):
        self.novnc_port = -1
        self.playwright_port = -1
        self.browser_context = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockUrlStatusManager:
    """Mock URL status manager."""
    
    def __init__(self, url_statuses=None, url_block_list=None):
        self.url_statuses = url_statuses or {}
        self.url_block_list = url_block_list or []
    
    def is_url_allowed(self, url: str) -> bool:
        return True
    
    def is_url_blocked(self, url: str) -> bool:
        return False
    
    def is_url_rejected(self, url: str) -> bool:
        return False
    
    def set_url_status(self, url: str, status: str):
        self.url_statuses[url] = status


class InteractiveRegion:
    """Mock interactive region class."""
    
    def __init__(self, **kwargs):
        self.rects = kwargs.get('rects', [])
        self.role = kwargs.get('role', '')
        self.aria_name = kwargs.get('aria_name', '')
        self.tag_name = kwargs.get('tag_name', '')
        self.contenteditable = kwargs.get('contenteditable', 'false')


class PlaywrightController:
    """Browser controller for WebSurfer agent using Playwright."""
    
    def __init__(
        self,
        animate_actions: bool = False,
        downloads_folder: Optional[str] = None,
        viewport_width: int = 1440,
        viewport_height: int = 1440,
        _download_handler=None,
        to_resize_viewport: bool = True,
        single_tab_mode: bool = False,
        url_status_manager=None,
        url_validation_callback=None,
    ):
        self.animate_actions = animate_actions
        self.downloads_folder = downloads_folder
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._download_handler = _download_handler
        self.to_resize_viewport = to_resize_viewport
        self.single_tab_mode = single_tab_mode
        self.url_status_manager = url_status_manager or MockUrlStatusManager()
        self.url_validation_callback = url_validation_callback
        self.logger = logging.getLogger(__name__)
        
        # Mock browser components
        self._browser = MockPlaywrightBrowser()
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
    
    async def __aenter__(self):
        await self._browser.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._browser.__aexit__(exc_type, exc_val, exc_tb)
    
    async def visit_url(self, url: str):
        """Navigate to a URL."""
        if self._page is None:
            self._page = Page()
        await self._page.goto(url, wait_until="load")
        return f"Navigated to {url}"
    
    async def get_page_content(self) -> str:
        """Get the current page content."""
        if self._page is None:
            return "No page loaded"
        title = await self._page.title()
        return f"Page title: {title}, URL: {self._page.url}"
    
    async def on_new_page(self, page: Page):
        """Handle new page creation."""
        self._page = page
        if self.to_resize_viewport:
            await page.set_viewport_size({
                "width": self.viewport_width,
                "height": self.viewport_height
            })
        
        if self.downloads_folder:
            # Set download behavior
            await page.set_download_behavior(
                behavior="allow",
                download_path=self.downloads_folder
            )
        
        # Set up download handler
        if self._download_handler:
            page.on("download", self._download_handler)
    
    async def visit_page(self, page: Page, url: str) -> Tuple[bool, bool]:
        """Visit a URL on the given page."""
        try:
            await page.goto(url, wait_until="domcontentloaded")
            return True, True  # reset_prior_metadata, reset_last_download
        except Exception as e:
            self.logger.error(f"Error visiting page {url}: {e}")
            return False, False
    
    async def get_screenshot(self, page: Page) -> bytes:
        """Get screenshot of the current page."""
        try:
            screenshot = await page.screenshot(full_page=True)
            return screenshot
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            # Return a mock screenshot
            return self._create_mock_screenshot()
    
    def _create_mock_screenshot(self) -> bytes:
        """Create a mock screenshot for testing."""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple mock screenshot
        img = Image.new('RGB', (self.viewport_width, self.viewport_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some mock content
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((50, 50), "Mock Web Page", fill='black', font=font)
        draw.text((50, 100), "This is a mock screenshot for testing", fill='gray', font=font)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    async def get_interactive_rects(self, page: Page) -> Dict[str, InteractiveRegion]:
        """Get interactive regions from the page."""
        try:
            # This is a simplified mock implementation
            # In a real implementation, this would use JavaScript to find interactive elements
            mock_rects = {
                "1": InteractiveRegion(
                    rects=[{"left": 100, "top": 100, "right": 200, "bottom": 150, "width": 100, "height": 50}],
                    role="button",
                    aria_name="Click Me",
                    tag_name="button"
                ),
                "2": InteractiveRegion(
                    rects=[{"left": 100, "top": 200, "right": 300, "bottom": 230, "width": 200, "height": 30}],
                    role="textbox",
                    aria_name="Search",
                    tag_name="input"
                )
            }
            return mock_rects
        except Exception as e:
            self.logger.error(f"Error getting interactive rects: {e}")
            return {}
    
    async def get_visible_text(self, page: Page) -> str:
        """Get visible text from the page."""
        try:
            text = await page.evaluate("""
                () => {
                    return document.body.innerText || document.body.textContent || '';
                }
            """)
            return text or "Mock page content - this is a test implementation"
        except Exception as e:
            self.logger.error(f"Error getting visible text: {e}")
            return "Mock page content - error getting text"
    
    async def get_page_markdown(self, page: Page) -> str:
        """Get page content as markdown."""
        try:
            # Simplified markdown extraction
            title = await page.title()
            url = page.url
            text = await self.get_visible_text(page)
            
            markdown = f"# {title}\n\nURL: {url}\n\n{text}"
            return markdown
        except Exception as e:
            self.logger.error(f"Error getting page markdown: {e}")
            return "# Mock Page\n\nMock content for testing"
    
    async def describe_page(self, page: Page, get_screenshot: bool = True) -> Tuple[str, Optional[bytes], str]:
        """Describe the current page."""
        try:
            title = await page.title()
            url = page.url
            text = await self.get_visible_text(page)
            
            description = f"Page: {title}\nURL: {url}\nContent: {text[:500]}..."
            
            screenshot = None
            if get_screenshot:
                screenshot = await self.get_screenshot(page)
            
            # Create a simple hash for metadata
            metadata_hash = f"page_{hash(title + url)}"
            
            return description, screenshot, metadata_hash
        except Exception as e:
            self.logger.error(f"Error describing page: {e}")
            return "Error describing page", None, "error_hash"
    
    async def click_id(self, context: BrowserContext, page: Page, target_id: str) -> Optional[Page]:
        """Click on an element by ID."""
        try:
            # Mock implementation - in reality this would find and click the element
            self.logger.info(f"Mock clicking element with ID: {target_id}")
            return None  # No new page opened
        except Exception as e:
            self.logger.error(f"Error clicking element {target_id}: {e}")
            return None
    
    async def fill_id(
        self, 
        page: Page, 
        input_field_id: str, 
        text_value: str, 
        press_enter: bool = True, 
        delete_existing_text: bool = True
    ):
        """Fill text into an input field by ID."""
        try:
            # Mock implementation
            self.logger.info(f"Mock filling field {input_field_id} with '{text_value}'")
        except Exception as e:
            self.logger.error(f"Error filling field {input_field_id}: {e}")
    
    async def hover_id(self, page: Page, target_id: str):
        """Hover over an element by ID."""
        try:
            # Mock implementation
            self.logger.info(f"Mock hovering over element {target_id}")
        except Exception as e:
            self.logger.error(f"Error hovering over element {target_id}: {e}")
    
    async def select_option(self, context: BrowserContext, page: Page, target_id: str) -> Optional[Page]:
        """Select an option from a dropdown by ID."""
        try:
            # Mock implementation
            self.logger.info(f"Mock selecting option {target_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error selecting option {target_id}: {e}")
            return None
    
    async def scroll_mousewheel(self, page: Page, direction: str, pixels: int = 400):
        """Scroll the page using mouse wheel."""
        try:
            if direction == "down":
                await page.evaluate(f"window.scrollBy(0, {pixels})")
            else:
                await page.evaluate(f"window.scrollBy(0, -{pixels})")
        except Exception as e:
            self.logger.error(f"Error scrolling {direction}: {e}")
    
    async def go_back(self, page: Page) -> bool:
        """Go back in browser history."""
        try:
            await page.go_back()
            return True
        except Exception as e:
            self.logger.error(f"Error going back: {e}")
            return False
    
    async def refresh_page(self, page: Page):
        """Refresh the current page."""
        try:
            await page.reload()
        except Exception as e:
            self.logger.error(f"Error refreshing page: {e}")
    
    async def sleep(self, page: Page, duration: float):
        """Wait for a specified duration."""
        await asyncio.sleep(duration)
    
    async def keypress(self, page: Page, keys: List[str]):
        """Press keyboard keys."""
        try:
            for key in keys:
                await page.keyboard.press(key)
        except Exception as e:
            self.logger.error(f"Error pressing keys {keys}: {e}")
    
    async def create_new_tab(self, context: BrowserContext, url: str) -> Page:
        """Create a new tab and navigate to URL."""
        try:
            new_page = await context.new_page()
            await self.on_new_page(new_page)
            await self.visit_page(new_page, url)
            return new_page
        except Exception as e:
            self.logger.error(f"Error creating new tab: {e}")
            raise
    
    async def switch_tab(self, context: BrowserContext, tab_index: int) -> Page:
        """Switch to a different tab by index."""
        try:
            pages = context.pages
            if 0 <= tab_index < len(pages):
                page = pages[tab_index]
                await page.bring_to_front()
                return page
            else:
                raise ValueError(f"Invalid tab index: {tab_index}")
        except Exception as e:
            self.logger.error(f"Error switching tab: {e}")
            raise
    
    async def close_tab(self, context: BrowserContext, tab_index: int) -> Page:
        """Close a tab and switch to an adjacent one."""
        try:
            pages = context.pages
            if 0 <= tab_index < len(pages) and len(pages) > 1:
                page_to_close = pages[tab_index]
                await page_to_close.close()
                
                # Switch to an adjacent tab
                new_index = min(tab_index, len(pages) - 2)
                return pages[new_index]
            else:
                raise ValueError("Cannot close the last tab or invalid index")
        except Exception as e:
            self.logger.error(f"Error closing tab: {e}")
            raise
    
    async def get_focused_rect_id(self, page: Page) -> Optional[str]:
        """Get the ID of the currently focused element."""
        try:
            # Mock implementation
            return None
        except Exception as e:
            self.logger.error(f"Error getting focused element: {e}")
            return None
    
    async def get_tabs_information(self, context: BrowserContext, current_page: Page) -> List[Dict[str, Any]]:
        """Get information about all open tabs."""
        try:
            tabs_info = []
            pages = context.pages
            for i, page in enumerate(pages):
                title = await page.title()
                url = page.url
                is_active = page == current_page
                is_controlled = page == current_page  # Simplified
                
                tabs_info.append({
                    "index": i,
                    "title": title,
                    "url": url,
                    "is_active": is_active,
                    "is_controlled": is_controlled
                })
            return tabs_info
        except Exception as e:
            self.logger.error(f"Error getting tabs information: {e}")
            return []
    
    async def get_current_url_title(self, page: Page) -> Tuple[str, str]:
        """Get current URL and title."""
        try:
            url = page.url
            title = await page.title()
            return url, title
        except Exception as e:
            self.logger.error(f"Error getting URL and title: {e}")
            return "about:blank", "Mock Page"
    
    async def cleanup_animations(self, page: Page):
        """Clean up any animations or highlights."""
        try:
            # Mock implementation - remove any CSS animations or highlights
            pass
        except Exception as e:
            self.logger.error(f"Error cleaning up animations: {e}")
    
    async def preview_action(self, page: Page, element_id: str):
        """Preview an action by highlighting the element."""
        try:
            # Mock implementation - add visual feedback
            self.logger.info(f"Previewing action on element {element_id}")
        except Exception as e:
            self.logger.error(f"Error previewing action: {e}")
    
    async def _ensure_page_ready(self, page: Page):
        """Ensure the page is ready for interaction."""
        try:
            await page.wait_for_load_state("domcontentloaded")
        except Exception as e:
            self.logger.error(f"Error ensuring page ready: {e}")
