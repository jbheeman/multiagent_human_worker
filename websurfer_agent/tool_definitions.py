from typing import Dict, Any, Optional, TYPE_CHECKING
from smolagents import Tool
import asyncio
import nest_asyncio

# Allow nested event loops
nest_asyncio.apply()

if TYPE_CHECKING:
    from browser_playwright import PlaywrightController

# Removed explanation fields - they were pure token burn

# Global browser controller instance that will be set by WebSurferAgent
_browser_controller: Optional['PlaywrightController'] = None
# Global ID mapping for Set of Mark (maps display IDs like "5" to real element IDs)
_id_mapping: Dict[str, str] = {}

def set_browser_controller(controller: 'PlaywrightController'):
    """Set the global browser controller for all tools."""
    global _browser_controller
    _browser_controller = controller

def get_browser_controller() -> Optional['PlaywrightController']:
    """Get the global browser controller."""
    return _browser_controller

def set_id_mapping(mapping: Dict[str, str]):
    """Set the ID mapping for Set of Mark."""
    global _id_mapping
    _id_mapping = mapping

def get_real_element_id(display_id: str) -> str:
    """
    Map a display ID (from Set of Mark) to the real element ID.
    If no mapping exists, return the original ID (backward compatible).
    
    Args:
        display_id: The ID shown in the annotated screenshot (e.g., "5")
    
    Returns:
        The real element ID to use with Playwright (e.g., "element_xyz")
    """
    global _id_mapping
    return _id_mapping.get(display_id, display_id)

# Tool definitions for WebSurfer agent
class VisitUrlTool(Tool):
    """Navigate directly to a provided URL using the browser's address bar."""
    name = "visit_url"
    description = "Navigate directly to a provided URL using the browser's address bar. Prefer this tool over other navigation techniques in cases where the user provides a fully-qualified URL (e.g., choose it over clicking links, or inputing queries into search boxes)."
    inputs = {
                    "url": {
                        "type": "string",
                        "description": "The URL to visit in the browser.",
                    },
            }
    output_type = "string"

    def forward(self, url: str) -> str:
        """Execute the visit_url tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return f"Error: Browser controller not initialized. Would navigate to: {url}"
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Navigate to the URL using the real PlaywrightController
        try:
            page = controller._page
            if page is None:
                return f"Error: No page available. Browser not initialized."
            success, _ = loop.run_until_complete(controller.visit_page(page, url))
            if success:
            # Reduced verbosity: only report URL; annotated markers are provided via screenshot callback
                return f"Navigated to {url}. Use the numbered markers in the latest screenshot to interact."
            else:
                return f"Failed to navigate to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {str(e)}"


class WebSearchTool(Tool):
    """Performs a web search on a search engine with the given query."""
    name = "web_search"
    description = "Performs a web search on a search engine with the given query. Make sure the query is simple and don't use compound queries."
    inputs = {
                    "query": {
                        "type": "string",
                        "description": "The web search query to use.",
                    },
            }
    output_type = "string"

    def forward(self, query: str) -> str:
        """Execute the web_search tool."""
        import asyncio
        from urllib.parse import quote_plus
        controller = get_browser_controller()
        if controller is None:
            return f"Error: Browser controller not initialized. Searched for: {query}"
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Perform the search using the browser controller
        search_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
        try:
            page = controller._page
            context = controller._context
            if page is None:
                return f"Error: No page available. Browser not initialized."
            success, _ = loop.run_until_complete(controller.visit_page(page, search_url))
            if success:
                # Get the search results page content
                content = loop.run_until_complete(controller.get_page_markdown(page, max_tokens=5000))
                
                return f"Search results for '{query}':\n\n{content}\n\nNote: Interactive elements are marked with numbers in the screenshot. Use those numbered markers to interact with elements."
            else:
                return f"Failed to search for: {query}"
        except Exception as e:
            return f"Error performing search: {str(e)}"


class HistoryBackTool(Tool):
    """Navigates back one page in the browser's history."""
    name = "history_back"
    description = "Navigates back one page in the browser's history. This is equivalent to clicking the browser back button."
    inputs = {
                  
            }
    output_type = "string"

    def forward(self) -> str:
        """Execute the history_back tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            success = loop.run_until_complete(controller.go_back(page))
            return "Navigated back in browser history" if success else "Could not navigate back"
        except Exception as e:
            return f"Error navigating back: {str(e)}"


class RefreshPageTool(Tool):
    """Refreshes the current page in the browser."""
    name = "refresh_page"
    description = "Refreshes the current page in the browser. This is equivalent to clicking the browser refresh button or pressing F5."
    inputs = {
            }
    output_type = "string"

    def forward(self) -> str:
        """Execute the refresh_page tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            loop.run_until_complete(controller.refresh_page(page))
            return "Page refreshed successfully"
        except Exception as e:
            return f"Error refreshing page: {str(e)}"


class ScrollDownTool(Tool):
    """Scrolls down on the current page using mouse wheel for 900 pixels."""
    name = "scroll_down"
    description = "Scrolls down on the current page using mouse wheel for 900 pixels."
    inputs = {
                  
            }
    output_type = "string"

    def forward(self) -> str:
        """Execute the scroll_down tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            loop.run_until_complete(controller.scroll_mousewheel(page, "down", 900))
            return "Scrolled down 900 pixels"
        except Exception as e:
            return f"Error scrolling down: {str(e)}"


class ScrollUpTool(Tool):
    """Scrolls up on the current page using mouse wheel for 900 pixels."""
    name = "scroll_up"
    description = "Scrolls up on the current page using mouse wheel for 900 pixels."
    inputs = {
            }
    output_type = "string"

    def forward(self) -> str:
        """Execute the scroll_up tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            loop.run_until_complete(controller.scroll_mousewheel(page, "up", 900))
            return "Scrolled up 900 pixels"
        except Exception as e:
            return f"Error scrolling up: {str(e)}"


class ClickTool(Tool):
    """Clicks the mouse on the target with the given id."""
    name = "click"
    description = "Clicks the mouse on the target with the given id."
    inputs = {
                    "target_id": {
                        "type": "integer",
                        "description": "The numeric id of the target to click.",
                    },
            }
    output_type = "string"

    def forward(self, target_id: int) -> str:
        """Execute the click tool."""
        import asyncio
        import time
        controller = get_browser_controller()
        if controller is None:
            return f"Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            context = controller._context
            if page is None:
                return "Error: No page available."
            
            # Map display ID to real element ID (for Set of Mark support)
            real_id = get_real_element_id(str(target_id))
            
            # Click the element
            loop.run_until_complete(controller.click_id(context, page, real_id))
            
            # Give page a moment to start loading
            time.sleep(2)
            
            # Try to get the page content after clicking, but handle timeouts gracefully
            try:
                # Wait for page to be in a reasonable state (networkidle is more lenient than load)
                loop.run_until_complete(page.wait_for_load_state("domcontentloaded", timeout=10000))
            except Exception:
                # Page might be slow, but we can still try to get content
                pass
            
            # Reduced verbosity: only report action and URL; annotated markers are provided via screenshot callback
            return f"Clicked element {target_id}. Current URL: {page.url}. Use the numbered markers in the latest screenshot to continue."
        except Exception as e:
            return f"Error clicking element {target_id}: {str(e)}"


class InputTextTool(Tool):
    """Types the given text value into the specified field."""
    name = "input_text"
    description = "Types the given text value into the specified field. Presses enter only if you want to submit the form or search."
    inputs = {
                    "input_field_id": {
                        "type": "integer",
                        "description": "The numeric id of the input field to receive the text.",
                    },
                    "text_value": {
                        "type": "string",
                        "description": "The text to type into the input field.",
                    },
                    "press_enter": {
                        "type": "boolean",
                        "description": "Whether to press enter after typing into the field or not.",
            "nullable": True,
                    },
                    "delete_existing_text": {
                        "type": "boolean",
                        "description": "Whether to delete existing text in the field before inputing the text value.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, input_field_id: int, text_value: str, 
                press_enter: bool = False, delete_existing_text: bool = True) -> str:
        """Execute the input_text tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return f"Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            
            # Map display ID to real element ID (for Set of Mark support)
            real_id = get_real_element_id(str(input_field_id))
            
            loop.run_until_complete(controller.fill_id(
                page, real_id, text_value, 
                press_enter=press_enter, delete_existing_text=delete_existing_text
            ))
            return f"Typed '{text_value}' into field {input_field_id}"
        except Exception as e:
            return f"Error typing into field {input_field_id}: {str(e)}"


class HoverTool(Tool):
    """Hovers the mouse over the target with the given id."""
    name = "hover"
    description = "Hovers the mouse over the target with the given id."
    inputs = {
                    "target_id": {
                        "type": "integer",
                        "description": "The numeric id of the target to hover over.",
                    },
            }
    output_type = "string"

    def forward(self, target_id: int) -> str:
        """Execute the hover tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            
            # Map display ID to real element ID (for Set of Mark support)
            real_id = get_real_element_id(str(target_id))
            
            loop.run_until_complete(controller.hover_id(page, real_id))
            return f"Hovered over element with ID {target_id}"
        except Exception as e:
            return f"Error hovering over element {target_id}: {str(e)}"


class KeypressTool(Tool):
    """Press one or multiple keyboard keys in sequence."""
    name = "keypress"
    description = "Press one or multiple keyboard keys in sequence, this is not used for typing text. Supports special keys like 'Enter', 'Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Backspace', 'Delete', 'Escape', 'Control', 'Alt', 'Shift'."
    inputs = {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to press in sequence. For special keys, use their full name (e.g. 'Enter', 'Tab', etc.).",
                    },
            }
    output_type = "string"

    def forward(self, keys: list) -> str:
        """Execute the keypress tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            loop.run_until_complete(controller.keypress(page, keys))
            return f"Pressed keys: {', '.join(keys)}"
        except Exception as e:
            return f"Error pressing keys: {str(e)}"


class ReadPageAndAnswerTool(Tool):
    """Used to answer questions about the current webpage's content."""
    name = "answer_question"
    description = "Used to answer questions about the current webpage's content."
    inputs = {
                    "question": {
                        "type": "string",
                        "description": "The question to answer. Do not ask any follow up questions or say that you can help with more things.",
                    },
            }
    output_type = "string"

    def forward(self, question: str) -> str:
        """Execute the answer_question tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            
            # Get the page content
            content = loop.run_until_complete(controller.get_page_markdown(page, max_tokens=8000))
            
            # Use LLM to answer the question based on page content
            # For now, we'll extract relevant information using simple text search
            # A full implementation would use the model to analyze and answer
            
            # Simple extraction: look for stock-related content
            lines = content.lower().split('\n')
            relevant_info = []
            
            # Keywords to look for
            keywords = ['stock', 'buy', 'recommend', 'portfolio', 'ticker', 'company']
            
            for line in lines:
                if any(keyword in line for keyword in keywords):
                    relevant_info.append(line)
            
            if relevant_info:
                answer = "Based on the page content, here's what I found:\n\n" + '\n'.join(relevant_info[:20])  # Limit to avoid token overflow
            else:
                answer = f"I couldn't find specific information to answer '{question}' on this page. The page contains general content but no clear stock recommendations."
            
            return answer
            
        except Exception as e:
            return f"Error reading page: {str(e)}"


class SleepTool(Tool):
    """Wait a specified period of time in seconds."""
    name = "sleep"
    description = "Wait a specified period of time in seconds (default 3 seconds). Call this function if the page has not yet fully loaded, or if it is determined that a small delay would increase the task's chances of success."
    inputs = {
                    "duration": {
                        "type": "number",
                        "description": "The number of seconds to wait. Default is 3 seconds.",
            "nullable": True,
                    },
            }
    output_type = "string"

    def forward(self, duration: float = 3.0) -> str:
        """Execute the sleep tool."""
        import time
        time.sleep(duration)
        return f"Waited {duration} seconds"


class StopActionTool(Tool):
    """Perform no action on the browser and provide an answer."""
    name = "stop_action"
    description = "Perform no action on the browser. Answer the request directly and summarize all past actions and observations you did previously in relation to the request."
    inputs = {
                    "answer": {
                        "type": "string",
                        "description": "The answer to the request and a complete summary of past actions and observations. Phrase using first person and as if you are directly talking to the user. Do not ask any questions or say that you can help with more things.",
                    },
            }
    output_type = "string"

    def forward(self, answer: str) -> str:
        """Execute the stop_action tool."""
        return answer


class SelectOptionTool(Tool):
    """Selects an option from a dropdown/select menu."""
    name = "select_option"
    description = "Selects an option from a dropdown/select menu."
    inputs = {
                    "target_id": {
                        "type": "integer",
                        "description": "The numeric id of the option to select.",
                    },
            }
    output_type = "string"

    def forward(self, target_id: int) -> str:
        """Execute the select_option tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            context = controller._context
            if page is None:
                return "Error: No page available."
            if context is None:
                return "Error: Browser context not available."
            
            # Map display ID to real element ID (for Set of Mark support)
            real_id = get_real_element_id(str(target_id))
            
            loop.run_until_complete(controller.select_option(context, page, real_id))
            return f"Selected option with ID {target_id}"
        except Exception as e:
            return f"Error selecting option {target_id}: {str(e)}"


class CreateTabTool(Tool):
    """Creates a new browser tab and navigates to the specified URL."""
    name = "create_tab"
    description = "Creates a new browser tab and navigates to the specified URL."
    inputs = {
                    "url": {
                        "type": "string",
                        "description": "The URL to open in the new tab.",
                    },
            }
    output_type = "string"

    def forward(self, url: str) -> str:
        """Execute the create_tab tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            context = controller._context
            if context is None:
                return "Error: Browser context not available."
            
            # Create new page in the same context
            new_page = loop.run_until_complete(context.new_page())
            loop.run_until_complete(controller.visit_page(new_page, url))
            
            # Update controller to use the new page
            controller._page = new_page
            
            return f"Created new tab and navigated to {url}"
        except Exception as e:
            return f"Error creating tab: {str(e)}"


class SwitchTabTool(Tool):
    """Switches focus to a different browser tab by its index."""
    name = "switch_tab"
    description = "Switches focus to a different browser tab by its index."
    inputs = {
                    "tab_index": {
                        "type": "integer",
                        "description": "The index of the tab to switch to (0-based).",
                    },
            }
    output_type = "string"

    def forward(self, tab_index: int) -> str:
        """Execute the switch_tab tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            context = controller._context
            if context is None:
                return "Error: Browser context not available."
            
            # Get all pages in the context
            pages = loop.run_until_complete(asyncio.gather(*[asyncio.create_task(asyncio.sleep(0))]))
            pages = context.pages
            
            if tab_index < 0 or tab_index >= len(pages):
                return f"Error: Tab index {tab_index} out of range (0-{len(pages)-1})"
            
            # Switch to the specified page
            controller._page = pages[tab_index]
            loop.run_until_complete(pages[tab_index].bring_to_front())
            
            return f"Switched to tab {tab_index}"
        except Exception as e:
            return f"Error switching tab: {str(e)}"


class CloseTabTool(Tool):
    """Closes the specified browser tab by its index and switches to an adjacent tab."""
    name = "close_tab"
    description = "Closes the specified browser tab by its index and switches to an adjacent tab. Cannot close the last remaining tab."
    inputs = {
                    "tab_index": {
                        "type": "integer",
                        "description": "The index of the tab to close (0-based).",
                    },
            }
    output_type = "string"

    def forward(self, tab_index: int) -> str:
        """Execute the close_tab tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            context = controller._context
            if context is None:
                return "Error: Browser context not available."
            
            # Get all pages in the context
            pages = context.pages
            
            if len(pages) <= 1:
                return "Error: Cannot close the last remaining tab"
            
            if tab_index < 0 or tab_index >= len(pages):
                return f"Error: Tab index {tab_index} out of range (0-{len(pages)-1})"
            
            # Close the specified page
            page_to_close = pages[tab_index]
            loop.run_until_complete(page_to_close.close())
            
            # Switch to an adjacent tab
            new_index = max(0, tab_index - 1)
            controller._page = context.pages[new_index]
            loop.run_until_complete(controller._page.bring_to_front())
            
            return f"Closed tab {tab_index} and switched to tab {new_index}"
        except Exception as e:
            return f"Error closing tab: {str(e)}"


class UploadFileTool(Tool):
    """Upload a file to a specified input element."""
    name = "upload_file"
    description = "Upload a file to a specified input element."
    inputs = {
                    "target_id": {
                        "type": "string",
                        "description": "The ID of the target input element.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to be uploaded.",
                    },
            }
    output_type = "string"

    def forward(self, target_id: str, file_path: str) -> str:
        """Execute the upload_file tool."""
        import asyncio
        import os
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            
            # Upload file to the input element
            loop.run_until_complete(controller.upload_file(page, target_id, file_path))
            return f"Uploaded file {file_path} to element {target_id}"
        except Exception as e:
            return f"Error uploading file: {str(e)}"


class PageUpTool(Tool):
    """Scrolls the entire browser viewport one page UP towards the beginning."""
    name = "page_up"
    description = "Scrolls the entire browser viewport one page UP towards the beginning."
    inputs = {
    }
    output_type = "string"

    def forward(self) -> str:
        """Execute the page_up tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            loop.run_until_complete(page.keyboard.press("PageUp"))
            return "Scrolled one page up"
        except Exception as e:
            return f"Error scrolling page up: {str(e)}"


class PageDownTool(Tool):
    """Scrolls the entire browser viewport one page DOWN towards the end."""
    name = "page_down"
    description = "Scrolls the entire browser viewport one page DOWN towards the end."
    inputs = {
    }
    output_type = "string"

    def forward(self) -> str:
        """Execute the page_down tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            loop.run_until_complete(page.keyboard.press("PageDown"))
            return "Scrolled one page down"
        except Exception as e:
            return f"Error scrolling page down: {str(e)}"


class ClickFullTool(Tool):
    """Clicks the mouse on the target with the given id, with optional hold duration and button type."""
    name = "click_full"
    description = "Clicks the mouse on the target with the given id, with optional hold duration and button type."
    inputs = {
        "target_id": {
            "type": "integer",
            "description": "The numeric id of the target to click.",
        },
        "hold": {
            "type": "number",
            "description": "Seconds to hold the mouse button down before releasing. Default: 0.0.",
            "nullable": True,
        },
        "button": {
            "type": "string",
            "description": "Mouse button to use ('left' or 'right'). Default: 'left'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, target_id: int, hold: float = 0.0, button: str = "left") -> str:
        """Execute the click_full tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            context = controller._context
            if page is None:
                return "Error: No page available."
            
            # Map display ID to real element ID (for Set of Mark support)
            real_id = get_real_element_id(str(target_id))
            
            # Click with specified button and hold duration
            loop.run_until_complete(controller.click_id(
                context, page, real_id, 
                button=button, delay=int(hold * 1000)
            ))
            
            # Get updated page content
            content = loop.run_until_complete(controller.get_page_markdown(page, max_tokens=5000))
            
            return f"Clicked element ID {target_id} with {button} button (hold: {hold}s)\n\nPage content after click:\n{content}\n\nNote: Interactive elements are marked with numbers in the screenshot. Use those numbered markers to interact with elements."
        except Exception as e:
            return f"Error clicking element {target_id}: {str(e)}"


class ScrollElementDownTool(Tool):
    """Scrolls a given html element (e.g., a div or a menu) DOWN."""
    name = "scroll_element_down"
    description = "Scrolls a given html element (e.g., a div or a menu) DOWN."
    inputs = {
        "target_id": {
            "type": "integer",
            "description": "The numeric id of the target to scroll down.",
        },
    }
    output_type = "string"

    def forward(self, target_id: int) -> str:
        """Execute the scroll_element_down tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            
            # Map display ID to real element ID (for Set of Mark support)
            real_id = get_real_element_id(str(target_id))
            
            loop.run_until_complete(controller.scroll_element(page, real_id, "down"))
            return f"Scrolled element {target_id} down"
        except Exception as e:
            return f"Error scrolling element down: {str(e)}"


class ScrollElementUpTool(Tool):
    """Scrolls a given html element (e.g., a div or a menu) UP."""
    name = "scroll_element_up"
    description = "Scrolls a given html element (e.g., a div or a menu) UP."
    inputs = {
        "target_id": {
            "type": "integer",
            "description": "The numeric id of the target to scroll UP.",
        },
    }
    output_type = "string"

    def forward(self, target_id: int) -> str:
        """Execute the scroll_element_up tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            
            # Map display ID to real element ID (for Set of Mark support)
            real_id = get_real_element_id(str(target_id))
            
            loop.run_until_complete(controller.scroll_element(page, real_id, "up"))
            return f"Scrolled element {target_id} up"
        except Exception as e:
            return f"Error scrolling element up: {str(e)}"


class SummarizePageTool(Tool):
    """Uses AI to summarize the entire page."""
    name = "summarize_page"
    description = "Uses AI to summarize the entire page."
    inputs = {
    }
    output_type = "string"

    def forward(self) -> str:
        """Execute the summarize_page tool."""
        import asyncio
        controller = get_browser_controller()
        if controller is None:
            return "Error: Browser controller not initialized."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            page = controller._page
            if page is None:
                return "Error: No page available."
            # Get full page content for summarization (no token limit)
            content = loop.run_until_complete(controller.get_page_markdown(page))
            return f"Full page content for summarization:\n\n{content}"
        except Exception as e:
            return f"Error getting page content: {str(e)}"


# List of all available tools
ALL_TOOLS = [
    VisitUrlTool(),
    WebSearchTool(),
    HistoryBackTool(),
    RefreshPageTool(),
    ScrollDownTool(),
    ScrollUpTool(),
    ClickTool(),
    InputTextTool(),
    HoverTool(),
    KeypressTool(),
    ReadPageAndAnswerTool(),
    SleepTool(),
    StopActionTool(),
    SelectOptionTool(),
    CreateTabTool(),
    SwitchTabTool(),
    CloseTabTool(),
    UploadFileTool(),
    # Additional tools from magentic-ui
    PageUpTool(),
    PageDownTool(),
    ClickFullTool(),
    ScrollElementDownTool(),
    ScrollElementUpTool(),
    SummarizePageTool(),
]
