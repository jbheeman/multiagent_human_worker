"""
Defines the toolset and the shared browser instance for the FileSurfer agent.
"""
import os
from smolagents import tool
from .markdown_file_browser import MarkdownFileBrowser

# This browser instance will be shared by all the tool functions.
# It is initialized by the FileSurfer agent.
browser: MarkdownFileBrowser | None = None

def get_browser_state() -> str:
    """Helper function to get the current view of the browser."""
    if browser is None:
        return "Error: Browser is not initialized."
    header = (
        f"Current Path: {os.path.relpath(browser.path, browser._base_path)}\n"
        f"Page Title: {browser.page_title}\n"
        f"Page: {browser.viewport_current_page + 1} of {len(browser.viewport_pages)}.\n"
    )
    return header.strip() + "\n=======================\n" + browser.viewport

@tool
def open_path(path: str) -> str:
    """
    Opens a file or a directory to view its contents. Paths are relative.

    Args:
        path (str): The relative path to the file or directory to open.
    """
    browser.open_path(path)
    return get_browser_state()

@tool
def page_up() -> str:
    """Scrolls the viewport up one page in the currently open file."""
    browser.page_up()
    return get_browser_state()

@tool
def page_down() -> str:
    """Scrolls the viewport down one page in the currently open file."""
    browser.page_down()
    return get_browser_state()

@tool
def find_on_page(search_string: str) -> str:
    """
    Searches for text in the open file and jumps the viewport to the first match.

    Args:
        search_string (str): The text to search for within the file.
    """
    browser.find_on_page(search_string)
    return get_browser_state()

@tool
def find_next() -> str:
    """Jumps to the next search result after a search has been performed."""
    browser.find_next()
    return get_browser_state()