"""URL Status Manager for controlling which URLs can be accessed."""

from typing import Dict, List, Optional


class UrlStatusManager:
    """Manages allowed and blocked URLs for the browser."""
    
    def __init__(
        self,
        url_statuses: Optional[Dict[str, str]] = None,
        url_block_list: Optional[List[str]] = None,
    ):
        """Initialize the URL status manager.
        
        Args:
            url_statuses: Dictionary mapping URLs to their status ("allowed" or "blocked")
            url_block_list: List of URLs to block
        """
        self.url_statuses = url_statuses or {}
        self.url_block_list = url_block_list or []
        
        # Add block list entries to statuses
        for url in self.url_block_list:
            self.url_statuses[url] = "blocked"
    
    def is_url_allowed(self, url: str) -> bool:
        """Check if a URL is allowed.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is allowed, False otherwise
        """
        # If no statuses are set, allow all URLs
        if not self.url_statuses:
            return True
        
        # Check exact match
        if url in self.url_statuses:
            return self.url_statuses[url] == "allowed"
        
        # Check domain match
        for status_url, status in self.url_statuses.items():
            if url.startswith(status_url) or status_url in url:
                return status == "allowed"
        
        # Default: allow if not explicitly blocked
        return True
    
    def is_url_blocked(self, url: str) -> bool:
        """Check if a URL is blocked.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is blocked, False otherwise
        """
        return not self.is_url_allowed(url)
    
    def set_url_status(self, url: str, status: str) -> None:
        """Set the status for a URL.
        
        Args:
            url: The URL to set status for
            status: Either "allowed" or "blocked"
        """
        self.url_statuses[url] = status
    
    def block_url(self, url: str) -> None:
        """Block a URL.
        
        Args:
            url: The URL to block
        """
        self.set_url_status(url, "blocked")
    
    def allow_url(self, url: str) -> None:
        """Allow a URL.
        
        Args:
            url: The URL to allow
        """
        self.set_url_status(url, "allowed")

