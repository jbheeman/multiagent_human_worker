from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

# Mock UrlStatus for standalone implementation
class UrlStatus:
    ALLOWED = "allowed"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class WebSurferConfig(BaseModel):
    """Configuration class for WebSurfer agent."""
    
    name: str = "WebSurferAgent"
    downloads_folder: Optional[str] = None
    description: Optional[str] = None
    debug_dir: Optional[str] = None
    start_page: str = "about:blank"
    animate_actions: bool = False
    to_save_screenshots: bool = False
    max_actions_per_step: int = 5
    to_resize_viewport: bool = True
    url_statuses: Optional[Dict[str, str]] = None
    url_block_list: Optional[List[str]] = None
    single_tab_mode: bool = False
    viewport_height: int = 1440
    viewport_width: int = 1440
    use_action_guard: bool = False
    search_engine: str = "duckduckgo"
    model_context_token_limit: Optional[int] = None
