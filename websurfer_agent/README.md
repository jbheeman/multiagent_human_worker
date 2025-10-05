# WebSurfer Agent for Smol Agents

This is a 1:1 copy of the magentic-ui WebSurfer agent implemented using the smol agents framework. It provides the same functionality as the original WebSurfer agent but integrates seamlessly with smol agents.

## Features

The WebSurfer agent provides comprehensive web browsing capabilities:

- **Navigation**: Visit URLs, go back/forward in history, refresh pages
- **Search**: Perform web searches using configurable search engines (Google, Bing, DuckDuckGo, Yahoo)
- **Interaction**: Click elements, type text, hover over elements, select options from dropdowns
- **Scrolling**: Scroll up/down on pages
- **Tab Management**: Create, switch, and close browser tabs
- **File Operations**: Upload files to web forms
- **Content Analysis**: Answer questions about page content, summarize pages
- **Screenshots**: Take and save screenshots of pages
- **Keyboard Input**: Press keyboard keys for advanced interactions
- **State Management**: Save and restore browser state

## Installation

```bash
# Install required dependencies
pip install smolagents playwright pillow tiktoken pydantic
```

## Quick Start

```python
import asyncio
from smolagents import OpenAIServerModel
from websurfer_agent import WebSurferAgent

async def main():
    # Initialize model
    model = OpenAIServerModel(
        model_id="gpt-4o",
        api_base="https://api.openai.com/v1",
        api_key="your-api-key-here"
    )
    
    # Create WebSurfer agent
    websurfer = WebSurferAgent(
        name="MyWebSurfer",
        model=model,
        debug_dir="./screenshots"
    )
    
    # Use the agent
    result = await websurfer.run("Search for the latest AI news")
    print(result)
    
    # Clean up
    await websurfer.close()

# Run the example
asyncio.run(main())
```

## Configuration

The WebSurfer agent accepts many configuration options:

```python
websurfer = WebSurferAgent(
    name="CustomWebSurfer",           # Agent name
    model=model,                      # Smol agents model
    downloads_folder="./downloads",   # Folder for downloads
    debug_dir="./debug",              # Debug directory for screenshots
    start_page="https://google.com",  # Starting page
    animate_actions=True,             # Animate browser actions
    to_save_screenshots=True,         # Save screenshots
    max_actions_per_step=5,           # Max actions per step
    to_resize_viewport=True,          # Resize viewport
    single_tab_mode=False,            # Allow multiple tabs
    viewport_height=1080,             # Viewport height
    viewport_width=1920,              # Viewport width
    search_engine="google",           # Default search engine
    use_action_guard=False            # Require approval for actions
)
```

## Available Tools

The agent includes all the tools from the original WebSurfer:

- `visit_url`: Navigate to a specific URL
- `web_search`: Perform a web search
- `click`: Click on page elements
- `input_text`: Type text into input fields
- `hover`: Hover over elements
- `scroll_up`/`scroll_down`: Scroll the page
- `history_back`: Go back in browser history
- `refresh_page`: Refresh the current page
- `keypress`: Press keyboard keys
- `sleep`: Wait for a specified duration
- `answer_question`: Answer questions about page content
- `select_option`: Select options from dropdowns
- `create_tab`: Create new browser tabs
- `switch_tab`: Switch between tabs
- `close_tab`: Close browser tabs
- `upload_file`: Upload files to forms
- `stop_action`: Stop and provide a final answer

## Advanced Usage

### State Management

Save and restore the agent's state:

```python
# Save state
state = await websurfer.save_state()

# Create new agent and load state
new_websurfer = WebSurferAgent(name="Restored", model=model)
await new_websurfer.load_state(state)
```

### Context Manager

Use the agent as a context manager for automatic cleanup:

```python
async with WebSurferAgent(model=model) as websurfer:
    result = await websurfer.run("Navigate to a website")
    # Automatic cleanup when exiting the context
```

### Custom Search Engines

Configure different search engines:

```python
websurfer = WebSurferAgent(
    model=model,
    search_engine="bing"  # Options: google, bing, duckduckgo, yahoo
)
```

### URL Filtering

Control which URLs the agent can access:

```python
websurfer = WebSurferAgent(
    model=model,
    url_statuses={"example.com": "allowed"},
    url_block_list=["malicious-site.com"]
)
```

## Architecture

The implementation follows the same architecture as the original WebSurfer:

- **WebSurferAgent**: Main agent class that coordinates everything
- **PlaywrightController**: Handles browser automation (using Playwright)
- **Tool Definitions**: Individual tool classes for each browser action
- **Prompts**: System prompts and tool prompts for the LLM
- **Set of Mark**: Screenshot annotation for interactive elements

## Differences from Original

While this is a 1:1 copy, there are some differences due to the smol agents framework:

1. **Model Integration**: Uses smol agents' model interface instead of autogen's
2. **Tool Execution**: Tools are wrapped to integrate with smol agents' execution model
3. **State Management**: Simplified state management for smol agents compatibility
4. **Async Handling**: Fully async implementation following smol agents patterns

## Examples

See the `example.py` file for comprehensive examples including:

- Basic usage
- Advanced configuration
- State management
- Context manager usage
- Error handling

## Requirements

- Python 3.8+
- smolagents
- playwright
- pillow
- tiktoken
- pydantic

## License

This implementation follows the same license as the original magentic-ui project.

## Contributing

This is a direct port of the magentic-ui WebSurfer agent. For improvements to the core WebSurfer functionality, please contribute to the original magentic-ui project.
