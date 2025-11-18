# HumanLLM - Smol Agent Orchestrator

A powerful multi-agent orchestration system using smolagents framework with specialized agents for web browsing, file surfing, and code execution.

## Features


### Available Agents

1. **WebSurferAgent** - Browse the web and interact with web pages
2. **FileSurferAgent** - Navigate and analyze local file systems (read-only)
3. **CoderAgent** - Write and execute code to solve tasks (NEW!)

### Orchestrator

The `Orchestrator.py` provides a unified interface that combines all agents and allows them to work together to solve complex tasks.

## Quick Start

### Installation

```bash
pip install smolagents autogen-ext
# For additional functionality
pip install numpy pandas matplotlib pillow requests beautifulsoup4
```

### Setup

```bash
export NAUTILUS_API_KEY="your_api_key_here"
```

### Using Individual Agents

#### CoderAgent

```python
from smolagents import LiteLLMModel
from coder_agent import CoderAgent

model = LiteLLMModel(model_id="openai/llama3")
coder = CoderAgent(model=model)

result = coder.run("Calculate the factorial of 10")
print(result)
```

#### FileSurferAgent

```python
from file_surfer_agent import FileSurfer

file_surfer = FileSurfer(model=model, base_path=".")
result = file_surfer.run("List all Python files in this directory")
```

#### WebSurferAgent

```python
from websurfer_agent import WebSurferAgent

web_surfer = WebSurferAgent(model=model)
result = await web_surfer.run("Search for Python tutorials")
```

### Using the Orchestrator

```python
from smolagents import CodeAgent, OpenAIServerModel
from websurfer_agent import WebSurferTool
from file_surfer_agent import FileSurferTool
from coder_agent import CoderTool

model = OpenAIServerModel(
    model_id="gemma3",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key=os.environ["NAUT_API_KEY"],
)

# Create tools
web_surfer_tool = WebSurferTool(model=model)
file_surfer_tool = FileSurferTool(model=model, base_path=".")
coder_tool = CoderTool(model=model)

# Create orchestrator
manager_agent = CodeAgent(
    tools=[web_surfer_tool, file_surfer_tool, coder_tool],
    model=model,
)

# Run complex tasks
result = manager_agent.run(
    "Find good stocks to buy and create a CSV file with the analysis"
)
```

## Project Structure

```
HumanLLM/
├── Orchestrator.py           # Main orchestrator script
├── websurfer_agent/          # Web browsing agent
│   ├── web_surfer_agent.py
│   ├── web_surfer_tool.py
│   └── ...
├── file_surfer_agent/        # File system agent
│   ├── file_surfer.py
│   ├── file_surfer_tool.py
│   └── ...
├── coder_agent/              # Code execution agent
│   ├── coder_agent.py
│   ├── coder_tool.py
│   ├── demo.py
│   └── ...
└── browser_playwright/       # Browser automation backend
```

## Agent Capabilities

| Feature | CoderAgent | FileSurfer | WebSurfer |
|---------|-----------|------------|-----------|
| Code Execution | ✓ | ✗ | ✗ |
| File Read/Write | ✓ | Read-only | ✗ |
| Web Browsing | ✗ | ✗ | ✓ |
| Data Analysis | ✓ | Limited | ✗ |
| Debugging | ✓ (iterative) | N/A | N/A |
| Multi-step Tasks | ✓ | ✓ | ✓ |

## Examples

See individual agent directories for detailed examples:
- `coder_agent/example.py` - CoderAgent examples
- `coder_agent/demo.py` - Quick CoderAgent demo
- `file_surfer_agent/example.py` - FileSurfer examples
- `websurfer_agent/example.py` - WebSurfer examples

## Testing

Run tests for individual agents:

```bash
# Test CoderAgent
cd coder_agent && python test_coder.py

# Test FileSurfer
cd file_surfer_agent && python test_file_surfer.py
```

## License

MIT
