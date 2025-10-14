# Simulated Human Agents

A system of LLM-powered simulated human agents that provide human-in-the-loop decision making for multi-agent systems. These agents think through decisions and return structured outputs, enabling quality gates, approvals, and preferences without requiring actual human interaction.

## Overview

The simulated human system provides three role-specific agents:
- **Web Human**: Specializes in web navigation, source selection, and research decisions
- **Code Human**: Focuses on code review, security, and development decisions
- **File Human**: Handles file organization, naming, and documentation decisions

Each agent has a distinct persona with specific preferences and decision criteria, making them suitable for different types of tasks.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                          │
│                 (ToolCallingAgent)                       │
└────────────┬──────────────┬──────────────┬──────────────┘
             │              │              │
    ┌────────▼───────┐ ┌───▼────────┐ ┌──▼─────────┐
    │  Web Human     │ │ Code Human │ │ File Human │
    │     Tool       │ │    Tool    │ │    Tool    │
    └────────┬───────┘ └────┬───────┘ └──┬─────────┘
             │              │              │
    ┌────────▼──────────────▼──────────────▼─────────┐
    │        SimulatedHumanAgent                      │
    │    (LLM + Persona + Decision Logic)             │
    └─────────────────────────────────────────────────┘
```

## Installation & Setup

The simulated humans are already integrated into the HumanLLM system. To use them:

```python
import os
from smolagents import OpenAIServerModel, ToolCallingAgent
from simulated_humans import create_human_tools

# Create your model
model = OpenAIServerModel(
    model_id="gemma3",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key=os.environ["NAUT_API_KEY"],
)

# Create human tools
human_tools = create_human_tools(model=model, use_simulated=True)

# Add to orchestrator
agent = ToolCallingAgent(
    tools=[*your_other_tools, *human_tools],
    model=model,
)
```

## Usage

### 1. Using Tools in Orchestrator

The orchestrator can call human tools like any other tool:

```python
# The orchestrator decides when to call humans
agent.run("I found 3 Python libraries. Ask the code human which one is best for async web scraping.")
```

### 2. Using Tools Directly

You can also use the tools directly:

```python
from simulated_humans import CodeHumanTool

code_human = CodeHumanTool(model=model)

result = code_human.forward(
    task="Review this file deletion",
    context="About to delete /tmp/cache/*",
    needs="approval"
)
# Returns JSON: {"decision": "approve", "message": "...", ...}
```

### 3. Using Agents Directly

For more control, use the SimulatedHumanAgent directly:

```python
from simulated_humans import SimulatedHumanAgent

web_human = SimulatedHumanAgent(model=model, role="web")

decision = web_human.decide(
    task="Choose best documentation source",
    options=[
        "https://docs.python.org/3/library/asyncio.html",
        "https://medium.com/blog/asyncio-tutorial",
    ],
    needs="preference",
    hints={"user_level": "beginner"}  # Optional contextual hints
)
# Returns: HumanDecision dict
```

## Passing Hints (Credentials, Preferences, Context)

You can pass additional contextual information to simulated humans using the `hints` parameter:

```python
decision = web_human.decide(
    task="Help me login and book a room",
    context="On booking page, need credentials",
    needs="guidance",
    hints={
        "username": "ymalegao",
        "password": "my_password",
        "booking_details": {
            "building": "Science and Engineering",
            "time": "6pm-7pm",
            "capacity": "1-6"
        }
    }
)
```

### What to Put in Hints

- **Credentials**: Username, password, API keys
- **User Preferences**: Learning style, experience level, language
- **Session State**: Current progress, previous actions
- **Project Context**: Framework, dependencies, requirements
- **Constraints**: Time limits, budget, compliance requirements
- **Any other info**: Anything the simulated human might need to know

### Using Hints with Tools

When using tools (as the orchestrator does), pass hints as a JSON string:

```python
import json

hints_json = json.dumps({
    "username": "ymalegao",
    "preferences": {"capacity": "1-6"}
})

result = web_human_tool.forward(
    task="Book a room",
    hints=hints_json
)
```

## Decision Types (`needs` parameter)

- **`approval`**: Binary approve/deny/revise decision (default)
- **`preference`**: Choose between multiple options
- **`critique`**: Review and provide feedback
- **`clarification`**: Request more information
- **`takeover`**: Suggest when human intervention is needed

## Decision Response Format

All decisions return a `HumanDecision` dict:

```python
{
    "decision": "approve" | "deny" | "revise",
    "message": "Brief rationale (1-2 sentences)",
    "revisions": {"key": "value"} or None  # Only if decision is "revise"
}
```

## Personas

### Web Human
- **Role**: Decisive researcher/PM
- **Priorities**: Reputable sources, low risk, official documentation
- **Temperature**: 0.6 (moderate variability)
- **Use for**: Source selection, web navigation strategies, link choices

### Code Human
- **Role**: Senior code reviewer
- **Priorities**: Correctness, security, tests, maintainability
- **Temperature**: 0.3 (conservative)
- **Use for**: Code reviews, security checks, destructive operations

### File Human
- **Role**: Documentation owner/editor
- **Priorities**: Naming conventions, structure, completeness
- **Temperature**: 0.3 (conservative)
- **Use for**: File organization, naming, documentation review

## Examples

### Example 1: Web Source Selection

```python
web_human = SimulatedHumanAgent(model=model, role="web")

decision = web_human.decide(
    task="Choose best Python documentation",
    options=[
        "https://docs.python.org/3/library/asyncio.html",
        "https://medium.com/blog/asyncio-intro"
    ],
    needs="preference"
)

# Likely result:
# {
#   "decision": "approve",
#   "message": "Official docs are authoritative and comprehensive",
#   "revisions": {"selected": "https://docs.python.org/3/library/asyncio.html"}
# }
```

### Example 2: Code Security Review

```python
code_human = SimulatedHumanAgent(model=model, role="code")

code = '''
def process_input(user_data):
    return eval(user_data)  # Execute user input
'''

decision = code_human.decide(
    task="Review code for security issues",
    context=f"Code snippet:\n{code}",
    needs="critique"
)

# Likely result:
# {
#   "decision": "deny",
#   "message": "Using eval() on user input is a critical security vulnerability",
#   "revisions": {"suggestion": "Use ast.literal_eval() or json.loads() instead"}
# }
```

### Example 3: File Naming Review

```python
file_human = SimulatedHumanAgent(model=model, role="file")

decision = file_human.decide(
    task="Review file naming",
    context="Files: MainHelper.py, utils_FINAL.py, test.py",
    needs="critique"
)

# Likely result:
# {
#   "decision": "revise",
#   "message": "Inconsistent naming; use kebab-case or snake_case consistently",
#   "revisions": {
#       "MainHelper.py": "main_helper.py",
#       "utils_FINAL.py": "utils.py",
#       "test.py": "test_utils.py"
#   }
# }
```

## Configuration

### Enable/Disable in Orchestrator

In `Orchestrator.py`:

```python
# Set to False to disable simulated humans
USE_SIMULATED_HUMANS = True

if USE_SIMULATED_HUMANS:
    human_tools = create_human_tools(model=model, use_simulated=True)
else:
    human_tools = []
```

### Create Specific Roles Only

```python
# Only create web and code humans (no file human)
human_tools = create_human_tools(
    model=model,
    use_simulated=True,
    roles=["web", "code"]
)
```

## Advanced Usage

### Access Decision History

```python
web_human = SimulatedHumanAgent(model=model, role="web")

# Make some decisions...
web_human.decide(task="...", needs="approval")
web_human.decide(task="...", needs="preference")

# Get history
history = web_human.get_decision_history()
# [{"task": "...", "needs": "approval", "decision": {...}}, ...]
```

### Update Preferences

```python
code_human = SimulatedHumanAgent(model=model, role="code")

# Update preferences for this session
code_human.update_preferences({
    "require_tests": False,  # Override default
    "allow_destructive_ops": True
})
```

## When to Use Simulated Humans

The orchestrator can call human tools when:

1. **Ambiguity**: Multiple valid options → ask for preference
2. **Risk**: Destructive operations, auth, costs → ask for approval first
3. **Quality Gates**: Before final answer, merge, or export → get review
4. **Blocked**: Paywalls, CAPTCHAs, 403s → get guidance or takeover suggestion

## Future Extensions

- **Real Human Proxy**: CLI/Web UI integration for actual human input
- **Memory Persistence**: Save preferences across sessions
- **Multi-round Conversations**: Extended back-and-forth dialogues
- **Custom Personas**: User-defined roles and preferences
- **Metrics Dashboard**: Track approval rates, revision rates, etc.

## Files

- `human_decision.py`: TypedDict for decision format
- `personas.py`: System prompts and preferences for each role
- `simulated_human_agent.py`: Core agent with decision logic
- `human_tools.py`: SmolAgents tool wrappers
- `example.py`: Usage examples
- `design_doc.md`: Detailed design documentation

## Design Philosophy

1. **Orchestrator Agnostic**: Orchestrator doesn't know if it's talking to a real or simulated human
2. **Structured Output**: Always returns valid `HumanDecision` dict
3. **Single-Round**: One request, one decision (agent thinks internally)
4. **Persona-Driven**: Each role has distinct personality and preferences
5. **Minimal Friction**: Easy to enable/disable, no changes to orchestrator logic

## See Also

- Design document: `design_doc.md`
- Examples: `example.py`
- Main orchestrator: `../Orchestrator.py`

