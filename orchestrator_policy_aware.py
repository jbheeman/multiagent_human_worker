"""Policy-Aware Orchestrator with Cost-Conscious Human-in-the-Loop.

This orchestrator follows a clear escalation policy:
1. Prefer tools < specialists < humans
2. Hard rules: humans BEFORE risky actions
3. Soft rules: humans AFTER failures or low confidence
4. Keep human calls minimal unless risk is flagged

The LLM orchestrator makes these decisions based on the system prompt - no Python loops needed.
"""

import os
import json
from smolagents import ToolCallingAgent, OpenAIServerModel

from websurfer_agent import WebSurferTool
from file_surfer_agent import FileSurferTool
from coder_agent import CoderTool
from simulated_humans import create_human_tools


# Create the model
model = OpenAIServerModel(
    model_id="gemma3",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key=os.environ["NAUT_API_KEY"],
)

# No user hints needed for GAIA dataset tasks

# Create specialist tools
web_surfer_tool = WebSurferTool(model=model)
file_surfer_tool = FileSurferTool(
    model=model,
    base_path=".",
    viewport_size=2048
)
from pathlib import Path
coder_tool = CoderTool(
    model=model,
    max_debug_rounds=3,
    use_local_executor=True,
    work_dir=Path.cwd()
)

# Create human tools (simulated)
human_tools = create_human_tools(model=model, use_simulated=True)

# Policy-aware system prompt
POLICY_AWARE_SYSTEM_PROMPT = """You are a policy-aware orchestrator with access to specialist tools and human consultants.

**COST HIERARCHY (always prefer cheaper before expensive):**
1. Direct tool use (cheapest)
2. Specialist agents (moderate)
3. Human consultants (expensive)

**ESCALATION POLICY:**

You have three domain-specific human consultants (web_human, code_human, file_human). Each tool description 
contains HARD RULES and SOFT RULES. Follow them precisely:

**HARD RULES** (must call human with phase='guard' BEFORE the action):
- web_human: credentials, payments, account creation, personal data forms, TOS violations
- code_human: delete/overwrite files, destructive git ops, execute untrusted code, security changes
- file_human: delete/move critical files, bulk ops (>10 files), external distribution

**SOFT RULES** (call human when conditions are met):
- phase='help': After ≥2 failures on the same task
- phase='plan': If confidence < 0.6 or task is novel/unfamiliar
- phase='verify': If final result confidence < 0.7

**DECISION INTERPRETATION:**
When you call a human tool, you receive a JSON response with:
- decision: approve|deny|revise|suggest|verify_ok|verify_fail
- message: brief rationale
- revisions: optional corrections/hints

**How to respond:**
- approve → proceed with the action
- deny → stop, don't do it, explain to user
- revise → adjust your approach using the revisions dict
- suggest → follow the hints in revisions and message
- verify_ok → task complete, return result
- verify_fail → fix the issues in message, retry

**WORKFLOW EXAMPLE:**
1. User asks to find flights → use web_surfer directly (no human needed yet)
2. Hit CAPTCHA → tried twice, failed → call web_human(phase='help', needs='critique', ...)
3. Human suggests alternative site → try that
4. Need to create account → call web_human(phase='guard', needs='approval', ...) with credentials in hints
5. Human approves → proceed
6. Got result → call web_human(phase='verify', needs='critique', ...)
7. Human returns verify_ok → done

**KEEP HUMAN CALLS MINIMAL:** Only escalate when policy requires it or you're genuinely stuck. 
Most tasks should complete with just tools/specialists.

**CREDENTIALS:** When calling human tools with phase='guard' for credential-requiring actions, 
always pass the user hints as the 'hints' parameter.
"""

# Create the orchestrator agent
# Note: ToolCallingAgent doesn't accept system_prompt in __init__
# Instead, we include the policy in the task description when calling run()
orchestrator = ToolCallingAgent(
    tools=[web_surfer_tool, file_surfer_tool, coder_tool, *human_tools],
    model=model,
   
)


def run_task(task_description: str, verbose: bool = True):
    """Run a task with the policy-aware orchestrator.
    
    Args:
        task_description: The task to complete
        verbose: Whether to print the result
        
    Returns:
        The orchestrator's response
    """
    if verbose:
        print("\n" + "="*80)
        print("POLICY-AWARE ORCHESTRATOR")
        print("="*80)
        print(f"\nTASK: {task_description}\n")
        print("Running with cost-aware escalation policy...")
        print("="*80 + "\n")
    
    # Prepend the policy-aware system prompt to the task
    full_task = f"""{POLICY_AWARE_SYSTEM_PROMPT}

---

USER TASK:
{task_description}
"""
    
    result = orchestrator.run(full_task)
    
    if verbose:
        print("\n" + "="*80)
        print("RESULT:")
        print("="*80)
        print(result)
        print("="*80 + "\n")
    
    return result


if __name__ == "__main__":
    task = f"""What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris
patients from Jan-May 2018 as listed on the NIH website?.
"""
    
    run_task(task)

