import json
import sys
import argparse
import os
from pathlib import Path

# Try to import rich for pretty printing, fall back to standard if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.theme import Theme
    HAS_RICH = True
    theme = Theme({
        "assistant": "bold cyan",
        "user": "bold green",
        "tool": "bold yellow",
        "system": "italic dim",
        "error": "bold red",
        "success": "bold green",
        "header": "bold white on blue"
    })
    console = Console(theme=theme)
except ImportError:
    HAS_RICH = False

def print_plain(msg):
    role = msg.get("role", "unknown").upper()
    content = msg.get("content", "")
    print(f"[{role}]: {content}\n")

def print_rich(msg):
    role = msg.get("role", "assistant" if msg.get("role") == "assistant" else msg.get("role", "unknown"))
    content = msg.get("content", "")
    
    # Handle Tool calls
    if role == "assistant" and msg.get("tool_calls"):
        tcalls = msg.get("tool_calls")
        t_text = Text("\n".join([f"  > {tc['name']}({tc.get('arguments', '')})" for tc in tcalls]))
        console.print(Panel(t_text, title="Agent Calls Tools", border_style="tool"))
    
    if content:
        console.print(f"[{role}]{role.upper()}:[/] {content}")

def view_path(input_path, persona_id=None, task_id=None, last_only=False):
    path = Path(input_path)
    if not path.exists():
        print(f"Error: {input_path} not found.")
        return

    all_sims = []
    
    # Identify files to load
    files_to_load = []
    if path.is_dir():
        files_to_load = list(path.glob("*.json"))
        # Exclude comparison.json if it exists
        files_to_load = [f for f in files_to_load if f.name != "comparison.json"]
    else:
        files_to_load = [path]

    for f_path in files_to_load:
        with open(f_path, "r") as f:
            try:
                data = json.load(f)
                sims = data if isinstance(data, list) else [data]
                all_sims.extend(sims)
            except json.JSONDecodeError:
                continue

    if not all_sims:
        print(f"No valid simulation JSON found in {input_path}")
        return

    # Apply filters
    filtered_sims = []
    for sim in all_sims:
        if persona_id and sim.get("_persona_id") != persona_id:
            continue
        if task_id and str(sim.get("task_id")) != str(task_id):
            continue
        filtered_sims.append(sim)

    if not filtered_sims:
        print("No simulations matching your filters were found.")
        return

    if last_only:
        filtered_sims = [filtered_sims[-1]]

    for sim in filtered_sims:
        p_id = sim.get("_persona_id", "Unknown")
        t_id = sim.get("task_id", "Unknown")
        reward_info = sim.get("reward_info")
        reward = reward_info.get("reward", 0.0) if reward_info else 0.0
        
        if HAS_RICH:
            console.print(f"\n[header] SIMULATION: Persona {p_id} | Task {t_id} | Reward {reward} [/]")
        else:
            print(f"\n{'='*20} SIMULATION: {p_id} (Task {t_id}) Reward: {reward} {'='*20}")

        for msg in sim.get("messages", []):
            if HAS_RICH:
                print_rich(msg)
            else:
                print_plain(msg)
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretty print simulation transcripts.")
    parser.add_argument("path", help="Path to results JSON file or directory")
    parser.add_argument("--persona", help="Filter by Persona ID")
    parser.add_argument("--task", help="Filter by Task ID")
    parser.add_argument("--last", action="store_true", help="View only the last simulation")
    
    args = parser.parse_args()
    view_path(args.path, persona_id=args.persona, task_id=args.task, last_only=args.last)
