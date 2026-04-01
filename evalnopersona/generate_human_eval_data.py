import os
import json
import random
import csv
from pathlib import Path

def extract_trace(filepath):
    """Reads a trace JSON and returns the task goal and formatted dialogue."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the task goal
    try:
        task_goal = data['tasks'][0]['user_scenario']['instructions']['reason_for_call']
    except (KeyError, IndexError):
        task_goal = "Goal not found in trace."
        
    # Extract and format the dialogue
    try:
        messages = data['simulations'][0]['messages']
    except (KeyError, IndexError):
        return task_goal, "Error: No messages found in trace."

    trace_lines = []
    for msg in messages:
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content')
        
        if role == 'Assistant':
            if content:
                trace_lines.append(f"🤖 AGENT: {content}")
            if msg.get('tool_calls'):
                # Handle different tool call dictionary structures
                tools = []
                for t in msg['tool_calls']:
                    if 'function' in t:
                        tools.append(t['function']['name'])
                    elif 'name' in t:
                        tools.append(t['name'])
                    else:
                        tools.append('unknown_tool')
                trace_lines.append(f"⚙️  AGENT [Used Tools]: {', '.join(tools)}")
                
        elif role == 'User':
            if content:
                # We leave the <internal_monologue> in so you can see WHY the persona acts this way!
                trace_lines.append(f"👤 USER: {content}")
                
        elif role == 'Tool':
            # Skip dumping huge tool DB outputs to keep reading clean
            trace_lines.append(f"💻 SYSTEM [Database/Tool Returned Data]")
            
    formatted_trace = "\n\n".join(trace_lines)
    return task_goal, formatted_trace


def main():
    # --- CONFIGURATION ---
    # Update these paths if your actual drive letter or folder names are different
    BASE_EVAL_DIR = Path(r"H:\multiagent_human_worker\reddit\Eval\ScaledPersonas")
    YAML_DIR = Path(r"H:\multiagent_human_worker\reddit\ScaledPersonas\YAML")
    
    model1 = "kimi"
    model2 = "qwen3"
    domains = ["airline", "retail", "telecom"]
    num_samples = 50
    # ---------------------

    available_pairs = []

    print("Scanning directories for matching pairs...")
    for domain in domains:
        m1_dir = BASE_EVAL_DIR / domain / model1
        m2_dir = BASE_EVAL_DIR / domain / model2
        
        if not m1_dir.exists() or not m2_dir.exists():
            continue
            
        # Find all JSONs for model1
        for m1_file in m1_dir.glob("*.json"):
            filename = m1_file.name
            
            # Filename format expected: {persona}_{model}_{domain}.json
            # We need to extract the persona name to find the matching model2 file and YAML
            suffix_to_remove = f"_{model1}_{domain}.json"
            if filename.endswith(suffix_to_remove):
                persona_name = filename[:-len(suffix_to_remove)]
                
                # Check if model2 has the exact same scenario
                m2_filename = f"{persona_name}_{model2}_{domain}.json"
                m2_file = m2_dir / m2_filename
                
                # Check if YAML exists
                yaml_file = YAML_DIR / f"{persona_name}.yaml"
                
                if m2_file.exists() and yaml_file.exists():
                    available_pairs.append({
                        "domain": domain.capitalize(),
                        "persona_name": persona_name,
                        "m1_path": m1_file,
                        "m2_path": m2_file,
                        "yaml_path": yaml_file
                    })

    print(f"Found {len(available_pairs)} valid pairs across all domains.")
    
    if len(available_pairs) < num_samples:
        print(f"Warning: Only found {len(available_pairs)} pairs, proceeding with all of them.")
        sampled_pairs = available_pairs
    else:
        sampled_pairs = random.sample(available_pairs, num_samples)

    eval_data = []
    answer_key = []

    print("Extracting traces and building evaluation JSON...")
    for idx, pair in enumerate(sampled_pairs, 1):
        pair_id = f"pair_{idx:03d}"
        
        # Read the YAML
        with open(pair['yaml_path'], 'r', encoding='utf-8') as f:
            persona_yaml = f.read()
            
        # Extract traces
        task_goal, trace_m1 = extract_trace(pair['m1_path'])
        _, trace_m2 = extract_trace(pair['m2_path'])
        
        # Randomize which model is A and which is B (Blinding)
        is_m1_trace_A = random.choice([True, False])
        
        if is_m1_trace_A:
            trace_A = trace_m1
            trace_B = trace_m2
            answer_key.append({"pair_id": pair_id, "Trace_A": model1, "Trace_B": model2})
        else:
            trace_A = trace_m2
            trace_B = trace_m1
            answer_key.append({"pair_id": pair_id, "Trace_A": model2, "Trace_B": model1})

        # Compile the JSON record
        eval_data.append({
            "pair_id": pair_id,
            "domain": pair['domain'],
            "persona_profile": persona_yaml,
            "task_goal": task_goal,
            "trace_A": trace_A,
            "trace_B": trace_B
        })

    # Save the JSON for the CLI grading tool
    with open("evaluation_pairs.json", "w", encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
    # Save the answer key
    with open("answer_key.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "Trace_A", "Trace_B"])
        writer.writeheader()
        writer.writerows(answer_key)

    print("\n✅ Done! Successfully created:")
    print("  1. 'evaluation_pairs.json' (Use this with the CLI grading script)")
    print("  2. 'answer_key.csv' (Keep this hidden until you finish grading!)")

if __name__ == "__main__":
    main()