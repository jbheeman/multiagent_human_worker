from smolagents.models import OpenAIServerModel
import os
from typing import Any
import json
import re
import time
from functools import wraps
from dataclasses import dataclass
from cleanpersona import _clean_persona

persona_model = OpenAIServerModel(
        model_id="qwen3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )



def retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0):
    """Decorator to retry a function with exponential backoff on timeout or connection errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, Exception) as e:
                    error_str = str(e).lower()
                    is_timeout = "timeout" in error_str or "timed out" in error_str
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed, raise the exception
                        raise
                    
                    if is_timeout or "connection" in error_str:
                        print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        # Non-timeout error, don't retry
                        raise
            return None
        return wrapper
    return decorator




REDDIT_PROMPT = """
You are an expert Psychological Profiler.
Generate a persona definition that is self-explanatory. The persona description must be so coherent and psychologically vivid that an AI acting as this person will naturally deduce how to behave in any situation purely by reading the description.

Do not write specific rules (e.g., 'Do not give zip code'). Instead, write the psychological reasoning (e.g., 'He is deeply skeptical of digital surveillance and treats personal data as a currency to be hoarded').

=== INPUT DATA ===
1. DEMOGRAPHIC ANCHOR:
{anchor_demographics}

2. PSYCHOLOGICAL SHIFT (Context: r/{subreddit}):
{psych_vector_str}

3. BEHAVIORAL SAMPLES:
{history_str}

=== OUTPUT FORMAT ===
You must output the persona in the following strict format:

### 1. CORE IDENTITY
(A first-person introduction: "I am a [Age] year old [Job]...")

### 2. PSYCHOLOGICAL DRIVERS
(A narrative explanation of *why* they act the way they do. Connect their background to their values.)

### 3. SCHWARTZ VALUES (JSON)
(Provide the raw values in a valid JSON block for parsing)
```json
{{
  "Security": 0.8,
  "Conformity": 0.4,
  ...
}}

=== YOUR RESPONSE ===
"""

CLAUDE_PROMPT = """
You are an expert Persona Compiler for Multi-Agent Simulation Environments. Your objective is to ingest raw, narrative-heavy human personas and compile them into strict, machine-readable YAML behavioral specifications. 

These YAML specifications will be used to govern the behavior of a simulated user interacting with a target agent in an objective, state-tracking benchmark (e.g., Dec-POMDP environments like Tau-bench).

YOUR DIRECTIVES:
1. Strip all narrative fluff, backstory, and "internal monologue" rules.
2. CRITICAL - DYNAMIC KEYS: For `cognitive_profile` and `interaction_policy`, you MUST invent custom, highly specific keys tailored to the specific Schwartz values of the persona. DO NOT use generic keys like "technical_competence", "patience_level", or "adaptability" for everyone. Invent keys like "authority_defiance", "novelty_seeking", "bureaucracy_tolerance", or "empathy_capacity".
3. Translate abstract values into concrete, observable Interaction Policies.
4. Define strict State-Transition Rules (If/Then heuristics) that dictate exactly how the persona reacts to specific agent behaviors.
5. Output ONLY valid YAML. Do not include any conversational filler, preamble, or postscript.

YAML SCHEMA TO POPULATE:

persona_profile:
  id: [Generate a descriptive string, e.g., "28_High_Stimulation"]
  demographics: [Brief string summarizing age, role, region]
  communication_style:
    formality: [Low/Medium/High] # Provide specific, actionable writing instructions instead of abstract traits
    sentence_structure: "[Describe sentence length, pacing, and whether they use run-ons, fragments, or perfect grammar]"
    vocabulary_and_lexicon: "[Describe specific word choices, industry jargon, slang, or idioms they rely on]"
    punctuation_and_formatting: "[Describe their typing habits: do they ignore capitalization? Overuse ellipses? Use emojis or ALL CAPS?]"
    example_utterances:
        - "[Invent a highly specific quote demonstrating the above 3 traits perfectly]"
        - "[Invent a second quote showing how they ask for help]"


cognitive_profile:
  [INVENT_CUSTOM_TRAIT_1]: [Low/Medium/High/Specific Descriptor]
  [INVENT_CUSTOM_TRAIT_2]: [Low/Medium/High/Specific Descriptor]
  [INVENT_CUSTOM_TRAIT_3]: [Low/Medium/High/Specific Descriptor]

interaction_policy:
  [INVENT_CUSTOM_POLICY_1]: [Specific Descriptor]
  [INVENT_CUSTOM_POLICY_2]: [Specific Descriptor]
  escalation_trigger: [Specific condition that causes them to demand a human/supervisor]

state_transition_rules:
  - "IF the agent asks you to wait or delays, THEN [Specific behavioral reaction]"
  - "IF the agent denies a request based on strict policy, THEN [Specific behavioral reaction]"
  - "IF the agent makes a mistake, THEN [Specific behavioral reaction]"
  - "[Add 1-2 more IF/THEN rules specific to this persona's dominant Schwartz values]"

termination_conditions:
  success: "The final database state matches the initial goal."
  abandonment: "[Specific condition where this persona gives up, e.g., 'Agent repeats the same question 3 times' or 'Task takes more than 5 turns']"
"""

CONFORMANCE_PROMPT = """
You are an Automated Conformance Evaluator for Multi-Agent Behavioral Specs.
Your task is to verify that a compiled YAML behavioral profile strictly aligns with its source Schwartz Values vector.

INPUT A (Source Values JSON):
{schwartz_json}

INPUT B (Compiled YAML):
{yaml_file}

EVALUATION CRITERIA:
Analyze the YAML against the source values and output a JSON evaluation with binary pass/fail scores.

1. Dominant Values Check: Identify the top 2 highest Schwartz values in INPUT A. Does the YAML explicitly codify behavioral policies, triggers, or tone that manifest these specific dominant values? 
2. Inferior Values Check: Identify the 1 lowest Schwartz value in INPUT A. Does the YAML explicitly show a lack of concern, or resistance, related to this lowest value?
3. State-Transition Verifiability: Do the state_transition_rules contain strict "IF/THEN" behavioral heuristics rather than vague narrative guidelines?
4. Termination Bounds Check: Are there strict, numerical bounds on when the persona abandons the task (e.g., specific turn limits, repetition limits)?

OUTPUT FORMAT (output ONLY valid JSON, no preamble):
{{
  "Dominant_Values_Check": {{"pass": true/false, "identified_values": "[List the 2 values]", "reason": "..."}},
  "Inferior_Value_Check": {{"pass": true/false, "identified_value": "[List the 1 value]", "reason": "..."}},
  "State_Transition_Check": {{"pass": true/false, "reason": "..."}},
  "Termination_Check": {{"pass": true/false, "reason": "..."}},
  "OVERALL_STATUS": "APPROVE or REJECT"
}}
"""




@dataclass
class PersonaDataInst:
    user_id: str             # "lumenation"
    subreddit: str           # "r/KotakuInAction" (The Context)
    
    # INPUTS FOR THE AGENT
    posts: list[str]         # The 5 posts from THIS subreddit only
    anchor_demographics: str # "28M, Developer, St. Louis" (extracted globally)
    shift_vector: dict       # {"POWER": 0.8, ...} (extracted locally from these posts)
    
    # GROUND TRUTH (For Evaluation)
    # We test if the agent matches THIS vector, not the global average
    target_vector: dict      # Same as shift_vector



def load_persona_dataset(path: str) -> list[PersonaDataInst]:
    with open(path, "r", encoding="utf-8") as f:
        examples: list[PersonaDataInst] = []
        for row in f:
            try:
                data = json.loads(row)
                examples.append(PersonaDataInst(user_id=data["user_id"], subreddit=data["subreddit"], posts=data["posts"], anchor_demographics=data["anchor_demographics"], shift_vector=data["shift_vector"], target_vector=data["target_vector"]))
            except Exception as e:
                print(f"Error loading row: {e}")
                continue
    return examples


@retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
def call_persona_model(prompt: str) -> str:
    response_message = persona_model([{"role": "user", "content": prompt}])
    return response_message.content or ""


def extract_schwartz_json(persona_text: str) -> dict:
    """Parse the ```json block from the raw REDDIT_PROMPT output."""
    match = re.search(r'```json\s*(\{.*?\})\s*```', persona_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: grab any bare JSON object in the text
    match = re.search(r'\{[^{}]*"[A-Za-z]+":\s*[\d.]+.*?\}', persona_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


@retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
def compile_persona(persona_description: str, critique: str = None) -> str:
    prompt = CLAUDE_PROMPT
    if critique:
        prompt += (
            "\n\n=== REVISION CONSTRAINTS (must address before outputting YAML) ===\n"
            f"{critique}\n"
            "Address every constraint above. Do NOT rewrite from scratch; only adjust the fields that failed.\n"
        )
    prompt += f"\n\n=== RAW PERSONA ===\n{persona_description}\n\n=== YOUR YAML OUTPUT ==="
    response = persona_model([{"role": "user", "content": prompt}])
    return response.content or ""


@retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
def call_conformance_model(prompt: str) -> dict:
    response = persona_model([{"role": "user", "content": prompt}])
    text = response.content or ""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def extract_critique(conformance_result: dict) -> str:
    """Build a targeted critique from the failed checks in a conformance result."""
    lines = []
    for key, val in conformance_result.items():
        if key == "OVERALL_STATUS":
            continue
        if isinstance(val, dict) and not val.get("pass", True):
            lines.append(f"- {key}: {val.get('reason', 'No reason provided.')}")
    return "\n".join(lines) if lines else "General conformance failure — tighten IF/THEN rules and termination bounds."


#Steps:
#1. Generate a persona using the REDDIT_PROMPT
#2. Compile the persona into a YAML file using the CLAUDE_PROMPT
#3. Evaluate the YAML file using the CONFORMANCE_PROMPT - which needs schwartz values as input from original persona <- can parse from original persona or from the data we use for persona generation
#4. If the YAML file is not approved, go back to step 2 and generate a new persona with critique Pass the Critique as a Direct Constraint, not a Rewrite from Scratch
    #Implement a Hard Cutoff ($k=3$) LLMs can occasionally get stuck in a loop where fixing one constraint breaks another. To prevent infinite API calls, implement a max_retries counter. If a persona fails the conformance check 3 times, kick it out of the automated pipeline and log it as an "edge-case failure" for your methodology section. Documenting why certain value combinations fail to compile cleanly is actually a great finding for the paper.


if __name__ == "__main__":

    #validation_personas.jsonl - write userID+generated persona as {userID: persona}

    # trainset = load_persona_dataset("train_gdelt_enriched.jsonl")
    testset = load_persona_dataset("H:/multiagent_human_worker/reddit/personasforpaper.jsonl")
    total_users = len(testset)
    print(f"Loaded {testset} users from personasforpaper.jsonl")
    
    # Load existing user_ids from output file to skip already processed users
    output_file = "pipeline_personas.jsonl"
    eval_folder = "H:/multiagent_human_worker/reddit/eval_personas"
    os.makedirs(eval_folder, exist_ok=True)

    
    
    print(f"Starting pipeline...\n")
    
    MAX_CONFORMANCE_RETRIES = 3

    successful = 0
    failed = 0
    skipped = 0
    edge_case_failures = []

    with open(output_file, "a", encoding="utf-8") as f:
        for i in range(len(testset)):
            user_id = testset[i].user_id
            # yaml_file = f"{eval_folder}/{user_id}.yaml" #yaml file for the persona
            yaml_output_path = os.path.join(eval_folder, f"{user_id}.yaml")

            try:
                print(f"[{i+1}/{total_users}] Processing user: {user_id}...", end=" ", flush=True)

                anchor_demographics = testset[i].anchor_demographics
                subreddit = testset[i].subreddit
                psych_vector_str = testset[i].shift_vector
                posts = testset[i].posts

                prompt = REDDIT_PROMPT.format(
                    history_str=posts,
                    anchor_demographics=anchor_demographics,
                    subreddit=subreddit,
                    psych_vector_str=psych_vector_str,
                )
                response_message = call_persona_model(prompt)
                persona_description = _clean_persona(response_message)

                schwartz_json = extract_schwartz_json(persona_description)
                if not schwartz_json:
                    schwartz_json = testset[i].target_vector

                # k=3 compile → conformance → critique retry loop
                critique = None
                conformance_result = None
                approved = False

                for attempt in range(MAX_CONFORMANCE_RETRIES):
                    yaml_content = compile_persona(persona_description, critique=critique)

                    conformance_result = call_conformance_model(
                        CONFORMANCE_PROMPT.format(schwartz_json=schwartz_json, yaml_file=yaml_content)
                    )
                    if not conformance_result:
                        print(f"\n  WARNING: Empty conformance result on attempt {attempt + 1}")
                        break

                    print(f"\n  Conformance attempt {attempt + 1}: {conformance_result.get('OVERALL_STATUS')}")

                    if conformance_result.get("OVERALL_STATUS") == "APPROVE":
                        approved = True
                        break

                    critique = extract_critique(conformance_result)
                    print(f"  Critique: {critique}")

                if not approved:
                    print(f"  EDGE CASE: {user_id} failed conformance after {MAX_CONFORMANCE_RETRIES} attempts — logging.")
                    edge_case_failures.append({
                        "user_id": user_id,
                        "last_conformance": conformance_result,
                    })
                    failed += 1
                    continue

                # Write the yaml file to the eval_folder so we can evaluate the personas after the pipeline
                yaml_output_path = os.path.join(eval_folder, f"{user_id}.yaml")
                with open(yaml_output_path, "w", encoding="utf-8") as file2:
                    file2.write(yaml_content)

                f.write(json.dumps({"user_id": user_id, "persona": persona_description, "yaml": yaml_output_path}, ensure_ascii=False) + "\n")
                f.flush()

                successful += 1
                # if i == 1:
                #     break
                print(f"  ✓ Success")

            except Exception as e:
                failed += 1
                print(f"✗ Failed: {e}")
                continue

    if edge_case_failures:
        with open("edge_case_failures.jsonl", "a", encoding="utf-8") as ef:
            for rec in edge_case_failures:
                ef.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Successful: {successful}/{total_users}")
    print(f"Skipped: {skipped}/{total_users}")
    print(f"Failed: {failed}/{total_users}")
    print(f"Edge-case failures logged: {len(edge_case_failures)}")
    print(f"{'='*50}")
