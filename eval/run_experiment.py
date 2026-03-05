"""Run persona simulations on any τ²-bench domain.

Loads or generates personas, runs τ²-bench simulations, and saves per-persona
result files. Supports three experimental conditions and all τ²-bench domains.

Conditions:
- "layer": 3-layer personas (L0 orientations + L1 background + L2 situational state)
- "legacy": Old Schwartz-values-based YAML personas
- "none": No persona (stock τ²-bench user simulator baseline)

Domains: airline (50 tasks), retail (114 tasks), telecom (2285 tasks),
         telecom-workflow (2285 tasks, workflow policy variant)

Usage:
    # Layer personas on retail:
    python -m eval.run_experiment \
        --condition layer \
        --personas reddit/eval_personas_all/ \
        --domain retail \
        --task-ids 0 1 \
        --max-concurrency 20

    # Layer personas on airline with task split:
    python -m eval.run_experiment \
        --condition layer \
        --personas reddit/eval_personas_all/ \
        --domain airline \
        --task-split test

    # Baseline (no persona) on telecom:
    python -m eval.run_experiment \
        --condition none \
        --domain telecom \
        --task-split small

    # After running, analyze results:
    python -m eval.analyze eval/results/
    python -m eval.run_comparison --layer eval/results/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


def _add_tau2_to_path():
    """Add tau2-bench/src to sys.path so tau2 imports work."""
    tau2_src = Path(__file__).resolve().parent.parent / "tau2-bench" / "src"
    if str(tau2_src) not in sys.path:
        sys.path.insert(0, str(tau2_src))

_add_tau2_to_path()


def load_layer_personas(
    persona_dir: str,
    specs_dir: str,
) -> list[dict]:
    """Load layer-based persona YAML files and generate assembled prompts.

    Returns list of dicts, each with:
        persona_id: code string like "1A-2E-3C-4D-5J/C8-D3-S5-F3/E8-B3-G4"
        persona_prompt: assembled system prompt
        persona_codes: {dimension: code} dict for trait sensitivity analysis
        yaml_path: path to source YAML
    """
    from persona.registry import LayerRegistry
    from persona.schema import (
        StateTransitionRule,
        TerminationConditions,
        GenerationMode,
    )
    from persona.assembler import assemble_system_prompt

    registry = LayerRegistry(specs_dir)
    personas = []

    yaml_files = sorted(Path(persona_dir).glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {persona_dir}")
        return []

    for yaml_path in yaml_files:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            continue

        profile = data.get("persona_profile", {})
        layer_codes = profile.get("layer_codes", {})

        # If this is a new-format persona with layer_codes, reconstruct PersonaConfig
        if layer_codes:
            config = _reconstruct_config(registry, data)
            if config is None:
                print(f"  Skipping {yaml_path.name}: failed to reconstruct config")
                continue
            prompt = assemble_system_prompt(registry, config)
            personas.append({
                "persona_id": profile.get("id", yaml_path.stem),
                "persona_prompt": prompt,
                "persona_codes": layer_codes,  # {dim: code} for trait sensitivity
                "yaml_path": str(yaml_path),
            })
        else:
            # Old-format persona — load raw YAML for legacy mode
            print(f"  {yaml_path.name}: old format (no layer_codes), use --condition legacy")
            continue

    return personas


def _reconstruct_config(registry, yaml_data: dict):
    """Reconstruct a PersonaConfig from saved YAML data.

    Uses registry.build_persona_config() for layer selections,
    then manually adds state transition rules and termination conditions.
    """
    from persona.schema import (
        StateTransitionRule,
        TerminationConditions,
        GenerationMode,
    )

    profile = yaml_data.get("persona_profile", {})
    layer_codes = profile.get("layer_codes", {})
    rules_data = yaml_data.get("state_transition_rules", {})
    term_data = yaml_data.get("termination_conditions", {})

    # Validate codes before building
    errors = registry.validate_codes(layer_codes)
    if errors:
        for err in errors:
            print(f"    WARNING: {err}")
        return None

    # Build config from layer codes using the registry's method
    try:
        mode_str = profile.get("mode", "synthetic")
        try:
            mode = GenerationMode(mode_str)
        except ValueError:
            mode = GenerationMode.SYNTHETIC

        config = registry.build_persona_config(
            layer_codes,
            mode=mode,
            demographics=profile.get("demographics", ""),
            source_user_id=profile.get("source_user_id"),
            source_subreddit=profile.get("source_subreddit"),
        )
    except KeyError as e:
        print(f"    WARNING: {e}")
        return None

    # Rebuild state transition rules from YAML
    for rule_dict in rules_data.get("core", []):
        config.state_transition_rules.append(StateTransitionRule(
            trigger=rule_dict.get("trigger", ""),
            behavior=rule_dict.get("behavior", ""),
            derived_from=rule_dict.get("derived_from", ""),
            is_core=True,
        ))
    for rule_dict in rules_data.get("persona_specific", []):
        config.state_transition_rules.append(StateTransitionRule(
            trigger=rule_dict.get("trigger", ""),
            behavior=rule_dict.get("behavior", ""),
            derived_from=rule_dict.get("derived_from", ""),
            is_core=False,
        ))

    # Termination conditions
    config.termination = TerminationConditions(
        success=term_data.get("success", "The final database state matches the initial goal."),
        abandonment=term_data.get("abandonment", ""),
    )

    if not config.is_complete():
        print(f"    WARNING: incomplete config (missing dimensions)")
        return None

    return config


def load_legacy_personas(persona_dir: str) -> list[dict]:
    """Load old Schwartz-format YAML files as raw yaml_content.

    Returns list of {"persona_id": ..., "yaml_content": ..., "yaml_path": ...}
    """
    personas = []
    yaml_files = sorted(Path(persona_dir).glob("*.yaml"))

    for yaml_path in yaml_files:
        with open(yaml_path, "r") as f:
            content = f.read()
        personas.append({
            "persona_id": yaml_path.stem,
            "yaml_content": content,
            "yaml_path": str(yaml_path),
        })

    return personas


def run_condition(
    condition: str,
    domain: str,
    personas: list[dict],
    output_dir: str,
    num_tasks: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
    task_split: Optional[str] = None,
    task_persona_filter: Optional[str] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    seed: int = 300,
    llm_agent: str = "gpt-4.1",
    llm_user: str = "gpt-4.1",
    agent_type: str = "llm_agent",
    max_concurrency: int = 1,
) -> str:
    """Run tau-bench simulations for one condition.

    Args:
        condition: "layer", "legacy", or "none"
        domain: tau-bench domain name (e.g., "retail", "airline", "telecom")
        personas: List of persona dicts from load_*_personas()
        output_dir: Directory to save results
        num_tasks: Limit number of tasks (None = all)
        task_ids: Specific task IDs to run
        task_split: Named task split (e.g., "base", "small", "train", "test")
        num_trials: Number of trials per task
        max_steps: Max conversation steps
        seed: Random seed
        llm_agent: Model for the agent
        llm_user: Model for the user simulator
        agent_type: "llm_agent" or "llm_agent_gt"
        max_concurrency: Max parallel simulations

    Returns:
        Path to results directory.
    """
    from tau2.run import run_tasks, get_tasks
    from tau2.evaluator.evaluator import EvaluationType
    import litellm

    # Setup Nautilus API if needed
    api_key = os.getenv("NAUTILUS_API_KEY")
    api_base = os.getenv("NAUTILUS_API_BASE", "https://ellm.nrp-nautilus.io/v1")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = api_base

    # Register Nautilus models (bare names, prefixed names, and gateway-resolved names)
    models = [
        "openai/llama3-sdsc", "llama3-sdsc", "meta-llama/Llama-3.3-70B-Instruct",
        "openai/gemma3", "gemma3", "google/gemma-3-27b-it",
        "openai/qwen3", "qwen3",
    ]
    for m in models:
        litellm.model_cost[m] = {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0, "max_tokens": 32768}
    litellm.request_timeout = 120

    llm_args = {
        "base_url": api_base,
        "api_key": api_key,
        "custom_llm_provider": "openai"
    }

    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=task_split,
        task_ids=task_ids,
        num_tasks=num_tasks,
    )
    if task_persona_filter is not None:
        tag = f"PERSONA:{task_persona_filter}"
        tasks = [t for t in tasks if tag in t.id]
        print(f"Filtered to {len(tasks)} tasks with {tag}")
    print(f"\n{'='*60}")
    print(f"CONDITION: {condition}")
    print(f"Domain: {domain}, Tasks: {len(tasks)}, Trials: {num_trials}, Agent: {agent_type}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{condition}.json")

    all_simulations = []

    if condition == "none":
        # Baseline: no persona
        print(f"Running baseline (no persona)...")
        results = run_tasks(
            domain=domain,
            tasks=tasks,
            agent=agent_type,
            user="user_simulator",
            llm_agent=llm_agent,
            llm_args_agent=llm_args,
            llm_user=llm_user,
            llm_args_user=llm_args,
            num_trials=num_trials,
            max_steps=max_steps,
            save_to=None,
            console_display=True,
            evaluation_type=EvaluationType.ALL,
            max_concurrency=max_concurrency,
            seed=seed,
            persona_prompt=None,
            yaml_content=None,
        )
        for sim in results.simulations:
            sim_dict = json.loads(sim.model_dump_json())
            sim_dict["_condition"] = "none"
            sim_dict["_persona_id"] = "baseline"
            sim_dict["_persona_codes"] = {}
            all_simulations.append(sim_dict)

    elif condition in ("layer", "legacy"):
        from concurrent.futures import ThreadPoolExecutor
        
        def _run_single_persona(persona):
            pid = persona.get("persona_id", "unknown")
            pcodes = persona.get("persona_codes", {})
            persona_prompt = persona.get("persona_prompt")
            yaml_content = persona.get("yaml_content")
            
            print(f"Starting Persona: {pid}")
            results = run_tasks(
                domain=domain,
                tasks=tasks,
                agent=agent_type,
                user="user_simulator",
                llm_agent=llm_agent,
                llm_args_agent=llm_args,
                llm_user=llm_user,
                llm_args_user=llm_args,
                num_trials=num_trials,
                max_steps=max_steps,
                save_to=None,
                console_display=False, # Disable rich console noise in parallel
                evaluation_type=EvaluationType.ALL,
                max_concurrency=1, # One task at a time PER thread
                seed=seed,
                persona_prompt=persona_prompt,
                yaml_content=yaml_content,
            )
            
            persona_sims = []
            for sim in results.simulations:
                sim_dict = json.loads(sim.model_dump_json())
                sim_dict["_condition"] = condition
                sim_dict["_persona_id"] = pid
                sim_dict["_persona_codes"] = pcodes
                persona_sims.append(sim_dict)
            print(f"Finished Persona: {pid}")
            return persona_sims

        print(f"Running {len(personas)} personas in parallel (concurrency={max_concurrency})...")
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            batch_results = list(executor.map(_run_single_persona, personas))
            
        # Save one file per persona — the directory IS the result set
        for persona_sims in batch_results:
            if not persona_sims:
                continue
            pid = persona_sims[0]["_persona_id"]
            safe_pid = pid.replace("/", "_")
            p_output_path = os.path.join(output_dir, f"{safe_pid}.json")
            with open(p_output_path, "w") as f:
                json.dump(persona_sims, f, indent=2)
            all_simulations.extend(persona_sims)

    print(f"\nSaved {len(all_simulations)} simulations to {output_dir}/")
    return output_dir  # return the directory, not a consolidated file


def run_evaluation(result_files: dict[str, str], output_path: str):
    """Run comparison evaluation on saved result files.

    Args:
        result_files: {"layer": "path.json", "none": "path.json", ...}
        output_path: Where to save comparison JSON.
    """
    from eval.run_comparison import evaluate_from_files
    evaluate_from_files(result_files, output_path=output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full persona comparison experiment"
    )

    # What to run
    parser.add_argument(
        "--condition", nargs="+", required=True,
        choices=["layer", "legacy", "none"],
        help="Which conditions to run"
    )

    # Persona sources
    parser.add_argument("--personas", help="Path to persona YAML directory")
    parser.add_argument("--specs", default="specs", help="Path to specs/ directory")

    # Tau-bench config
    parser.add_argument("--domain", default="retail",
                        choices=["airline", "retail", "telecom", "telecom-workflow"],
                        help="tau-bench domain (default: retail)")
    parser.add_argument("--num-tasks", type=int, default=None, help="Limit tasks")
    parser.add_argument("--task-ids", nargs="+", default=None, help="Specific task IDs")
    parser.add_argument("--task-split", default=None,
                        choices=["base", "small", "train", "test"],
                        help="Named task split (e.g., 'small', 'test')")
    parser.add_argument("--task-persona-filter", default=None,
                        choices=["None", "Easy", "Hard"],
                        help="Filter tasks by built-in tau2 persona tag (None/Easy/Hard)")
    parser.add_argument("--num-trials", type=int, default=1, help="Trials per task")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per sim")
    parser.add_argument("--seed", type=int, default=300, help="Random seed")
    parser.add_argument("--llm-agent", default="gpt-4.1", help="Agent model")
    parser.add_argument("--llm-user", default="gpt-4.1", help="User simulator model")
    parser.add_argument("--agent-type", default="llm_agent", choices=["llm_agent", "llm_agent_gt"], help="Agent implementation")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Max parallel sims")

    # Output
    parser.add_argument("--output", default="eval/results", help="Output directory")

    # Evaluation only (skip simulation)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip simulation, just run evaluation on existing result files")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    result_files = {}

    if args.eval_only:
        # Look for existing result files
        for cond in args.condition:
            path = os.path.join(args.output, f"{cond}.json")
            if os.path.exists(path):
                result_files[cond] = path
            else:
                print(f"WARNING: {path} not found, skipping condition '{cond}'")
    else:
        # Run simulations for each condition
        for cond in args.condition:
            personas = []

            if cond == "layer":
                if not args.personas:
                    print("ERROR: --personas required for 'layer' condition")
                    sys.exit(1)
                print(f"Loading layer personas from {args.personas}...")
                personas = load_layer_personas(args.personas, args.specs)
                print(f"  Loaded {len(personas)} layer personas")
                if not personas:
                    print("ERROR: No layer personas loaded")
                    sys.exit(1)

            elif cond == "legacy":
                if not args.personas:
                    print("ERROR: --personas required for 'legacy' condition")
                    sys.exit(1)
                print(f"Loading legacy personas from {args.personas}...")
                personas = load_legacy_personas(args.personas)
                print(f"  Loaded {len(personas)} legacy personas")

            elif cond == "none":
                # No personas needed
                pass

            result_path = run_condition(
                condition=cond,
                domain=args.domain,
                personas=personas,
                output_dir=args.output,
                num_tasks=args.num_tasks,
                task_ids=args.task_ids,
                task_split=args.task_split,
                task_persona_filter=args.task_persona_filter,
                num_trials=args.num_trials,
                max_steps=args.max_steps,
                seed=args.seed,
                llm_agent=args.llm_agent,
                llm_user=args.llm_user,
                agent_type=args.agent_type,
                max_concurrency=args.max_concurrency,
            )
            result_files[cond] = result_path

    # Run evaluation if we have results
    if len(result_files) >= 1:
        eval_output = os.path.join(args.output, "comparison.json")
        print(f"\nRunning comparison evaluation...")
        run_evaluation(result_files, eval_output)
    else:
        print("No result files available for evaluation")


if __name__ == "__main__":
    main()
