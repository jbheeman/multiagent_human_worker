"""Layer-based persona generation pipeline.

Replaces the old Schwartz-based pipeline with the 3-layer persona system.

Modes:
- FULLY_GROUNDED: Extract L0 + L1 from Reddit, sample L2
- PARTIALLY_GROUNDED: Extract L0 from Reddit, sample L1 + L2
- SYNTHETIC: Sample all layers (no Reddit data needed)

Steps (grounded modes):
1. Extract layer codes from Reddit user data (LLM call)
2. Sample remaining layers randomly
3. Compile IF/THEN behavioral rules (LLM call)
4. Conformance check + retry loop (LLM call)
5. Assemble system prompt and write YAML output
"""

import json
import os
import time
from dataclasses import dataclass
from functools import wraps
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

from persona.assembler import assemble_system_prompt
from persona.compiler import compile_rules
from persona.conformance import check_conformance, extract_critique
from persona.extractor import extract_layers
from persona.registry import LayerRegistry
from persona.sampler import PersonaSampler
from persona.schema import GenerationMode, PersonaConfig

# LLM backend: use smolagents if available, otherwise fallback to openai client
try:
    from smolagents.models import OpenAIServerModel
    _HAS_SMOLAGENTS = True
except ImportError:
    _HAS_SMOLAGENTS = False


# ── LLM configuration ──────────────────────────────────────────────────────

class _OpenAIFallbackModel:
    """Minimal OpenAI-compatible chat model for when smolagents is not installed."""

    def __init__(self, model_id: str, api_base: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_id = model_id

    def __call__(self, messages: list[dict]) -> "types.SimpleNamespace":
        import types
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )
        result = types.SimpleNamespace()
        result.content = response.choices[0].message.content
        return result


def make_llm_model(
    model_id: str = "qwen3",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Create an LLM model for pipeline calls.

    Uses smolagents.OpenAIServerModel if available, falls back to openai client.
    """
    base = api_base or os.getenv("NAUTILUS_API_BASE") or "https://ellm.nrp-nautilus.io/v1"
    key = api_key or os.getenv("NAUTILUS_API_KEY") or ""
    if _HAS_SMOLAGENTS:
        return OpenAIServerModel(model_id=model_id, api_base=base, api_key=key)
    return _OpenAIFallbackModel(model_id=model_id, api_base=base, api_key=key)


# ── Retry utility ──────────────────────────────────────────────────────────

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
                    is_retryable = (
                        "timeout" in error_str
                        or "timed out" in error_str
                        or "connection" in error_str
                    )
                    if attempt == max_retries - 1:
                        raise
                    if is_retryable:
                        print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        raise
            return None
        return wrapper
    return decorator


# ── Data loading ───────────────────────────────────────────────────────────

@dataclass
class RedditUserData:
    """Input data for a single Reddit user."""
    user_id: str
    subreddit: str
    posts: list[str]
    anchor_demographics: dict  # {"age": 28, "gender": "male", "occupation": "...", "location": "..."}


def load_reddit_dataset(path: str) -> list[RedditUserData]:
    """Load Reddit user data from JSONL file."""
    examples = []
    with open(path, "r") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue
            try:
                data = json.loads(row)
                examples.append(RedditUserData(
                    user_id=data["user_id"],
                    subreddit=data["subreddit"],
                    posts=data["posts"],
                    anchor_demographics=data.get("anchor_demographics", {}),
                ))
            except Exception as e:
                print(f"Error loading row: {e}")
                continue
    return examples


def demographics_to_string(demographics: dict) -> str:
    """Convert demographics dict to readable string."""
    if isinstance(demographics, str):
        return demographics
    parts = []
    if "age" in demographics:
        parts.append(str(demographics["age"]))
    if "gender" in demographics:
        g = demographics["gender"]
        parts.append("M" if g.lower().startswith("m") else "F" if g.lower().startswith("f") else g)
    if "occupation" in demographics:
        parts.append(demographics["occupation"])
    if "location" in demographics:
        parts.append(demographics["location"])
    return ", ".join(parts) if parts else "Unknown"


# ── Pipeline core ──────────────────────────────────────────────────────────

class PersonaPipeline:
    """Generates personas using the 3-layer system.

    Args:
        specs_dir: Path to the specs/ directory with layer spec markdown files.
        model_id: LLM model to use for extraction/compilation/conformance.
        api_base: LLM API base URL.
        api_key: LLM API key (defaults to NAUTILUS_API_KEY env var).
        seed: Random seed for sampler reproducibility.
        max_conformance_retries: Max attempts for compile → conform loop.
    """

    def __init__(
        self,
        specs_dir: str,
        model_id: str = "qwen3",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
        max_conformance_retries: int = 3,
        tier: Optional["Valence"] = None,
    ):
        from persona.sampler import PersonaSampler
        self.registry = LayerRegistry(specs_dir)
        self.sampler = PersonaSampler(self.registry, seed=seed, tier=tier)
        self.model = make_llm_model(model_id, api_base, api_key)
        self.max_conformance_retries = max_conformance_retries

        # Verify registry parsed everything
        stats = self.registry.stats()
        mismatches = {k: v for k, v in stats.items() if not v["ok"]}
        if mismatches:
            print(f"WARNING: Registry parsing mismatches: {mismatches}")

    @retry_with_backoff(max_retries=3)
    def _llm_call(self, prompt: str) -> str:
        """Make an LLM call with retry."""
        response = self.model([{"role": "user", "content": prompt}])
        return response.content or ""

    def generate_grounded(
        self,
        user_data: RedditUserData,
        mode: GenerationMode,
    ) -> PersonaConfig:
        """Generate a persona grounded in Reddit data.

        Args:
            user_data: Reddit user data (posts, demographics, subreddit).
            mode: FULLY_GROUNDED or PARTIALLY_GROUNDED.

        Returns:
            PersonaConfig with all 12 dimensions + compiled rules.
        """
        demographics_str = demographics_to_string(user_data.anchor_demographics)

        # Step 1: Extract layer codes from Reddit data
        print(f"    Extracting layers ({mode.value})...", end=" ", flush=True)
        extracted_codes, extraction_info = extract_layers(
            registry=self.registry,
            demographics=demographics_str,
            subreddit=user_data.subreddit,
            posts=user_data.posts,
            mode=mode,
            llm_call=self._llm_call,
        )
        print(f"got {len(extracted_codes)} codes")

        if not extracted_codes:
            print(f"    WARNING: Extraction returned no codes, falling back to synthetic")
            return self.generate_synthetic(demographics=demographics_str)

        # Step 2: Sample remaining layers + build PersonaConfig
        config = self.sampler.sample_persona(
            mode=mode,
            fixed_codes=extracted_codes,
            demographics=demographics_str,
            source_user_id=user_data.user_id,
            source_subreddit=user_data.subreddit,
            extraction_justification=extraction_info,
        )

        # Step 3: Compile rules + conformance check
        config = self._compile_with_conformance(config)
        return config

    def generate_synthetic(
        self,
        demographics: str = "",
    ) -> PersonaConfig:
        """Generate a fully synthetic persona (no Reddit data).

        Returns:
            PersonaConfig with all 12 dimensions + compiled rules.
        """
        config = self.sampler.sample_persona(
            mode=GenerationMode.SYNTHETIC,
            demographics=demographics,
        )
        config = self._compile_with_conformance(config)
        return config

    def _compile_with_conformance(self, config: PersonaConfig) -> PersonaConfig:
        """Compile rules and run conformance check with retry loop.

        Up to max_conformance_retries attempts:
        1. Compile IF/THEN rules
        2. Check conformance
        3. If rejected, compile again with critique
        """
        critique = None

        for attempt in range(self.max_conformance_retries):
            # Compile rules
            print(f"    Compiling rules (attempt {attempt + 1})...", end=" ", flush=True)
            config = compile_rules(config, self._llm_call)

            if not config.state_transition_rules:
                print("WARNING: compiler returned no rules")
                continue

            # Check conformance
            conformance_result = check_conformance(config, self._llm_call)
            status = conformance_result.get("OVERALL_STATUS", "UNKNOWN")
            print(f"conformance: {status}")

            if status == "APPROVE":
                return config

            # Build critique for retry
            critique = extract_critique(conformance_result)
            print(f"    Critique: {critique[:200]}")

            # TODO: feed critique back into compiler on next iteration
            # For now the compiler doesn't accept critique yet — the retry
            # just re-runs with the same config hoping for different output.
            # A more robust approach would modify compile_rules to accept critique.

        print(f"    Edge case: failed conformance after {self.max_conformance_retries} attempts")
        return config

    def generate_and_save(
        self,
        config: PersonaConfig,
        output_dir: str,
        filename: Optional[str] = None,
    ) -> str:
        """Save a persona to YAML file.

        Args:
            config: Completed PersonaConfig.
            output_dir: Directory to write YAML to.
            filename: Optional filename (defaults to code_string or user_id).

        Returns:
            Path to the written YAML file.
        """
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            if config.source_user_id:
                filename = f"{config.source_user_id}.yaml"
            else:
                # Use code string with slashes replaced
                filename = config.code_string.replace("/", "_") + ".yaml"

        output_path = os.path.join(output_dir, filename)

        yaml_dict = config.to_yaml_dict()
        with open(output_path, "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return output_path

    def generate_assembled_prompt(
        self,
        config: PersonaConfig,
        scenario_instructions: str = "",
    ) -> str:
        """Generate the full assembled system prompt for the simulator.

        This combines vignettes, anti-imitation rules, ground rules,
        compiled IF/THEN rules, and scenario instructions.
        """
        return assemble_system_prompt(self.registry, config, scenario_instructions)


# ── CLI entry point ────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_dir: str,
    specs_dir: str,
    mode: GenerationMode = GenerationMode.FULLY_GROUNDED,
    num_synthetic: int = 0,
    seed: Optional[int] = None,
    log_path: Optional[str] = None,
    tier: Optional[str] = None,
    model_id: str = "qwen3",
):
    """Run the full persona generation pipeline.

    Args:
        input_path: Path to Reddit JSONL data.
        output_dir: Directory for output YAML files.
        specs_dir: Path to specs/ directory.
        mode: Generation mode for Reddit-grounded personas.
        num_synthetic: Number of additional synthetic personas to generate.
        seed: Random seed for reproducibility.
        log_path: Optional JSONL log file path.
        tier: Optional valence name (positive, neutral, negative).
        model_id: LLM model ID to use for generation.
    """
    from persona.valence import Valence
    tier_enum = Valence(tier) if tier else None
    pipeline = PersonaPipeline(specs_dir=specs_dir, seed=seed, tier=tier_enum, model_id=model_id)

    # Load Reddit data
    dataset = load_reddit_dataset(input_path)
    print(f"Loaded {len(dataset)} users from {input_path}")
    print(f"Mode: {mode.value}")
    print(f"Synthetic: {num_synthetic}")
    print()

    successful = 0
    failed = 0
    edge_cases = []
    log_file = open(log_path, "a") if log_path else None

    # Summary of intent
    print(f"Pipeline Mode: {mode.value}")
    if mode != GenerationMode.SYNTHETIC:
        print(f"Processing {len(dataset)} Reddit users...")
    if num_synthetic > 0:
        print(f"Generating {num_synthetic} additional synthetic personas...")
    print(f"Valence Tier: {tier if tier else 'none'}")
    print()

    successful = 0
    failed = 0
    edge_cases = []
    log_file = open(log_path, "a") if log_path else None

    try:
        # Generate grounded personas from Reddit data (only if not in purely synthetic mode)
        if mode != GenerationMode.SYNTHETIC:
            for i, user_data in enumerate(dataset):
                try:
                    print(f"[{i + 1}/{len(dataset)}] {user_data.user_id} (r/{user_data.subreddit})")
                    config = pipeline.generate_grounded(user_data, mode)
                    yaml_path = pipeline.generate_and_save(config, output_dir)
                    print(f"    Saved: {yaml_path}")
                    # ... log writing ...
                    successful += 1
                except Exception as e:
                    failed += 1
                    print(f"    FAILED: {e}")
                    edge_cases.append({"user_id": user_data.user_id, "error": str(e)})
                    continue

        # Generate synthetic personas
        for i in range(num_synthetic):
            try:
                label = f"synthetic_{i}"
                print(f"[Synthetic {i + 1}/{num_synthetic}]")
                config = pipeline.generate_synthetic()
                yaml_path = pipeline.generate_and_save(config, output_dir)
                print(f"    Saved: {yaml_path}")
                # ... log writing ...
                successful += 1
            except Exception as e:
                failed += 1
                print(f"    FAILED: {e}")
                edge_cases.append({"user_id": f"synthetic_{i}", "error": str(e)})
                continue

    finally:
        if log_file:
            log_file.close()

    # Summary
    total = len(dataset) + num_synthetic
    print(f"\n{'=' * 50}")
    print(f"Pipeline complete!")
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {failed}/{total}")
    if edge_cases:
        print(f"Edge cases: {len(edge_cases)}")
        for ec in edge_cases:
            print(f"  - {ec['user_id']}: {ec['error'][:100]}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer-based persona generation pipeline")
    parser.add_argument("--input", default="reddit/personasforpaper.jsonl",
                        help="Path to Reddit JSONL data")
    parser.add_argument("--output", default="reddit/eval_personas",
                        help="Output directory for YAML files")
    parser.add_argument("--specs", default="specs",
                        help="Path to specs/ directory")
    parser.add_argument("--mode", default="fully_grounded",
                        choices=["fully_grounded", "partially_grounded", "synthetic"],
                        help="Generation mode for Reddit-grounded personas")
    parser.add_argument("--num-synthetic", type=int, default=0,
                        help="Number of additional synthetic personas")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--log", default="pipeline_log.jsonl",
                        help="JSONL log file path")
    parser.add_argument("--tier", choices=["positive", "neutral", "negative"],
                        help="Valence bias for sampled layers")
    parser.add_argument("--model", default="qwen3",
                        help="LLM model ID to use for generation")

    args = parser.parse_args()

    mode_map = {
        "fully_grounded": GenerationMode.FULLY_GROUNDED,
        "partially_grounded": GenerationMode.PARTIALLY_GROUNDED,
        "synthetic": GenerationMode.SYNTHETIC,
    }

    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        specs_dir=args.specs,
        mode=mode_map[args.mode],
        num_synthetic=args.num_synthetic,
        seed=args.seed,
        log_path=args.log,
        tier=args.tier,
        model_id=args.model,
    )
