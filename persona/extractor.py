"""Reddit data to layer code extraction.

Extracts layer codes from Reddit user data (posts, demographics, subreddit).
Supports two modes:
- FULLY_GROUNDED: Extract Layer 0 + Layer 1 codes
- PARTIALLY_GROUNDED: Extract Layer 0 codes only

Layer 2 is never extracted — it's sampled per scenario.
"""

import json
import re
from typing import Callable

from persona.registry import LayerRegistry
from persona.schema import DIMENSIONS, GenerationMode, Layer


EXTRACTION_PROMPT = """You are an expert behavioral profiler mapping real human data to a structured persona framework.

=== REAL PERSON DATA ===
Demographics: {demographics}
Community context: r/{subreddit}

Their actual posts (from this subreddit):
{posts_formatted}

=== YOUR TASK ===
Based on this person's actual writing, demographics, and community context, select the SINGLE BEST option from each dimension below. You are profiling how this person would likely behave in a customer service interaction, based on who they are.

PHASE 1 — ANALYSIS (write 2-3 sentences):
What kind of person is this based on their writing? What are their communication patterns? What do their posts reveal about their typical interaction style, trust level, and emotional tendencies?

PHASE 2 — LAYER SELECTIONS:
For each dimension, pick the single best code and give a 1-sentence justification.

{dimensions_section}

=== OUTPUT FORMAT (valid JSON only, no other text) ===
{{
  "analysis": "2-3 sentence behavioral summary",
  "selections": {{
{selections_template}
  }}
}}"""


def _format_posts(posts: list[str], max_posts: int = 5) -> str:
    """Format Reddit posts for the extraction prompt."""
    formatted = []
    for i, post in enumerate(posts[:max_posts], 1):
        # Truncate very long posts
        text = post[:500] + "..." if len(post) > 500 else post
        formatted.append(f"Post {i}:\n{text}")
    return "\n\n".join(formatted)


def _build_dimensions_section(
    registry: LayerRegistry,
    mode: GenerationMode,
) -> str:
    """Build the dimension options section for the extraction prompt."""
    lines = []

    # Determine which layers to extract
    if mode == GenerationMode.FULLY_GROUNDED:
        extract_layers = {Layer.L0, Layer.L1}
    elif mode == GenerationMode.PARTIALLY_GROUNDED:
        extract_layers = {Layer.L0}
    else:
        return ""  # Synthetic mode doesn't extract

    layer_names = {
        Layer.L0: "LAYER 0: DEEP ORIENTATIONS",
        Layer.L1: "LAYER 1: STABLE BACKGROUND",
    }

    for layer in [Layer.L0, Layer.L1]:
        if layer not in extract_layers:
            continue
        lines.append(f"=== {layer_names[layer]} ===")
        lines.append("")

        for dim_name, (dim_layer, prefix, _) in DIMENSIONS.items():
            if dim_layer != layer:
                continue
            # Build human-readable dimension name
            readable_name = dim_name.replace("_", " ").title()
            options_summary = registry.get_option_summary(dim_name)
            lines.append(f"{readable_name}:")
            lines.append(f"Options: {options_summary}")
            lines.append("")

    return "\n".join(lines)


def _build_selections_template(
    registry: LayerRegistry,
    mode: GenerationMode,
) -> str:
    """Build the JSON selections template for the expected output format."""
    lines = []

    if mode == GenerationMode.FULLY_GROUNDED:
        extract_layers = {Layer.L0, Layer.L1}
    elif mode == GenerationMode.PARTIALLY_GROUNDED:
        extract_layers = {Layer.L0}
    else:
        return ""

    for dim_name, (dim_layer, prefix, _) in DIMENSIONS.items():
        if dim_layer not in extract_layers:
            continue
        example_code = f"{prefix}X"
        lines.append(
            f'    "{dim_name}": {{"code": "{example_code}", "justification": "..."}}'
        )

    return ",\n".join(lines)


def _build_extraction_prompt(
    registry: LayerRegistry,
    demographics: str,
    subreddit: str,
    posts: list[str],
    mode: GenerationMode,
) -> str:
    """Build the complete extraction prompt."""
    return EXTRACTION_PROMPT.format(
        demographics=demographics,
        subreddit=subreddit,
        posts_formatted=_format_posts(posts),
        dimensions_section=_build_dimensions_section(registry, mode),
        selections_template=_build_selections_template(registry, mode),
    )


def _parse_extraction_response(response_text: str) -> dict:
    """Parse the extraction JSON response."""
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _validate_extracted_codes(
    raw_selections: dict,
    registry: LayerRegistry,
    mode: GenerationMode,
) -> tuple[dict[str, str], dict[str, str]]:
    """Validate extracted codes and return (valid_codes, justifications).

    Returns:
        Tuple of (codes dict, justifications dict).
        Invalid codes are dropped (caller should fall back to sampling).
    """
    codes = {}
    justifications = {}

    if mode == GenerationMode.FULLY_GROUNDED:
        extract_layers = {Layer.L0, Layer.L1}
    elif mode == GenerationMode.PARTIALLY_GROUNDED:
        extract_layers = {Layer.L0}
    else:
        return {}, {}

    for dim_name, (dim_layer, prefix, _) in DIMENSIONS.items():
        if dim_layer not in extract_layers:
            continue

        sel = raw_selections.get(dim_name, {})
        if not isinstance(sel, dict):
            continue

        code = sel.get("code", "")
        justification = sel.get("justification", "")

        # Validate the code exists in the registry
        valid_codes = registry.get_all_codes(dim_name)
        if code in valid_codes:
            codes[dim_name] = code
            justifications[dim_name] = justification

    return codes, justifications


def extract_layers(
    registry: LayerRegistry,
    demographics: str,
    subreddit: str,
    posts: list[str],
    mode: GenerationMode,
    llm_call: Callable[[str], str],
) -> tuple[dict[str, str], dict]:
    """Extract layer codes from Reddit user data.

    Args:
        registry: LayerRegistry for validation.
        demographics: Demographic string (e.g., "28M developer, SF").
        subreddit: Source subreddit (e.g., "learnprogramming").
        posts: List of user's posts from this subreddit.
        mode: FULLY_GROUNDED or PARTIALLY_GROUNDED.
        llm_call: LLM call function.

    Returns:
        Tuple of (codes_dict, extraction_info).
        codes_dict: dimension → code for extracted layers.
        extraction_info: {"analysis": str, "justifications": dict, "raw_response": dict}
    """
    if mode == GenerationMode.SYNTHETIC:
        return {}, {"analysis": "Synthetic mode: no extraction needed", "justifications": {}, "raw_response": {}}

    prompt = _build_extraction_prompt(registry, demographics, subreddit, posts, mode)
    response = llm_call(prompt)
    parsed = _parse_extraction_response(response)

    if not parsed:
        return {}, {"analysis": "", "justifications": {}, "raw_response": {}}

    raw_selections = parsed.get("selections", {})
    codes, justifications = _validate_extracted_codes(raw_selections, registry, mode)

    return codes, {
        "analysis": parsed.get("analysis", ""),
        "justifications": justifications,
        "raw_response": parsed,
    }


def get_extraction_prompt(
    registry: LayerRegistry,
    demographics: str,
    subreddit: str,
    posts: list[str],
    mode: GenerationMode,
) -> str:
    """Expose the extraction prompt for debugging."""
    return _build_extraction_prompt(registry, demographics, subreddit, posts, mode)
