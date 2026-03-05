"""Random layer sampling with configurable weights and plausibility filtering.

Supports three generation modes:
- FULLY_GROUNDED: L0+L1 from Reddit, only L2 sampled
- PARTIALLY_GROUNDED: L0 from Reddit, L1+L2 sampled
- SYNTHETIC: All layers sampled

Sampling weights can be biased toward positive (easy), negative (difficult),
or neutral personas using the canonical valence mapping from persona.valence.
"""

import random
from typing import Optional

from persona.registry import LayerRegistry
from persona.schema import (
    DIMENSIONS,
    GenerationMode,
    Layer,
    PersonaConfig,
)
from persona.valence import CODE_VALENCE, Valence

# Implausible combinations to filter out.
IMPLAUSIBLE_PAIRS: list[tuple[str, str, str, str]] = [
    ("relational_stance", "2G", "agency", "3E"),
]


# Sampling weight table: maps (tier, code_valence) → sampling weight.
# Rows = target tier, Columns = code's valence.
# Positive tier strongly favors positive/slightly_positive codes.
# Negative tier strongly favors negative/slightly_negative codes.
# Neutral tier favors the middle, penalizes extremes.
_TIER_WEIGHTS: dict[Valence, dict[Valence, float]] = {
    Valence.POSITIVE: {
        Valence.POSITIVE: 5.0,
        Valence.SLIGHTLY_POSITIVE: 3.0,
        Valence.NEUTRAL: 1.0,
        Valence.SLIGHTLY_NEGATIVE: 0.3,
        Valence.NEGATIVE: 0.1,
    },
    Valence.NEGATIVE: {
        Valence.POSITIVE: 0.1,
        Valence.SLIGHTLY_POSITIVE: 0.3,
        Valence.NEUTRAL: 1.0,
        Valence.SLIGHTLY_NEGATIVE: 3.0,
        Valence.NEGATIVE: 5.0,
    },
    Valence.NEUTRAL: {
        Valence.POSITIVE: 0.5,
        Valence.SLIGHTLY_POSITIVE: 1.0,
        Valence.NEUTRAL: 5.0,
        Valence.SLIGHTLY_NEGATIVE: 1.0,
        Valence.NEGATIVE: 0.5,
    },
}


class PersonaSampler:
    """Samples random layer codes with configurable weights and valence tiers.

    Args:
        registry: The LayerRegistry to sample from.
        seed: Optional random seed for reproducibility.
        tier: Optional Valence to bias the sampling (POSITIVE, NEUTRAL, or NEGATIVE).
    """

    def __init__(
        self,
        registry: LayerRegistry,
        seed: Optional[int] = None,
        tier: Optional[Valence] = None,
    ):
        self.registry = registry
        self.rng = random.Random(seed)
        self.tier = tier

    def _weighted_choice(self, dimension: str) -> str:
        """Pick a random code, biased by the selected tier."""
        codes = self.registry.get_all_codes(dimension)

        if self.tier is None:
            return self.rng.choice(codes)

        tier_weights = _TIER_WEIGHTS[self.tier]
        weights = [
            tier_weights.get(
                CODE_VALENCE.get(code, Valence.NEUTRAL), 1.0
            )
            for code in codes
        ]
        return self.rng.choices(codes, weights=weights, k=1)[0]

    def _is_plausible(self, codes: dict[str, str]) -> bool:
        """Check if a combination passes plausibility filters."""
        for dim1, code1, dim2, code2 in IMPLAUSIBLE_PAIRS:
            if codes.get(dim1) == code1 and codes.get(dim2) == code2:
                return False
        return True

    def sample_layer(self, layer: Layer) -> dict[str, str]:
        """Sample all dimensions for a single layer.

        Returns dict of dimension → code.
        """
        codes = {}
        for dim_name, (dim_layer, _, _) in DIMENSIONS.items():
            if dim_layer == layer:
                codes[dim_name] = self._weighted_choice(dim_name)
        return codes

    def sample_full(self, max_attempts: int = 50) -> dict[str, str]:
        """Sample all 12 dimensions, respecting plausibility constraints.

        Retries if the combination is implausible (up to max_attempts).
        """
        for _ in range(max_attempts):
            codes = {}
            for dim_name in DIMENSIONS:
                codes[dim_name] = self._weighted_choice(dim_name)
            if self._is_plausible(codes):
                return codes
        # If we can't find a plausible combo, return the last attempt
        return codes

    def sample_persona(
        self,
        mode: GenerationMode = GenerationMode.SYNTHETIC,
        fixed_codes: Optional[dict[str, str]] = None,
        demographics: str = "",
        source_user_id: Optional[str] = None,
        source_subreddit: Optional[str] = None,
        extraction_justification: Optional[dict] = None,
    ) -> PersonaConfig:
        """Sample a complete PersonaConfig.

        Args:
            mode: Generation mode (determines which layers are sampled vs. fixed).
            fixed_codes: Pre-determined codes (from Reddit extraction or manual selection).
                         For FULLY_GROUNDED: must include L0 + L1 codes.
                         For PARTIALLY_GROUNDED: must include L0 codes.
                         For SYNTHETIC: ignored (all sampled).
            demographics: Demographic string for the persona.
            source_user_id: Reddit user ID (if grounded).
            source_subreddit: Source subreddit (if grounded).
            extraction_justification: LLM extraction reasoning (if grounded).

        Returns:
            PersonaConfig with all 12 dimensions filled.
        """
        fixed_codes = fixed_codes or {}
        all_codes = {}

        if mode == GenerationMode.SYNTHETIC:
            all_codes = self.sample_full()

        elif mode == GenerationMode.PARTIALLY_GROUNDED:
            # L0 from fixed_codes, L1 + L2 sampled
            for dim_name, (layer, _, _) in DIMENSIONS.items():
                if layer == Layer.L0:
                    if dim_name not in fixed_codes:
                        raise ValueError(
                            f"PARTIALLY_GROUNDED requires L0 code for {dim_name}"
                        )
                    all_codes[dim_name] = fixed_codes[dim_name]
                else:
                    all_codes[dim_name] = self._weighted_choice(dim_name)

        elif mode == GenerationMode.FULLY_GROUNDED:
            # L0 + L1 from fixed_codes, L2 sampled
            for dim_name, (layer, _, _) in DIMENSIONS.items():
                if layer in (Layer.L0, Layer.L1):
                    if dim_name not in fixed_codes:
                        raise ValueError(
                            f"FULLY_GROUNDED requires L0+L1 code for {dim_name}"
                        )
                    all_codes[dim_name] = fixed_codes[dim_name]
                else:
                    all_codes[dim_name] = self._weighted_choice(dim_name)

        # Plausibility check (resample clashing dimensions if needed)
        attempts = 0
        while not self._is_plausible(all_codes) and attempts < 20:
            # Only resample the sampled dimensions, not the fixed ones
            for dim_name in DIMENSIONS:
                if dim_name not in fixed_codes:
                    all_codes[dim_name] = self._weighted_choice(dim_name)
            attempts += 1

        return self.registry.build_persona_config(
            all_codes,
            mode=mode,
            demographics=demographics,
            source_user_id=source_user_id,
            source_subreddit=source_subreddit,
            extraction_justification=extraction_justification,
        )
