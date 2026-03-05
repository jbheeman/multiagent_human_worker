"""Layer registry: parses spec markdown files and provides vignette lookups.

The spec files (specs/layer{0,1,2}_persona_spec.md) are the single source of
truth. This module parses them at init time and builds a lookup table from
dimension/code to vignette text. It also extracts the system prompt frames.
"""

import os
import re
from dataclasses import dataclass, field

from persona.schema import DIMENSIONS, Layer, LayerSelection


@dataclass
class DimensionInfo:
    """Metadata + all options for one dimension."""
    name: str                          # e.g., "situation_construal"
    layer: Layer
    code_prefix: str                   # e.g., "1" or "C"
    options: dict[str, dict] = field(default_factory=dict)
    # options: code -> {"label": str, "vignette": str}


class LayerRegistry:
    """Loads and indexes all layer vignettes from the spec markdown files.

    Usage:
        registry = LayerRegistry("specs/")
        selection = registry.get("situation_construal", "1A")
        frame = registry.get_system_prompt_frame(Layer.L0)
    """

    def __init__(self, specs_dir: str):
        self.specs_dir = specs_dir
        self.dimensions: dict[str, DimensionInfo] = {}
        self.system_prompt_frames: dict[Layer, str] = {}

        # Initialize dimension metadata from schema
        for dim_name, (layer, prefix, _num) in DIMENSIONS.items():
            self.dimensions[dim_name] = DimensionInfo(
                name=dim_name, layer=layer, code_prefix=prefix
            )

        # Parse each spec file
        self._parse_spec(
            os.path.join(specs_dir, "layer0_persona_spec.md"), Layer.L0
        )
        self._parse_spec(
            os.path.join(specs_dir, "layer1_persona_spec.md"), Layer.L1
        )
        self._parse_spec(
            os.path.join(specs_dir, "layer2_persona_spec.md"), Layer.L2
        )

    def _parse_spec(self, path: str, layer: Layer) -> None:
        """Parse a spec markdown file, extracting system prompt frame and vignettes."""
        with open(path, "r") as f:
            content = f.read()

        # Extract system prompt frame: the first fenced code block in the file
        # (appears in the "System Prompt" / "System Prompt Addition" section)
        self._extract_system_prompt_frame(content, layer)

        # Extract answer vignettes: **CODE — Label** followed by ```...```
        self._extract_vignettes(content, layer)

    def _extract_system_prompt_frame(self, content: str, layer: Layer) -> None:
        """Extract the system prompt frame template from the spec content.

        The spec files use double braces {{VAR}} for readability in markdown.
        We convert to single braces {VAR} for Python .format().

        For Layer 0, the spec frame includes a GROUND RULES section that we
        strip out since the assembler provides its own (improved) version.
        """
        code_blocks = re.findall(r"```\n(.*?)```", content, re.DOTALL)
        for block in code_blocks:
            if "{{" in block:
                # Convert {{PLACEHOLDER}} to {PLACEHOLDER} for .format()
                frame = re.sub(r"\{\{(\w+)\}\}", r"{\1}", block.strip())

                # For Layer 0: strip the embedded GROUND RULES section.
                # The assembler provides a separate, improved ground rules block.
                if layer == Layer.L0:
                    ground_rules_start = frame.find("--- GROUND RULES ---")
                    if ground_rules_start != -1:
                        frame = frame[:ground_rules_start].rstrip()

                self.system_prompt_frames[layer] = frame
                break

    def _extract_vignettes(self, content: str, layer: Layer) -> None:
        """Extract all **CODE — Label** + code block pairs from the spec.

        Pattern: **CODE — Label** followed by a fenced code block.
        The code prefix determines which dimension this belongs to.
        """
        # Match patterns like **1A — Rights exercise** or **C1 — Standard American English, fluent and flexible**
        pattern = r"\*\*(\w+)\s*(?:—|--|-)\s*(.+?)\*\*\s*\n```\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)

        for code, label, vignette in matches:
            code = code.strip()
            label = label.strip()
            vignette = vignette.strip()

            # Determine which dimension this code belongs to
            dim_name = self._code_to_dimension(code, layer)
            if dim_name is None:
                continue

            self.dimensions[dim_name].options[code] = {
                "label": label,
                "vignette": vignette,
            }

    def _code_to_dimension(self, code: str, layer: Layer) -> str | None:
        """Map a code like '1A' or 'C3' to its dimension name."""
        for dim_name, (dim_layer, prefix, _) in DIMENSIONS.items():
            if dim_layer != layer:
                continue
            # Check if the code starts with this dimension's prefix
            if prefix.isdigit():
                # Numeric prefix: code is like "1A", "2B"
                if code.startswith(prefix) and len(code) >= 2 and code[1:].isalpha():
                    return dim_name
            else:
                # Letter prefix: code is like "C1", "C12"
                if code.startswith(prefix) and len(code) >= 2 and code[1:].isdigit():
                    return dim_name
        return None

    def get(self, dimension: str, code: str) -> LayerSelection:
        """Look up a vignette by dimension name and code.

        Args:
            dimension: e.g., "situation_construal"
            code: e.g., "1A"

        Returns:
            LayerSelection with dimension, code, label, and vignette.

        Raises:
            KeyError: If the dimension or code is not found.
        """
        if dimension not in self.dimensions:
            raise KeyError(f"Unknown dimension: {dimension}")
        dim = self.dimensions[dimension]
        if code not in dim.options:
            raise KeyError(
                f"Unknown code '{code}' for dimension '{dimension}'. "
                f"Valid codes: {list(dim.options.keys())}"
            )
        opt = dim.options[code]
        return LayerSelection(
            dimension=dimension,
            code=code,
            label=opt["label"],
            vignette=opt["vignette"],
        )

    def get_system_prompt_frame(self, layer: Layer) -> str:
        """Return the system prompt template for a given layer.

        The template contains {{PLACEHOLDER}} tags to be filled with vignettes.
        """
        if layer not in self.system_prompt_frames:
            raise KeyError(f"No system prompt frame found for {layer}")
        return self.system_prompt_frames[layer]

    def get_all_codes(self, dimension: str) -> list[str]:
        """Return all valid codes for a dimension, in order."""
        if dimension not in self.dimensions:
            raise KeyError(f"Unknown dimension: {dimension}")
        return sorted(
            self.dimensions[dimension].options.keys(),
            key=lambda c: self._sort_key(c),
        )

    def get_all_options(self, dimension: str) -> list[dict]:
        """Return all options for a dimension as list of {code, label, vignette}."""
        codes = self.get_all_codes(dimension)
        return [
            {"code": c, **self.dimensions[dimension].options[c]}
            for c in codes
        ]

    def get_option_summary(self, dimension: str) -> str:
        """Return compact summary for extraction prompts: '1A=Rights exercise, 1B=Asking a favor, ...'"""
        opts = self.get_all_options(dimension)
        return ", ".join(f"{o['code']}={o['label']}" for o in opts)

    def build_persona_config(self, codes: dict[str, str], **kwargs) -> "PersonaConfig":
        """Build a PersonaConfig from a dict of dimension→code mappings.

        Args:
            codes: dict like {"situation_construal": "1A", "relational_stance": "2E", ...}
            **kwargs: Additional PersonaConfig fields (mode, demographics, source_user_id, etc.)
        """
        from persona.schema import PersonaConfig

        config_kwargs = {}
        for dim_name, code in codes.items():
            selection = self.get(dim_name, code)
            config_kwargs[dim_name] = selection
        config_kwargs.update(kwargs)
        return PersonaConfig(**config_kwargs)

    def validate_codes(self, codes: dict[str, str]) -> list[str]:
        """Validate a dict of dimension→code, returning list of errors (empty if valid)."""
        errors = []
        for dim_name in DIMENSIONS:
            if dim_name not in codes:
                errors.append(f"Missing dimension: {dim_name}")
                continue
            code = codes[dim_name]
            valid = self.get_all_codes(dim_name)
            if code not in valid:
                errors.append(
                    f"Invalid code '{code}' for {dim_name}. Valid: {valid}"
                )
        return errors

    @staticmethod
    def _sort_key(code: str) -> tuple:
        """Sort codes naturally: 1A < 1B < ... < 1J, C1 < C2 < ... < C12."""
        # Split into prefix + suffix
        if code[0].isdigit():
            # Numeric prefix: "1A" -> (1, 'A')
            return (int(code[0]), code[1:])
        else:
            # Letter prefix: "C12" -> ('C', 12)
            return (code[0], int(code[1:]))

    def stats(self) -> dict:
        """Return parsing statistics for verification."""
        result = {}
        for dim_name, dim_info in self.dimensions.items():
            expected = DIMENSIONS[dim_name][2]  # num_options
            actual = len(dim_info.options)
            result[dim_name] = {
                "expected": expected,
                "actual": actual,
                "ok": expected == actual,
                "codes": sorted(dim_info.options.keys(), key=self._sort_key),
            }
        return result
