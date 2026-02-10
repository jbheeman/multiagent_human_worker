import json
from tau_bench.envs.retail.tools import ALL_TOOLS

# Extract the JSON schema for every tool
tools_schema = [tool.get_info() for tool in ALL_TOOLS]

# Convert to a readable string to paste into your "Persona Translation" prompt
tools_context_str = json.dumps(tools_schema, indent=2)

print(tools_context_str)