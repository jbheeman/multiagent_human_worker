from typing import List
from smolagents import OpenAIServerModel
from knowledge_base.models import UserProducts
import json
import re

class GoalGenerator:
    """
    Generates diverse shopping goals based on a user's purchase history.
    """
    def __init__(self, model: OpenAIServerModel):
        self.model = model

    def generate_goals(self, user_products: UserProducts, persona: str, n: int = 33) -> List[str]:
        """
        Generates n distinct, likely shopping goals for the user.
        """
        product_list_str = "\n".join([f"- {p.title} (${p.price})" for p in user_products.products.values() if p.title])

        prompt = f"""
        Based on the following persona and list of purchased products, generate {n} distinct, likely shopping goals for this user.
        
        **User Persona:**
        {persona}

        **Purchased Products:**
        {product_list_str}

        **Instructions:**
        1. Analyze the user's spending habits, interests, and potential needs based on the purchase history and persona.
        2. Generate exactly {n} specific, natural language shopping goals.
        3. Goals should vary in category (e.g., "Buy a new laptop", "Restock coffee pods", "Find a gift for a cyclist") so that the products in each category don't overlap.
        4. Return ONLY a JSON list of strings.

        **Example Output:**
        ```json
        [
            "Upgrade to a mechanical keyboard",
            "Buy a new pair of running shoes",
            ...
        ]
        ```
        """

        response_message = self.model([{"role": "user", "content": prompt}])
        content = response_message.content

        match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
        if match:
            try:
                goals = json.loads(match.group(1))
                if isinstance(goals, list):
                    return goals[:n] # Ensure we don't return more than n
            except json.JSONDecodeError:
                pass
        
        # Fallback if JSON parsing fails or structure is wrong
        # This is a basic fallback, in a real scenario we might retry or use a more robust parser
        return []
