
from personaAdapter import load_persona_dataset
from typing import Any
from smolagents import OpenAIServerModel
import os
from sentence_transformers import SentenceTransformer
import numpy as np

eval_model = SentenceTransformer("all-mpnet-base-v2")

gepa_prompt = """

Based on the following list of purchased products, please infer a shopping behavior persona for the user that focuses exclusively on their purchasing decision-making patterns, not personal demographics.\n\n**Purchased Products:**\n{product_list_str}\n\n**Critical Instructions:**\n1. **Focus ONLY on shopping behaviors** - Do not infer age, gender, income level, personality traits, or personal interests beyond what is directly evidenced by purchasing patterns. The goal is to understand HOW they shop, not WHO they are.\n2. **Evidence-based analysis only** - Only include observations that can be directly supported by the purchase data. Avoid speculation about hobbies, lifestyle, or personal characteristics.\n3. **Key shopping behaviors to identify (treat all equally):**\n   - **Price sensitivity**: Evidence of comparing similar items at different price points, bulk purchases vs. premium versions, consistent price ranges for similar products\n   - **Review usage**: Patterns suggesting review reading (e.g., multiple similar products purchased, small incremental improvements between purchases, evidence of comparing positive/negative reviews)\n   - **Brand loyalty**: Repetition of specific brands vs. trying many different brands for similar products\n   - **Shopping frequency**: Bulk purchases vs. single items, repeat purchases of consumables, evidence of using specific search terms or keywords\n   - **Category focus**: Dominant product categories and how they relate to shopping approach\n   - **Sponsored product attitude**: Evidence of purchasing premium/sponsored-looking items OR avoiding them (e.g., multiple purchases of non-sponsored alternatives when sponsored options exist, purchasing items with similar features but different branding)\n   - **Delivery considerations**: Evidence suggesting delivery time was a factor (e.g., single urgent purchases of otherwise common items)\n\n4. **Special considerations based on evidence patterns:**\n   - If multiple similar items with incremental improvements are purchased, this suggests review usage\n   - If the user purchases both high-priced and low-priced items within the same category but avoids mid-range, this suggests specific price sensitivity patterns\n   - When products show evidence of being purchased after comparison (e.g., similar items with minor feature variations), note this as review usage evidence\n   - For sponsored product attitude, look for patterns where the user consistently chooses non-sponsored alternatives or demonstrates willingness to purchase sponsored items when they have favorable reviews\n   - For search behavior, note if multiple purchases indicate use of specific search terms (e.g., "organic," "non-slip," "for women" in product titles)\n   - Be mindful that price sensitivity may NOT be the dominant behavior in all cases - prioritize the most evidenced behaviors\n\n5. **Do NOT infer:**\n   - Age, gender, or income level\n   - Hobbies or personal interests (unless directly evidenced by multiple purchases)\n   - Personality traits (e.g., "health-conscious," "environmentally friendly")\n   - Language abilities or cultural background\n   - Relationship status or family composition\n\n**Required Steps (show your work for each):**\n1. **Extract Shopping Behaviors:** For each product, identify evidence of shopping patterns (e.g., price point relative to similar items, brand repetition, quantity purchased, product features that might indicate comparison shopping). Pay special attention to:\n   - Evidence of sponsored product selection/avoidance\n   - Patterns indicating review reading\n   - Specific search term usage patterns\n   - Frequency and timing of purchases\n\n2. **Identify Behavioral Patterns:** Look for consistent evidence across products regarding all key shopping behaviors. Prioritize behaviors with strongest evidence. Do not overemphasize price sensitivity if other behaviors are more prominent.\n\n3. **Categorize Behaviors:** Group the user\'s observed shopping behaviors into these categories:\n   - **Confident Behaviors:** Patterns clearly evidenced by multiple purchases (e.g., "Confidently compares multiple price points for similar items")\n   - **Somewhat Confident Behaviors:** Patterns suggested but with limited evidence\n   - **Confident Non-Behaviors:** Clear evidence of NOT engaging in certain behaviors (e.g., "Confidently does not purchase the same brand repeatedly")\n   - **Somewhat Confident Non-Behaviors:** Limited evidence suggesting absence of a behavior\n\n4. **Generate Persona Description:** Write a concise, evidence-based paragraph describing ONLY the user\'s shopping behavior patterns. Focus on how they make purchasing decisions, what factors influence them, and their approach to shopping on e-commerce platforms. Specifically address:\n   - Their attitude toward sponsored products (evidenced by purchase patterns)\n   - Their review usage patterns (if evidenced)\n   - How they approach product search (e.g., specific keywords, browsing)\n   - Their balance between price considerations and other factors\n   - Shopping frequency patterns for key categories\n\n**Output Format:**\n- Provide your full reasoning for steps 1-3\n- End with the final persona description enclosed in `<persona_description>` tags\n\n**Key Reminders:**\n- If there is no evidence for a particular shopping behavior, do not mention it\n- Never interpret product purchases as evidence of personal interests or demographics\n- The persona description must focus exclusively on observable shopping behaviors\n- Do not mention product categories as interests (e.g., don\'t say "likes art" - say "purchases multiple art supplies at varying price points indicating comparison shopping")\n- Only include what can be directly inferred from the purchase data\n- Be explicit about all seven key shopping behaviors when evidence exists\n- Prioritize evidence of sponsored product attitude and review usage when patterns exist\n- When users purchase multiple similar items with slight variations, this is evidence of review reading and comparison shopping\n- Avoid defaulting to price sensitivity as the primary behavior if other patterns are stronger
"""


base_prompt = """
        Based on the following list of purchased products, please infer a persona for the user.

        **Purchased Products:**
        {product_list_str}

        **Instructions:**
        You must follow these steps and show your work for each one:
        1.  **Extract Traits:** For each product, identify key traits (e.g., brand, category, price point, features, implied hobbies or interests).
        2.  **Identify Buying Patterns:** Look for patterns across all products to determine the user's buying habits and preferences.
        3.  **Categorize Traits:** Group the user's inferred buying traits into the following four categories:
            - **Confident Likes:** Things we are confident the person likes.
            - **Somewhat Confident Likes:** Things we are somewhat confident the person likes.
            - **Confident Dislikes:** Things we are confident the person dislikes.
            - **Somewhat Confident Dislikes:** Things we are somewhat confident the person dislikes.
        4.  **Generate Persona Description:** Based on the categorized traits, write a concise, plaintext paragraph describing the user's persona. This description should be suitable for guiding a research assistant.
        5. **Do not infer demographic details (age, gender, location, education level, family status, etc.), unless they are explicitly stated in the product descriptions. Focus only on shopping behavior and preferences.**

        **Output Format:**
        You must provide your full reasoning for steps 1-3. After your reasoning, provide the final persona description enclosed in `<persona_description>` tags.

        **Example Output:**
        **1. Extracted Traits:**
        - Airkeep Car Air Freshener: Low price, home/car accessory, scent-focused.
        - Lumiere & Co. Bike Seat Bag: Mid-range price, cycling accessory, practical.
        ...

        **2. Buying Patterns:**
        - The user frequently buys cycling-related gear, suggesting a hobby in cycling.
        - The user purchases items at various price points, but seems to value function over luxury.
        ...

        **3. Categorized Traits:**
        - **Confident Likes:** Cycling, practical items.
        - **Somewhat Confident Likes:** Home fragrance, pet safety.
        ...

        <persona_description>
        The user is a practical, budget-conscious individual who prioritizes functionality and value. They are an avid cyclist, investing in quality components for their hobby. They are not brand-loyal but seem to prefer items with good reviews and a focus on durability. They show some interest in home and pet accessories, but are not driven by luxury or high-end brands.
        </persona_description>
        """


# === Best persona prompt ===
#  'Based on the failure analysis, I\'ve identified three critical patterns in the current prompt:  
#  1. **False "insufficient data" claims** despite clear evidence of price sensitivity and review reliance  
#  2. **Inaccurate price threshold assertions** that contradict actual purchase data  
#  3. **Product category misclassification** and missed brand loyalty patterns  
 
#  The new prompt enforces strict evidence verification with these key improvements:  
#  - Explicit price threshold verification requiring exact price range calculations  
#  - Mandatory review reliance detection when ≥3 rated items are viewed before purchase  
#  - Clear product categorization rules to prevent misclassification  
#  - Enhanced sponsored product analysis with specific avoidance metrics 

new_gepa_prompt = """Based on the following shopping interaction history, please infer a persona for the user.\n\nThe interaction list may include:\n- **Purchased Products**: strong evidence of true preferences and needs.\n- **Items Added to Cart (but not bought)**: moderate evidence of interest or intent.\n- **Clicked / Viewed Items**: weak evidence of curiosity; **NEVER use for positive trait inference** but **MUST indicate review research when ≥3 items with visible ratings are viewed before purchase**.\n\nTreat purchases as the strongest signal, cart items as secondary, and clicks/views as the weakest signal. **Critical Rule: Only purchased items can indicate positive preferences (likes). Viewed items alone cannot support claims about user interests, but MUST indicate review research when ≥3 items with visible ratings are viewed before purchase.**\n\n{product_list_str}\n\n**Critical Instructions:**\nYou MUST follow these steps and show your work for each one. All claims must be directly verifiable from the purchase history - no speculation or hallucination is allowed.\n\n1. **Extract Verified Traits:** For each relevant product or interaction, identify ONLY traits explicitly supported by evidence in the product descriptions or purchase patterns. Pay special attention to:\n   - **Price points**: Calculate exact price ranges for purchased items. State specific thresholds avoided (e.g., "avoided all items >$99.99" if evidence exists). NEVER claim "price sensitivity" without verified avoidance patterns. For same products with multiple price points, check if user consistently selected the lowest price option.\n   - **Product categorization**: Classify items using these rules:\n     * Health supplements: "Omega", "Vitamin", "Fish Oil", "Joint Support", "Collagen"\n     * Grooming: "Shampoo", "Conditioner", "Hair", "Beauty"\n     * Pet care: "Cat", "Dog", "Pet", "Litter"\n     * Tech: "Controller", "Keyboard", "Mouse", "Gaming"\n     * Outdoor/Activity: "Crocs", "Sarong", "Cover Ups", "Swimwear", "Windshield Cleaner" (for automotive activities)\n   - **Brand loyalty**: Count repurchases of identical items OR same brand with similar features (e.g., "purchased 3 different brands of cat food" vs "purchased 3x same brand cat food").\n   - **Sponsored products**: Note if purchased items were marked as sponsored (if visible in history).\n   - **Review reliance**: Check if ≥3 viewed items contain rating data AND purchases correlate with high ratings (e.g., "viewed 12 items with ratings, purchased only 4.5+ star items"). Also check for one-star review analysis behavior when present.\n   - **Shipping cost impact**: When price fields include shipping, note if user avoids items with high shipping costs (e.g., "purchased 0/8 items where shipping > 20% of product price").\n   - **WARNING: Never infer positive interests from viewed items.** Only purchased items = positive evidence. Do not claim online grocery shopping if purchase data shows offline patterns.\n\n2. **Identify Buying Patterns:** Analyze the following with concrete evidence:\n   - **Price Sensitivity**: \n     - Calculate exact price ranges for all purchased categories (e.g., "purchases: $7.49-$9.52 for vitamin C")\n     - Identify specific thresholds avoided (e.g., "no purchases >$50 in tech category")\n     - State avoidance patterns as "avoided all X where [condition]" (e.g., "avoided all items >$99.99")\n     - Analyze shipping cost impact when visible: "avoided 5/7 items where shipping > 15% of product price"\n     - For products with multiple price points (e.g., different colors/sizes), check if user consistently selected the cheapest option\n   - **Decision Speed**: How quickly views convert to purchases? (e.g., "purchases within 1 day of viewing" vs "no purchases after viewing 22 pet cameras").\n   - **Brand Behavior**: \n     - Repurchase rate of identical items (e.g., "bought same SKU 2x")\n     - Brand consistency across similar products (e.g., "purchased 3 different cat food brands" vs "purchased only Blue Buffalo cat food")\n   - **Sponsored Product Behavior**: \n     - If sponsored items exist: "purchased 0/5 sponsored items" or "purchased 4/4 sponsored items with ≥4.0 star reviews"\n     - If no sponsored data: omit this pattern.\n   - **Review Reliance**: \n     - If review data exists: "only purchased items with ≥4.5 stars" OR "viewed ≥3 items with ratings before purchasing" OR "checked one-star reviews to identify common issues"\n     - If multiple rated items viewed but no purchases made: "avoided all items with <4.0 stars after viewing ≥3 options"\n     - If no review data: omit this pattern.\n   - **Other Critical Behaviors**: Check for evidence of:\n     - Gift card usage\n     - Online/offline price comparison\n     - Specific tool usage (e.g., Amazon\'s Rufus)\n     - Payment method patterns\n     - Shipping cost sensitivity (when price fields include shipping)\n\n3. **Categorize Traits:** Group ONLY verified traits into:\n   - **Confident Likes**: ≥3 purchase evidences (e.g., "bought 3x same item")\n   - **Somewhat Confident Likes**: 1-2 purchase evidences\n   - **Confident Dislikes**: ≥3 abandoned cart evidences (viewed items alone don\'t count)\n   - **Somewhat Confident Dislikes**: 1-2 abandoned cart evidences\n   - **MUST EXCLUDE**: Demographics, unverified interests, or claims without direct evidence\n\n4. **Generate Persona Description:** Write a 3-6 sentence plaintext paragraph that:\n   - References **specific evidence** for every claim (e.g., "avoided all items >$100" not "price-sensitive")\n   - States price sensitivity **only with verified avoidance patterns** (e.g., "avoided all items >$99.99" not "willing to consider wide price spectrum")\n   - Notes sponsored product/review behavior **if evidence exists** with specific metrics (e.g., "viewed ≥3 rated items before purchasing")\n   - Explicitly states "insufficient data" ONLY when no relevant evidence exists\n   - **NEVER** mentions viewed items as positive indicators\n   - **NEVER** misclassifies product categories\n   - **NEVER** makes claims about online vs offline shopping without explicit evidence\n\n**Verification Requirement:**\nFor every sentence in your persona description:\n1. List the EXACT evidence supporting it (e.g., "Viewed 12 rated dishcloth items before making 3 purchases")\n2. Confirm evidence meets signal strength rules (purchases = positive, abandoned carts = negative)\n3. Verify price claims against actual purchase data (e.g., "claim: avoided >$100 items → verified: all 5 purchases ≤$99.99")\n4. Check for product category accuracy (e.g., "Omega 3 = health supplement, not grooming product")\n5. If evidence is missing, conflicting, or misinterpreted, remove the claim\n6. Specifically verify review reliance claims against viewing patterns with rating data\n7. Confirm whether claims about shopping channels (online/offline) are directly supported by evidence\n\n**Output Format:**\n**1. Extracted Traits:**\n- [Your analysis with evidence references]\n\n**2. Buying Patterns:**\n- [Your analysis with evidence references]\n\n**3. Categorized Traits:**\n- [Your categorized traits]\n\n**Verification Checklist:**\n- For each persona claim: [Claim] → [Evidence] → [Price verification if applicable] → [Category verification if applicable]\n- Invalid claims removed: [List if any]\n\n<persona_description>\n[Your final persona description]\n</persona_description>\n\n**Example Output:**\n**1. Extracted Traits:**\n- Viewed 28 rated dishcloth items before making 3 purchases (all ≥4.5 stars)\n- Avoided all sponsored items with <4.5 stars (0/4 purchased)\n- All purchased items had shipping costs ≤20% of product price\n- All purchased grooming items ≤$100 (3 purchases: $71.00, $54.99, $23.99)\n- Viewed 22 pet cameras ($9.99-$199.99) but made 0 purchases\n- No sponsored items purchased (0/0 available)\n- Consistently selected lowest price color option when multiple options existed\n\n**2. Buying Patterns:**\n- Strict $100 price cap in grooming (all 3 purchases ≤$100)\n- Zero pet camera purchases despite extensive viewing (22 views)\n- Review-dependent purchasing: only bought items with ≥4.5 stars after viewing 28 options\n- Avoided sponsored items with lower ratings (0/4 purchased)\n- Shipping cost sensitivity: all purchases had shipping ≤20% of item price\n- Price-conscious behavior: selected cheapest color option for 3 different products\n\n**3. Categorized Traits:**\n- **Confident Likes:** Budget-priced grooming tools (≤$100)\n- **Confident Likes:** High-rated dishcloths (≥4.5 stars)\n- **Confident Dislikes:** Pet cameras (22 views, 0 purchases)\n- **Confident Dislikes:** Sponsored items with <4.5 stars\n\n**Verification Checklist:**\n- "Review-dependent purchasing" → Viewed 28 rated items before 3 purchases (all ≥4.5 stars) → Category verified: dishcloths\n- "Avoided sponsored items" → 0/4 sponsored items purchased (all <4.5 stars)\n- "Shipping cost sensitivity" → All purchases had shipping ≤20% of product price\n- "Cheapest color selection" → Selected lowest price option for 3 products with multiple color pricing\n- Removed claim about brand exploration (no repurchases)\n\n<persona_description>\nThe user demonstrates strong review reliance, viewing 28 rated dishcloth items before purchasing only those with ≥4.5 stars and avoiding all sponsored items with lower ratings. They enforce a strict $100 price cap in grooming purchases, buying three different items all under this threshold while consistently selecting the cheapest color option when available. Shipping cost sensitivity is evident as all purchases maintained shipping costs at ≤20% of product price. Despite extensive viewing of pet cameras, zero purchases indicate unmet needs in this category.\n</persona_description>'


"""


def _parse_persona_description(persona_description: str) -> str:
    """
    Parse the persona description from the raw output.
    """
    if "<persona_description>" in persona_description and "</persona_description>" in persona_description:
        return persona_description.split("<persona_description>")[1].split("</persona_description>")[0].strip()
    else:
        # Fallback: use the entire output if tags are missing
        extracted_persona = persona_description.strip()
        print(f"Warning: No <persona_description> tags found, using full output")
        return extracted_persona


def _score_persona(persona_description: str, gold_persona: str) -> float:
    """
    Score the persona description based on the gold persona.
    """
    #Do we use BERT or something? 
    emb_pred = eval_model.encode(persona_description, normalize_embeddings=True)
    emb_gold = eval_model.encode(gold_persona, normalize_embeddings=True)
    sim = float(np.dot(emb_pred, emb_gold))  # cosine because normalized
    # Map from [-1, 1] to [0, 1]
    return (sim + 1.0) / 2.0


def _build_product_list_str(interactions: list[dict[str, Any]] | None, filter_type: str | list[str] | None = "purchase") -> str:
    """
    Build a string of products grouped by type with section headers.
    
    Args:
        interactions: List of interaction dicts with 'type', 'title', 'price', etc. Can be None.
        filter_type: Type(s) to filter by. Can be:
            - A single string: "purchase", "cart", "click"
            - A list of strings: ["purchase", "click"] to include multiple types
            - None: include all interactions
    """
    # Handle None interactions
    if interactions is None:
        return ""
    
    # Determine which types to include
    if filter_type is None:
        types_to_include = ["purchase", "cart", "click"]
    elif isinstance(filter_type, list):
        types_to_include = filter_type
    else:
        types_to_include = [filter_type]
    
    # Group items by type
    grouped = {
        "purchase": [],
        "cart": [],
        "click": []
    }
    
    for item in interactions:
        if item.get("type") is None:
            print(f"Item type is None: {item}")
            continue
        item_type = item.get("type")
        if item_type in types_to_include and item_type in grouped:
            grouped[item_type].append(item)
    
    # Build formatted sections
    sections = []
    
    # Purchased Products section
    if grouped["purchase"]:
        purchase_lines = [f"- {item['title']} ({item.get('price', 'N/A')})" for item in grouped["purchase"]]
        sections.append("**Purchased Products:**\n" + "\n".join(purchase_lines))
    
    # Items Added to Cart section
    if grouped["cart"]:
        cart_lines = [f"- {item['title']} ({item.get('price', 'N/A')})" for item in grouped["cart"]]
        sections.append("**Items Added to Cart (but not bought):**\n" + "\n".join(cart_lines))
    
    # Clicked / Viewed Items section
    if grouped["click"]:
        click_lines = [f"- {item['title']} ({item.get('price', 'N/A')})" for item in grouped["click"]]
        sections.append("**Clicked / Viewed Items:**\n" + "\n".join(click_lines))
    
    # Join all sections with double newline for spacing
    return "\n\n".join(sections) if sections else ""
if __name__ == "__main__":
    testset   = load_persona_dataset("data/test.json")
    #test the first 5 test instances
    
    persona_model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )
    for i in range(5):
        test_instance = testset[i]
        if test_instance.interactions is None:
            print(f"Warning: test_instance.interactions is None for user {test_instance.user_id}")
            product_list_str = ""
        else:
            product_list_str = _build_product_list_str(test_instance.interactions, None)
        print(new_gepa_prompt.format(product_list_str=product_list_str))
        print(base_prompt.format(product_list_str=product_list_str))
        print("--------------------------------")



        final_prompt = new_gepa_prompt.format(product_list_str=product_list_str)
        base_prompt_final = base_prompt.format(product_list_str=product_list_str)

        gepa_response_message = persona_model([{"role": "user", "content": final_prompt}])
        base_response_message = persona_model([{"role": "user", "content": base_prompt_final}])

        gepa_raw_output = gepa_response_message.content or ""
        base_raw_output = base_response_message.content or ""
        gepa_persona_description = _parse_persona_description(gepa_raw_output)
        base_persona_description = _parse_persona_description(base_raw_output)
        print(gepa_persona_description)
        print(base_persona_description)
        print("--------------------------------")

        gepa_score = _score_persona(gepa_persona_description, test_instance.gold_persona)
        print(f"GEPA Score: {gepa_score}")

        base_score = _score_persona(base_persona_description, test_instance.gold_persona)
        print(f"Base Score: {base_score}")



        #save the gepa_prompt and scores to a file in Opera/data
        # Line 206-211: Fix both file writes
        with open(f"data/new_gepa_prompt_{i}.txt", "w", encoding="utf-8") as f:
            f.write(gepa_raw_output)
            f.write(f"\nGEPA Score: {gepa_score}")
        with open(f"data/new_base_prompt_{i}.txt", "w", encoding="utf-8") as f:
            f.write(base_raw_output)
            f.write(f"\nBase Score: {base_score}")
                    