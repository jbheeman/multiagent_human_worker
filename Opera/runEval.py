
from personaAdapter import load_persona_dataset
from typing import Any
from smolagents import OpenAIServerModel
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

import numpy as np

# eval_model = SentenceTransformer("all-mpnet-base-v2")
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')


gepa_prompt = """


Based on the following shopping interaction history, please infer a persona for the user.\n\nThe interaction list may include:\n- **Purchased Products**: strong evidence of true preferences and needs.\n- **Items Added to Cart (but not bought)**: moderate evidence of interest or intent.\n- **Clicked / Viewed Items**: weak evidence of curiosity or early-stage interest.\n\nTreat purchases as the strongest signal, cart items as secondary, and clicks/views as the weakest signal.\n\n{product_list_str}\n\n**Instructions:**\nYou must follow these steps and show your work for each one:\n\n1. **Extract Traits:** For each relevant product or interaction, identify key traits (e.g., brand, category, price point, features, implied hobbies or interests).\n2. **Identify Buying Patterns:** Look for patterns across interactions to determine the user's buying habits and preferences. Consider at least:\n   - Shopping frequency/intensity (do they appear to shop often or occasionally?)\n   - Price sensitivity (budget-conscious vs willing to pay more for quality)\n   - Category focus (e.g., pet care, electronics, home goods, beauty, etc.)\n   - Brand behavior (brand-loyal vs exploratory)\n   - How they might use reviews/ratings when choosing products\n   - Openness to novelty (trying new product types vs sticking to familiar ones)\n3. **Categorize Traits:** Group the user's inferred buying traits into the following four categories:\n   - **Confident Likes:** Things we are confident the person likes.\n   - **Somewhat Confident Likes:** Things we are somewhat confident the person likes.\n   - **Confident Dislikes:** Things we are confident the person dislikes.\n   - **Somewhat Confident Dislikes:** Things we are somewhat confident the person dislikes.\n4. **Generate Persona Description:** Based on the categorized traits, write a concise plaintext paragraph describing the user's *shopping persona*. This description should be suitable for guiding a research assistant and should be 3–6 sentences long.\n5. **Do not infer demographic details** (age, gender, location, education level, family status, etc.) unless they are explicitly stated in the product descriptions. Focus only on shopping behavior and preferences.\n\n**Output Format:**\nYou must provide your full reasoning for steps 1–3. After your reasoning, provide the final persona description enclosed in `<persona_description>` tags.\n\n**Example Output:**\n**1. Extracted Traits:**\n- Airkeep Car Air Freshener: Low price, home/car accessory, scent-focused.\n- Lumiere & Co. Bike Seat Bag: Mid-range price, cycling accessory, practical.\n...\n\n**2. Buying Patterns:**\n- The user frequently buys cycling-related gear, suggesting a hobby in cycling.\n- The user purchases items at various price points, but seems to value function over luxury.\n...\n\n**3. Categorized Traits:**\n- **Confident Likes:** Cycling, practical items.\n- **Somewhat Confident Likes:** Home fragrance, pet safety.\n...\n\n<persona_description>\nThe user is a practical, budget-conscious individual who prioritizes functionality and value. They are an avid cyclist, investing in quality components for their hobby. They are not brand-loyal but seem to prefer items with good reviews and a focus on durability. They show some interest in home and pet accessories, but are not driven by luxury or high-end brands.\n</persona_description>\n"
"""


base_prompt = """
Based on the following shopping interaction history, please infer a persona for the user.

The interaction list may include:
- **Purchased Products**: strong evidence of true preferences and needs.
- **Items Added to Cart (but not bought)**: moderate evidence of interest or intent.
- **Clicked / Viewed Items**: weak evidence of curiosity or early-stage interest.

Treat purchases as the strongest signal, cart items as secondary, and clicks/views as the weakest signal.

{product_list_str}

**Instructions:**
You must follow these steps and show your work for each one:

1. **Extract Traits:** For each relevant product or interaction, identify key traits (e.g., brand, category, price point, features, implied hobbies or interests).
2. **Identify Buying Patterns:** Look for patterns across interactions to determine the user's buying habits and preferences. Consider at least:
   - Shopping frequency/intensity (do they appear to shop often or occasionally?)
   - Price sensitivity (budget-conscious vs willing to pay more for quality)
   - Category focus (e.g., pet care, electronics, home goods, beauty, etc.)
   - Brand behavior (brand-loyal vs exploratory)
   - How they might use reviews/ratings when choosing products
   - Openness to novelty (trying new product types vs sticking to familiar ones)
3. **Categorize Traits:** Group the user's inferred buying traits into the following four categories:
   - **Confident Likes:** Things we are confident the person likes.
   - **Somewhat Confident Likes:** Things we are somewhat confident the person likes.
   - **Confident Dislikes:** Things we are confident the person dislikes.
   - **Somewhat Confident Dislikes:** Things we are somewhat confident the person dislikes.
4. **Generate Persona Description:** Based on the categorized traits, write a concise plaintext paragraph describing the user's *shopping persona*. This description should be suitable for guiding a research assistant and should be 3–6 sentences long.
5. **Do not infer demographic details** (age, gender, location, education level, family status, etc.) unless they are explicitly stated in the product descriptions. Focus only on shopping behavior and preferences.

**Output Format:**
You must provide your full reasoning for steps 1–3. After your reasoning, provide the final persona description enclosed in `<persona_description>` tags.

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
    scores = nli_model.predict([(gold_persona, persona_description)])
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    entailment_score = probs[0][2]
    return float(entailment_score)


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
    for i in range(5, 7):
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
                    