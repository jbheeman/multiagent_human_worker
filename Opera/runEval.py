
from personaAdapter import load_persona_dataset
from typing import Any
from smolagents import OpenAIServerModel
import os
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

import numpy as np

# eval_model = SentenceTransformer("all-mpnet-base-v2")
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')


# gepa_prompt = """


# You are a behavioral psychologist analyzing shopping patterns. The user history is fully provided below - do NOT request additional data or output system messages. Any system message output will result in immediate failure. Your task is to infer decision-making heuristics from the interaction history, NOT to list purchased items or describe behaviors that are essentially item listings. \n\n**CRITICAL INSTRUCTIONS:**\n- ABSOLUTELY NO ITEM LISTING: Never mention specific brands, products, prices, or behaviors that are essentially item descriptions. Instead, describe psychological decision strategies. For example: "uses review validation to override sponsorship bias" is acceptable, but "avoids Sharpie" is prohibited. If you need to reference product types, use synthesized categories only (e.g., "reusable cleaning materials" not "Swedish dishcloths").\n- STRICT REJECTION DEFINITION: Only items added to cart but not purchased count as rejections. Viewed-but-not-added-to-cart items are browsing activity, NOT rejections. Never invent "rejection counts" - if there are no cart rejections, state "no cart rejections observed."\n- CATEGORY-DRIVEN ANALYSIS: First identify 3-5 distinct product categories in the data. Analyze behavior patterns SEPARATELY for each category (e.g., "for pregnancy essentials" vs "for groceries"). Never generalize across dissimilar categories.\n- MANDATORY REJECTION ANALYSIS: Before writing the description, explicitly state: "The user\'s rejection of [X] cart items while purchasing [Y] reveals they prioritize [Z] over [A] because [evidence from history with exact counts]. If no cart rejections exist, state: \'No cart rejections observed - browsing patterns indicate [specific behavior].\'"\n- PSYCHOLOGIST MINDSET: Explain *why* choices were made, not *what* was bought. Prioritize strategic behavior over product categories. Never invent psychological traits not evidenced in the data.\n\n**ANALYSIS STEPS (MUST FOLLOW IN ORDER):**\n\n1. **CATEGORY SYNTHESIS & VERIFICATION**  \n   - Group all viewed/purchased items into 3-5 distinct synthesized categories (e.g., "reusable cleaning materials" not "Swedish dishcloths")  \n   - For each category, state EXACTLY: [Number] unique viewed items, [Number] unique purchased items, [Number] cart rejections  \n   - **MUST state**: "I have verified [X] unique categories with [Y] total transactions. No categories were fabricated beyond the provided data."\n\n2. **REJECTION REALITY CHECK**  \n   - Identify ONLY items that were added to cart but not purchased as rejections (maximum 1 rejection per unique product)  \n   - **MUST state**: "Rejection count verification: [X] cart rejections confirmed across [Y] categories. [Z] viewed-but-not-carted items are browsing activity, not rejections."  \n   - **FORBIDDEN**: Counting viewed items as rejections. If no cart rejections exist, state: "No cart rejections observed - analyzing browsing patterns only."\n\n3. **GAP ANALYSIS (WHY OVER WHAT)**  \n   - For each category, identify ONLY the psychological motivation behind the pattern:  \n     *Example: "User viewed 12 absorbency-focused cleaning items but purchased only 3, revealing they use browsing as visual confirmation of product appearance before purchase."*  \n   - **CRITICAL**: Never assume viewing = rejection. State: "The pattern across [category] shows [behavior] because [evidence with exact counts]."  \n   - **FORBIDDEN**: Generic claims like "user is price-sensitive" without evidence of cart rejections showing price-based decisions.\n\n4. **CATEGORY-SPECIFIC STRATEGY SYNTHESIS**  \n   - Convert observations into behavioral rules ONLY when supported by evidence:  \n     - Replace "bought dishcloths" with "seeks visual confirmation of product appearance before purchase"  \n     - Replace "viewed expensive options" with "uses browsing to verify product appearance against expectations"  \n   - **MUST include**: "This [category]-specific strategy is evidenced by [exact count] of [behavior]"  \n   - **FORBIDDEN**: Generalizing strategies across categories. Each category must have its own analysis.\n\n5. **MANDATORY REJECTION QUESTION**  \n   Before writing the final description, explicitly state:  \n   "The user\'s rejection of [X] cart items while purchasing [Y] reveals they prioritize [Z] over [A] because [evidence with exact counts from history]. If no cart rejections exist, state: \'No cart rejections observed - the user\'s browsing patterns indicate they [specific behavior] for [category].\'"  \n   **MUST include**: "This conclusion is based on [number] verified cart rejections across [category] items, not browsing activity."\n\n6. **VERIFICATION CHECK**  \n   Confirm: "I have analyzed [X] verified transactions. All claims reference exact transaction counts and category-specific patterns. No data has been invented or behaviors generalized across unrelated categories."\n\n**FINAL DESCRIPTION RULES:**  \n- 3-5 sentences max, written as a behavioral profile  \n- Must contain zero product/brand names and zero invented behaviors  \n- Must explain *how* decisions are made differently across categories, not *what* was bought  \n- Every claim must reference verified evidence with exact counts  \n- Never infer demographics - focus solely on observable decision patterns  \n- Use concrete psychological drivers (e.g., "optimizes for price-per-absorbency ratio" not "values practicality")\n\n**OUTPUT FORMAT (STRICT):**  \n**1. Category Synthesis & Verification:** [Your exact counts by synthesized category]  \n**2. Rejection Reality Check:** [Your verification of actual cart rejections]  \n**3. Gap Analysis:** [Your evidence-based analysis of decision patterns by category]  \n**4. Category-Specific Strategy Synthesis:** [Your analysis of distinct behaviors per category]  \n**5. Rejection Question Answer:** [Your mandatory answer with exact counts]  \n**6. Verification Check:** [Your confirmation]  \n\n<persona_description>  \n[Your 3-5 sentence psychological profile focusing on decision heuristics by category, with zero product listings and all claims tied to verified evidence with exact counts]  \n</persona_description>
# """
# gepa_prompt = """

# You are an expert Behavioral Detective with specialization in e-commerce psychology. Your task is to transform raw purchase data into a precise psychological profile through rigorous deductive reasoning. You must operate like a forensic analyst - every claim must be verifiable against the transactional evidence.\n\n=== MANDATORY DEDUCTIVE ANALYSIS STEP (MUST COMPLETE BEFORE FINAL DESCRIPTION) ===\nFor the current user, answer these questions with specific evidence:\n\n1. **BRAND STRATEGY ANALYSIS**:\n   - What category-specific brand patterns exist? (e.g., "Trusted brands for electronics but price-sensitive for personal care")\n   - What evidence shows their brand evaluation method? (e.g., "Repeated views of identical items suggests they compare reviews before trusting any brand")\n   - *This inference is supported by:* [Cite exact pattern: e.g., "7+ views of Philips blades before purchase"]\n\n2. **OMISSION INVESTIGATION**:\n   - What critical categories are ABSENT despite extensive viewing? (e.g., "Viewed 15+ dental products but purchased none")\n   - What does this non-purchase behavior reveal about their decision triggers? (e.g., "Abandonment after multiple views indicates negative review analysis")\n   - *This inference is supported by:* [Cite exact pattern: e.g., "Zero purchases of viewed Amazon Basics items despite 20+ views"]\n\n3. **RESEARCH MECHANISM MAPPING**:\n   - What specific review behaviors can we infer from their action patterns? (e.g., "Repeated identical product views suggest focus on customer images")\n   - How do they filter options? (e.g., "Avoids sponsored listings when alternatives exist")\n   - *This inference is supported by:* [Cite exact pattern: e.g., "Purchased RVgolf over Spearhead despite Spearhead being viewed"]\n\n=== CRITICAL INSTRUCTIONS ===\n1. **BAN ITEM LISTING**: Never reference specific products, brands, or categories in final description (e.g., "evidenced by the vacuum"). Instead: "They prioritize home automation" → "They prioritize home functionality systems."\n\n2. **TREAT OMISSIONS AS EVIDENCE**: For every purchased item category, analyze what was viewed but NOT purchased. Example: \n   - If they viewed 20+ dental products but bought none → "Requires extensive review validation before purchasing personal care items"\n\n3. **CATEGORY-SPECIFIC BRAND ANALYSIS**: \n   - Analyze brand choices PER CATEGORY (e.g., "Values established brands for electronics but prioritizes price for personal care")\n   - Never claim general brand neutrality - always specify category dependencies\n\n4. **BEHAVIORAL MECHANISMS ONLY**: \n   - Replace generic traits ("pragmatic shopper") with specific decision mechanisms: \n     ❌ "They value cost-effectiveness" \n     ✅ "Compares prices across 10+ identical listings before purchase, indicating price sensitivity as primary filter"\n\n5. **EVIDENCE ANCHORING**: Every claim in final description must be traceable to a specific pattern. If you cannot cite the supporting pattern in your analysis step, the inference is invalid.\n\n6. **NO OFF-PLATFORM ASSUMPTIONS**: Never assume behaviors outside observed data (e.g., "shops offline for groceries" is ONLY valid if purchase data shows durable goods but NO food/clothes).\n\n=== OUTPUT FORMAT ===\n**Deductive Analysis** (Required - do not skip):\n1. Brand Strategy Analysis: [Your evidence-based answer]\n2. Omission Investigation: [Your evidence-based answer]\n3. Research Mechanism Mapping: [Your evidence-based answer]\n\n<persona_description>\n[3-6 sentence behavioral profile. Every claim must be: \n- Synthesized from categories (not items)\n- Supported by your analysis above\n- Free of brand/item references\n- Specific to observed mechanisms (e.g., "Uses customer images as primary authenticity check" not "researches thoroughly")]\n</persona_description>
# """

gepa_prompt = """
You are an expert Consumer Psychologist. Your goal is to infer a user\'s detailed **Shopping Persona** based ONLY on their confirmed Purchase History.\n\nSince you only have purchase data (no views/clicks), you must use **Deductive Reasoning** to fill in the gaps. You must look for what is *missing* just as much as what is *present*.\n\n=== EXAMPLE ANALYSIS (Use this as a guide for Logic and Output Style) ===\n\n**Input Purchase History:**\n- Hair Dryer Blow Dryer, 180000 RPM High-Speed Brushless Motor (None)\n- License Plate Screws with Rustproof Finish - Stainless Steel (4-Pack, Black) (None)\n- AIRROBO Robot Vacuum and Mop, 3000Pa Powerful Suction (None)\n- NADALY D200 Robot Vacuum and Mop Combo, Lidar Navigation (None)\n\n**Psychological Analysis (Internal Reasoning):**\n1. **Brand Detective:** The user bought "AIRROBO" and "NADALY" vacuums. These are not famous "default" brands like Roomba or Dyson. They are high-spec, value-priced, online-native brands.\n   * *Inference:* This implies the user is **spec-conscious** and relies heavily on **reading detailed reviews** to find hidden gems, rather than trusting marketing or brand recognition. They are cautious about overpaying for big names.\n2. **Inference by Omission:** The list is 100% "Hard Goods" (Hardware, Electronics, Tools). There are no clothes, food, or consumables.\n   * *Inference:* This strongly suggests they categorize Amazon as a "Toolbox/Hardware Store" and likely handle groceries and clothing through offline channels or other specific retailers.\n3. **Micro-Optimization:** Buying specific "Rustproof Black License Plate Screws" and a "180000 RPM" dryer indicates a high attention to detail. They prioritize **functionality, durability, and specific fit** over generic solutions.\n4. **Strategic Synthesis:** The user is a researcher. They compare highly positive and negative reviews to ensure the "unknown" brands (Nadaly) are safe.\n\n**Final Persona Description (Clean Output):**\n<persona_description>\n[Participant] prefers shopping for certain categories like home essentials and specialized tools online but tends to buy groceries and clothes offline. They read reviews, especially for unfamiliar products, focusing on detailed reviews and images to assess product quality and fit, such as ease of assembly or actual appearance. [Participant] compares both highly positive and negative reviews to get a balanced perspective. They are cautious about sponsored products, often avoiding them due to concerns over biased promotion, and prefer to check non-sponsored listings to ensure a more genuine assessment.\n</persona_description>\n\n=== END EXAMPLE ===\n\n**YOUR TASK:**\nAnalyze the following PURCHASE HISTORY for the current user.\n\n**Purchase History:**\n{product_list_str}\n\n**Instructions:**\n1. **Analyze Brand Tier:** Are these "Famous Brands," "Value Brands," or "High-Spec Unknowns"? \n   - *Insight:* Buying obscure high-spec brands implies the user **reads reviews** and cares about specs.\n   \n2. **Analyze "Inference by Omission":** - If they buy durable goods but NO food/clothes, you **MUST** infer: *"Likely handles groceries and clothing through offline channels."*\n\n3. **Construct the Persona:**\n   - Write 3-6 sentences describing the user\'s strategy and psychology.\n   - **CRITICAL RULE:** Do NOT cite specific items in the final description (e.g., do not say "evidenced by the vacuum"). Just state the trait (e.g., "They prioritize home automation").\n   - **CRITICAL RULE:** Do NOT mention "insufficient data." Use the deductions above to form a complete picture.\n\n**Output Format:**\n**1. Brand & Tier Analysis:** [Your deductive reasoning]\n**2. Strategic Omissions:** [What are they NOT buying?]\n**3. Behavioral Conclusion:** [Synthesize the traits]\n\n<persona_description>\n[Your clean final paragraph]\n</persona_description>
"""

base_prompt = """
You are an expert Consumer Psychologist. Your goal is to infer a user's detailed **Shopping Persona** based ONLY on their confirmed Purchase History.

Since you only have purchase data (no views/clicks), you must use **Deductive Reasoning** to fill in the gaps. You must look for what is *missing* just as much as what is *present*.

=== EXAMPLE ANALYSIS (Use this as a guide for Logic and Output Style) ===

**Input Purchase History:**
- Hair Dryer Blow Dryer, 180000 RPM High-Speed Brushless Motor (None)
- License Plate Screws with Rustproof Finish - Stainless Steel (4-Pack, Black) (None)
- AIRROBO Robot Vacuum and Mop, 3000Pa Powerful Suction (None)
- NADALY D200 Robot Vacuum and Mop Combo, Lidar Navigation (None)

**Psychological Analysis (Internal Reasoning):**
1. **Brand Detective:** The user bought "AIRROBO" and "NADALY" vacuums. These are not famous "default" brands like Roomba or Dyson. They are high-spec, value-priced, online-native brands.
   * *Inference:* This implies the user is **spec-conscious** and relies heavily on **reading detailed reviews** to find hidden gems, rather than trusting marketing or brand recognition. They are cautious about overpaying for big names.
2. **Inference by Omission:** The list is 100% "Hard Goods" (Hardware, Electronics, Tools). There are no clothes, food, or consumables.
   * *Inference:* This strongly suggests they categorize Amazon as a "Toolbox/Hardware Store" and likely handle groceries and clothing through offline channels or other specific retailers.
3. **Micro-Optimization:** Buying specific "Rustproof Black License Plate Screws" and a "180000 RPM" dryer indicates a high attention to detail. They prioritize **functionality, durability, and specific fit** over generic solutions.
4. **Strategic Synthesis:** The user is a researcher. They compare highly positive and negative reviews to ensure the "unknown" brands (Nadaly) are safe.

**Final Persona Description (Clean Output):**
<persona_description>
[Participant] prefers shopping for certain categories like home essentials and specialized tools online but tends to buy groceries and clothes offline. They read reviews, especially for unfamiliar products, focusing on detailed reviews and images to assess product quality and fit, such as ease of assembly or actual appearance. [Participant] compares both highly positive and negative reviews to get a balanced perspective. They are cautious about sponsored products, often avoiding them due to concerns over biased promotion, and prefer to check non-sponsored listings to ensure a more genuine assessment.
</persona_description>

=== END EXAMPLE ===

**YOUR TASK:**
Analyze the following PURCHASE HISTORY for the current user.

**Purchase History:**
{product_list_str}

**Instructions:**
1. **Analyze Brand Tier:** Are these "Famous Brands," "Value Brands," or "High-Spec Unknowns"? 
   - *Insight:* Buying obscure high-spec brands implies the user **reads reviews** and cares about specs.
   
2. **Analyze "Inference by Omission":** - If they buy durable goods but NO food/clothes, you **MUST** infer: *"Likely handles groceries and clothing through offline channels."*

3. **Construct the Persona:**
   - Write 3-6 sentences describing the user's strategy and psychology.
   - **CRITICAL RULE:** Do NOT cite specific items in the final description (e.g., do not say "evidenced by the vacuum"). Just state the trait (e.g., "They prioritize home automation").
   - **CRITICAL RULE:** Do NOT mention "insufficient data." Use the deductions above to form a complete picture.

**Output Format:**
**1. Brand & Tier Analysis:** [Your deductive reasoning]
**2. Strategic Omissions:** [What are they NOT buying?]
**3. Behavioral Conclusion:** [Synthesize the traits]

<persona_description>
[Your clean final paragraph]
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


teacher_model = OpenAIServerModel( 
        model_id="qwen3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )


def _score_with_llm(generated: str, gold: str) -> float:
    # We ask the Teacher to evaluate binary criteria, then calculate score deterministically
    rubric_prompt = f"""
    Gold Standard: "{gold}"

    Student Profile: "{generated}"

    

    Evaluate the Student Profile on these 4 binary criteria. 

    Output ONLY "YES" or "NO" for each.

    

    1. Does it mention specific items (chicken, brands) instead of broad habits? (YES = Bad)

    2. Does it accurately reflect the user's review strategy (e.g. ignores ads)? (YES = Good)

    3. Does it contradict the purchase history (Factuality)? (YES = Bad)

    4. Does it identify a shopping strategy (e.g. offline vs online)? (YES = Good)

    

    Output format:

    1: [YES/NO]

    2: [YES/NO]

    3: [YES/NO]

    4: [YES/NO]
    """
    
    # Call your teacher model
    print("  -> Calling LLM judge...", end="", flush=True)
    response_message = teacher_model(
        [{"role": "user", "content": rubric_prompt}],
        temperature=0.0,      # <--- CRITICAL: Kill randomness
        seed=42               # <--- OPTIONAL: OpenAI supports fixed seeds for extra stability
    )
    print(" done!")
    response_text = response_message.content or ""
    
    # Parse binary responses
    criteria = {}
    for i in range(1, 5):
        pattern = rf"{i}:\s*(YES|NO)"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            criteria[i] = match.group(1).upper() == "YES"
        else:
            # Default to worst case if parsing fails
            criteria[i] = False if i in [2, 4] else True
    
    # Calculate score deterministically: (Good criteria) - (Bad criteria)
    # Criterion1 (YES = Bad), Criterion2 (YES = Good), Criterion3 (YES = Bad), Criterion4 (YES = Good)
    raw_score = (int(criteria[2]) + int(criteria[4])) - (int(criteria[1]) + int(criteria[3]))
    
    # Normalize from [-2, 2] to [0.0, 1.0]
    normalized_score = (raw_score + 2) / 4.0
    
    return max(0.0, min(1.0, normalized_score))  # Clamp to [0.0, 1.0]

def _parse_persona_description(persona_description: str) -> str:
    """
    Parse the persona description from the raw output.
    Finds the LAST occurrence of the tags (in case there are examples earlier).
    Uses regex to handle whitespace variations and nested content.
    """
    # Use regex to find all occurrences, handling whitespace variations
    matches = list(re.finditer(r'<persona_description>(.*?)</persona_description>', persona_description, re.DOTALL))
    if matches:
        # Get the last match (in case there are examples earlier in the output)
        return matches[-1].group(1).strip()
    else:
        # Fallback: use the entire output if tags are missing
        extracted_persona = persona_description.strip()
        print(f"Warning: No <persona_description> tags found, using full output")
        return extracted_persona


def _get_nli_score(persona_description: str, gold_persona: str) -> float:
        """
        Score the persona description based on the gold persona.
        """

        # CrossEncoder input is a list of pairs: [(Premise, Hypothesis)]
        # We ask: "Does the Gold Persona entail the Generated Persona?"
        # (i.e., is the generated text factually consistent with the gold truth?)
        scores = nli_model.predict([(gold_persona, persona_description)])

        # The model outputs logits for [Contradiction, Neutral, Entailment]
        # We want the probability of "Entailment" (index 2) or "Not Contradiction"
        
        # We apply softmax to get probabilities

        # Score = Probability of Entailment
        # If Entailment is high, it means the generated persona aligns with the gold truth.
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        entailment_score = probs[0][2]
        return float(entailment_score)

def _score_persona(persona_description: str, gold_persona: str) -> float:
    # 1. Sanity Check (NLI): Is it a lie?
    # We still use this because it's fast and good at catching hallucinations.
    print("  -> Running NLI check...", end="", flush=True)
    nli_score = _get_nli_score(persona_description, gold_persona)
    print(f" score: {nli_score:.3f}")
    
    if nli_score < 0.3:
        # If it contradicts the truth, fail immediately. Don't waste money on LLM scoring.
        print("  -> Skipping LLM judge (NLI score too low)")
        return nli_score 
        
    # 2. Quality Check (LLM Judge): Is it insightful?
    llm_quality_score = _score_with_llm(persona_description, gold_persona)
    print(f"  -> LLM judge score: {llm_quality_score:.3f}")
    
    return llm_quality_score


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
    for i in range(len(testset)):
        test_instance = testset[i]
        if test_instance.interactions is None:
            print(f"Warning: test_instance.interactions is None for user {test_instance.user_id}")
            product_list_str = ""
        else:
            product_list_str = _build_product_list_str(test_instance.interactions, "purchase")
        print(gepa_prompt.format(product_list_str=product_list_str))
        print(base_prompt.format(product_list_str=product_list_str))
        print("--------------------------------")



        final_prompt = gepa_prompt.format(product_list_str=product_list_str)
        base_prompt_final = base_prompt.format(product_list_str=product_list_str)

        gepa_response_message = persona_model([{"role": "user", "content": final_prompt}])
        base_response_message = persona_model([{"role": "user", "content": base_prompt_final}])

        gepa_raw_output = gepa_response_message.content or ""
        base_raw_output = base_response_message.content or ""
        gepa_persona_description = _parse_persona_description(gepa_raw_output)
        base_persona_description = _parse_persona_description(base_raw_output)
        # print(gepa_persona_description)
        # print(base_persona_description)
        # print("--------------------------------")

        # print("Scoring GEPA persona...")
        # gepa_score = _score_persona(gepa_persona_description, test_instance.gold_persona)
        # print(f"GEPA Score: {gepa_score}")

        # print("Scoring base persona...")
        # base_score = _score_persona(base_persona_description, test_instance.gold_persona)
        # print(f"Base Score: {base_score}")



        #save the gepa_prompt and scores to a file in Opera/data
        # Line 206-211: Fix both file writes
        with open(f"data/gepa_prompt_{i}.txt", "w", encoding="utf-8") as f:
            f.write(gepa_raw_output)
            # f.write(f"\nGEPA Score: {gepa_score}")
        with open(f"data/base_prompt_{i}.txt", "w", encoding="utf-8") as f:
            f.write(base_raw_output)
            # f.write(f"\nBase Score: {base_score}")
                    