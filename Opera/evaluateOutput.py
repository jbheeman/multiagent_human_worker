"""
This script is used to evaluate the output of the orchestrator.
For a user-ID we can get their persona information, purchases, maybe even color preferences or other information.
The Orchestrator has also had this information and has chosen an item to recommend.
We want to evaluate if the recommended item is appropriate for the user based on their persona information and purchases.

We can use a Judge 
The Judgemental Agent will be given the persona information, purchases, and the recommended item.
It will then evaluate if the recommended item is appropriate for the user based on their persona information and purchases.
It will then return a score between 0 and 100 indicating how appropriate the recommended item is.
It will also return a justification for the score.


Ideally the judge LLM can think about the user's persona like if the system reccomended a 10000 dollar item and is a phd student with a family of 4 and 2 kids, it might not be appropriate.
Would the judge need to be a multi step LLM? Or can it be a single step LLM?

"""

import pandas as pd
import json
import os
from smolagents import OpenAIServerModel
# Cache for dataframes to avoid reloading
_user_df_cache = None
_session_df_cache = None
_action_df_cache = None

def clear_cache():
    """Clear the cached dataframes to force reloading."""
    global _user_df_cache, _session_df_cache, _action_df_cache
    _user_df_cache = None
    _session_df_cache = None
    _action_df_cache = None
    survey_df_cache = None
    print("Cache cleared. Dataframes will be reloaded on next access.")


def _load_dataframes():
    """Lazy load dataframes with caching."""
    global _user_df_cache, _session_df_cache, _action_df_cache
    if _user_df_cache is None:
        print("Loading user dataframe...")
        _user_df_cache = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/user/train/train.parquet")
    if _session_df_cache is None:
        print("Loading session dataframe...")
        _session_df_cache = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/session/train/train.parquet")
    if _action_df_cache is None:
        print("Loading action dataframe...")
        _action_df_cache = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_filtered/action/train/train.parquet")


    return _user_df_cache, _session_df_cache, _action_df_cache


def load_persona_info_for_user(user_id: str, user_df: pd.DataFrame) -> str:
    """
    Load the persona information (interview transcript) for a specific user.
    """
    user_row = user_df[user_df["user_id"] == user_id]
    if len(user_row) == 0:
        # Show available user IDs for debugging
        available_users = user_df["user_id"].head(5).tolist()
        raise ValueError(f"User {user_id} not found. Available users (first 5): {available_users}")
    
    persona_info = user_row["interview_transcript_processed"].values[0]
    
    # Check if persona is empty/null
    if pd.isna(persona_info) or persona_info is None:
        return None
    
    persona_str = str(persona_info).strip()
    if not persona_str:
        return None
    
    return persona_str


def load_survey_info_for_user(user_id: str, user_df: pd.DataFrame) -> dict:
    """
    Load the survey information for a specific user.
    Returns a dictionary with survey data, or None if not available.
    """
    user_row = user_df[user_df["user_id"] == user_id]
    if len(user_row) == 0:
        return None
    
    # Check if survey column exists
    if "survey" not in user_row.columns:
        return None
    
    survey_info = user_row["survey"].values[0]
    
    # Check if survey is empty/null
    if pd.isna(survey_info) or survey_info is None:
        return None
    
    # Try to parse as JSON if it's a string
    if isinstance(survey_info, str):
        try:
            survey_dict = json.loads(survey_info)
            return survey_dict
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, return as string wrapped in dict
            return {"raw_survey": survey_info}
    
    # If it's already a dict, return as is
    if isinstance(survey_info, dict):
        return survey_info
    
    return None


def format_survey_for_prompt(survey: dict) -> str:
    """
    Format survey dictionary into a readable string for the prompt.
    """
    if not survey:
        return ""
    
    lines = []
    lines.append("### Survey Information")
    
    # Demographic Information
    if "Demographic Information" in survey:
        demos = survey["Demographic Information"]
        lines.append("\n**Demographics:**")
        for key, value in demos.items():
            lines.append(f"- {key}: {value}")
    
    # Self Description
    if "Self Description" in survey:
        self_desc = survey["Self Description"]
        lines.append("\n**Self Description:**")
        if isinstance(self_desc, dict):
            for key, value in self_desc.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append(f"- {self_desc}")
    
    # Shopping Preferences
    if "Shopping Preference" in survey:
        shop_pref = survey["Shopping Preference"]
        lines.append("\n**Shopping Preferences:**")
        if isinstance(shop_pref, dict):
            for key, value in shop_pref.items():
                if isinstance(value, dict):
                    lines.append(f"- {key}:")
                    for sub_key, sub_value in value.items():
                        lines.append(f"  - {sub_key}: {sub_value}")
                else:
                    lines.append(f"- {key}: {value}")
        else:
            lines.append(f"- {shop_pref}")
    
    # Personality
    if "Personality" in survey:
        personality = survey["Personality"]
        lines.append("\n**Personality:**")
        if isinstance(personality, dict):
            for key, value in personality.items():
                if isinstance(value, dict):
                    lines.append(f"- {key}:")
                    for sub_key, sub_value in value.items():
                        lines.append(f"  - {sub_key}: {sub_value}")
                else:
                    lines.append(f"- {key}: {value}")
        else:
            lines.append(f"- {personality}")
    
    # Handle raw_survey if present
    if "raw_survey" in survey:
        lines.append("\n**Survey Data:**")
        lines.append(str(survey["raw_survey"]))
    
    return "\n".join(lines)








def _extract_products_vectorized(products_series):
    """Vectorized extraction of products from JSON strings."""
    def parse_single(prod_str):
        if pd.isna(prod_str):
            return []
        try:
            products = json.loads(prod_str)
            return products if isinstance(products, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    
    return products_series.apply(parse_single)



def load_purchases_info_for_user(user_id: str, session_df: pd.DataFrame, action_df: pd.DataFrame) -> str:
    """
    Load the purchases for a specific user.
    """
    
    # Get user sessions
    user_sessions = session_df[session_df["user_id"] == user_id]["session_id"].tolist()
    if not user_sessions:
        return "No sessions found for this user."

    # Filter actions for this user's sessions and purchase clicks (more efficient)
    purchase_actions = action_df[
        (action_df["session_id"].isin(user_sessions)) &
        (action_df["action_type"] == "click") &
        (action_df["click_type"] == "purchase")
    ].copy()

    if len(purchase_actions) == 0:
        return "No purchases found for this user."

    # Vectorized product extraction
    products_list = _extract_products_vectorized(purchase_actions["products"])
    
    # Flatten products and create records using list comprehension (faster than nested loops)
    purchased_items = [
        {
            "session_id": purchase_actions.loc[idx, "session_id"],
            "asin": p.get("asin"),
            "title": p.get("title"),
            "price": p.get("price"),
            "options": p.get("options")
        }
        for idx, products in products_list.items()
        for p in products
    ]

    purchases_df = pd.DataFrame(purchased_items)
    
    if len(purchases_df) == 0:
        return "No valid purchase data found."

    # Convert to string
    purchases_text = str(purchases_df.to_string(index=False))
    return purchases_text

    

def judge_recommendation(persona_info: str = None, purchases: str = None, recommendation: str = "", survey_info: dict = None) -> str:
    """
    Judge the recommendation for a specific user.
    
    Args:
        persona_info: Interview transcript/persona (optional)
        purchases: Past purchases information (optional)
        recommendation: The recommended item to evaluate
        survey_info: Survey data dictionary (optional)
    """

    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )
    
    # Build the prompt sections dynamically based on available data
    prompt_sections = []
    
    prompt_sections.append("""# [System]
# You are an impartial evaluation agent. Your job is to judge whether a recommended shopping item is appropriate for a specific user, given available information about:
# 1) the user's persona/survey information and shopping preferences,
# 2) the items the user actually bought in past sessions (if available).

# You must reason about fit, not popularity. Consider price sensitivity, product domain (bike gear vs pet food), brand/official-store preference, and practicality for the user's described lifestyle.

# Return a JSON object with:
# - "score": integer from 0 to 100 (higher = better match)
# - "reasoning": short explanation (3–6 sentences)
# - "penalties": list of strings describing mismatches (empty list if none)

# Be strict: if the item is off-domain, overpriced for the persona, or ignores past behavior, lower the score.

# [User]""")
    
    # Add persona section if available
    if persona_info:
        prompt_sections.append(f"""
# Persona/Interview Information:
{persona_info}""")
    
    # Add survey section if available
    if survey_info:
        survey_formatted = format_survey_for_prompt(survey_info)
        if survey_formatted:
            prompt_sections.append(f"""
# Survey Information:
{survey_formatted}""")
    
    # Add purchases section if available
    if purchases and purchases.strip() and not purchases.startswith("No"):
        prompt_sections.append(f"""
# Past purchases (most recent first):
{purchases}""")
    else:
        prompt_sections.append("""
# Past purchases: No purchase history available for this user.""")
    
    # Add recommendation
    prompt_sections.append(f"""
# Model's recommended item:
{recommendation}

# Now evaluate.""")
    
    # Combine all sections
    prompt = "\n".join(prompt_sections)
    messages = [{"role": "user", "content": prompt}]

    #call LLM here 
    response = model(messages)
    print(response.content)
    return response.content




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load user persona and purchases from OPeRA")
    parser.add_argument("--user-id", type=str, default="85aeec61-2d4e-489d-93bf-76c928d2d795", 
                       help="User ID to load")
    parser.add_argument("--show-cache", action="store_true", 
                       help="Show cache size information")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear cached dataframes before loading")
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
    
    user_id = args.user_id
    user_df, session_df, action_df = _load_dataframes()
    
    # Show cache info if requested
    if args.show_cache:
        get_cache_size_info()
    
    try:
        persona_info = load_persona_info_for_user(user_id, user_df)
        print("--------------------------------")
        print("Persona info:")
        if persona_info:
            print(persona_info)
        else:
            print("[No persona/interview information available]")
        print("--------------------------------")
    except ValueError as e:
        print(f"ERROR loading persona: {e}")
        persona_info = None
    
    # Load survey information
    survey_info = load_survey_info_for_user(user_id, user_df)
    if survey_info:
        print("--------------------------------")
        print("Survey info loaded:")
        print(f"Survey keys: {list(survey_info.keys())}")
        print("--------------------------------")
    else:
        print("--------------------------------")
        print("Survey info: [No survey information available]")
        print("--------------------------------")
    
    purchases_info = load_purchases_info_for_user(user_id, session_df, action_df)
    print("Purchases info:")
    print(purchases_info)
    print("--------------------------------")

    recommendation = "Green Pea Protein Powder"

    judge_result = judge_recommendation(
        persona_info=persona_info,
        purchases=purchases_info,
        recommendation=recommendation,
        survey_info=survey_info
    )

    print("--------------------------------")
    print("Judge recommendation result:")
    print(judge_result)
    print("--------------------------------")



# [System]
# You are an impartial evaluation agent. Your job is to judge whether a recommended shopping item is appropriate for a specific user, given:
# 1) the user's persona and shopping preferences,
# 2) the items the user actually bought in past sessions.

# You must reason about fit, not popularity. Consider price sensitivity, product domain (bike gear vs pet food), brand/official-store preference, and practicality for the user's described lifestyle.

# Return a JSON object with:
# - "score": integer from 0 to 100 (higher = better match)
# - "reasoning": short explanation (3–6 sentences)
# - "penalties": list of strings describing mismatches (empty list if none)

# Be strict: if the item is off-domain, overpriced for the persona, or ignores past behavior, lower the score.

# [User]
# Persona:
# {{persona_text}}

# Past purchases (most recent first):
# {{purchases_text}}  # e.g. "1. ROCKBROS Smart Bike Tail Light ...; 2. Bike Saddle Bag ...; 3. Primal Freeze Dried Raw Cat Food ..."
# Model's recommended item:
# {{model_recommendation}}

# Now evaluate.



###

# Demographic Information": {"Gender": "Male", "Age": "25-34 years old", "City": "Boston", "Education level": "Doctoral degree or current doctoral student (PhD, JD, MD, DDS etc.)", "Living situation": "Live together with roommate", "Yearly household income or stipend": "$75,000-$99,999", "Employment status": "Full-time employee"}, "Self Description": {"Two sentence description": "I am a research scientist in Natural Language Processing and Human-Computer Interaction. I conduct scientific research projects, mentor students, and publish academic papers. "}, "Shopping Preference": {"Online shopping frequency": "Once every couple of weeks", "Monthly online shopping spend $": "500", "Amazon Prime membership": "Yes", "To what extent do you agree with the following statements": {"I tend to shop more during holidays.": "Somewhat agree", "Online Ads attract my attention and are a good source of information.": "Strongly disagree", "I usually do a lot research (e.g. reading online reviews) before making purchase.": "Strongly agree", "I prioritize delivery speed and delivery fee of the product.": "Somewhat disagree", "Getting high-quality online products is very important for me.": "Somewhat agree", "The more expensive online product brands are usually my choice.": "Somewhat disagree", "The more I learn about online products, the harder it seems to choose the best.": "Somewhat agree", "I shop quickly for online products, buying the first product or brand I find that seems good enough.": "Strongly disagree", "Once I find a brand I like, I stick with it.": "Somewhat agree", "I would buy a new or different brand of product just to see what it is like.": "Neither agree or disagree", "I enjoy shopping for online products just for the fun of it.": "Somewhat disagree", "I look carefully to find the best value for money when shopping online.": "Strongly agree"}}, "Personality": {"MBTI personality type": "INTJ", "Big Five Scores": {"Extraversion": "Low", "Agreeableness": "Low", "Conscientiousness": "Extremely high", "Emotional Stability": "High", "Intellect": "High"}}}
###