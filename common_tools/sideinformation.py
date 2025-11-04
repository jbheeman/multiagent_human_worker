from typing import Optional
import json
import re
import pandas as pd

# Cache for user dataframe to avoid reloading
_user_df_cache = None

def get_side_info(string: str) -> str:
   
    return _format_to_json(string)
    


def load_side_info_for_user(user_id: str, parquet_path: str = "hf://datasets/NEU-HAI/OPeRA/OPeRA_full/user/train/train.parquet") -> Optional[str]:
    """
    Load persona information for a specific user from the OPeRA dataset.
    
    Args:
        user_id: The user ID to look up
        parquet_path: Path to the OPeRA user parquet file (default uses HuggingFace datasets)
        
    Returns:
        The persona/interview transcript as a string, or None if user not found
    """
    global _user_df_cache
    
    # Load user dataframe (cache it to avoid reloading)
    if _user_df_cache is None:
        _user_df_cache = pd.read_parquet(parquet_path)
    
    # Find the user
    user_row = _user_df_cache[_user_df_cache["user_id"] == user_id]
    
    if len(user_row) == 0:
        return None
    
    # Extract the persona/interview transcript
    persona_info = user_row["interview_transcript_processed"].values[0]
    
    # Return as string (handle NaN/None cases)
    if pd.isna(persona_info):
        return None
    
    return str(persona_info)


def _format_to_json(string)->str:

    formatted_json = {}
    formatted_json["steps"] = []

    line = re.search(r"{(.*)}", string, re.DOTALL)
    if line:
        stripped_string = line.group(1)
        print(f"Stripped string: {stripped_string}")
    else:
        raise ValueError("No JSON found in the string")

    # Parse the JSON to get the Steps value
    try:
        json_data = json.loads("{" + stripped_string + "}")
        steps_text = json_data.get("Steps", "")
        print(f"Steps text: {steps_text}")
        
        # Split by numbered steps (1., 2., 3., etc.)
        steps = re.split(r'\d+\.', steps_text)
        # Remove empty strings and clean up
        formatted_json["steps"] = [step.strip() for step in steps if step.strip()]
        
    except json.JSONDecodeError:
        # Fallback: try to extract steps manually if JSON parsing fails
        steps_text = re.search(r'"Steps":\s*"([^"]*(?:\\.[^"]*)*)"', stripped_string)
        if steps_text:
            steps_content = steps_text.group(1)
            steps = re.split(r'\d+\.', steps_content)
            formatted_json["steps"] = [step.strip() for step in steps if step.strip()]
    
    print(f"Formatted JSON: {formatted_json}")
    return formatted_json


# if __name__ == "__main__":

#     input_string = """Annotator Metadata: {"Steps": "1. Go to arxiv.org and navigate to the Advanced Search page.\n2. Enter \"AI regulation\" in the search box and select \"All fields\" from the dropdown.\n3. Enter 2022-06-01 and 2022-07-01 into the date inputs, select \"Submission date (original)\", and submit the search.\n4. Go through the search results to find the article that has a figure with three axes and labels on each end of the axes, titled \"Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation\".\n5. Note the six words used as labels: deontological, egalitarian, localized, standardized, utilitarian, and consequential.\n6. Go back to arxiv.org\n7. Find \"Physics and Society\" and go to the page for the \"Physics and Society\" category.\n8. Note that the tag for this category is \"physics.soc-ph\".\n9. Go to the Advanced Search page.\n10. Enter \"physics.soc-ph\" in the search box and select \"All fields\" from the dropdown.\n11. Enter 2016-08-11 and 2016-08-12 into the date inputs, select \"Submission date (original)\", and submit the search.\n12. Search for instances of the six words in the results to find the paper titled \"Phase transition from egalitarian to hierarchical societies driven by competition between cognitive and social constraints\", indicating that \"egalitarian\" is the correct answer."}"""
    
    
#     get_side_info(input_string)