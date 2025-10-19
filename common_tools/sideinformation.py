# from typing import (
#     str
# )
import json
import re
def get_side_info(string: str) -> str:
   
    return _format_to_json(string)
    

def load_side_info_from_metadata(file_path="gaia/metadata.jsonl"):
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        if line:
            return json.loads(line)
    return None


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