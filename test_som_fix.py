#!/usr/bin/env python3
"""
Test script to verify the SOM marker interpretation fix.
This script simulates the same scenario that failed before.
"""

import asyncio
import os
from websurfer_agent.web_surfer_agent import WebSurferAgent
from smolagents.models import OpenAIServerModel

async def test_arxiv_pdf_download():
    """Test the arXiv PDF download scenario that previously failed."""
    
    # Initialize the model (you'll need to set your API key)
  

       # 1. Initialize the shared model
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

    
    # Create WebSurfer agent with debug directory
    debug_dir = os.path.join(os.getcwd(), "websurfer_agent", "debug_screenshots")
    agent = WebSurferAgent(
        name="TestWebSurfer",
        model=model,
        debug_dir=debug_dir,
        to_save_screenshots=True,
        max_actions_per_step=10
    )
    
    try:
        # Test the same scenario that failed before
        request = """
      

        Go to arxiv.org and navigate to the Advanced Search page.\n2. Enter \"AI regulation\" in the search box and select \"All fields\" from the dropdown.\n3. Enter 2022-06-01 and 2022-07-01 into the date inputs, select \"Submission date (original)\", and submit the search.\n4. Go through the search results to find the article that has a figure with three axes and labels on each end of the axes, titled \"Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation\".\n5. Note the six words used as labels: deontological, egalitarian, localized, standardized, utilitarian, and consequential.\n6. Go back to arxiv.org\n7. Find \"Physics and Society\" and go to the page for the \"Physics and Society\" category.\n8. Note that the tag for this category is \"physics.soc-ph\".\n9. Go to the Advanced Search page.\n10. Enter \"physics.soc-ph\" in the search box and select \"All fields\" from the dropdown.\n11. Enter 2016-08-11 and 2016-08-12 into the date inputs, select \"Submission date (original)\", and submit the search.\n12. Search for instances of the six words in the results to find the paper titled \"Phase transition from egalitarian to hierarchical societies driven by competition between cognitive and social constraints\", indicating that \"egalitarian\" is the correct answer."
        """
        
        print("Starting arXiv PDF download test...")
        print("Request:", request)
        print("\n" + "="*50)
        
        result = await agent.run(request)
        
        print("\n" + "="*50)
        print("RESULT:")
        print(result)
        
        # Check if the agent correctly identified marker 21 as the PDF download button
        # and didn't mistakenly use marker 14 (author name) or marker 3 (donate button)
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await agent.close()

if __name__ == "__main__":
    print("Testing SOM marker interpretation fix...")
    print("Make sure to set OPENAI_API_KEY environment variable")
    asyncio.run(test_arxiv_pdf_download())
