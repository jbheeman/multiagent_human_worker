import os
import json
from pathlib import Path
import sys
import shutil
import asyncio

from dotenv import load_dotenv
load_dotenv()
import json

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    WebSearchTool,
    WikipediaSearchTool,
    AgentMemory,
    tool, # Added tool import
)

import yaml


# Tools
from websurfer_agent.web_surfer_tool import WebSurferTool
from file_surfer_agent.file_surfer_tool import FileSurferTool
from coder_agent.coder_tool import CoderTool
from common_tools.llm_chat_tool import create_llm_chat_tool
from common_tools.logger import Logger
from common_tools.read_links_tool import read_links

from knowledge_base.models import KnowledgeBase, ProductKnowledge, PurchasedProduct
from persona_agent.persona_agent import PersonaAgent


async def run_orchestrator_task(user_goal, max_refinement_loops: int = 3):
    """
    Run the orchestrator task with a given user goal and optional side information.
    
    Args:
        user_goal (str): The user's goal/question to solve
        max_refinement_loops (int): The maximum number of refinement loops to perform.
    
    Returns:
        The final answer from the PersonaAgent.
    """
    user_products_json = """
{
    "B0DBZ3JYRF": {
        "title": "Airkeep Car Air Freshener - White Jasmine Handmade Scented Ceramic for Drawers and Closets, Car Air Freshener Gifts for Men Women Car Fragrance",
        "price": "$5.99",
        "asin": "B0DBZ3JYRF"
    },
    "B0CYZKP9CF": {
        "title": "Airkeep Car Air Freshener - White Tea Handmade Scented Ceramic for Drawers and Closets, Car Air Freshener Gifts for Men Women Car Fragrance",
        "price": "$4.99",
        "asin": "B0CYZKP9CF"
    },
    "B0CMQZ2HFJ": {
        "title": "Airkeep Car Air Freshener - Ocean Mist&Sea Salt Handmade Scented Ceramic for Drawers and Closets,Air Freshener Gifts for Men Women Car Fragrance",
        "price": "$5.99",
        "asin": "B0CMQZ2HFJ"
    },
    "B0D418HHPL": {
        "title": "Airkeep Car Air Freshener -Ocean Mist&Sea Salt Handmade Scented Ceramic for Drawers and Closets, Air Freshener Gifts for Men Women Car Fragrance",
        "price": "$3.99",
        "asin": "B0D418HHPL"
    },
    "B0DJFQB41P": {
        "title": "Lumiere & Co. Bike Seat Bag, Bike Saddle Bag, Saddle Bag Bicycle, Bicycle Saddle Bag, Bike Bag Under Seat, Saddle Bag, Bike Saddle Bags, Small Bike Bag - Maximus",
        "price": "$27.32",
        "asin": "B0DJFQB41P"
    },
    "B0DH7PNPPM": {
        "title": "Lumiere & Co. Maximus Large Magnetic Bike Seat Bag – Quick-Attach Saddle Bag with AquaGuard Zippers, Secure Under-Seat Storage for Cycling Gear & Accessories",
        "price": "$26.45",
        "asin": "B0DH7PNPPM"
    },
    "B07XRBL9QR": {
        "title": "Shimano GRX BL-RX810 1 x 11-Speed Left Drop-Bar Hydraulic Brake Lever without hose or caliper",
        "price": "$171.00",
        "asin": "B07XRBL9QR"
    },
    "B07HY3WF49": {
        "title": "Fizik Performance Bicycle Bar Tape - Soft, Tacky & Classic Professional Bike Handlebar Tape (2mm, 2.7mm, 3mm)",
        "price": "$40.31",
        "asin": "B07HY3WF49"
    },
    "B01MY8Y0S4": {
        "title": "Sram Apex 1 Crankset (165 mm, Black)",
        "price": "$109.00",
        "asin": "B01MY8Y0S4"
    },
    "B0BT9K54D6": {
        "title": "ROCKBROS Bike Water Bottle Cage Holder Lightweight Alloy Aluminum Bicycle Water Bottle Holder Cages Secure Hold Brackets",
        "price": "$14.39",
        "asin": "B0BT9K54D6"
    },
    "B0BCK29WBT": {
        "title": "ROCKBROS Smart Bike Tail Light for Night Riding Brake Sensing Bicycle Rear Lights USB Rechargeable IPX6 Waterproof Bright LED Bike Light 260mah Cycling Safety Road BikeTaillight Accessories 4 Modes",
        "price": "$15.99",
        "asin": "B0BCK29WBT"
    },
    "B0BP69X89N": {
        "title": "Lumiere & Co. Road Bike Saddle Bag, Bike Bag, Mountain Bike Saddle Bag, Small Bike Seat Bags, Bike Under Seat Bag, Bicycle accessories, Bike Bags, Bike Seat Bag,",
        "price": "$19.56",
        "asin": "B0BP69X89N"
    },
    "B0DH77HWMC": {
        "title": "Lumiere & Co. Road Bike Saddle Bag, Bike Bag, Mountain Bike Saddle Bag, Small Bike Seat Bags, Bike Under Seat Bag, Bicycle accessories, Bike Bags, Bike Seat Bag,",
        "price": "$18.86",
        "asin": "B0DH77HWMC"
    },
    "B0BP746BTN": {
        "title": "Lumiere & Co. Road Bike Saddle Bag, Bike Bag, Mountain Bike Saddle Bag, Small Bike Seat Bags, Bike Under Seat Bag, Bicycle accessories, Bike Bags, Bike Seat Bag,",
        "price": "$18.34",
        "asin": "B0BP746BTN"
    },
    "B0BP6Q2X3X": {
        "title": "Lumiere & Co. Road Bike Saddle Bag, Bike Bag, Mountain Bike Saddle Bag, Small Bike Seat Bags, Bike Under Seat Bag, Bicycle accessories, Bike Bags, Bike Seat Bag,",
        "price": "",
        "asin": "B0BP6Q2X3X"
    },
    "B0CLKN4WZF": {
        "title": "Lumiere & Co. Road Bike Saddle Bag, Bike Bag, Mountain Bike Saddle Bag, Small Bike Seat Bags, Bike Under Seat Bag, Bicycle accessories, Bike Bags, Bike Seat Bag,",
        "price": "$19.25",
        "asin": "B0CLKN4WZF"
    },
    "B0DJFQ5L13": {
        "title": "Lumiere & Co. Bike Seat Bag, Bike Saddle Bag, Saddle Bag Bicycle, Bicycle Saddle Bag, Bike Bag Under Seat, Saddle Bag, Bike Saddle Bags, Small Bike Bag - Maximus",
        "price": "$31.32",
        "asin": "B0DJFQ5L13"
    },
    "B0DJFQMSB4": {
        "title": "Lumiere & Co. Bike Seat Bag, Bike Saddle Bag, Saddle Bag Bicycle, Bicycle Saddle Bag, Bike Bag Under Seat, Saddle Bag, Bike Saddle Bags, Small Bike Bag - Maximus",
        "price": "$31.32",
        "asin": "B0DJFQMSB4"
    },
    "B0DJFQ136X": {
        "title": "Lumiere & Co. Bike Seat Bag, Bike Saddle Bag, Saddle Bag Bicycle, Bicycle Saddle Bag, Bike Bag Under Seat, Saddle Bag, Bike Saddle Bags, Small Bike Bag - Maximus",
        "price": "$27.32",
        "asin": "B0DJFQ136X"
    },
    "B0DJFPW1QY": {
        "title": "Lumiere & Co. Bike Seat Bag, Bike Saddle Bag, Saddle Bag Bicycle, Bicycle Saddle Bag, Bike Bag Under Seat, Saddle Bag, Bike Saddle Bags, Small Bike Bag - Maximus",
        "price": "$31.32",
        "asin": "B0DJFPW1QY"
    },
    "B0BKZKL1F7": {
        "title": "Shappy Balcony Net for Pets 16.5 ft x 2.5 ft Anti Fall Cat Dogs Netting Balcony Mesh Fence Net Screen Protection Crib Mesh for Pets Patios Stairway Clear Stair Banister Guard Apartment(Black)",
        "price": "$19.99",
        "asin": "B0BKZKL1F7"
    },
    "B09MJVT8FG": {
        "title": "Shappy Balcony Net for Pets 16.5 ft x 2.5 ft Anti Fall Cat Dogs Netting Balcony Mesh Fence Net Screen Protection Crib Mesh for Pets Patios Stairway Clear Stair Banister Guard Apartment(White)",
        "price": "$18.99",
        "asin": "B09MJVT8FG"
    },
    "B0CJTWQNPN": {
        "title": "Philips Sonicare Protective Clean 9900 Rechargeable Electric Power Toothbrush, Charging Travel Case with USB Charging, Soft Brush Head, Midnight with Accessories",
        "price": "$279.99",
        "asin": "B0CJTWQNPN"
    }
}
    """
    user_products_data = json.loads(user_products_json)
    user_products = {asin: PurchasedProduct(**data) for asin, data in user_products_data.items() if data.get("asin")}

    logger = Logger()
    logger.log_overview(f"--- 🚀 Initializing Orchestrator with Dynamic Task Loop ---")
    logger.log_overview(f"Logs for this run will be saved in: {logger.run_dir}")

    # Create a workspace directory for this run inside the logs directory
    work_dir = os.path.abspath(os.path.join(logger.run_dir, "workspace"))
    os.makedirs(work_dir, exist_ok=True)
    workspace_source_dir = "workspace"
    if not os.path.exists(workspace_source_dir):
        os.makedirs(workspace_source_dir)
    if os.path.exists(workspace_source_dir):
        shutil.copytree(workspace_source_dir, work_dir, dirs_exist_ok=True)
    logger.log_overview(f"Working directory for this run: {work_dir}")

    orchestrator_prompt = yaml.safe_load(open("orchestrator_agent.yaml").read())

    # 1. Define models for different roles
    gemma_model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )
    planning_model = OpenAIServerModel( # Still used for persona agent
        model_id="qwen3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

    # 2. Initialize Tools
    web_surfer_tool = WebSurferTool(model=gemma_model)
    file_surfer_tool = FileSurferTool(model=gemma_model, base_path=work_dir)
    coder_tool = CoderTool(model=gemma_model, work_dir=Path(work_dir))
    llm_tool = create_llm_chat_tool(model=gemma_model)

    # 3. Initialize Agents and KnowledgeBase
    persona_agent = PersonaAgent(model=gemma_model, user_products=user_products, logger=logger)
    knowledge_base = KnowledgeBase()
    
    # The ManagerAgent is now stateless and created per-loop, but we define its instructions here
    manager_instructions = orchestrator_prompt.get("instructions", "")
    personality = persona_agent.personality_prompt
    manager_instructions = (
        f"**Persona Directive**\n"
        f"You are to embody the following persona. All of your actions, decisions, and responses must be guided by this persona's traits and perspective.\n\n"
        f"**Persona Profile:**\n{personality}\n\n"
        f"---\n\n"
        f"{manager_instructions}"
    )

    # --- Start of the Dynamic Task Loop ---
    logger.log_overview(f"\n🎯 User Goal: {user_goal}")

    # Formulate the initial research task using the PersonaAgent
    initial_chat_message = await asyncio.to_thread(persona_agent.initial_task, user_goal)
    current_research_task = initial_chat_message.content
    last_refinement_decision = None
    
    for i in range(max_refinement_loops):
        logger.log_overview("\n" + "=" * 50)
        logger.log_overview(f"▶️ Starting Refinement Loop {i+1}/{max_refinement_loops}")
        logger.log_overview(f"🔎 Current Research Task: {current_research_task}")
        logger.log_overview("=" * 50)

        # 1. MANAGER ACTION
        # Define the tool to ask the persona agent questions for this loop iteration
        @tool
        def ask_persona_agent(question: str, context: str = "") -> str:
            """
            Ask a clarifying question to the persona agent to get more information or guidance on the task.

            Args:
                question: The specific question to ask the persona agent.
                context: You must explicitly pass any relevant information, such as the output of your previous
                         tool calls, in this parameter. The persona agent does not have access to your history.
            """
            logger.log_overview("--- Manager Agent asking Persona Agent ---")
            logger.log_overview(f"Question: {question}")
            logger.log_overview(f"Context: {context}")
            logger.log_overview("-----------------------------------------")
            
            # Also write to the manager's specific log file
            print(f"Question to PersonaAgent:\n{question}\nContext:\n{context}", file=log_file)
            
            return persona_agent.answer_question(question, context, knowledge_base, last_refinement_decision)

        manager_agent = ToolCallingAgent(
            tools=[llm_tool, WebSearchTool(), WikipediaSearchTool(), read_links, ask_persona_agent],
            model=gemma_model,
            instructions=manager_instructions,
        )
        log_file = logger.get_log_file(f"ManagerAgent_Loop{i+1}")
        old_stdout = sys.stdout
        sys.stdout = log_file
        # Run synchronous manager agent in a separate thread to not block the event loop
        manager_result = await asyncio.to_thread(manager_agent.run, current_research_task)
        sys.stdout = old_stdout
        log_file.close()
        logger.log_overview(f"📝 Manager Result: {str(manager_result)[:500]}...")

        # 2. KNOWLEDGE EXTRACTION
        logger.log_overview("🧠 PersonaAgent: Identifying products from manager results...")
        product_names = persona_agent.identify_products_from_text(manager_result)
        
        newly_added_products = []
        for name in product_names:
            if name not in knowledge_base.products:
                knowledge_base.products[name] = ProductKnowledge()
                newly_added_products.append(name)
        
        if newly_added_products:
            logger.log_overview(f"➕ Added new products to KnowledgeBase: {newly_added_products}")

        # Concurrently update knowledge for all identified products
        update_tasks = []
        for name in product_names:
            task = persona_agent.update_knowledge(
                product_name=name,
                existing_knowledge=knowledge_base.products[name],
                new_info=manager_result
            )
            update_tasks.append(task)
        
        updated_knowledge_list = await asyncio.gather(*update_tasks)

        # The `update_knowledge` method modifies the objects in-place, so no further action is needed here.
        
        # 3. STRATEGIC DIRECTION
        log_file = logger.get_log_file(f"PersonaAgent_Refine_Loop{i+1}")
        old_stdout = sys.stdout
        sys.stdout = log_file
        refinement_decision = persona_agent.refine(knowledge_base, user_goal, last_decision=last_refinement_decision)
        sys.stdout = old_stdout
        log_file.close()
        logger.log_overview(f"🤔 Persona Refinement Decision: {refinement_decision.thought}")

        # 4. STATE UPDATE
        last_refinement_decision = refinement_decision
        knowledge_base.thought_history.append(refinement_decision.thought)
        if refinement_decision.options_to_prune:
            for option_name in refinement_decision.options_to_prune:
                if option_name in knowledge_base.products:
                    knowledge_base.products[option_name].status = "pruned"
                    logger.log_overview(f"✂️ Pruned option: {option_name}")

        # 5. LOOP OR EXIT
        if refinement_decision.status == "ready_to_choose":
            logger.log_overview("✅ Persona is ready to choose. Exiting refinement loop.")
            break
        
        current_research_task = refinement_decision.next_research_task

    # --- Final Decision Phase ---
    logger.log_overview("\n--- 🏆 Final Decision Phase ---")
    
    log_file = logger.get_log_file("PersonaAgent_Choose")
    old_stdout = sys.stdout
    sys.stdout = log_file
    final_answer = persona_agent.choose(knowledge_base)
    sys.stdout = old_stdout
    log_file.close()
    
    logger.log_overview("\n" + "="*50)
    logger.log_overview(f"🎉 Final Answer: {final_answer}")
    logger.log_overview("="*50)
    
    return final_answer

if __name__ == "__main__":
    # Example usage when running Orchestrator.py directly
    sample_user_goal = "I'm looking for a new laptop. I want something that is good for the environment, but also powerful enough for some light video editing. My budget is around $800."
    sample_max_loops = 3 # You can adjust this for testing

    print(f"Starting Orchestrator with user goal: {sample_user_goal}")
    final_recommendation = asyncio.run(run_orchestrator_task(sample_user_goal, max_refinement_loops=sample_max_loops))
    
    print("\n--- FINAL RECOMMENDATION ---")
    print(final_recommendation)
