import os
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from persona_agent.persona_agent import PersonaAgent

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Orchestrator import run_orchestrator_task
from common_tools.goal_generator import GoalGenerator
from common_tools.logger import Logger
from knowledge_base.models import UserProducts, PurchasedProduct
from smolagents import OpenAIServerModel

# Explicitly load .env from the current directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

def load_m2_sample(file_path: str) -> Dict[str, Any]:
    """
    Loads a sample from the Amazon M2 dataset (mock implementation for now).
    """
    # TODO: Replace with actual M2 dataset loading logic
    # For now, returning a mock structure based on the hardcoded example in Orchestrator
    return {
        "user_id": "mock_user",
        "history": {
             "B0DBZ3JYRF": {
                "title": "Airkeep Car Air Freshener - White Jasmine",
                "price": "$5.99",
                "asin": "B0DBZ3JYRF"
            },
            # ... (Add more mock items if needed)
        },
        "ground_truth_next_item": {
            "asin": "B0CJTWQNPN",
            "title": "Philips Sonicare Protective Clean 9900",
            "category": "Electric Toothbrush"
        }
    }

async def train_orchestrator(sample: Dict[str, Any], output_file: str, main_logger: Logger, timestamp: str, max_loops: int, top_k: int):
    """
    Runs the Orchestrator in training mode (SimRAG).
    """
    user_products_data = sample["history"]
    # Convert to UserProducts model
    user_products = UserProducts(products={
        asin: PurchasedProduct(**data) for asin, data in user_products_data.items()
    })
    
    ground_truth = sample["ground_truth_next_item"]
    target_category = ground_truth.get("category", "Item")
    
    # Construct a goal that points to the ground truth category
    goal = f"Find the best {target_category} for me."
    
    main_logger.log_overview(f"--- Starting Training Run ---", to_stdout=True)
    main_logger.log_overview(f"Goal: {goal}", to_stdout=True)
    main_logger.log_overview(f"Ground Truth: {ground_truth['title']} ({ground_truth['asin']})\n", to_stdout=True)
    main_logger.log_overview(f"Note: --n_goals argument is ignored in 'train_orchestrator' mode, as this mode focuses on a single ground truth goal.\n", to_stdout=True)

    # Run Orchestrator
    # Note: We need to modify Orchestrator to return the trajectory/log_dir for critique
    # For now, we assume run_orchestrator_task returns the final answer string
    # In the real implementation, we might need to capture the log directory or return the PersonaAgent instance
    
    # We pass the ground truth to the orchestrator so it can be used for critique (in a real blind setting we wouldn't, 
    # but here we are training the stopping condition, so the environment/critic needs to know)
    # Actually, the Orchestrator shouldn't know, but the *Critic* (PersonaAgent) called *after* should know.
    # But Orchestrator.py instantiates the PersonaAgent internally. 
    # To make this work without massive refactoring, we might need to pass the ground truth to run_orchestrator_task
    # and have it call the critique method internally at the end, OR return the agent/knowledge base.
    
    # Let's assume we modify run_orchestrator_task to return (final_answer, knowledge_base, log_dir)
    # For now, relying on the existing return (final_answer) and we might need to parse logs or change Orchestrator return type.
    
    # To keep it simple for this step, we will rely on the Orchestrator changes we plan to make.
    # We will pass 'ground_truth' as a special argument if we want internal critique, 
    # OR we accept that we need to change Orchestrator return signature.
    
    # Let's assume we pass ground_truth to run_orchestrator_task for the "training_mode"
    
    final_answer = await run_orchestrator_task(
        user_goal=goal,
        max_refinement_loops=max_loops,
        products_list=user_products_data,
        top_k=top_k,
        logger=main_logger
    )
    
    main_logger.log_overview(f"Final Answer: {final_answer}\n", to_stdout=True)

    # Instantiate PersonaAgent for critique
    # We need to recreate the model here as it's not returned by run_orchestrator_task
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )
    persona_agent = PersonaAgent(model=model, user_products=user_products.products, logger=main_logger)
    
    reward = persona_agent.critique_task(
        user_goal=goal,
        final_answer=final_answer,
        ground_truth=ground_truth
    )
    
    main_logger.log_overview(f"Reward: {reward}\n", to_stdout=True)

    # Save result to training data
    with open(output_file, "a") as f:
        record = {
            "goal": goal,
            "ground_truth": ground_truth,
            "final_answer": final_answer,
            "reward": reward,
            "run_timestamp": timestamp
            # "trajectory": ... # Ideally we save the path to the logs or the logs themselves
        }
        f.write(json.dumps(record) + "\n")

import contextlib
import os
from datetime import datetime

async def run_task_with_logging(run_dir: str, goal_id: int, user_goal: str, max_refinement_loops: int, products_list: dict, top_k: int, persona: str):
    """
    Wrapper to run orchestrator task with the new logger.
    """
    goal_str = f"goal_{goal_id}"
    logger = Logger(run_dir=run_dir, goal_id=goal_str)
    
    logger.log_overview(f"Starting Goal {goal_id}: {user_goal}\n", to_stdout=True)
    
    return await run_orchestrator_task(
        user_goal=user_goal,
        max_refinement_loops=max_refinement_loops,
        products_list=products_list,
        top_k=top_k,
        task_id=goal_str,
        persona=persona,
        logger=logger
    )

async def run_parallel_inference(sample: Dict[str, Any], output_file: str, main_logger: Logger, timestamp: str, n_goals: int = 33, max_loops: int = 3, top_k: int = 3):
    """
    Runs parallel inference for generated goals.
    """
    user_products_data = sample["history"]
    user_products = UserProducts(products={
        asin: PurchasedProduct(**data) for asin, data in user_products_data.items()
    })
    
    # Initialize model
    model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )
    
    # Use the main_logger passed from the main function
    run_dir = main_logger.run_dir
    
    # Infer Persona ONCE
    # We create a temporary PersonaAgent just to infer the persona
    # The logger passed to this agent is only used for the duration of persona inference
    temp_agent = PersonaAgent(model=model, user_products=user_products.products, logger=main_logger)
    persona = temp_agent.personality_prompt
    
    # Generate Goals using the inferred persona
    main_logger.log_overview(f"--- Generating {n_goals} Goals ---", to_stdout=True)
    
    generator = GoalGenerator(model)
    goals = generator.generate_goals(user_products, persona, n=n_goals)
    
    tasks = []
    for i, goal in enumerate(goals):
        goal_id = i + 1
        tasks.append(
            run_task_with_logging(
                run_dir=run_dir,
                goal_id=goal_id,
                user_goal=goal,
                max_refinement_loops=max_loops,
                products_list=user_products_data,
                top_k=top_k,
                persona=persona,
            )
        )
    
    results = await asyncio.gather(*tasks)
    
    # Save results
    with open(output_file, "a") as f:
        for i, (goal, result) in enumerate(zip(goals, results)):
            record = {
                "goal": goal,
                "final_answer": result, # This will be a list of 3 items
                "task_id": f"goal_{i+1}",
                "run_timestamp": timestamp
            }
            f.write(json.dumps(record) + "\n")
    
    main_logger.log_overview(f"--- Parallel Inference Complete. Results saved to {output_file} ---", to_stdout=True)

async def main():
    parser = argparse.ArgumentParser(description="SimRAG Driver - A tool for training and running multi-agent orchestrators.")
    parser.add_argument(
        "--mode", 
        choices=["train_orchestrator", "generate_goals", "parallel_inference"], 
        default="train_orchestrator",
        help="Sets the operational mode of the driver. 'train_orchestrator' trains the agent, 'generate_goals' creates new research goals, and 'parallel_inference' runs multiple research goals concurrently."
    )
    parser.add_argument(
        "--output", 
        default="simrag_training_data.jsonl",
        help="Specifies the output file path where results (e.g., training data, inference results) will be saved."
    )
    parser.add_argument(
        "--n_goals", 
        type=int, 
        default=5,
        help="Defines the number of research goals to generate or run in parallel inference mode. Defaults to 5."
    )
    parser.add_argument(
        "--max_loops", 
        type=int, 
        default=3,
        help="Sets the maximum number of refinement loops the orchestrator will perform for each goal before making a final decision. Defaults to 3."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Specifies the number of top entries to choose as the final answer. Defaults to 3."
    )
    args = parser.parse_args()

    # Generate timestamp for this run and create the main run directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = f"logs/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create a main logger for initial logging and passing to sub-functions
    main_logger = Logger(run_dir)
    main_logger.log_overview(f"Running simrag_driver with arguments: {args}\n", to_stdout=True)

    # Mock sample load
    sample = load_m2_sample("path/to/mock/file")

    if args.mode == "train_orchestrator":
        await train_orchestrator(sample, args.output, main_logger, timestamp, args.max_loops, args.top_k)
    elif args.mode == "parallel_inference":
        await run_parallel_inference(sample, args.output, main_logger, timestamp, args.n_goals, args.max_loops, args.top_k)
    elif args.mode == "generate_goals":
        main_logger.log_overview("--- Testing Goal Generation ---", to_stdout=True)
        user_products_data = sample["history"]
        user_products = UserProducts(products={
            asin: PurchasedProduct(**data) for asin, data in user_products_data.items()
        })
        
        # Initialize model
        model = OpenAIServerModel(
            model_id="gemma3",
            api_base="https://ellm.nrp-nautilus.io/v1",
            api_key=os.getenv("NAUT_API_KEY"),
        )
        
        # Initialize Generator
        generator = GoalGenerator(model)
        
        # Mock Persona (since we don't have the full PersonaAgent here, just passing a string)
        # In a real scenario, we'd infer it or load it.
        mock_persona = "The user is a practical, budget-conscious shopper who values functionality."
        
        goals = generator.generate_goals(user_products, mock_persona, n=5)
        main_logger.log_overview(f"Generated {len(goals)} goals:\n", to_stdout=True)
        for i, g in enumerate(goals):
            main_logger.log_overview(f"{i+1}. {g}", to_stdout=True)

if __name__ == "__main__":
    asyncio.run(main())
