#!/usr/bin/env python3
"""
Example usage of the WebSurferAgent implemented with smol agents.

This demonstrates how to use the WebSurfer agent as a 1:1 copy of the magentic-ui WebSurfer.
"""

import asyncio
import os
from smolagents import OpenAIServerModel
from .web_surfer_agent import WebSurferAgent


async def basic_example():
    """Basic example of using WebSurferAgent."""
    print("🚀 Starting Basic WebSurfer Agent Example")
    
    # Initialize model (you'll need to set up your API key)
    model = OpenAIServerModel(
        model_id="gpt-4o",  # or your preferred model
        api_base="https://api.openai.com/v1",  # or your API base
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
    )
    
    # Create WebSurfer agent
    websurfer = WebSurferAgent(
        name="MyWebSurfer",
        model=model,
        debug_dir="./debug_screenshots",
        to_save_screenshots=True,
        max_actions_per_step=3,
        search_engine="duckduckgo"
    )
    
    try:
        # Use the agent
        result = await websurfer.run("Search for the latest news about artificial intelligence")
        print(f"✅ Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await websurfer.close()


async def advanced_example():
    """Advanced example with custom configuration."""
    print("🚀 Starting Advanced WebSurfer Agent Example")
    
    # Initialize model
    model = OpenAIServerModel(
        model_id="gpt-4o",
        api_base="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
    )
    
    # Create WebSurfer agent with custom configuration
    websurfer = WebSurferAgent(
        name="AdvancedWebSurfer",
        model=model,
        downloads_folder="./downloads",
        debug_dir="./debug_screenshots",
        start_page="https://www.google.com",
        animate_actions=True,
        to_save_screenshots=True,
        max_actions_per_step=5,
        to_resize_viewport=True,
        single_tab_mode=False,
        viewport_height=1080,
        viewport_width=1920,
        use_action_guard=False,
        search_engine="google"
    )
    
    try:
        # Multiple tasks
        tasks = [
            "Navigate to a news website",
            "Find articles about technology",
            "Summarize the main points",
            "Take a screenshot of the page"
        ]
        
        for i, task in enumerate(tasks, 1):
            print(f"📋 Task {i}: {task}")
            result = await websurfer.run(task)
            print(f"✅ Result: {result}\n")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await websurfer.close()


async def state_management_example():
    """Example showing state saving and loading."""
    print("🚀 Starting State Management Example")
    
    model = OpenAIServerModel(
        model_id="gpt-4o",
        api_base="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
    )
    
    # Create WebSurfer agent
    websurfer = WebSurferAgent(
        name="StatefulWebSurfer",
        model=model,
        debug_dir="./debug_screenshots"
    )
    
    try:
        # Perform some actions
        await websurfer.run("Navigate to a search engine")
        
        # Save state
        state = await websurfer.save_state()
        print("💾 State saved")
        
        # Create a new agent and load state
        new_websurfer = WebSurferAgent(
            name="RestoredWebSurfer",
            model=model,
            debug_dir="./debug_screenshots"
        )
        
        await new_websurfer.load_state(state)
        print("📂 State loaded")
        
        # Continue from where we left off
        result = await new_websurfer.run("Search for Python programming tutorials")
        print(f"✅ Result: {result}")
        
        await new_websurfer.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await websurfer.close()


async def context_manager_example():
    """Example using context manager."""
    print("🚀 Starting Context Manager Example")
    
    model = OpenAIServerModel(
        model_id="gpt-4o",
        api_base="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
    )
    
    # Use context manager for automatic cleanup
    async with WebSurferAgent(
        name="ContextWebSurfer",
        model=model,
        debug_dir="./debug_screenshots"
    ) as websurfer:
        
        result = await websurfer.run("Visit a website and take a screenshot")
        print(f"✅ Result: {result}")


def main():
    """Run all examples."""
    print("🌐 WebSurfer Agent Examples")
    print("=" * 50)
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Examples will use mock data.")
        print("   Set OPENAI_API_KEY environment variable for real API calls.")
        print()
    
    # Run examples
    examples = [
        ("Basic Example", basic_example),
        ("Advanced Example", advanced_example),
        ("State Management Example", state_management_example),
        ("Context Manager Example", context_manager_example),
    ]
    
    for name, example_func in examples:
        print(f"\n{name}:")
        print("-" * len(name))
        try:
            asyncio.run(example_func())
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
    
    print("\n🎉 All examples completed!")


if __name__ == "__main__":
    main()
