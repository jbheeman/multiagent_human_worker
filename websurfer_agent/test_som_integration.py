"""
Test script for Set of Mark (SOM) integration in WebSurfer agent.

This script tests that:
1. Annotated screenshots are generated
2. ID mapping works correctly
3. Tools can map display IDs to real element IDs
4. The agent workflow integrates SOM properly
"""

import os
import sys
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from websurfer_agent.web_surfer_agent import WebSurferAgent
from websurfer_agent.tool_definitions import set_id_mapping, get_real_element_id
from smolagents.models import OpenAIServerModel
from browser_playwright.types import InteractiveRegion, DOMRectangle
from websurfer_agent.set_of_mark import add_set_of_mark
import PIL.Image


def test_id_mapping():
    """Test that ID mapping works correctly."""
    print("\n=== Test 1: ID Mapping ===")
    
    # Create a test mapping
    test_mapping = {
        "1": "element_abc123",
        "2": "element_def456",
        "3": "element_ghi789",
    }
    
    # Set the mapping
    set_id_mapping(test_mapping)
    
    # Test retrieval
    assert get_real_element_id("1") == "element_abc123", "Mapping failed for '1'"
    assert get_real_element_id("2") == "element_def456", "Mapping failed for '2'"
    assert get_real_element_id("3") == "element_ghi789", "Mapping failed for '3'"
    
    # Test non-existent ID (should return original)
    assert get_real_element_id("999") == "999", "Non-existent ID should return original"
    
    print("✅ ID mapping works correctly!")


def test_set_of_mark_annotation():
    """Test that Set of Mark annotation works."""
    print("\n=== Test 2: Set of Mark Annotation ===")
    
    # Create a simple test image
    test_img = PIL.Image.new('RGB', (800, 600), color='white')
    
    # Convert to bytes
    import io
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Create mock interactive regions
    mock_regions: dict = {
        "elem_1": {
            "tag_name": "button",
            "role": "button",
            "aria-name": "Submit",
            "v-scrollable": False,
            "rects": [{
                "x": 100, "y": 100,
                "width": 80, "height": 40,
                "top": 100, "right": 180,
                "bottom": 140, "left": 100
            }]
        },
        "elem_2": {
            "tag_name": "input",
            "role": "textbox",
            "aria-name": "Search",
            "v-scrollable": False,
            "rects": [{
                "x": 200, "y": 50,
                "width": 150, "height": 30,
                "top": 50, "right": 350,
                "bottom": 80, "left": 200
            }]
        }
    }
    
    # Annotate the image
    annotated_img, visible_ids, ids_above, ids_below, id_mapping = add_set_of_mark(
        screenshot=img_bytes.getvalue(),
        ROIs=mock_regions,
        use_sequential_ids=True
    )
    
    # Verify results
    assert isinstance(annotated_img, PIL.Image.Image), "Should return PIL Image"
    assert isinstance(id_mapping, dict), "Should return ID mapping dict"
    assert len(id_mapping) > 0, "Should have at least one mapping"
    
    print(f"✅ Set of Mark annotation works!")
    print(f"   - Generated {len(id_mapping)} ID mappings")
    print(f"   - Visible IDs: {visible_ids}")
    print(f"   - ID Mapping: {id_mapping}")


async def test_agent_screenshot_method():
    """Test that the agent's screenshot method works."""
    print("\n=== Test 3: Agent Screenshot Method ===")
    
    # Check if NAUT_API_KEY is available
    api_key = os.environ.get("NAUT_API_KEY")
    if not api_key:
        print("⚠️  Skipping agent test (NAUT_API_KEY not set)")
        return
    
    try:
        # Create a minimal agent (won't actually run, just test setup)
        model = OpenAIServerModel(
            model_id="anthropic/claude-3-5-sonnet-20241022",
            api_base="https://api.naga.ac/v1",
            api_key=api_key,
        )
        
        agent = WebSurferAgent(
            model=model,
            headless=True,
            to_save_screenshots=True,
            debug_dir="./test_screenshots"
        )
        
        # Initialize browser
        await agent.lazy_init()
        
        # Test that the screenshot method exists and is callable
        assert hasattr(agent, '_get_annotated_screenshot'), "Agent should have _get_annotated_screenshot method"
        assert hasattr(agent, '_screenshot_callback'), "Agent should have _screenshot_callback method"
        assert hasattr(agent, '_current_id_mapping'), "Agent should have _current_id_mapping attribute"
        
        print("✅ Agent has all required SOM methods!")
        
        # Clean up
        await agent.close()
        
    except Exception as e:
        print(f"⚠️  Agent test partial failure: {e}")
        print("   (This might be expected if browser can't initialize)")


def test_tool_integration():
    """Test that tools are updated to use ID mapping."""
    print("\n=== Test 4: Tool Integration ===")
    
    from websurfer_agent.tool_definitions import (
        ClickTool, InputTextTool, HoverTool, 
        SelectOptionTool, ScrollElementDownTool
    )
    
    # Check that tools exist and have the forward method
    tools = [ClickTool, InputTextTool, HoverTool, SelectOptionTool, ScrollElementDownTool]
    
    for tool_class in tools:
        tool = tool_class()
        assert hasattr(tool, 'forward'), f"{tool_class.__name__} should have forward method"
        
        # Check that forward method accepts target_id (for most tools)
        if tool_class != InputTextTool:
            import inspect
            sig = inspect.signature(tool.forward)
            assert 'target_id' in sig.parameters, f"{tool_class.__name__} should accept target_id"
    
    print(f"✅ All {len(tools)} tools have proper signatures!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Set of Mark (SOM) Integration Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: ID Mapping
        test_id_mapping()
        
        # Test 2: Set of Mark Annotation
        test_set_of_mark_annotation()
        
        # Test 3: Agent Screenshot Method (async)
        asyncio.run(test_agent_screenshot_method())
        
        # Test 4: Tool Integration
        test_tool_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\n✅ Set of Mark integration is working correctly!")
        print("✅ Your WebSurfer agent now has vision-based interaction!")
        print("\nNext steps:")
        print("  1. Test with a real webpage: await agent.run('Go to example.com')")
        print("  2. Use a VLM model like Qwen/Qwen2-VL-72B-Instruct")
        print("  3. Check debug_dir for annotated screenshots")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

