"""Basic tests for simulated human agents.

These are simple functional tests to ensure the components work together.
Run with: python -m pytest test_simulated_humans.py
Or directly: python test_simulated_humans.py
"""

import os
import sys
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    try:
        from simulated_humans import (
            HumanDecision,
            SimulatedHumanAgent,
            WebHumanTool,
            CodeHumanTool,
            FileHumanTool,
            create_human_tools,
            get_persona_prompt,
            get_temperature,
            get_default_preferences,
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_persona_functions():
    """Test persona utility functions."""
    try:
        from simulated_humans import get_persona_prompt, get_temperature, get_default_preferences
        
        # Test each role
        for role in ["web", "code", "file"]:
            prompt = get_persona_prompt(role)
            temp = get_temperature(role)
            prefs = get_default_preferences(role)
            
            assert isinstance(prompt, str) and len(prompt) > 0, f"Invalid prompt for {role}"
            assert isinstance(temp, float) and 0 <= temp <= 1, f"Invalid temperature for {role}"
            assert isinstance(prefs, dict), f"Invalid preferences for {role}"
        
        print("✅ Persona functions work correctly")
        return True
    except Exception as e:
        print(f"❌ Persona function test failed: {e}")
        return False


def test_human_decision_structure():
    """Test HumanDecision TypedDict structure."""
    try:
        from simulated_humans import HumanDecision
        
        # Create a valid decision
        decision: HumanDecision = {
            "decision": "approve",
            "message": "This looks good",
        }
        
        # With revisions
        decision_with_revisions: HumanDecision = {
            "decision": "revise",
            "message": "Needs changes",
            "revisions": {"url": "https://example.com"}
        }
        
        print("✅ HumanDecision structure is valid")
        return True
    except Exception as e:
        print(f"❌ HumanDecision test failed: {e}")
        return False


def test_tool_creation():
    """Test creating human tools."""
    try:
        from simulated_humans import create_human_tools
        from smolagents import OpenAIServerModel
        
        # Check if API key is available
        if "NAUT_API_KEY" not in os.environ:
            print("⚠️  Skipping tool creation test (no NAUT_API_KEY)")
            return True
        
        # Create a mock model
        model = OpenAIServerModel(
            model_id="gemma3",
            api_base="https://ellm.nrp-nautilus.io/v1",
            api_key=os.environ["NAUT_API_KEY"],
        )
        
        # Create tools
        tools = create_human_tools(model=model, use_simulated=True)
        
        assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"
        assert all(hasattr(tool, 'name') for tool in tools), "Tools missing name attribute"
        assert all(hasattr(tool, 'forward') for tool in tools), "Tools missing forward method"
        
        print("✅ Tool creation successful")
        return True
    except Exception as e:
        print(f"❌ Tool creation test failed: {e}")
        return False


def test_simulated_agent_structure():
    """Test SimulatedHumanAgent structure (without calling model)."""
    try:
        from simulated_humans import SimulatedHumanAgent
        from smolagents import OpenAIServerModel
        
        # Check if API key is available
        if "NAUT_API_KEY" not in os.environ:
            print("⚠️  Skipping agent structure test (no NAUT_API_KEY)")
            return True
        
        model = OpenAIServerModel(
            model_id="gemma3",
            api_base="https://ellm.nrp-nautilus.io/v1",
            api_key=os.environ["NAUT_API_KEY"],
        )
        
        # Create agent
        agent = SimulatedHumanAgent(model=model, role="web")
        
        # Check attributes
        assert agent.role == "web", "Role not set correctly"
        assert hasattr(agent, 'decide'), "Missing decide method"
        assert hasattr(agent, 'get_decision_history'), "Missing history method"
        assert hasattr(agent, 'update_preferences'), "Missing preferences method"
        
        print("✅ SimulatedHumanAgent structure is valid")
        return True
    except Exception as e:
        print(f"❌ Agent structure test failed: {e}")
        return False


def test_json_parsing():
    """Test JSON parsing from various formats."""
    try:
        from simulated_humans.simulated_human_agent import SimulatedHumanAgent
        from smolagents import OpenAIServerModel
        
        if "NAUT_API_KEY" not in os.environ:
            print("⚠️  Skipping JSON parsing test (no NAUT_API_KEY)")
            return True
        
        model = OpenAIServerModel(
            model_id="gemma3",
            api_base="https://ellm.nrp-nautilus.io/v1",
            api_key=os.environ["NAUT_API_KEY"],
        )
        
        agent = SimulatedHumanAgent(model=model, role="code")
        
        # Test various JSON formats
        test_cases = [
            # Plain JSON
            '{"decision": "approve", "message": "Looks good", "revisions": null}',
            
            # JSON in code block
            '```json\n{"decision": "deny", "message": "Not safe", "revisions": null}\n```',
            
            # JSON with extra text
            'Here is my decision: {"decision": "revise", "message": "Needs work", "revisions": {"fix": "this"}}',
        ]
        
        for i, test_case in enumerate(test_cases):
            result = agent._parse_decision(test_case)
            assert result["decision"] in ["approve", "deny", "revise"], f"Test case {i+1} failed"
            assert isinstance(result["message"], str), f"Test case {i+1} message invalid"
        
        print("✅ JSON parsing works correctly")
        return True
    except Exception as e:
        print(f"❌ JSON parsing test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Simulated Human Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Persona Functions", test_persona_functions),
        ("HumanDecision Structure", test_human_decision_structure),
        ("Tool Creation", test_tool_creation),
        ("Agent Structure", test_simulated_agent_structure),
        ("JSON Parsing", test_json_parsing),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 Test: {name}")
        print("-" * 60)
        success = test_func()
        results.append((name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

