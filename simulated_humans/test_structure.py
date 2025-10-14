"""Basic structure tests that don't require external dependencies.

Run with: python test_structure.py
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_human_decision_type():
    """Test HumanDecision TypedDict."""
    try:
        from simulated_humans.human_decision import HumanDecision
        
        # Test valid decision
        decision = HumanDecision(
            decision="approve",
            message="This is fine"
        )
        
        assert decision["decision"] == "approve"
        assert decision["message"] == "This is fine"
        
        print("✅ HumanDecision type works")
        return True
    except Exception as e:
        print(f"❌ HumanDecision test failed: {e}")
        return False


def test_personas_module():
    """Test personas module functions."""
    try:
        from simulated_humans.personas import (
            get_persona_prompt,
            get_temperature,
            get_default_preferences,
            SHARED_POLICY,
            WEB_HUMAN_PERSONA,
            CODE_HUMAN_PERSONA,
            FILE_HUMAN_PERSONA,
        )
        
        # Test constants exist
        assert len(SHARED_POLICY) > 0, "SHARED_POLICY is empty"
        assert len(WEB_HUMAN_PERSONA) > 0, "WEB_HUMAN_PERSONA is empty"
        assert len(CODE_HUMAN_PERSONA) > 0, "CODE_HUMAN_PERSONA is empty"
        assert len(FILE_HUMAN_PERSONA) > 0, "FILE_HUMAN_PERSONA is empty"
        
        # Test functions
        for role in ["web", "code", "file"]:
            prompt = get_persona_prompt(role)
            assert isinstance(prompt, str) and len(prompt) > 100, f"Invalid prompt for {role}"
            
            temp = get_temperature(role)
            assert isinstance(temp, float) and 0 <= temp <= 1, f"Invalid temp for {role}"
            
            prefs = get_default_preferences(role)
            assert isinstance(prefs, dict) and len(prefs) > 0, f"Invalid prefs for {role}"
        
        print("✅ Personas module works")
        return True
    except Exception as e:
        print(f"❌ Personas test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_structure():
    """Test that all expected files exist."""
    try:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        
        required_files = [
            "human_decision.py",
            "personas.py",
            "simulated_human_agent.py",
            "human_tools.py",
            "__init__.py",
            "README.md",
            "design_doc.md",
        ]
        
        for filename in required_files:
            filepath = os.path.join(base_dir, filename)
            assert os.path.exists(filepath), f"Missing file: {filename}"
        
        print("✅ All required files exist")
        return True
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False


def test_init_exports():
    """Test __init__.py exports."""
    try:
        import simulated_humans
        
        expected_exports = [
            "HumanDecision",
            "SimulatedHumanAgent",
            "HumanApprovalTool",
            "WebHumanTool",
            "CodeHumanTool",
            "FileHumanTool",
            "create_human_tools",
            "get_persona_prompt",
            "get_temperature",
            "get_default_preferences",
        ]
        
        for export in expected_exports:
            assert hasattr(simulated_humans, export), f"Missing export: {export}"
        
        print("✅ __init__.py exports correct")
        return True
    except Exception as e:
        print(f"❌ Init exports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests():
    """Run all structure tests."""
    print("\n" + "=" * 60)
    print("Simulated Humans - Structure Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("Module Structure", test_module_structure),
        ("HumanDecision Type", test_human_decision_type),
        ("Personas Module", test_personas_module),
        ("Init Exports", test_init_exports),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 {name}")
        print("-" * 60)
        success = test_func()
        results.append((name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    print(f"\n{passed}/{total} tests passed\n")
    
    if passed == total:
        print("🎉 All structure tests passed!")
        print("\nNote: Full integration tests require smolagents and model API access.")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

