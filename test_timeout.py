#!/usr/bin/env python3
"""
Test script for timeout functionality in benchmark runner
"""

import os
import sys
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_runner import capture_console_output_with_timeout

def slow_function():
    """A function that takes a long time to complete"""
    print("Starting slow function...")
    time.sleep(10)  # Sleep for 10 seconds
    print("Slow function completed!")
    return "Success"

def test_timeout():
    """Test the timeout functionality"""
    print("Testing timeout functionality...")
    
    # Test with 3-second timeout on a 10-second function
    print("Running function with 3-second timeout (should timeout)...")
    result, stdout, stderr, error = capture_console_output_with_timeout(
        slow_function, timeout_seconds=3
    )
    
    print(f"Result: {result}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")
    print(f"Error: {error}")
    
    if error and "timed out" in error.lower():
        print("✅ Timeout test passed!")
    else:
        print("❌ Timeout test failed!")
    
    print("\n" + "="*50 + "\n")
    
    # Test with 15-second timeout on a 10-second function (should complete)
    print("Running function with 15-second timeout (should complete)...")
    result, stdout, stderr, error = capture_console_output_with_timeout(
        slow_function, timeout_seconds=15
    )
    
    print(f"Result: {result}")
    print(f"Stdout: {stdout}")
    print(f"Stderr: {stderr}")
    print(f"Error: {error}")
    
    if result == "Success" and error is None:
        print("✅ Completion test passed!")
    else:
        print("❌ Completion test failed!")

if __name__ == "__main__":
    test_timeout()
