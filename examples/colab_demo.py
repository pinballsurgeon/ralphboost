
"""
RalphBoost Colab Demo
To run in Google Colab:
1. !pip install git+https://github.com/pinballsurgeon/ralphboost.git
2. Set GEMINI_API_KEY environment variable.
"""

import os
import sys
import json
import argparse
import ralphboost
from ralphboost.loop import run_loop
from ralphboost.verifier import verify_code_state
from ralphboost.contract import CODE_ONLY_INSTRUCTIONS
from ralphboost.config import Settings

def mock_generate_fn(prompt, settings=None):
    """Simulates a smart agent fixing the code."""
    # Check if code is already fixed
    if "return a / b" not in prompt: 
        # Assume fixed if we don't see the bug (simplified check)
        pass 
    
    # In this demo, the bug is 'return a / b' which fails for 0.
    # We want to change it to handle 0.
    
    # Iteration 1 response: Apply fix
    if "return a / b" in prompt:
        return {
            "text": json.dumps({
                "iteration_summary": "Handle zero division by returning None.",
                "patch": [{
                    "path": "math_ops.py",
                    "value": "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b"
                }],
                "completion_signal": {"done": True, "reason": "Fixed zero division"}
            }),
            "model": "mock-model",
            "usage": {"total_tokens": 100}
        }
    
    return {"text": "{}", "model": "mock", "usage": {}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run with mock LLM")
    args = parser.parse_args()

    # Ensure API key is set if not mocking
    if not args.mock and "GEMINI_API_KEY" not in os.environ:
        print("Please set GEMINI_API_KEY environment variable or use --mock.")
        return

    # Define a task
    task = "Fix the divide function to handle zero division by returning None."

    # Initial code state
    files = {
        "math_ops.py": """
def divide(a, b):
    return a / b
""",
        "test_math_ops.py": """
import unittest
from math_ops import divide

class TestDivide(unittest.TestCase):
    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)
        
    def test_divide_zero(self):
        self.assertIsNone(divide(10, 0))
"""
    }

    print(f"Starting RalphBoost loop for task: '{task}'")
    print("-" * 50)
    
    try:
        if args.mock:
            print("Running in MOCK mode...")
            # We use run_loop directly to inject mock
            result = run_loop(
                base_task=task,
                system_instructions=CODE_ONLY_INSTRUCTIONS,
                verifier=verify_code_state,
                settings=Settings(api_key="mock", model="mock"),
                max_iter=3,
                initial_state=files,
                generate_fn=mock_generate_fn
            )
        else:
            result = ralphboost.solve(task, files, max_iter=3)

        print("-" * 50)
        print(f"Status: {result.status}")
        print(f"Best Score: {result.best_score:.2f}")
        print(f"Iterations: {result.iterations}")

        # Inspect history
        print("\nHistory Analysis:")
        for i, rec in enumerate(result.history):
            print(f"\n[Iteration {i+1}]")
            print(f"  Loss: {rec.loss:.2f} (Visible: {rec.visible_loss:.2f})")
            print(f"  Failing Tests: {rec.failing_tests_count}")
            print(f"  Thrash Index: {rec.thrash_index:.2f}")
            if rec.patch:
                print(f"  Patch Summary: {rec.patch.get('iteration_summary')}")
            if rec.verification.complete:
                print("  >> VERIFIED! All tests passed.")
                
    except Exception as e:
        print(f"Error running loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
