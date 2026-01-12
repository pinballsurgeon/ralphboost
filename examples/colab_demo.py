
"""
RalphBoost Colab Demo
To run in Google Colab:
1. !pip install git+https://github.com/pinballsurgeon/ralphboost.git
2. Set GEMINI_API_KEY environment variable.
"""

import os
import ralphboost

# Ensure API key is set
if "GEMINI_API_KEY" not in os.environ:
    print("Please set GEMINI_API_KEY environment variable.")
    # In Colab: from google.colab import userdata; os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')

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

print(f"Starting RalphBoost loop for task: {task}")
try:
    result = ralphboost.solve(task, files, max_iter=3)

    print(f"Status: {result.status}")
    print(f"Best Score: {result.best_score}")

    # Inspect history
    for i, rec in enumerate(result.history):
        print(f"Iteration {i+1}: Loss={rec.loss:.2f}, Thrash={rec.thrash_index:.2f}")
        if rec.verification.complete:
            print("Verified!")
            if rec.patch:
                print("Patch applied:", rec.patch.get("patch"))
except Exception as e:
    print(f"Error running loop: {e}")
