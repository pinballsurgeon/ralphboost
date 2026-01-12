
import unittest
import json
from unittest.mock import MagicMock
from ralphboost.loop import run_loop
from ralphboost.verifier import ExecutableVerifier
from ralphboost.types import VerificationResult

class TestLoopExec(unittest.TestCase):
    def test_fix_loop(self):
        # Initial state: broken code
        initial_state = {
            "main.py": "def add(a, b): return a - b",
            "test_main.py": """
import unittest
from main import add
class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
"""
        }
        
        # Mock LLM response function
        def mock_generate(prompt, settings=None):
            if "return a - b" in prompt:
                return {
                    "text": json.dumps({
                        "iteration_summary": "Fix add function",
                        "patch": [{"path": "main.py", "value": "def add(a, b): return a + b"}],
                        "completion_signal": {"done": True, "reason": "Fixed bug"}
                    }),
                    "model": "test-model",
                    "usage": {}
                }
            return {
                "text": json.dumps({"patch": []}), "model": "test", "usage": {}
            }
        
        verifier = ExecutableVerifier().verify
        settings = {}
        
        res = run_loop(
            base_task="Fix the bug",
            system_instructions="You are a coder.",
            verifier=verifier,
            settings=settings,
            initial_state=initial_state,
            max_iter=3,
            generate_fn=mock_generate
        )
        
        self.assertEqual(res.status, "verified")
        self.assertEqual(res.iterations, 1)
        self.assertEqual(res.best_score, 1.0)
        
        last_iter = res.history[-1]
        self.assertTrue(last_iter.verification.complete)

    def test_loop_ignores_agent_done(self):
        initial_state = {"test_foo.py": "code"}
        
        def mock_generate(prompt, settings=None):
            return {
                "text": json.dumps({
                    "iteration_summary": "I am done",
                    "patch": [],
                    "completion_signal": {"done": True, "reason": "Trust me"}
                }),
                "model": "test-model",
                "usage": {}
            }
            
        mock_verifier = MagicMock()
        # Always fail
        mock_verifier.return_value = VerificationResult(
            complete=False, score=0.0, reasons=["Tests failed"], tests_run=1, failing_tests=["test_1"]
        )
        
        settings = {}
        res = run_loop(
            base_task="Fix bug",
            system_instructions="sys",
            verifier=mock_verifier,
            settings=settings,
            initial_state=initial_state,
            max_iter=3,
            generate_fn=mock_generate
        )
        
        # Should NOT be verified even though agent sent completion_signal
        self.assertEqual(res.status, "max-iterations")
        self.assertEqual(res.iterations, 3)

    def test_metrics_telemetry(self):
        initial_state = {"main.py": "broken"}
        
        def mock_generate(prompt, settings=None):
             return {
                 "text": json.dumps({
                     "patch": [{"path":"main.py", "value":"fixed"}], 
                     "completion_signal": {"done": True, "reason": "fixed"}
                 }),
                 "model": "model",
                 "usage": {}
             }
        
        mock_verifier = MagicMock()
        # Iteration 1: Score 0.5 (failing)
        # Iteration 2: Score 1.0 (pass)
        mock_verifier.side_effect = [
             VerificationResult(complete=False, score=0.5, reasons=["fail"], tests_run=2, failing_tests=["test_1"]),
             VerificationResult(complete=True, score=1.0, reasons=[], tests_run=2, failing_tests=[])
        ]
        
        res = run_loop(
            "fix", "sys", mock_verifier, {}, initial_state=initial_state, generate_fn=mock_generate, max_iter=2
        )
        
        self.assertEqual(len(res.history), 2)
        
        # Check iteration 1
        it1 = res.history[0]
        self.assertEqual(it1.visible_loss, 0.5)
        self.assertEqual(it1.failing_tests_count, 1)
        self.assertEqual(it1.thrash_index, 0.0)
        
        # Check iteration 2
        it2 = res.history[1]
        self.assertEqual(it2.visible_loss, 0.0)
        self.assertEqual(it2.failing_tests_count, 0)
        # Loss delta = 0.5. Patch chars > 0. Thrash should be > 0.
        self.assertGreater(it2.thrash_index, 0.0)

if __name__ == '__main__':
    unittest.main()
