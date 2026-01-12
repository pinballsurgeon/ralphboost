
import unittest
from ralphboost.verifier import ExecutableVerifier

class TestExecutableVerifier(unittest.TestCase):
    def test_passing_code(self):
        # Code that passes tests
        state = {
            "main.py": "def add(a, b): return a + b",
            "test_main.py": """
import unittest
from main import add
class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
"""
        }
        verifier = ExecutableVerifier()
        res = verifier.verify(state)
        self.assertTrue(res.complete)
        self.assertEqual(res.score, 1.0)
        self.assertFalse(res.failing_tests)
        self.assertGreater(res.tests_run, 0)

    def test_failing_code(self):
        # Code that fails tests
        state = {
            "main.py": "def add(a, b): return a - b", # Bug
            "test_main.py": """
import unittest
from main import add
class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
"""
        }
        verifier = ExecutableVerifier()
        res = verifier.verify(state)
        self.assertFalse(res.complete)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("test_add" in r for r in res.reasons))
        self.assertGreater(res.tests_run, 0)

    def test_zero_tests(self):
        # Code with no tests
        state = {
            "main.py": "def add(a, b): return a + b"
        }
        verifier = ExecutableVerifier()
        res = verifier.verify(state)
        self.assertFalse(res.complete)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("Zero tests run" in r for r in res.reasons))

if __name__ == '__main__':
    unittest.main()
