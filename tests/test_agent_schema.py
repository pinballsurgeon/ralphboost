import os
import unittest
import json
import numpy as np
from pathlib import Path

try:
    from google import genai
except ImportError:
    pass

from ralphboost.agents.gemini import GeminiAgent
from ralphboost.core.state import RalphState


def _maybe_load_env_local():
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env.local"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"}:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


class TestAgentSchema(unittest.TestCase):
    def setUp(self):
        _maybe_load_env_local()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not found in environment. Skipping real API schema test.")

        model = os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"
        
        try:
            # Use a stable model for schema verification
            self.agent = GeminiAgent(model=model, api_key=self.api_key, fail_hard=True)
        except Exception as e:
            self.skipTest(f"Failed to initialize GeminiAgent: {e}")

    def test_propose_blueprints_schema_validity(self):
        """
        Test that the real Gemini API returns a valid Blueprint JSON matching our schema.
        """
        # Create a dummy state with a simple signal
        t = np.linspace(0, 10, 100)
        signal = np.sin(t) + np.random.normal(0, 0.1, 100)
        
        state = RalphState(residual=signal, fitted=np.zeros_like(signal))
        
        context = {
            "domain_name": "Signal Processing",
            "hint": "Look for a sine wave."
        }
        
        blueprint = self.agent.propose_blueprints(state, context=context)
        
        # Validation
        self.assertIsInstance(blueprint, dict, "Blueprint must be a dict")
        
        # Check required keys
        self.assertIn("iteration", blueprint)
        self.assertIn("focus", blueprint)
        self.assertIn("candidates", blueprint)
        self.assertIn("constraints", blueprint)
        
        # Check Focus
        focus = blueprint["focus"]
        self.assertIsInstance(focus, dict)
        self.assertIn("window_policy", focus)
        self.assertIn("windows", focus)
        self.assertIsInstance(focus["windows"], list)
        
        # Check Candidates
        candidates = blueprint["candidates"]
        self.assertIsInstance(candidates, list)
        if candidates:
            cand = candidates[0]
            self.assertIn("family", cand)
            self.assertIn("parameters", cand)
            self.assertIn("rationale", cand)
            
        # Check Constraints
        constraints = blueprint["constraints"]
        self.assertIsInstance(constraints, dict)
        
        print("\n[Real API] Blueprint received:")
        print(json.dumps(blueprint, indent=2))

if __name__ == '__main__':
    unittest.main()
