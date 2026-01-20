import os
import unittest
from pathlib import Path

from ralphboost.agents.gemini import GeminiAgent


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


class TestGeminiConnection(unittest.TestCase):
    def setUp(self):
        _maybe_load_env_local()
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            self.skipTest("No Gemini API key found (GEMINI_API_KEY/GOOGLE_API_KEY).")

        self.model = os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"

    def test_validate_connection(self):
        agent = GeminiAgent(model=self.model, api_key=self.api_key, fail_hard=True)
        result = agent.validate_connection()
        self.assertIsInstance(result, dict)
        self.assertIn("ok", result)
        self.assertTrue(bool(result["ok"]))


if __name__ == "__main__":
    unittest.main()
