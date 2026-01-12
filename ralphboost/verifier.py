
import os
import sys
import subprocess
import tempfile
import re
from typing import Dict
from .types import VerificationResult

class ExecutableVerifier:
    def verify(self, state: Dict[str, str]) -> VerificationResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write state to temp dir
            for path, content in state.items():
                full_path = os.path.join(tmpdir, path)
                dir_name = os.path.dirname(full_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
            
            # Run unittest discovery in subprocess
            # Use -v to see test names, which helps debugging but parsing "Ran N tests" is standard
            cmd = [sys.executable, "-m", "unittest", "discover", "-s", tmpdir, "-p", "test_*.py"]
            
            result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
            
            output = result.stdout + "\n" + result.stderr
            
            # Parse tests run
            tests_run = 0
            match = re.search(r"Ran (\d+) tests?", output)
            if match:
                tests_run = int(match.group(1))
            
            # Check completion: must have exit code 0 AND run at least 1 test
            complete = (result.returncode == 0) and (tests_run > 0)
            
            reasons = []
            failing_tests = []
            
            # Rudimentary parsing for failing tests count if not complete
            # We do this before calculating score to have accurate count
            if not complete and tests_run > 0:
                 # ... parsing logic will append to failing_tests ...
                 pass # Logic below handles parsing
                 
            # We need to parse failing tests *before* determining score if we want granular score.
            # Moving parsing logic up.
            if tests_run == 0:
                reasons.append("Zero tests run. You must define and run valid tests.")
                complete = False
                score = 0.0
            elif not complete:
                reasons.append(f"Tests failed (exit code {result.returncode}):\n{output}")
                
                for line in output.splitlines():
                    if line.startswith("FAIL:") or line.startswith("ERROR:"):
                        parts = line.split(" ", 2)
                        if len(parts) > 1:
                            failing_tests.append(parts[1])
                
                # Calculate score
                failures = len(failing_tests)
                if failures == 0:
                    failures = 1 # Fallback
                score = max(0.0, (tests_run - failures) / tests_run)
            else:
                score = 1.0
            
            return VerificationResult(
                complete=complete,
                score=score,
                reasons=reasons,
                hidden_reasons=[], 
                failing_tests=failing_tests,
                tests_run=tests_run
            )

def verify_code_state(state: Dict[str, str]) -> VerificationResult:
    return ExecutableVerifier().verify(state)
