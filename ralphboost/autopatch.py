
import json
from typing import Any, Dict, Optional

from .types import VerificationResult


def build_patch_spec(
    parsed: Optional[Dict[str, Any]], vr: VerificationResult
) -> Dict[str, Any]:
    fixes = []
    if not parsed:
        fixes.append(
            {"path": "root", "must_include": ["valid JSON", "schema compliance"]}
        )
        return {"fixes": fixes}

    if vr.failing_tests:
        for test in vr.failing_tests:
            fixes.append({"path": "test failure", "must_fix": test})

    for reason in vr.reasons:
        # Deduplicate if redundant with failing_tests, but for now include all
        fixes.append({"path": "error", "message": reason})

    for reason in vr.hidden_reasons or []:
        fixes.append({"path": "hidden", "message": reason})

    if not fixes and not vr.complete:
        fixes.append({"path": "root", "must_include": ["address verifier feedback"]})

    return {"fixes": fixes}


def format_patch_spec(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, ensure_ascii=False)
