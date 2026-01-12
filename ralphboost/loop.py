
import json
import time
from typing import Callable, List, Dict, Any, Optional

from .gemini_client import generate_response as default_generate_response
from .types import IterationRecord, LoopResult, VerificationResult
from .patch import apply_code_patch
from .autopatch import build_patch_spec, format_patch_spec

def safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return None

def run_loop(
    base_task: str,
    system_instructions: str,
    verifier: Callable[[Dict[str, str]], VerificationResult],
    settings: Any,
    max_iter: int = 6,
    initial_state: Dict[str, str] = None,
    allowed_paths: Optional[List[str]] = None,
    generate_fn: Callable = default_generate_response
) -> LoopResult:
    history: List[IterationRecord] = []
    best_score = -1.0
    stagnant = 0
    feedback = ""
    
    state = initial_state.copy() if initial_state else {}
    
    for i in range(1, max_iter + 1):
        prompt = system_instructions + "\n\nTASK:\n" + base_task.strip()
        
        prompt += "\n\nCURRENT_CODE_STATE:\n"
        for path, content in state.items():
            prompt += f"--- {path} ---\n{content}\n"
            
        if feedback:
            prompt += "\n\nVERIFIER FEEDBACK:\n" + feedback

        t0 = time.time()
        resp = generate_fn(prompt, settings=settings)
        raw = resp["text"]
        dt = time.time() - t0

        parsed = safe_json_load(raw)
        
        patch_payload = None
        patch_ops = []
        patch_applied = 0
        patch_chars = 0
        
        if parsed:
            patch_ops = parsed.get("patch", [])
            state, patch_applied, _ = apply_code_patch(state, patch_ops, allowed_paths)
            patch_chars = sum(len(str(op.get("value", ""))) for op in patch_ops)
            
        # Verify
        vr = verifier(state)
        
        # Loss calculation
        visible_loss = 1.0 - vr.score
        hidden_loss = 0.0
        loss = visible_loss + hidden_loss
        
        failing_tests_count = len(vr.failing_tests) if vr.failing_tests else 0
        if not vr.complete and failing_tests_count == 0 and vr.tests_run > 0:
             # Fallback if list empty but score < 1
             failing_tests_count = int((1.0 - vr.score) * vr.tests_run)
        
        # Thrash Index Calculation
        thrash_index = 0.0
        if i > 1:
            prev_loss = history[-1].loss
            loss_delta = prev_loss - loss
            if loss_delta <= 1e-6:
                # Loss didn't improve significantly
                thrash_index = float(patch_chars)
            else:
                # Improved. Normalized cost.
                thrash_index = float(patch_chars) / (loss_delta * 100.0 + 1e-6)

        history.append(
            IterationRecord(
                iteration=i,
                prompt=prompt,
                raw_text=raw,
                parsed=parsed,
                verification=vr,
                duration_sec=dt,
                model=resp.get("model"),
                prompt_chars=len(prompt),
                response_chars=len(raw),
                prompt_tokens=resp.get("usage", {}).get("prompt_tokens", 0),
                response_tokens=resp.get("usage", {}).get("response_tokens", 0),
                total_tokens=resp.get("usage", {}).get("total_tokens", 0),
                patch=parsed,
                patch_ops=len(patch_ops),
                patch_applied_ops=patch_applied,
                patch_chars=patch_chars,
                visible_loss=visible_loss,
                hidden_loss=hidden_loss,
                loss=loss,
                thrash_index=thrash_index,
                failing_tests_count=failing_tests_count,
            )
        )

        if vr.score > best_score + 1e-12:
            best_score = vr.score
            stagnant = 0
        else:
            stagnant += 1

        # Check completion signal: ONLY verifier decides!
        if vr.complete:
            return LoopResult("verified", best_score, len(history), history)

        reasons = list(vr.reasons)
        if vr.failing_tests:
            reasons.extend([f"FAIL: {t}" for t in vr.failing_tests])
            
        patch_spec = build_patch_spec(parsed, vr)
        feedback = "\n".join(f"- {r}" for r in reasons)
        if patch_spec:
            feedback += "\nPATCH_SPEC:\n" + format_patch_spec(patch_spec)
            
    return LoopResult("max-iterations", best_score, len(history), history)
