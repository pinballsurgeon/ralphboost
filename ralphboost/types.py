from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    complete: bool
    score: float  # 0..1 visible score
    reasons: List[str]
    hidden_reasons: Optional[List[str]] = None


@dataclass
class IterationRecord:
    iteration: int
    prompt: str
    raw_text: str
    parsed: Optional[Dict[str, Any]]
    verification: VerificationResult
    duration_sec: float
    model: Optional[str] = None
    prompt_chars: int = 0
    response_chars: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    patch: Optional[Dict[str, Any]] = None
    patch_ops: int = 0
    patch_applied_ops: int = 0
    patch_chars: int = 0
    visible_loss: float = 0.0
    hidden_loss: float = 0.0
    loss: float = 0.0


@dataclass
class LoopResult:
    status: str  # verified | max-iterations | no-progress
    best_score: float
    iterations: int
    history: List[IterationRecord]
