
from typing import Dict, Any, Optional, List
from .config import load_settings, Settings
from .contract import CODE_ONLY_INSTRUCTIONS
from .loop import run_loop
from .verifier import verify_code_state
from .types import LoopResult

def solve(
    task: str,
    files: Dict[str, str],
    max_iter: int = 6,
    model: str = None,
    api_key: str = None,
    allowed_paths: Optional[List[str]] = None
) -> LoopResult:
    """
    High-level API to run RalphBoost on a set of files.
    
    Args:
        task: Description of the task/bug to fix.
        files: Dictionary of file paths to content (initial state).
        max_iter: Maximum iterations to try.
        model: Gemini model name (default from env or config).
        api_key: Gemini API key (default from env).
        allowed_paths: List of allowed file paths to modify.
        
    Returns:
        LoopResult object containing history and status.
    """
    try:
        settings = load_settings()
    except RuntimeError:
        # Fallback if env vars missing but might be passed as args
        settings = Settings(api_key="", model="gemini-3-flash-preview")
    
    if api_key:
        settings.api_key = api_key
    if model:
        settings.model = model
        
    if not settings.api_key:
        raise ValueError("API Key not provided. Set GEMINI_API_KEY env var or pass api_key argument.")

    result = run_loop(
        base_task=task,
        system_instructions=CODE_ONLY_INSTRUCTIONS,
        verifier=verify_code_state,
        settings=settings,
        max_iter=max_iter,
        initial_state=files,
        allowed_paths=allowed_paths
    )
    
    return result
