from typing import Optional, Dict, Any

from .config import Settings


def _require_genai():
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "google-genai is required. Install with: pip install -r requirements.txt"
        ) from exc
    return genai, types


def _usage_counts(resp: Any) -> Dict[str, int]:
    prompt_tokens = 0
    response_tokens = 0
    total_tokens = 0
    meta = getattr(resp, "usage_metadata", None)
    if meta is not None:
        prompt_tokens = int(getattr(meta, "prompt_token_count", 0) or 0)
        response_tokens = int(getattr(meta, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(meta, "total_token_count", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
    }


def generate_response(
    prompt: str, settings: Settings, thinking_level: Optional[str] = None
) -> Dict[str, Any]:
    genai, types = _require_genai()
    client = genai.Client(api_key=settings.api_key)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level or settings.thinking_level
        ),
    )
    resp = client.models.generate_content(
        model=settings.model, contents=contents, config=cfg
    )
    text = resp.text or ""
    usage = _usage_counts(resp)
    return {"text": text, "usage": usage, "model": settings.model}


def generate_text(prompt: str, settings: Settings, thinking_level: Optional[str] = None) -> str:
    return generate_response(prompt, settings, thinking_level)["text"]
