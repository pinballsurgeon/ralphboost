import os
from dataclasses import dataclass


def _read_env_file(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            data[k.strip()] = v.strip().strip('"').strip("'")
    return data


@dataclass
class Settings:
    api_key: str
    model: str = "gemini-3-flash-preview"
    thinking_level: str = "HIGH"


def load_settings() -> Settings:
    file_env = _read_env_file(".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or file_env.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment.")
    model = (os.getenv("RALPHBOOST_MODEL", "").strip() or file_env.get("RALPHBOOST_MODEL", "") or "gemini-3-flash-preview")
    thinking_level = (os.getenv("RALPHBOOST_THINKING", "").strip() or file_env.get("RALPHBOOST_THINKING", "") or "HIGH")
    return Settings(api_key=api_key, model=model, thinking_level=thinking_level)
