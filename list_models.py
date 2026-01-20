import os
from pathlib import Path

from google import genai
from google.genai import types


def _load_env_local():
    env_path = Path(__file__).resolve().parent / ".env.local"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def main():
    _load_env_local()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY/GOOGLE_API_KEY (env or .env.local).")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(client_args={"timeout": 30}),
    )

    for model in client.models.list():
        print(model.name)


if __name__ == "__main__":
    main()

