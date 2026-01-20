import os
import json
import re
import numpy as np
from typing import Any, Optional, Dict

try:
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

from .base import Agent

class GeminiAgent(Agent):
    """
    RalphBoost v3.5 Gemini Agent.
    Implements the "Ralph Wiggum Loop" strategy requirements.
    """
    def __init__(
        self,
        model: str = "gemini-flash-latest",
        api_key: Optional[str] = None,
        *,
        api_version: Optional[str] = None,
        timeout_s: int = 30,
        fail_hard: bool = False,
    ):
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("google-genai library not installed. Install with `pip install google-genai`.")
            
        self.model = model
        self.fail_hard = bool(fail_hard)

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            os.environ.setdefault("GEMINI_API_KEY", api_key)

        if not resolved_key:
            print("Warning: GEMINI_API_KEY not set in environment.")

        # NOTE: google-genai's internal default timeout can be too aggressive on some networks.
        # `client_args` is the most reliable way to configure the underlying httpx timeout.
        http_options = types.HttpOptions(client_args={"timeout": int(timeout_s)})
        if api_version:
            http_options.api_version = api_version
        self.client = genai.Client(api_key=resolved_key, http_options=http_options)

    @staticmethod
    def _coerce_array(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        try:
            return np.asarray(x, dtype=float)
        except Exception:
            return np.asarray(x)

    @staticmethod
    def _parse_json_response(response: Any) -> Dict[str, Any]:
        if getattr(response, "parsed", None) is not None:
            if isinstance(response.parsed, dict):
                return response.parsed
            if isinstance(response.parsed, str):
                return json.loads(response.parsed)

        text = getattr(response, "text", "") or ""
        clean_text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not match:
            raise ValueError("Gemini returned no JSON object.")
        return json.loads(match.group(0))

    def validate_connection(self, *, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a minimal real API call to validate credentials + model access.
        """
        response = self.client.models.generate_content(
            model=model or self.model,
            contents='Return ONLY JSON: {"ok": true}',
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.0,
                max_output_tokens=64,
            ),
        )
        return self._parse_json_response(response)

    def propose(self, state, k=1, context=None):
        """
        Legacy/Simple interface: Propose components directly.
        Internally calls propose_blueprints and extracts candidates.
        """
        blueprint = self.propose_blueprints(state, context=context)
        if blueprint and "candidates" in blueprint:
            return blueprint["candidates"][:k]
        return []

    def propose_blueprints(self, state, context=None) -> dict:
        """
        Generate a 'Blueprint' for the next boosting iteration.
        Returns strict JSON adhering to the RalphBoost v3.5 schema.
        """
        context = context or {}
        iteration = context.get("iteration", 0)
        domain_name = context.get('domain_name', 'General Decomposition')
        hint = context.get('hint', '')
        
        residual_data = self._coerce_array(getattr(state, "residual", []))
        residual_flat = residual_data.flatten() if residual_data.ndim > 1 else residual_data
        preview_len = int(min(len(residual_flat), 256))
        residual_preview = residual_flat[:preview_len]
        residual_preview_str = "[" + ", ".join(f"{float(v):.4f}" for v in residual_preview.tolist()) + "]"
        
        # Stats
        res_mean = float(np.mean(residual_flat)) if residual_flat.size else 0.0
        res_std = float(np.std(residual_flat)) if residual_flat.size else 0.0
        res_min = float(np.min(residual_flat)) if residual_flat.size else 0.0
        res_max = float(np.max(residual_flat)) if residual_flat.size else 0.0

        prompt = f"""
ROLE: RalphBoost Strategy Agent (Scientific Discovery)
TASK: Analyze residual signal and propose a Blueprint for the next boosting iteration.

CONTEXT:
Domain: {domain_name}
Iteration: {iteration}
Signal Stats: Mean={res_mean:.4f}, Std={res_std:.4f}, Min={res_min:.4f}, Max={res_max:.4f}
Hint: {hint}

RESIDUAL SAMPLE (first {preview_len} points, rounded):
{residual_preview_str}

INSTRUCTIONS:
1. Decide a windowing policy: global/local/micro.
2. Provide 1-5 candidates with clear rationales.
3. Set constraints (max_components, complexity_penalty).
4. Window indices: start/end are integer indices into the residual array preview; start inclusive, end exclusive.
Return ONLY valid JSON matching the required JSON schema.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=128),
                    temperature=0.2,
                    max_output_tokens=2048,
                )
            )
            data = self._parse_json_response(response)
            if not isinstance(data, dict):
                raise TypeError("Gemini returned non-object JSON for blueprint.")

            # Backfill minimal defaults so downstream code can be stable.
            data.setdefault("iteration", int(iteration))
            data.setdefault("focus", {"window_policy": "global", "windows": []})
            data.setdefault("candidates", [])
            data.setdefault("constraints", {"max_components": 1, "complexity_penalty": 0.0})
            return data
        except Exception as e:
            print(f"Gemini Agent Error: {e}")
            if self.fail_hard:
                raise
            return {}  # StrategyLoop treats this as no-candidates.

    def design_domain(self, data_sample, task_description="General Scientific Discovery"):
        """
        Analyze the data sample and task to generate a custom RalphBoost Domain class.
        """
        # (Existing implementation kept for completeness)
        data_type = type(data_sample).__name__
        shape = getattr(data_sample, 'shape', 'unknown')
        if hasattr(data_sample, 'flatten'):
            sample_vals = str(data_sample.flatten()[:50])
        else:
            sample_vals = str(data_sample)[:200]
        
        prompt = f"""
You are the Lead Architect for RalphBoost.
TASK: Create a Python Domain class for: "{task_description}"
Data: {data_type}, Shape: {shape}
Sample: {sample_vals}

REQUIREMENTS:
1. Inherit from `ralphboost.domains.base.Domain`.
2. Implement `propose`, `refine`, `apply`, `fingerprint`, `complexity`.
3. Return ONLY valid Python code.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="text/x-python" 
                )
            )
            text = response.text
            code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                code = text.replace("```", "").strip()
            return code
        except Exception as e:
            print(f"Gemini Agent Design Domain Error: {e}")
            return None
