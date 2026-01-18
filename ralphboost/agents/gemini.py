import os
import json
import re
try:
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

from .base import Agent

class GeminiAgent(Agent):
    def __init__(self, model="gemini-2.0-flash-exp", api_key=None):
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("google-genai library not installed. Install with `pip install google-genai`.")
            
        self.model = model
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        
        if "GEMINI_API_KEY" not in os.environ:
             # Do not hardcode keys in source control. 
             # Users must provide GEMINI_API_KEY via environment variable.
             print("Warning: GEMINI_API_KEY not set in environment.")
        
        self.client = genai.Client(http_options={'api_version': 'v1alpha'})

    def propose(self, state, k=1, context=None):
        # Build prompt
        context = context or {}
        domain_name = context.get('domain_name', 'General Decomposition')
        schema_desc = context.get('schema', 'JSON')
        hint = context.get('hint', '')
        
        # We need residual stats
        residual = state.residual
        # Summarize residual (don't send full array if huge)
        # For signal, maybe send first 100 points or stats?
        # Or assumes LLM can take full list if small?
        # The user example passed length 200. That's fine.
        
        # Limit residual display to avoid token limit issues in prototype
        preview_len = min(len(residual), 200)
        residual_preview = residual[:preview_len]
        
        prompt = f"""
OBJECTIVE: {domain_name}
Task: Analyze the residual signal and identify the dominant component.

Context:
{hint}

Residual Data (first {preview_len} points):
{residual_preview}

Format:
{schema_desc}

Return ONLY valid JSON.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            text = response.text
            # Extract JSON if needed (though response_mime_type helps)
            # Sometimes model adds extra text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                # Ensure it's a list if k>1, or wrap single item
                # RalphEngine expects a list of candidates
                return [data]
            else:
                return []
        except Exception as e:
            print(f"Gemini Agent Error: {e}")
            return []
