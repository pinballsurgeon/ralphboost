
from typing import Any, Dict, List, Tuple, Optional

def apply_code_patch(
    state: Dict[str, str],
    patch_ops: List[Dict[str, str]],
    allowed_paths: Optional[List[str]] = None
) -> Tuple[Dict[str, str], int, int]:
    """
    Applies patches to code state.
    state: Dict mapping file paths to file content.
    patch_ops: List of dicts with "path" and "value".
    allowed_paths: Optional list of allowed file paths (or prefixes?). 
                   For now, we enforce exact match or just directory check?
                   "Allowed paths only: src/ + tests/" implies directories.
                   But "Freeze passing files" implies specific files.
                   Let's support exact paths for now.
    """
    applied = 0
    skipped = 0
    
    new_state = state.copy()
    
    for op in patch_ops:
        path = op.get("path")
        value = op.get("value")
        
        if not path or value is None:
            skipped += 1
            continue
            
        # Check allowed paths
        if allowed_paths is not None:
            # Simple check: path must be in allowed_paths list
            # Or if allowed_paths contains directories, verify prefix?
            # User said "Allowed paths only: src/ + tests/".
            # If allowed_paths is ["src/", "tests/"], we check startsWith.
            # If it is ["main.py"], we check exact.
            # Let's support both: if allowed path ends with /, treat as dir.
            
            is_allowed = False
            for allowed in allowed_paths:
                if allowed.endswith("/"):
                    if path.startswith(allowed):
                        is_allowed = True
                        break
                else:
                    if path == allowed:
                        is_allowed = True
                        break
            
            if not is_allowed:
                skipped += 1
                continue
        
        new_state[path] = value
        applied += 1
        
    return new_state, applied, skipped
