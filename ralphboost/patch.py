from typing import Any, Dict, List, Tuple


def seed_state() -> Dict[str, Any]:
    return {
        "iteration_summary": "",
        "hypotheses": ["", "", ""],
        "metrics": [
            {"name": "", "definition": "", "formula_or_calc": ""},
            {"name": "", "definition": "", "formula_or_calc": ""},
            {"name": "", "definition": "", "formula_or_calc": ""},
        ],
        "micro_tasks": [],
        "anti_gaming": {"mechanism": "", "hidden_checks": []},
        "actions": [],
        "completion_signal": {"done": False, "reason": ""},
    }


def _split_path(path: str) -> List[str]:
    return [p for p in path.strip().split(".") if p]


def _is_index(seg: str) -> bool:
    return seg.isdigit()


def _ensure_list_size(lst: List[Any], idx: int) -> None:
    while len(lst) <= idx:
        lst.append(None)


def set_path(root: Any, path: str, value: Any) -> bool:
    parts = _split_path(path)
    if not parts:
        return False
    cur = root
    for i, seg in enumerate(parts):
        is_last = i == len(parts) - 1
        if _is_index(seg):
            if not isinstance(cur, list):
                return False
            idx = int(seg)
            _ensure_list_size(cur, idx)
            if is_last:
                cur[idx] = value
                return True
            if cur[idx] is None:
                cur[idx] = {}
            cur = cur[idx]
        else:
            if not isinstance(cur, dict):
                return False
            if is_last:
                cur[seg] = value
                return True
            if seg not in cur or cur[seg] is None:
                cur[seg] = {}
            cur = cur[seg]
    return False


def apply_patch(
    state: Dict[str, Any],
    patch_ops: List[Dict[str, Any]],
    allowed_roots: List[str],
) -> Tuple[Dict[str, Any], int, int]:
    applied = 0
    skipped = 0
    for op in patch_ops:
        path = str(op.get("path", "")).strip()
        value = op.get("value")
        if not path:
            skipped += 1
            continue
        root = path.split(".")[0]
        if allowed_roots and root not in allowed_roots:
            skipped += 1
            continue
        ok = set_path(state, path, value)
        if ok:
            applied += 1
        else:
            skipped += 1
    return state, applied, skipped
