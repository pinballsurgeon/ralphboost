import os
import subprocess
from typing import Any, Dict, List, Tuple


def _is_safe_path(path: str) -> bool:
    if not path or os.path.isabs(path):
        return False
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    return ".." not in parts


def execute_actions(
    actions: List[Dict[str, Any]],
    root_dir: str,
    allow_commands: bool,
) -> Tuple[int, int, List[str]]:
    write_count = 0
    cmd_count = 0
    errors: List[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        a_type = action.get("type")
        if a_type == "write_file":
            path = action.get("path", "")
            if not isinstance(path, str) or not _is_safe_path(path):
                errors.append(f"unsafe path: {path}")
                continue
            content = action.get("content")
            if content is None:
                content = action.get("text", "")
            if not isinstance(content, str):
                errors.append(f"non-string content for {path}")
                continue
            abs_path = os.path.join(root_dir, path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            write_count += 1
        elif a_type == "command":
            if not allow_commands:
                continue
            cmd = action.get("cmd")
            if not isinstance(cmd, str) or not cmd.strip():
                errors.append("empty command")
                continue
            result = subprocess.run(cmd, shell=True, cwd=root_dir)
            if result.returncode != 0:
                errors.append(f"command failed: {cmd}")
            cmd_count += 1
    return write_count, cmd_count, errors
