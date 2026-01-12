# ENV Parser

Goal
Implement `parse_env(text: str) -> dict[str,str]`.

Rules
- Lines "KEY=VALUE"
- Ignore empty lines and comments (#).
- Whitespace around keys and values trimmed.

Tests
- "A=1\\nB=2" -> {"A":"1","B":"2"}
- "A=1\\n#x\\nB=2" -> {"A":"1","B":"2"}

Hidden checks
- duplicate keys last wins
- keys with underscores
