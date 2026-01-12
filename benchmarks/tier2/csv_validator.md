# CSV Schema Validator

Goal
Implement `validate_csv(csv_text: str, schema: dict) -> list[str]` returning error messages.

Schema
- {"columns": ["a","b"], "types": {"a":"int","b":"str"}}

Tests
- Valid rows return []
- Missing column returns error with row number
- Type mismatch returns error

Hidden checks
- empty rows
- trailing commas
