# Top-K Streaming

Goal
Implement `topk(stream: list[int], k: int) -> list[int]` returning k largest values.

Constraints
- Must handle duplicates.
- Output sorted descending.

Tests
- stream=[3,1,5,2], k=2 -> [5,3]
- stream=[], k=3 -> []
- stream=[-1,-2,-3], k=2 -> [-1,-2]

Hidden checks
- stream size 10_000 with duplicates
