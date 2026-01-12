# Prime Sieve

Goal
Implement `sieve(n: int) -> list[int]` returning all primes <= n.

Constraints
- n can be 0..1_000_000
- Must be O(n log log n) or better.

Tests
- n=0 -> []
- n=1 -> []
- n=2 -> [2]
- n=10 -> [2,3,5,7]
- n=100 -> length 25

Hidden checks
- n=3
- n=97
