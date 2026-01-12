# LRU Cache

Goal
Implement `LRUCache(capacity)` with `get(key)` and `put(key, value)`.

Constraints
- get returns -1 if missing
- O(1) average operations

Tests
- capacity=2
- put(1,1), put(2,2), get(1)=1
- put(3,3) evicts key 2
- get(2)=-1

Hidden checks
- repeated get on hot key updates recency
