# LeetCode in Python — Category → stdlib Checklist

Organized by the **NeetCode 150** categories, mirroring the Zig and C++ toolkits.
Verified against **Python 3.12**; core idioms were run-tested. Where the standard library
gives you nothing, it says **(hand-roll)**.

The headline differences from C++/Zig: **integers never overflow** (arbitrary precision),
memory is fully automatic, and the heavy lifting lives in built-in types (`list`, `dict`,
`set`) plus a few batteries-included modules (`collections`, `heapq`, `bisect`,
`functools`). The thing to stay aware of is **speed and recursion depth** — Python is slow
and the default recursion limit (~1000) is shallow for deep trees/graphs.

---

## 0. Setup & gotchas

```python
from collections import Counter, defaultdict, deque
import heapq, bisect, math, sys
from functools import cache          # memoization decorator (3.9+)
# Deep DFS on trees/graphs? Raise the recursion ceiling:
sys.setrecursionlimit(10**6)
# LeetCode gives you `class Solution: def fn(self, ...)` — no I/O needed.
```

The gotchas that actually cost you:

| Trap | Why it bites |
|---|---|
| `[[0]*c]*r` | Makes **r references to one row** — writing `grid[0][0]` changes every row. Use `[[0]*c for _ in range(r)]`. |
| `+=` on strings in a loop | `str` is immutable → O(n²). Collect in a `list`, then `''.join()`. |
| Default mutable arg | `def f(x, seen=[])` shares one list across calls. Use `None` + create inside. |
| Recursion limit | Default ~1000; deep recursion hits `RecursionError`. Raise it (above) or go iterative. |
| Speed | Tight numeric loops are ~50× C++. Lean on built-ins/`Counter`/comprehensions. |

The flip side — things that are *free* in Python: no integer overflow, built-in modular
`pow`, `@cache` memoization, and slicing.

---

## Fundamentals — lists, memory, strings

### "Arrays" = lists

- `list` is the dynamic array. `.append`, `.pop()` (end) / `.pop(0)` (front, O(n) — use a
  `deque` instead), `a[i]`, `len(a)`, negative indices `a[-1]`.
- **Slicing** copies: `a[1:3]`, `a[::-1]` (reverse), `a[:]` (shallow copy).
- 2-D grid — **always** the comprehension form to avoid aliasing:
  ```python
  grid = [[0]*c for _ in range(r)]
  ```
- Comprehensions are idiomatic and fast: `[x*x for x in a if x > 0]`.

### Memory & mutability

- Fully garbage-collected — no allocation/free, ever.
- Mutable: `list`, `dict`, `set`. Immutable: `int`, `str`, `tuple`, `frozenset`.
  Only immutable (hashable) things can be dict keys / set members — so a coordinate key is a
  `tuple` `(r, c)`, never a `list`.

### Strings

`str` is immutable Unicode.

- Concatenate small bits with `+`; for loops, **collect in a list and `''.join(parts)`**.
- `s[i]` is a 1-char `str`, `s[1:3]` slices, `"sub" in s` tests containment,
  `s.find("sub")` → index or `-1`.
- `s.split(",")`, `" ".join(parts)`, `s.strip()`, `s.replace(a, b)`.
- Number ↔ string: `int("42")`, `str(7)`. Char ↔ code: `ord('a')` → 97, `chr(98)` → `'b'`.
  Char-index math: `ord(c) - ord('a')`.
- Classify: `c.isdigit()`, `c.isalpha()`, `c.lower()`. f-strings for formatting: `f"{x:03d}"`.

---

## 1. Arrays & Hashing
- `list`; `dict` / `set`; plus `collections.Counter` (frequency map with `.most_common()`)
  and `collections.defaultdict(int|list|set)` (no missing-key handling).
- Frequency count: `Counter(s)` or `cnt[x] += 1` with a `defaultdict(int)`.
- Composite keys are trivial: any `tuple` is hashable, so `seen[(r, c)] = ...` just works.

## 2. Two Pointers
- Index into a `list`, usually after `a.sort()` / `sorted(a)`. No special structure.

## 3. Sliding Window
- `list` + a running `Counter`/`dict` for window contents.
- Monotonic window (sliding-window maximum) ⇒ `collections.deque` of indices.

## 4. Stack
- A plain `list`: `.append(x)` (push), `.pop()` (pop+return), `a[-1]` (peek).
  Covers monotonic stack, parentheses, iterative DFS.

## 5. Binary Search
- `bisect` module: `bisect_left` / `bisect_right` → insertion index, `insort` to insert.
  Search by key with the `key=` parameter (3.10+).
- "Binary search on the answer" ⇒ **hand-roll** the `lo/hi/mid` loop.

## 6. Linked List
- LeetCode's `ListNode(val, next)`. Plain attribute assignment (`cur.next = ...`); a dummy
  head simplifies edits. No allocator — `ListNode(v)` just constructs.

## 7. Trees
- LeetCode's `TreeNode(val, left, right)`. Recursion for DFS (mind the recursion limit);
  `collections.deque` for BFS / level-order.

## 8. Tries
- **Hand-roll** with nested dicts: `root = {}`, then `node = node.setdefault(ch, {})`, with
  a sentinel key like `node['#'] = True` to mark word ends. Or a small `class TrieNode`.

## 9. Heap / Priority Queue
- `heapq` is a **min-heap only**, operating on a plain `list`:
  ```python
  h = []; heapq.heappush(h, x); smallest = heapq.heappop(h); h[0]  # peek
  ```
- **Max-heap**: push negatives (`heappush(h, -x)`), negate on pop.
- Keyed priority: push tuples `(priority, item)`; ties break on the next tuple element, so
  include a tiebreaker (e.g. a counter) if items aren't comparable.
- "k largest" ⇒ `heapq.nlargest(k, ...)` or a size-k min-heap.

## 10. Backtracking
- Recursion + a `list` path: `path.append(x); backtrack(); path.pop()`.
- Visited via a `set` or `list`. `itertools` (`permutations`, `combinations`, `product`)
  often replaces hand-written backtracking outright.

## 11. Graphs
- Adjacency list `defaultdict(list)`; grids as a 2-D `list` + a
  `[(1,0),(-1,0),(0,1),(0,-1)]` delta tuple list.
- BFS: `collections.deque` (`popleft()`). DFS: recursion or a `list` stack.
  Visited: a `set`.

## 12. Advanced Graphs
- Dijkstra/Prim: `heapq` of `(dist, node)`.
- **Union-Find (hand-roll)**: a `parent` list/dict with path compression (a tiny class is tidy).
- Topological sort: in-degree `dict` + `deque` (Kahn's), or DFS post-order.

## 13. 1-D DP
- A `list` `dp = [0]*n`; or memoize a recursion with **`@cache`** / `@lru_cache(None)` —
  often the entire solution.

## 14. 2-D DP
- 2-D `list` via comprehension (avoid the aliasing trap); or `@cache` on a 2-arg recursive
  function, which sidesteps building the table at all.
- Manual memo: a `dict` keyed by a `tuple` state.

## 15. Greedy
- `sorted(..., key=...)` + a linear pass; sometimes a `heapq`.

## 16. Intervals
- `intervals.sort(key=lambda x: x[0])`, then linear merge. "Meeting Rooms II" ⇒ `heapq` of
  end times.

## 17. Math & Geometry
- `math.gcd`, `math.lcm`, `math.isqrt`, `math.comb`, `math.perm`.
- **Built-in modular exponentiation**: `pow(base, exp, mod)` — no need to hand-roll.
- Infinity sentinels: `float('inf')` / `float('-inf')`.
- **No overflow** — integers are arbitrary precision, so `2**100` and huge factorials Just
  Work. This removes the single biggest C++ footgun.

## 18. Bit Manipulation
- Operators `& | ^ ~ << >>`. Population count: `(x).bit_count()` (3.10+) or
  `bin(x).count('1')`. Bit length: `x.bit_length()`.
- Shifts are arbitrary-precision (`1 << 200` is fine). Lowest set bit: `x & -x`.
- Subset-mask DP: iterate `for mask in range(1 << n)`.

---

### Quick mental model: what's built-in vs what you build

| You get from Python | You hand-roll |
|---|---|
| list, dict, set, Counter, defaultdict, deque, heapq, bisect, sort/sorted, `@cache` memo, modular `pow`, itertools, big ints | linked-list & tree nodes (LeetCode-provided), trie, union-find, DP recurrences, "binary search on answer" |

### Speed note (the Python tax)
Python trades raw speed for expressiveness. Two habits keep you under time limits:
push numeric work into built-ins/comprehensions/`Counter` instead of explicit loops, and
use `@cache` so recursive DP isn't recomputed. When a problem is genuinely loop-bound and
TLE-ing, that's usually a signal to switch languages — not to micro-optimize Python.