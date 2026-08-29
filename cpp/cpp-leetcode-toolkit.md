# LeetCode in C++ — Category → STL Checklist

Organized by the **NeetCode 150** categories, mirroring the Zig toolkit so you can
compare. Verified against **g++ 13 / C++20**; core idioms were compile-and-run tested.
Where the standard library gives you nothing, it says **(hand-roll)**.

The headline difference from Zig: the STL gives you far more out of the box (a real
`stack`, `queue`, `priority_queue`, ordered + unordered maps), memory is mostly automatic
via RAII, and `==` compares string contents. The thing to stay paranoid about is **integer
overflow** — `int` is 32-bit and silently wraps.

---

## 0. Setup & gotchas

```cpp
#include <bits/stdc++.h>   // pulls in the whole STL (GCC/Clang; fine for contests)
using namespace std;
// LeetCode hands you a method signature — no I/O needed. For raw competitive I/O:
//   ios::sync_with_stdio(false); cin.tie(nullptr);   // fast cin/cout
```

The gotchas that actually cost you points:

| Trap | Why it bites |
|---|---|
| `int` overflow | 32-bit, silently wraps at ~2.1e9. Sums/products often need `long long`. Use `1LL << k`. |
| Uninitialized locals | `int x;` is garbage; `vector<int> v(n);` *is* zero-initialized, but `int a[n];` is not. |
| `[]` on a map **inserts** | `if (mp[k])` creates `k` with value 0. Use `.count(k)` / `.find(k)` to test membership. |
| Iterator invalidation | mutating a container during iteration invalidates iterators/refs. |
| `unordered_map` worst case | adversarial inputs can force O(n) buckets; ordered `map` (O(log n)) is the safe fallback. |

---

## Fundamentals — arrays, vectors, memory, strings

### Arrays vs. vectors

- C arrays `int a[n]` are fixed and don't carry a length — avoid them; prefer the STL.
- `std::array<T, N>` — fixed compile-time size, knows its `.size()`.
- **`std::vector<T>`** — the workhorse dynamic array. `.push_back`, `.pop_back`, `.back()`,
  `.size()`, `v[i]`, range-for `for (int x : v)`.
  ```cpp
  vector<int> v(n, 0);             // n zeros
  vector<vector<int>> g(r, vector<int>(c, 0));   // r×c grid
  ```
- `std::span<T>` (C++20) is a non-owning view (pointer + length) — the closest thing to a
  Zig slice — but for LeetCode you mostly just pass `vector<T>&`.

### Memory

- **RAII**: containers and `std::string` free themselves at scope exit. No manual `free`.
- You only touch raw `new`/`delete` for the linked-list / tree nodes LeetCode defines, and
  even then leaks don't matter for judging. Smart pointers (`unique_ptr`, `shared_ptr`)
  exist but LeetCode node types use raw `*`.
- Pass big objects by reference (`vector<int>& v`) to avoid copies; `const&` for read-only.

### Strings

`std::string` is a real, mutable string type — much friendlier than Zig's byte slice.

- Build with `+` / `+=` (amortized O(1) append), index with `s[i]` (a `char`).
- **`==` compares contents** (`s == "abc"` works) — unlike C strings.
- `s.substr(i, len)`, `s.find("x")` → index or `string::npos`, `s.size()`.
- Number ↔ string: `stoi`, `stoll`, `to_string`. Char math: `c - 'a'`, `c - '0'`.
- Classify: `isdigit(c)`, `isalpha(c)`, `tolower(c)` (from `<cctype>`).
- Split: `stringstream ss(s); while (getline(ss, tok, ','))`.

---

## 1. Arrays & Hashing
- `vector<T>`; `unordered_map<K,V>` / `unordered_set<K>` (avg O(1)); ordered `map`/`set`
  (O(log n), sorted iteration, `lower_bound`).
- Frequency count: `mp[x]++`. Membership test: `mp.count(x)` (not `mp[x]`, which inserts).
- Composite keys: `map<pair<int,int>,V>` works directly; `unordered_map<pair<...>>` needs a
  custom hash, so reach for ordered `map` with `pair`/`tuple` keys.

## 2. Two Pointers
- Index into a `vector`, usually after `sort(v.begin(), v.end())`. No special container.

## 3. Sliding Window
- `vector` + a running `unordered_map`/`Counter`-style map for window contents.
- Monotonic window (sliding-window maximum) ⇒ `std::deque<int>` of indices.

## 4. Stack
- `std::stack<T>`: `.push(x)`, `.top()`, `.pop()` (pop returns void — read `.top()` first).
  A `vector` works too. Covers monotonic stack, parentheses, iterative DFS.

## 5. Binary Search
- `binary_search(b, e, x)` → bool; `lower_bound`/`upper_bound` → iterators (subtract
  `v.begin()` for an index); `equal_range`. Comparator optional (3rd arg).
- "Binary search on the answer" ⇒ **hand-roll** the `lo/hi/mid` loop.

## 6. Linked List
- Use LeetCode's `struct ListNode { int val; ListNode* next; };`. Raw pointers, a dummy
  head node for clean inserts. `new ListNode(v)` to allocate.

## 7. Trees
- LeetCode's `struct TreeNode { int val; TreeNode *left, *right; };`. Recursion for DFS;
  `std::queue<TreeNode*>` for BFS / level-order.

## 8. Tries
- **Hand-roll**: `struct Node { Node* ch[26]{}; bool end=false; };` (lowercase), or
  `unordered_map<char, Node*>` for arbitrary alphabets.

## 9. Heap / Priority Queue
- `std::priority_queue<T>` is a **max-heap by default**. Min-heap:
  ```cpp
  priority_queue<int, vector<int>, greater<int>> minh;
  ```
- `.push(x)`, `.top()`, `.pop()`. Custom order via a comparator type or lambda-with-decltype.
- "k largest" ⇒ min-heap of size k. Keyed entries ⇒ `priority_queue<pair<int,int>>`.

## 10. Backtracking
- Recursion + a `vector<T>` path: `path.push_back(x); recurse(); path.pop_back();`.
- Visited via `vector<bool>` or a bitmask.

## 11. Graphs
- Adjacency list `vector<vector<int>> adj(n)`; grids as `vector<vector<int>>` + a
  `{{1,0},{-1,0},{0,1},{0,-1}}` delta table.
- BFS: `std::queue`. DFS: recursion or `std::stack`. Visited: `vector<bool>`.

## 12. Advanced Graphs
- Dijkstra/Prim: `priority_queue` of `{dist, node}` (min-heap via `greater`).
- **Union-Find (hand-roll)**: `parent`/`rank` vectors with path compression.
- Topological sort: in-degree `vector` + `queue` (Kahn's), or DFS post-order.

## 13. 1-D DP
- `vector<int> dp(n)`; memo with `unordered_map` or a `vector<int>` of sentinels (`-1`).

## 14. 2-D DP
- `vector<vector<int>> dp(r, vector<int>(c))`; memo `map<pair<int,int>,int>`.

## 15. Greedy
- `sort` (often with a lambda comparator) + a linear pass; sometimes a `priority_queue`.

## 16. Intervals
- Sort by start with a lambda:
  ```cpp
  sort(iv.begin(), iv.end(), [](auto& a, auto& b){ return a[0] < b[0]; });
  ```
  then linear merge. "Meeting Rooms II" ⇒ `priority_queue` of end times.

## 17. Math & Geometry
- `<numeric>`: `std::gcd`, `std::lcm` (C++17). `<cmath>`: `pow`, `sqrt`, `abs`.
- Bounds: `INT_MAX`/`INT_MIN` (`<climits>`), `LLONG_MAX`, or `numeric_limits<T>::max()`.
- **Overflow discipline**: promote to `long long` before multiplying; modular arithmetic
  (`% 1000000007`) on `long long`. There is no free big-integer type.

## 18. Bit Manipulation
- Operators `& | ^ ~ << >>`. Builtins: `__builtin_popcount` (`__builtin_popcountll` for 64-bit),
  `__builtin_clz`, `__builtin_ctz`.
- Use `1LL << k` for shifts past bit 31. Lowest set bit: `x & -x`.
- Fixed-size sets / subset DP: `std::bitset<N>`.

---

### Quick mental model: what's STL vs what you build

| You get from the STL | You hand-roll |
|---|---|
| vector, array, unordered_map/set, ordered map/set, stack, queue, deque, priority_queue, sort + binary-search family, bitset, gcd/lcm | linked-list & tree nodes (LeetCode-provided), trie, union-find, DP recurrences, "binary search on answer" |

### Overflow note (the C++ tax)
`int` wraps silently at ~2.1×10⁹. The instant a sum or product *might* exceed that, switch
to `long long`. This is the single most common source of "passes samples, fails hidden
tests" in C++ — the opposite of Python, where ints never overflow.