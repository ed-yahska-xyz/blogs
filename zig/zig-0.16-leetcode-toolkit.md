# LeetCode in Zig 0.16 — Category → std API Checklist

Organized by the **NeetCode 150** categories. Every API name and snippet below was
verified against the std source shipped with **Zig 0.16.0**, and the core
data-structure idioms were compile-and-run tested. Where the standard library
gives you nothing, it says **(hand-roll)**.

---

## 0. Setup & the 0.16 gotchas that bite first

These are the breaking changes most likely to trip you when porting old tutorials:

| Thing | Old (pre-0.16) | **0.16** |
|---|---|---|
| Dynamic array | `ArrayList(T).init(a)` (managed) | `std.ArrayList(T)` is **unmanaged**: init `.empty`, methods take the allocator |
| General allocator | `GeneralPurposeAllocator(.{})` | renamed `std.heap.DebugAllocator(.{})`, init `.init` |
| Substring search | `std.mem.indexOf` / `indexOfScalar` | `std.mem.find` / `findScalar` (old names still exist but are **deprecated** → compile warnings) |
| Queue / fifo | `std.fifo.LinearFifo` | removed; use `std.Deque(T)` |
| Hash maps | `AutoHashMap(K,V).init(a)` | **unchanged** — still managed, `.init(a)` + `put(k,v)` |

Boilerplate for a standalone solution (no stdin/stdout needed — LeetCode hands you
typed args, so just write the function and call it from `main`):

```zig
const std = @import("std");

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();          // reports leaks in debug
    const a = gpa.allocator();
    // For throwaway solving, an arena is often nicer — allocate freely, free once:
    //   var arena = std.heap.ArenaAllocator.init(a);
    //   defer arena.deinit();
    //   const a2 = arena.allocator();
    _ = a;
    std.debug.print("debug output here\n", .{});  // your debugging channel
}
```

> **Allocator threading is the one thing LeetCode-in-Zig adds** that Python/Java/C++
> hide. Every dynamic structure below takes an allocator. An `ArenaAllocator` is the
> pragmatic cheat for contest-style code: you stop caring about individual `deinit`s.

---

## Fundamentals — arrays, slices, memory, strings

The category list below assumes these. If anything there looks unfamiliar, it's
probably one of these mechanics.

### Arrays vs. slices

This distinction is the one that trips up newcomers from Python/Java most.

- `[N]T` is a fixed-size **array** — length known at compile time, lives inline
  (stack or inside a struct). `var arr = [_]i32{ 1, 2, 3 };` infers `[3]i32`.
- `[]T` is a **slice** — a pointer **plus a runtime length** (`.ptr`, `.len`). This
  is what almost every function takes and returns. `[]const T` is the read-only form.
- Slicing an array yields a slice (half-open ranges): `arr[1..3]` (len 2), `arr[1..]`,
  or `&arr` for the whole thing. Use `.len` for the count, not a separate variable.
- A "2-D array" for DP is usually either a flat `[]T` indexed by `r*cols + c`, or a
  `[][]T` you allocate row by row.
- Two pointer flavors you'll meet in error messages: `[*]T` (many-item pointer, **no
  length** — comes from C) and sentinel-terminated `[:s]T` (e.g. a C string is `[:0]const u8`).

### Memory allocation

Everything dynamic goes through an `Allocator`. The handful you actually need:

```zig
const buf = try a.alloc(i32, n);    // []i32  — n-element slice
defer a.free(buf);
const p = try a.create(Node);       // *Node  — one value
defer a.destroy(p);
const copy = try a.dupe(u8, src);   // []u8   — owned copy of a slice
```

- Memory is **not** auto-zeroed. `@memset(buf, 0)` to clear, or initialize explicitly.
- Pair every `alloc`/`create` with `free`/`destroy` via `defer` (runs at scope exit).
- Pick the allocator once at the top. `DebugAllocator` reports leaks; wrapping it in an
  `ArenaAllocator` lets you allocate freely and `deinit()` everything at once — the
  pragmatic choice for solve-and-discard problems.

### Strings

There is no `String` type — a string is just `[]const u8` (a byte slice).

- String literals are `[]const u8`; char literals are `u8` (`'a'`). Multiline literals:
  ```zig
  const text =
      \\line one
      \\line two
  ;
  ```
- **Compare contents with `std.mem.eql(u8, x, y)`** — `==` does *not* compare bytes.
- Search: `std.mem.find(u8, hay, needle)` (substring → `?usize`),
  `std.mem.findScalar(u8, s, c)` (single char). *(These were `indexOf*` pre-0.16.)*
- Split/tokenize into an iterator: `std.mem.splitScalar(u8, s, ',')` /
  `std.mem.tokenizeScalar(...)`, then `while (it.next()) |part| { ... }`.
- Classify: `std.ascii.{ isDigit, isAlphabetic, toLower, toUpper, isWhitespace }`.
- Number ↔ string: `std.fmt.parseInt(i32, s, 10)`; build with
  `std.fmt.allocPrint(a, "{d}", .{x})` (allocates) or `std.fmt.bufPrint(buf, ...)` (into your buffer).
- Build incrementally with an `ArrayList(u8)`:
  ```zig
  var sb: std.ArrayList(u8) = .empty;
  defer sb.deinit(a);
  try sb.appendSlice(a, "id=");
  try sb.print(a, "{d}", .{n});   // format-append (0.16 style; no separate writer)
  const result = sb.items;         // []u8
  ```
- ASCII caveat (you've already hit this): byte indexing and `'a'..'z'` arithmetic are
  safe under LeetCode's ASCII constraints; reach for `std.unicode.Utf8View` only when a
  problem is genuinely multi-byte.

---

## 1. Arrays & Hashing

The workhorse category. You mainly need a growable array, a hash map, and a set.

- **Dynamic array** — `std.ArrayList(T)`
  ```zig
  var v: std.ArrayList(i32) = .empty;
  defer v.deinit(a);
  try v.append(a, 3);
  try v.appendSlice(a, &.{ 1, 2 });
  const slice = v.items;          // []i32
  _ = v.pop();                    // ?i32
  ```
- **Hash map** — `std.AutoHashMap(K, V)` (numeric/struct keys), `std.StringHashMap(V)` (`[]const u8` keys)
  ```zig
  var m = std.AutoHashMap(i32, i32).init(a);
  defer m.deinit();
  const gop = try m.getOrPut(key);          // frequency-count idiom
  if (gop.found_existing) gop.value_ptr.* += 1 else gop.value_ptr.* = 1;
  const got = m.get(key);                    // ?V
  _ = m.contains(key);
  ```
- **Set** — there is no dedicated set type. Use a map to `void`:
  `std.AutoHashMap(T, void)` / `std.StringHashMap(void)`, then `try set.put(x, {})`.
- **Composite keys** — `AutoHashMap` auto-hashes any struct/array of hashable fields,
  so grid coordinates work directly: `std.AutoHashMap([2]i32, V)`.
- **Sorting** — `std.mem.sort(T, slice, ctx, lessThanFn)` (stable). See §5 for comparator shape.

## 2. Two Pointers

No special structure — slice indexing on a (usually pre-sorted) array.
- Sort first when needed: `std.mem.sort`.
- Comparator helper for plain ascending order: `std.sort.asc(T)`.

## 3. Sliding Window

- Slice indexing for the window bounds.
- A running `std.AutoHashMap`/`StringHashMap` for "counts inside window."
- Monotonic-window problems (sliding-window maximum) want a **deque**: `std.Deque(T)` — see §11.

## 4. Stack

No `Stack` type — an `ArrayList` *is* the stack:
```zig
var st: std.ArrayList(i32) = .empty;
defer st.deinit(a);
try st.append(a, x);   // push
const top = st.pop();  // ?i32  (pop)
// peek: st.items[st.items.len - 1]
```
Covers monotonic-stack problems (next-greater-element, largest-rectangle) and iterative DFS.

## 5. Binary Search

std gives you the whole family. **Comparators here return `std.math.Order`, not a bool**
(this differs from sorting — easy to mix up):

```zig
fn cmpKey(key: i32, item: i32) std.math.Order { return std.math.order(key, item); }

const hit = std.sort.binarySearch(i32, slice, key, cmpKey);   // ?usize
const lo  = std.sort.lowerBound(i32, slice, key, cmpKey);     // usize
const hi  = std.sort.upperBound(i32, slice, key, cmpKey);     // usize
// also: std.sort.partitionPoint, std.sort.equalRange
```
For "binary search on the answer," **hand-roll** the `lo/hi/mid` loop — the boundary
logic is the whole problem and no library call expresses it.

Sorting comparator shape (note: **bool**, opposite of the search family):
```zig
fn lessThan(_: void, a_: i32, b_: i32) bool { return a_ < b_; }
std.mem.sort(i32, slice, {}, lessThan);
// or std.sort.pdq(...) for unstable+faster; std.sort.asc(i32)/desc(i32) for the trivial cases
```

## 6. Linked List

LeetCode hands you its own node, so **define your own** — don't reach for
`std.SinglyLinkedList`/`std.DoublyLinkedList` (those are *intrusive* in 0.16 and add
friction here):
```zig
const ListNode = struct { val: i32, next: ?*ListNode = null };
```
What you actually need from the language: **optional pointers** (`?*ListNode`), pointer
reassignment for reverse/merge/cycle-detect, and an allocator to `create` new nodes:
```zig
const node = try a.create(ListNode);
node.* = .{ .val = 5 };
```

## 7. Trees

Same story — define your own node, lean on recursion + optionals:
```zig
const TreeNode = struct { val: i32, left: ?*TreeNode = null, right: ?*TreeNode = null };
```
- DFS (pre/in/post): recursion over `?*TreeNode`.
- BFS / level-order: a queue of node pointers — `std.Deque(*TreeNode)` (§11).

## 8. Tries

**Hand-roll.** Lowercase-only alphabet is the common case:
```zig
const TrieNode = struct {
    children: [26]?*TrieNode = .{null} ** 26,
    is_end: bool = false,
};
```
For arbitrary alphabets swap `children` for `std.AutoHashMap(u8, *TrieNode)`. Allocator-backed.

## 9. Heap / Priority Queue

`std.PriorityQueue(T, Context, compareFn)`. The **comparator returns `std.math.Order`**;
ascending order ⇒ **min-heap**. The `Context` slot lets you compare by external data
(e.g. order indices by a `dist[]` array) — that's the idiomatic way to avoid packing
keys into `T`.
```zig
fn order(_: void, a_: i32, b_: i32) std.math.Order { return std.math.order(a_, b_); }
const PQ = std.PriorityQueue(i32, void, order);

var pq: PQ = .empty;            // or PQ.initContext(ctx) when Context isn't void
defer pq.deinit(a);
try pq.push(a, 7);
const top = pq.peek();          // ?T  (the min)
const x   = pq.pop();           // ?T
```
"k largest" ⇒ keep a **min-heap of size k**; "k smallest" ⇒ max-heap (flip the `order`).

## 10. Backtracking

Recursion + an `ArrayList` as the mutable path:
```zig
try path.append(a, choice);
try backtrack(...);
_ = path.pop();          // undo
```
Permutations/subsets/combinations/N-queens all fit this; visited-tracking via a bool
slice or `std.DynamicBitSet`.

## 11. Graphs

- **Adjacency list** — `std.ArrayList(std.ArrayList(u32))`, or `std.AutoHashMap(u32, std.ArrayList(u32))` for sparse/non-contiguous node ids. You build it from the edge list yourself.
- **Grid problems** — a 2-D `[][]T` or a flat slice with `r*cols + c` index math; neighbors via a `[_][2]i32{ .{1,0}, .{-1,0}, .{0,1}, .{0,-1} }` delta table.
- **BFS queue** — `std.Deque(T)`:
  ```zig
  var q: std.Deque(u32) = .empty;
  defer q.deinit(a);
  try q.pushBack(a, start);
  while (q.popFront()) |node| { ... }   // popFront / popBack return ?T
  ```
- **DFS** — recursion, or an `ArrayList` used as a stack.
- **Visited set** — `std.DynamicBitSet` (size known at runtime), a `[]bool`, or a hash set.

## 12. Advanced Graphs

- **Dijkstra / Prim** — `std.PriorityQueue` keyed by distance (§9).
- **Union-Find / DSU** — **hand-roll** parent + rank/size arrays with path compression:
  ```zig
  fn find(parent: []u32, x: u32) u32 {
      var r = x;
      while (parent[r] != r) : (r = parent[r]) parent[r] = parent[parent[r]];
      return r;
  }
  ```
- **Topological sort** — in-degree array + `std.Deque` (Kahn's), or DFS post-order.

## 13. 1-D DP

- **Tabulation** — a `[]T` (use `try a.alloc(T, n)`) or a fixed-size array.
- **Memoization** — `std.AutoHashMap(StateKey, V)`; for a single integer state a plain
  `[]?V` slice initialised to `null` is faster.

## 14. 2-D DP

- **Table** — either a flat `[]T` with manual `i*cols + j` indexing (fewer allocations,
  better cache behavior), or an allocated `[][]T`.
- **Memo** — `std.AutoHashMap([2]i32, V)` (struct/array keys auto-hash, §1).
- Interval DP (Burst Balloons, Matrix-Chain) lives here — and unlike row-sequential DP,
  its independent subproblems on each diagonal are where parallelism actually pays off.

## 15. Greedy

Usually `std.mem.sort` + a single linear pass; sometimes a `std.PriorityQueue` to always
pull the current best (task scheduling, etc.). No new primitives.

## 16. Intervals

- Sort by start with a struct comparator:
  ```zig
  const Interval = struct { start: i32, end: i32 };
  fn byStart(_: void, x: Interval, y: Interval) bool { return x.start < y.start; }
  std.mem.sort(Interval, intervals, {}, byStart);
  ```
- Then a linear merge. "Meeting Rooms II" type problems add a `std.PriorityQueue` of end times.

## 17. Math & Geometry

- `std.math.gcd`, `std.math.pow` / `std.math.powi`, `std.math.sqrt`, `std.math.divCeil`, `std.math.log2_int`.
- `@min` / `@max` are **builtins** now (not `std.math.min/max`).
- Bounds: `std.math.maxInt(T)` / `std.math.minInt(T)` for DP sentinels.
- **Overflow is a hard error in safe builds** — for sums that exceed `i32`, use `i64`/`u64`,
  or wrapping/saturating ops (`+%`, `+|`) where wraparound is intended. Modular counting
  (`% 1_000_000_007`) just uses `%` on a wide enough type.
- Parsing/formatting (rarely needed, but handy): `std.fmt.parseInt`, `std.fmt.bufPrint`, `std.fmt.allocPrint`.

## 18. Bit Manipulation

- Native operators: `& | ^ ~ << >>`.
- Builtins: `@popCount`, `@clz`, `@ctz`, `@bitReverse`.
- **Bitmask DP over subsets** — iterate `mask` over `0..(@as(usize,1) << n)`; watch shift-type
  rules (the shift amount needs a type that can't over-shift, so you'll see `@intCast`/`@truncate`
  around shifts more than in C).
- Fixed-size bitsets: `std.StaticBitSet(N)`; runtime size: `std.DynamicBitSet`.

---

### Quick mental model: what's std vs what you build

| You get from std | You hand-roll |
|---|---|
| dynamic array, hash map/set, deque, priority queue, sort + binary-search family, bitsets, gcd/pow/etc. | linked-list node, tree node, trie, union-find, the actual DP recurrences and "binary search on answer" loop |

### String note (you've hit this already)
Strings are `[]const u8` byte slices. LeetCode constraints almost always promise ASCII, so
byte indexing and `'a'..'z'` arithmetic are safe — you don't need `std.unicode.Utf8View`
unless a problem explicitly involves multi-byte text. Useful helpers: `std.mem.eql`,
`std.mem.find` (substring), `std.mem.findScalar` (char), `std.ascii.{isDigit,toLower,...}`,
`std.mem.{splitScalar,tokenizeScalar}`.
