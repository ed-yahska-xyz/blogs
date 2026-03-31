# Zig Interfaces Tutorial

Compile-Time, Runtime, and Tagged Union Approaches (Using Shapes)

This tutorial demonstrates Zig's three approaches to polymorphism using a single example domain:
- Rectangle
- Square
- Circle
- Triangle

Each shape must implement:
- `area()`
- `perimeter()`

We will cover:
1. Compile-time interface (generics / duck typing)
2. Runtime interface (vtable pattern like `std.mem.Allocator`)
3. Tagged union (closed set polymorphism)

For each approach we'll also show:
- How to create arrays of shapes
- Memory management patterns
- Gotchas and best practices

---

## 1. Compile-Time Interface (Generic / Duck Typing)

### Concept

Zig does not have a `trait` or `interface` keyword.

Instead, a function can accept `anytype`, and the compiler verifies that the required methods exist at compile time.

There is:
- Zero runtime overhead
- No dynamic dispatch
- One compiled version per concrete type

---

### Shape Implementations

```zig
const std = @import("std");

const Rectangle = struct {
    width: f64,
    height: f64,

    pub fn area(self: Rectangle) f64 {
        return self.width * self.height;
    }

    pub fn perimeter(self: Rectangle) f64 {
        return 2.0 * (self.width + self.height);
    }
};

const Square = struct {
    side: f64,

    pub fn area(self: Square) f64 {
        return self.side * self.side;
    }

    pub fn perimeter(self: Square) f64 {
        return 4.0 * self.side;
    }
};

const Circle = struct {
    radius: f64,

    pub fn area(self: Circle) f64 {
        return std.math.pi * self.radius * self.radius;
    }

    pub fn perimeter(self: Circle) f64 {
        return 2.0 * std.math.pi * self.radius;
    }
};

const Triangle = struct {
    a: f64,
    b: f64,
    c: f64,

    pub fn perimeter(self: Triangle) f64 {
        return self.a + self.b + self.c;
    }

    pub fn area(self: Triangle) f64 {
        const s = self.perimeter() / 2.0;
        return std.math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c));
    }
};
```

---

### Generic "Interface" Function

```zig
fn printShapeInfo(shape: anytype) void {
    std.debug.print("Area: {d}\n", .{shape.area()});
    std.debug.print("Perimeter: {d}\n\n", .{shape.perimeter()});
}
```

If a type does not implement `area()` and `perimeter()`, compilation fails.

---

### Restricting the Interface Explicitly

```zig
fn printShapeInfoStrict(shape: anytype) void {
    const T = @TypeOf(shape);

    if (!@hasDecl(T, "area"))
        @compileError("Type must implement area()");
    if (!@hasDecl(T, "perimeter"))
        @compileError("Type must implement perimeter()");

    std.debug.print("Area: {d}\n", .{shape.area()});
}
```

---

### Arrays in Compile-Time Approach

You cannot store mixed shapes in one array:

```zig
var arr = [_]Rectangle{ ... }; // OK
```

But this does NOT work:

```zig
// Cannot mix types
var arr = [_]{ Rectangle{...}, Circle{...} };
```

Because there is no shared runtime type.

**Pattern: Separate Arrays Per Type**

```zig
var rectangles = [_]Rectangle{
    .{ .width = 4, .height = 3 },
    .{ .width = 2, .height = 5 },
};
```

---

### Memory Management

There are no special memory concerns here because:
- Shapes are stored by value
- No pointers required
- No heap needed

This is the safest approach.

---

### When to Use Compile-Time

- Performance critical code
- Type known at compile time
- No heterogeneous collections needed

---

## 2. Runtime Interface (VTable Pattern)

### Concept

This is how `std.mem.Allocator` works.

We create a `Shape` struct that contains:
- `ctx: *anyopaque`
- `vtable: *const VTable`

This allows heterogeneous collections like `[]Shape`.

---

### Define Runtime Interface

```zig
const Shape = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        area: *const fn (ctx: *anyopaque) f64,
        perimeter: *const fn (ctx: *anyopaque) f64,
    };

    pub fn area(self: Shape) f64 {
        return self.vtable.area(self.ctx);
    }

    pub fn perimeter(self: Shape) f64 {
        return self.vtable.perimeter(self.ctx);
    }
};
```

---

### Implement Rectangle for Runtime Interface

```zig
const Rectangle = struct {
    width: f64,
    height: f64,

    fn vArea(ctx: *anyopaque) f64 {
        const self: *Rectangle = @ptrCast(@alignCast(ctx));
        return self.width * self.height;
    }

    fn vPerimeter(ctx: *anyopaque) f64 {
        const self: *Rectangle = @ptrCast(@alignCast(ctx));
        return 2.0 * (self.width + self.height);
    }

    const vtable = Shape.VTable{
        .area = vArea,
        .perimeter = vPerimeter,
    };

    pub fn asShape(self: *Rectangle) Shape {
        return .{
            .ctx = self,
            .vtable = &vtable,
        };
    }
};
```

Other shapes follow the same pattern.

---

### Array of Runtime Shapes

```zig
var shapes = std.ArrayList(Shape).init(allocator);
try shapes.append(rect.asShape());
try shapes.append(circle.asShape());
```

Now we can mix types freely.

---

### Memory Gotcha

This is WRONG:

```zig
try shapes.append(Rectangle{ .width = 4, .height = 3 }.asShape());
```

Because the temporary `Rectangle` is destroyed immediately.

The `Shape` stores a pointer to invalid memory.

---

### Correct Lifetime Pattern

Concrete shapes must outlive the `Shape` handles.

**Stack Allocation**

```zig
var r = Rectangle{ .width = 4, .height = 3 };
try shapes.append(r.asShape());
```

**Heap Allocation**

```zig
const r = try allocator.create(Rectangle);
r.* = .{ .width = 4, .height = 3 };
try shapes.append(r.asShape());
```

Remember to free heap allocations later.

---

### Best Practices (Runtime Interface)

- Use arena allocator for many small shapes
- Ensure concrete types outlive interface handles
- Avoid runtime interfaces unless necessary
- Use when heterogeneous arrays are required

---

## 3. Tagged Union (Closed Set Polymorphism)

### Concept

When the set of possible types is known, use a tagged union.

This is the most idiomatic Zig solution for AST-style problems.

---

### Define Tagged Union

```zig
const Shape = union(enum) {
    rectangle: Rectangle,
    square: Square,
    circle: Circle,
    triangle: Triangle,

    pub fn area(self: Shape) f64 {
        return switch (self) {
            .rectangle => |r| r.width * r.height,
            .square => |s| s.side * s.side,
            .circle => |c| std.math.pi * c.radius * c.radius,
            .triangle => |t| blk: {
                const s = (t.a + t.b + t.c) / 2.0;
                break :blk std.math.sqrt(s * (s - t.a) * (s - t.b) * (s - t.c));
            },
        };
    }

    pub fn perimeter(self: Shape) f64 {
        return switch (self) {
            .rectangle => |r| 2.0 * (r.width + r.height),
            .square => |s| 4.0 * s.side,
            .circle => |c| 2.0 * std.math.pi * c.radius,
            .triangle => |t| t.a + t.b + t.c,
        };
    }
};
```

---

### Array of Tagged Union Shapes

```zig
var shapes = [_]Shape{
    .{ .rectangle = .{ .width = 4, .height = 3 } },
    .{ .circle = .{ .radius = 2 } },
    .{ .square = .{ .side = 5 } },
};
```

This works because all elements are the same type: `Shape`.

---

### Why Tagged Union Is Powerful

- No pointers required
- No heap required
- Exhaustive switch
- No runtime dispatch
- Safer than vtables

---

### Memory Behavior

Tagged union stores:
- The largest variant
- Plus a small tag

Everything is stored inline.

No lifetime issues.

---

## Comparison Summary

| Feature | Compile-Time | Runtime | Tagged Union |
|---|---|---|---|
| Heterogeneous array | No | Yes | Yes |
| Runtime cost | None | Function pointer call | None |
| Requires pointers | No | Yes | No |
| Safe lifetimes | Yes | Must manage | Yes |
| Open for extension | Yes | Yes | No |
| Compiler exhaustiveness | No | No | Yes |
| Idiomatic Zig choice | Often | When needed | Very common |

---

## Practical Guidance

**Use Compile-Time When:**
- Performance matters
- Types known at compile time
- No mixed collections needed

**Use Tagged Union When:**
- Finite set of variants
- AST nodes
- Game entity types
- Parsers

**Use Runtime Interface When:**
- Need plugin-style extensibility
- Types not known in advance
- Allocators, writers, loggers

---

## Final Takeaway

Zig gives you three tools:
1. Compile-time polymorphism (fastest)
2. Tagged unions (most idiomatic for closed sets)
3. Runtime interfaces (most flexible)

In real Zig projects:
- AST - tagged union
- Allocator - runtime interface
- Math utilities - compile-time
- Game entities - tagged union

Mastering when to choose each is a core Zig skill.
