# Reference: NumPy Fundamentals (for numpy-transformer)

> A targeted NumPy guide covering only the operations actually used in
> `nn/numpy-transformer/`. Every concept is paired with a pointer to where
> it shows up in `src/numpy_transformer/layers.py`, `training.py`, or
> `inference.py`, so you can flip from theory to real usage in two clicks.

[< Back to Overview](00_overview.md)

---

## Table of Contents

1. [Mental model: what is an `ndarray`?](#1-mental-model-what-is-an-ndarray)
2. [Shapes, axes, and `dtype`](#2-shapes-axes-and-dtype)
3. [Creating arrays](#3-creating-arrays)
4. [Indexing and slicing](#4-indexing-and-slicing)
5. [Reshape, transpose, `swapaxes` (the MHA toolkit)](#5-reshape-transpose-swapaxes-the-mha-toolkit)
6. [Broadcasting](#6-broadcasting)
7. [Element-wise math](#7-element-wise-math)
8. [Reductions: `axis` and `keepdims`](#8-reductions-axis-and-keepdims)
9. [Matrix multiplication: `@` and `np.matmul`](#9-matrix-multiplication--and-npmatmul)
10. [Boolean masks and `np.where`](#10-boolean-masks-and-npwhere)
11. [Fancy indexing and scatter (embedding lookup)](#11-fancy-indexing-and-scatter-embedding-lookup)
12. [In-place operations](#12-in-place-operations)
13. [Random numbers](#13-random-numbers)
14. [Numerical stability patterns](#14-numerical-stability-patterns)
15. [Common pitfalls](#15-common-pitfalls)
16. [Quick-reference cheat sheet](#16-quick-reference-cheat-sheet)

---

## 1. Mental model: what is an `ndarray`?

An `np.ndarray` is a **typed, multi-dimensional, contiguous block of memory**
plus a small header telling NumPy how to walk through it (shape, strides, dtype).

The two consequences that matter most for this project:

- **Vectorized math is fast.** Loops in Python are slow; expressions like
  `x + y`, `x @ W`, `x.sum(axis=-1)` push the loop into compiled C code.
  *We never write `for i in range(...)` over array elements.*
- **Shape is everything.** Almost every bug you'll hit when porting math
  from a paper to NumPy is a shape bug: a `(B, n, d)` where you expected
  `(B, d, n)`, or a forgotten `keepdims=True` that broke broadcasting.

> **Convention used in the codebase**: leading dimensions are batch-like
> (`...` in docstrings), and the **last** axis is the "feature" axis
> (`d_model`, `vocab_size`, etc.). Most reductions use `axis=-1`.

---

## 2. Shapes, axes, and `dtype`

```python
import numpy as np

x = np.zeros((32, 5, 64))       # 3D array
x.shape                          # (32, 5, 64) -- a tuple
x.ndim                           # 3
x.size                           # 32 * 5 * 64 = 10240
x.dtype                          # dtype('float64')
```

### Axes are numbered from 0; negative indices count from the end

| Axis | Meaning in our code | Example (logits) |
|------|---------------------|-------------------|
| `0`  | batch              | `B = 32` |
| `1`  | sequence position  | `n = 6` |
| `-1` | feature / vocab    | `V = 13` |

`axis=-1` is the universal idiom for "operate over the feature axis,
regardless of how many leading batch dims I have." Used everywhere in
softmax, LayerNorm, log-softmax, argmax.

**Worked example -- `axis=0` vs `axis=-1`:**

```python
x = np.array([[1., 2., 3.],
              [4., 5., 6.]])              # shape (2, 3) -- 2 rows, 3 cols

x.sum(axis=0)     # array([5., 7., 9.])   shape (3,) -- one number per column
x.sum(axis=-1)    # array([6., 15.])      shape (2,) -- one number per row
x.sum()           # 21.0                    -- collapse everything
```

### Why "the axis of length `x.shape[k]` disappears"

When you call `x.sum(axis=k)`, NumPy walks along axis `k`, collapses
everything it finds into a single number per slice, then repeats for
every position along the **other** axes. The axis you walked over is
squashed to one value -- so it no longer counts as a dimension in the
output.

Picture the `(2, 3)` array above:

```
            col 0   col 1   col 2     <-- axis 1 runs left-right (length 3)
row 0:        1       2       3
row 1:        4       5       6
  ^
  axis 0 runs top-bottom (length 2)
```

- `x.sum(axis=0)` walks top-to-bottom (down each column) -> three sums
  (`1+4`, `2+5`, `3+6`) = `[5, 7, 9]`. The "down" direction (axis 0,
  length 2) is gone; only the "across" direction (axis 1, length 3)
  survives -> shape `(3,)`.
- `x.sum(axis=-1)` walks left-to-right (across each row) -> two sums
  (`1+2+3`, `4+5+6`) = `[6, 15]`. Axis 1 (length 3) is gone; axis 0
  (length 2) survives -> shape `(2,)`.

The general rule, for any shape `(d_0, d_1, ..., d_k, ..., d_n)`:

```
x.sum(axis=k).shape == (d_0, ..., d_{k-1}, d_{k+1}, ..., d_n)
                                          ^
                                   d_k is removed
```

3D sanity check:

```python
x = np.zeros((2, 3, 4))

x.sum(axis=0).shape       # (3, 4)    -- axis 0 (length 2) gone
x.sum(axis=1).shape       # (2, 4)    -- axis 1 (length 3) gone
x.sum(axis=-1).shape      # (2, 3)    -- last axis (length 4) gone
x.sum(axis=(0, 1)).shape  # (4,)      -- axes 0 and 1 both gone
x.sum().shape             # ()        -- all axes gone, scalar
```

This is also why `keepdims=True` is useful: it says "don't actually
remove the axis -- just shrink it to length 1," keeping the rank the
same so the result can still broadcast against the original input.

```python
x.sum(axis=0)                   # shape (3,)     -- axis 0 gone
x.sum(axis=0, keepdims=True)    # shape (1, 3)   -- axis 0 kept at length 1
```

Same numbers, different shape. You'll see this pattern in every
`Softmax` and `LayerNorm` because the reduced axis needs to broadcast
back into a subtraction or division.

### Why `(n,)` and not `(n, 1)` or `(1, n)`?

The trailing comma is **Python tuple syntax**, not a NumPy thing.
Parentheses alone don't make a tuple in Python -- commas do:

```python
(5)        # just the int 5 in parens
(5,)       # a tuple of length 1
(5, 3)     # a tuple of length 2
```

So `shape == (3,)` is saying "the shape is a tuple containing one
number." That single number is the length of the only axis. `(n,)`
means **a 1-dimensional array of length `n`** -- no row/column
distinction, just a flat vector.

`(n,)`, `(n, 1)`, and `(1, n)` can all hold the same `n` numbers, but
they are **genuinely different objects** with different ranks
(`ndim`):

| Shape    | `ndim` | Meaning                                  |
|----------|--------|------------------------------------------|
| `(3,)`   | 1      | A flat 1D vector, no orientation         |
| `(3, 1)` | 2      | A 2D column vector: 3 rows, 1 column     |
| `(1, 3)` | 2      | A 2D row vector: 1 row, 3 columns        |

Three places this matters in practice:

**1. `.T` does nothing on a 1D array.** A flat `(3,)` has no row/column
identity to swap, so `.T` returns the same shape. To get a column you
must be explicit: `a[:, None]` -> `(3, 1)`.

```python
a = np.array([1., 2., 3.])              # (3,)
a.T.shape                                # (3,)        -- unchanged

b = a[:, None]                           # (3, 1)
b.T.shape                                # (1, 3)      -- now it transposes
```

**2. Broadcasting right-aligns shapes, so `(n,)` and `(1, n)` are
equivalent in arithmetic, but `(n, 1)` is different.** A 1D array
"slots into" the last axis automatically:

```python
A = np.zeros((3, 4))
v = np.array([10., 20., 30., 40.])      # (4,)

A + v                                     # OK -> (3, 4); v aligned to axis -1
A + v[None, :]                            # OK -> (3, 4); same as above
A + v[:, None]                            # ERROR: (3, 4) + (4, 1), 3 != 4 in axis -2
```

**3. The PE `pos / divisor` trick relies on this exactly.** From
`SinusoidalPositionalEncoding._build_pe`:

```python
pos     = np.arange(max_len)[:, np.newaxis]    # (max_len, 1)  -- 2D column
divisor = np.power(10000.0, 2.0 * i / d_model) # (d_model//2,) -- 1D

angle   = pos / divisor                         # (max_len, d_model//2)
```

Right-aligning `(max_len, 1)` and `(d_model//2,)` gives
`(max_len, d_model//2)` -- the two axes don't collide, so we get a
full 2D grid. If `pos` had been left as 1D `(max_len,)`, NumPy would
have aligned it to the same axis as `divisor` and either errored or
done an element-wise op.

**Rule of thumb:**

- Default to 1D `(n,)`. Simpler, transposes are no-ops, broadcasts
  against any "feature axis" just works.
- Promote to `(n, 1)` only when you need to broadcast against an axis
  *other than the last* -- typically when building a 2D grid from two
  1D arrays (PE, distance matrices, attention masks).

### `dtype`s used in this project

| dtype | Where |
|-------|-------|
| `float64` (default) | All weights, activations, gradients |
| `int64` / `np.intp` | Token IDs, target indices |
| `bool` | Masks (causal mask, valid-token mask) |

```python
x.astype(np.float32)             # cast (returns a copy)
x.astype(x.dtype)                # no-op
(scores > 0).astype(x.dtype)     # bool mask -> float, e.g. ReLU
```

> **Used in:** `ReLU.forward` (`(x > 0).astype(x.dtype)`),
> `cross_entropy_with_label_smoothing` (`safe_targets.astype(np.intp)`).

---

## 3. Creating arrays

```python
np.zeros((d_model,))              # all zeros, shape (d_model,)
np.ones(d_model)                  # all ones
np.zeros_like(W)                  # same shape and dtype as W, zero-filled
np.full(shape, fill_value)        # constant
np.array([[1, 2], [3, 4]])        # from a Python list
np.arange(max_len)                # [0, 1, ..., max_len-1]
np.tril(np.ones((n, n), dtype=bool))   # lower-triangular causal mask
```

> **Used in:**
> - `Linear.__init__`: `self.b = np.zeros(out_features)`
> - `LayerNorm.__init__`: `self.gamma = np.ones(d_model)`
> - `Embedding.backward`: `np.zeros_like(self.weight)` (lazy grad allocation)
> - `SinusoidalPositionalEncoding._build_pe`: `np.arange(max_len)`
> - `Decoder.forward`: `np.tril(np.ones((n_q, n_q), dtype=bool))` (causal mask)
> - `cross_entropy_with_label_smoothing`: `np.full(logits.shape, eps/(V-1))`

**Worked example -- the causal mask:**

```python
np.tril(np.ones((4, 4), dtype=bool))
# array([[ True, False, False, False],
#        [ True,  True, False, False],
#        [ True,  True,  True, False],
#        [ True,  True,  True,  True]])
```

Row `i` is "what position `i` is allowed to attend to" -- only positions
`<= i`. Position 0 sees only itself; position 3 sees everything.

---

## 4. Indexing and slicing

### Basic slicing returns a **view** (no copy)

```python
x[1:5]                # rows 1..4
x[:, 0]               # all rows, column 0
x[..., -1]            # last element along the last axis (any rank)
x[..., None]          # same as x[..., np.newaxis] -- adds a length-1 axis
x[:, 0::2]            # every even column (used for sin in PE)
x[:, 1::2]            # every odd column  (used for cos in PE)
```

Two idioms used heavily in the codebase:

```python
# Add a new axis at the end (turn (B, n) into (B, n, 1) for broadcasting)
valid_mask[..., None]                                # cross_entropy

# Add a new axis after a 1D array (turn (max_len,) into (max_len, 1))
np.arange(max_len)[:, np.newaxis]                    # positional encoding
```

### `keepdims` vs an explicit `[..., None]`

These are equivalent for one reduction:

```python
x.max(axis=-1, keepdims=True)            # shape (B, n, 1)
x.max(axis=-1)[..., None]                # shape (B, n, 1) -- same
```

The `keepdims=True` form is preferred in this project because it survives
into broadcasting without an extra step.

> **Used in:** `Softmax.forward` (`x - x.max(axis=-1, keepdims=True)`),
> `LayerNorm.forward` (mean and var along the feature axis).

### Slicing the positional encoding table

```python
self.pe[:seq_len]    # rows 0..seq_len-1 from a (max_len, d_model) table
```

This is the heart of `SinusoidalPositionalEncoding.forward`: a precomputed
table indexed by sequence length.

**Worked example -- `[:, 0::2]` and `[:, 1::2]` for the PE sin/cos split:**

```python
pe = np.array([[10., 11., 12., 13.],
               [20., 21., 22., 23.],
               [30., 31., 32., 33.]])     # shape (3, 4) -- 3 positions, d_model=4

pe[:, 0::2]                                # even columns: 0, 2
# array([[10., 12.],
#        [20., 22.],
#        [30., 32.]])

pe[:, 1::2]                                # odd columns: 1, 3
# array([[11., 13.],
#        [21., 23.],
#        [31., 33.]])
```

In the real PE code: even columns get `sin(angle)`, odd columns get
`cos(angle)`, so each `(pos, i)` pair occupies neighbouring slots.

---

## 5. Reshape, transpose, `swapaxes` (the MHA toolkit)

Multi-head attention is mostly a shape ballet. Three operations are doing
all the work:

### `.reshape(new_shape)` -- relabel dimensions, no compute

```python
x.shape                                       # (32, 5, 32)        d_model = 32
x.reshape(32, 5, 4, 8).shape                  # (32, 5, h=4, d_k=8)
```

A `reshape` is a **view** (zero-copy) when the array is contiguous in the
right way; otherwise NumPy quietly makes a copy. You don't need to worry
about which case you're in -- the result is correct either way.

The two reshape idioms in `Linear.backward`:

```python
x_2d   = x.reshape(-1, x.shape[-1])           # flatten leading dims
grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
```

`-1` means "infer this dimension from the total size." Used here to flatten
`(B, n, d_in)` -> `(B*n, d_in)` so a 2D matmul gives `dL/dW`.

### `np.swapaxes(x, axis1, axis2)` -- swap two axes

```python
np.swapaxes(K, -1, -2)                        # transpose the LAST two axes
np.swapaxes(x, -2, -3)                        # promote/demote the head axis
```

Used in two places that are easy to confuse:

| Call | Purpose |
|------|---------|
| `np.swapaxes(K, -1, -2)` in SDPA | `K^T` for the `Q @ K^T` score matrix |
| `np.swapaxes(x, -2, -3)` in MHA `_split_heads` | move the new head axis from -2 to -3 so heads become a leading batch dim |

### `.T` -- transpose the *whole* array

For 2D arrays, `.T` swaps the only two axes -- handy for `dL/dW = x^T @ grad`:

```python
self.grad_W = x_2d.T @ grad_2d
```

For arrays with more than 2 dims, `.T` reverses **all** axes, which is
almost never what you want -- prefer `np.swapaxes` or `np.transpose(...,
axes=...)`.

### The split-heads / merge-heads pattern

```python
def _split_heads(self, x):                    # (..., n, d_model) -> (..., h, n, d_k)
    leading = x.shape[:-1]
    x = x.reshape(leading + (self.num_heads, self.d_k))
    return np.swapaxes(x, -2, -3)

def _merge_heads(self, x):                    # (..., h, n, d_k) -> (..., n, d_model)
    x = np.swapaxes(x, -2, -3)
    leading = x.shape[:-2]
    return x.reshape(leading + (self.d_model,))
```

This is *the* trick that lets multi-head attention be a single batched
matmul instead of `h` separate ones. Read it slowly -- it's worth it.

**Worked example -- splitting a single token vector across `h=2` heads, `d_k=2`:**

Imagine one token's `d_model=4` vector:

```python
x = np.array([[10., 11., 20., 21.]])         # shape (1, 4)  -- 1 token, d_model=4
#              \-- head 0 --/  \-- head 1 --/
```

Step 1 -- reshape `(1, 4) -> (1, h=2, d_k=2)`:

```python
x.reshape(1, 2, 2)
# array([[[10., 11.],         # head 0
#         [20., 21.]]])        # head 1
```

The bytes in memory didn't move; we just relabelled the four floats as a
2x2 grid. Now expand to a real batched example, `(B=1, n=3, d_model=4)`:

```python
x = np.array([[[10., 11., 20., 21.],         # token 0:  head0=[10,11]  head1=[20,21]
               [30., 31., 40., 41.],         # token 1:  head0=[30,31]  head1=[40,41]
               [50., 51., 60., 61.]]])       # token 2:  head0=[50,51]  head1=[60,61]
                                             # shape (1, 3, 4)

# Step 1: reshape to (1, 3, h=2, d_k=2) -- still token-major
x_grid = x.reshape(1, 3, 2, 2)
# x_grid[0, token, head, :] gives that head's slice of that token

# Step 2: swap axes -2 (n=3) and -3 (h=2) -> (1, h=2, 3, d_k=2)
x_split = np.swapaxes(x_grid, -2, -3)
# x_split[0, head, token, :] -- now head is the leading "batch" axis

x_split[0, 0]                                 # head 0's view of all 3 tokens
# array([[10., 11.],
#        [30., 31.],
#        [50., 51.]])

x_split[0, 1]                                 # head 1's view of all 3 tokens
# array([[20., 21.],
#        [40., 41.],
#        [60., 61.]])
```

Now `Q_h @ K_h^T` is a batched matmul over `(B, h)`, computing each
head's attention scores in parallel. `_merge_heads` does the inverse:
swap axes back to token-major, then reshape `(h, d_k) -> d_model`.

---

## 6. Broadcasting

When you add or multiply two arrays of different shapes, NumPy follows
**broadcasting** rules: align shapes from the right, and any dimension
that is `1` (or missing) gets virtually stretched to match.

```python
x  = np.zeros((32, 5, 64))                     # (B, n, d)
b  = np.zeros(64)                              # (d,)        -- aligns with d
x + b                                          # OK -> (32, 5, 64)

m  = np.zeros((32, 5, 1))                      # (B, n, 1)
x * m                                          # OK -> (32, 5, 64) -- the 1 broadcasts
```

### Broadcasting rules in plain English

1. **Right-align** the shapes.
2. Each axis must either be **equal** or **one of them is 1**.
3. Size-`1` axes are stretched (no copy, no compute).

### When does broadcasting fail?

```python
np.zeros((32, 5)) + np.zeros((5, 32))          # ERROR: 5 != 32 and neither is 1
```

To fix this, add a length-1 axis where you mean "broadcast me":

```python
a = np.zeros((32, 5))
b = np.zeros((5, 32))
a + b.T                                        # OK
a[:, :, None] + b[None, :, :]                  # also OK -> (32, 5, 32)
```

> **Used in:** `Linear.forward` (`x @ self.W + self.b`, where bias is 1D
> and broadcasts to `(B, n, out)`), `LayerNorm.forward`
> (`gamma * x_norm + beta`), `SinusoidalPositionalEncoding`
> (`pos / divisor` where `pos` is `(max_len, 1)` and `divisor` is
> `(d_model//2,)` -- result `(max_len, d_model//2)`).

**Worked example -- adding a bias across a batch:**

```python
x = np.array([[[1., 2., 3.],
               [4., 5., 6.]],
              [[7., 8., 9.],
               [10., 11., 12.]]])             # shape (B=2, n=2, d=3)

b = np.array([100., 200., 300.])              # shape (3,)

x + b
# array([[[101., 202., 303.],
#         [104., 205., 306.]],
#        [[107., 208., 309.],
#         [110., 211., 312.]]])               # shape (2, 2, 3)
```

Shape alignment: `(2, 2, 3)` and `(3,)` right-align to `(2, 2, 3)` and
`(_, _, 3)` -- the missing leading dims are virtually `1`, so `b` is
"copied" across all `(B, n)` positions. No actual copy happens.

**Worked example -- the positional-encoding broadcast:**

```python
pos = np.arange(3, dtype=float)[:, np.newaxis]   # shape (3, 1) -- column
# array([[0.],
#        [1.],
#        [2.]])

divisor = np.array([1., 10., 100.])              # shape (3,) -- row

pos / divisor                                     # (3, 1) / (3,) -> (3, 3)
# array([[0.   , 0.   , 0.   ],
#        [1.   , 0.1  , 0.01 ],
#        [2.   , 0.2  , 0.02 ]])
```

This is exactly how `angle = pos / divisor` builds the `(max_len, d_model/2)`
angle matrix in one go -- no Python loops over positions or dimensions.

---

## 7. Element-wise math

All standard arithmetic operators (`+ - * / **`) are element-wise and
broadcasting-aware. NumPy also exposes mathematical functions that
operate element-wise:

| Function | Used in |
|----------|---------|
| `np.exp(x)` | softmax, log-softmax |
| `np.log(x)` | log-softmax, label-smoothing floor |
| `np.sqrt(x)` | LayerNorm (`sqrt(var + eps)`), attention scaling, Adam |
| `np.power(base, exp)` | positional-encoding divisor |
| `np.sin(x)`, `np.cos(x)` | positional encoding |

Two patterns worth recognising:

```python
scale = 1.0 / np.sqrt(d_k)                     # SDPA scaling factor (a scalar)
scores = (Q @ np.swapaxes(K, -1, -2)) * scale  # broadcasts the scalar across all positions

divisor = np.power(10000.0, 2.0 * i / d_model) # (d_model//2,) -- element-wise power
angle   = pos / divisor                        # (max_len, d_model//2) via broadcasting
```

`x ** 2` is just `np.power(x, 2)` written shorter; both work.

---

## 8. Reductions: `axis` and `keepdims`

Reductions collapse one or more axes into a scalar (per remaining slice).

```python
x.sum()                            # scalar -- collapse all axes
x.sum(axis=0)                      # collapse axis 0
x.sum(axis=-1)                     # collapse last axis
x.sum(axis=(0, 1))                 # collapse multiple axes
x.sum(axis=-1, keepdims=True)      # keep a length-1 axis where the reduction was
```

Available reductions used in this project:

| Reduction | Where |
|-----------|-------|
| `.sum`    | grad_b, softmax-denominator, loss aggregation |
| `.mean`   | LayerNorm forward |
| `.var`    | LayerNorm forward |
| `.max`    | softmax stability shift |
| `.argmax` | greedy decoding |

### Why `keepdims=True` matters

Without it, the reduced axis vanishes and you can't broadcast back:

```python
m = x.max(axis=-1)                 # shape (B, n)        -- axis -1 is gone
x - m                              # ERROR if x is (B, n, V): trailing dims don't align

m = x.max(axis=-1, keepdims=True)  # shape (B, n, 1)
x - m                              # OK -- broadcasts to (B, n, V)
```

This shows up in **every** softmax, LayerNorm, and log-softmax in the
code.

### Reducing over "all leading axes" dynamically

```python
sum_axes = tuple(range(grad_output.ndim - 1))   # all axes except the last
self.grad_gamma = (grad_output * x_norm).sum(axis=sum_axes)
```

Used in `LayerNorm.backward` so the same code works for 2D `(B, d)` and
3D `(B, n, d)` inputs without special-casing.

**Worked example -- LayerNorm forward by hand:**

One token vector with `d_model = 4`:

```python
x = np.array([1., 3., 5., 7.])                  # shape (4,)

mean = x.mean(axis=-1, keepdims=True)           # array([4.])  -- shape (1,)
var  = x.var(axis=-1, keepdims=True)            # array([5.])  -- shape (1,)

eps = 1e-5
std_inv = 1.0 / np.sqrt(var + eps)              # array([0.4472...])

x_norm = (x - mean) * std_inv
# array([-1.3416, -0.4472,  0.4472,  1.3416])
# Note: x_norm has mean ~= 0 and var ~= 1 (the whole point of LayerNorm).

gamma = np.ones(4)
beta  = np.zeros(4)
out = gamma * x_norm + beta                     # same as x_norm here
```

Now stack into a batch -- the **exact same code** works:

```python
x = np.array([[1., 3., 5., 7.],
              [2., 4., 6., 8.]])                # shape (B=2, d=4)

x.mean(axis=-1, keepdims=True)
# array([[4.],
#        [5.]])                                  # shape (2, 1) -- per-row mean
```

`keepdims=True` is what lets `(x - mean)` broadcast cleanly: `(2, 4) - (2, 1) -> (2, 4)`.

---

## 9. Matrix multiplication: `@` and `np.matmul`

`@` is Python's matrix-multiplication operator; it calls `np.matmul`.

### The rule

`np.matmul` treats the **last two axes** as the matrix; everything before
them is treated as a batch and broadcast.

```python
# 2D x 2D: standard matmul
A = np.zeros((3, 4))
B = np.zeros((4, 5))
(A @ B).shape                                  # (3, 5)

# 3D x 2D: broadcast batch
x = np.zeros((32, 5, 64))                      # (B, n, in)
W = np.zeros((64, 128))                        # (in, out)
(x @ W).shape                                  # (32, 5, 128)

# 4D x 4D: broadcast multiple batch dims (this is what MHA does)
Q = np.zeros((32, 4, 5, 8))                    # (B, h, n_q, d_k)
K = np.zeros((32, 4, 5, 8))                    # (B, h, n_k, d_k)
scores = Q @ np.swapaxes(K, -1, -2)            # (32, 4, 5, 5)
```

### Two operators that look similar but differ

```python
A * B          # element-wise (Hadamard) -- shapes must broadcast
A @ B          # matrix multiply -- last two axes follow matmul rules
np.dot(A, B)   # older, sum-product over the last axis of A and second-to-last of B
```

Use `@` for matmul. Use `*` for element-wise. The codebase uses `np.matmul`
explicitly inside SDPA for clarity but `@` everywhere else; they're the
same function.

> **Used in:** `Linear.forward` (`x @ self.W + self.b`),
> `Linear.backward` (`x_2d.T @ grad_2d`),
> `ScaledDotProductAttention.forward`
> (`np.matmul(Q, np.swapaxes(K, -1, -2))`).

**Worked example -- standard 2D matmul:**

```python
A = np.array([[1., 2., 3.],
              [4., 5., 6.]])                    # shape (2, 3)

B = np.array([[1., 0.],
              [0., 1.],
              [1., 1.]])                        # shape (3, 2)

A @ B
# array([[ 4.,  5.],          # row 0:  1*1 + 2*0 + 3*1 = 4 ; 1*0 + 2*1 + 3*1 = 5
#        [10., 11.]])         # row 1:  4*1 + 5*0 + 6*1 = 10; 4*0 + 5*1 + 6*1 = 11
                              # shape (2, 2)
```

**Worked example -- batched matmul (the SDPA score computation):**

Tiny SDPA-shaped example: `B=1`, `n_q=2`, `n_k=2`, `d_k=2`.

```python
Q = np.array([[[1., 0.],
               [0., 1.]]])                      # shape (1, 2, 2)

K = np.array([[[1., 1.],
               [1., 0.]]])                      # shape (1, 2, 2)

K_T = np.swapaxes(K, -1, -2)                    # shape (1, 2, 2)
# K_T[0]:
# array([[1., 1.],
#        [1., 0.]])

scores = Q @ K_T                                 # shape (1, 2, 2)
# array([[[1., 1.],            # query 0 against keys 0, 1
#         [1., 0.]]])           # query 1 against keys 0, 1
```

Each `scores[b, i, j]` is `Q[b, i] . K[b, j]` -- a dot product between
query `i` and key `j`. The leading batch axis is broadcast through.

---

## 10. Boolean masks and `np.where`

A boolean array can be used as a **mask** to select or modify elements.

### `np.where(cond, a, b)` -- vectorized ternary

```python
np.where(mask, scores, -np.inf)     # SDPA: -inf at blocked positions
np.where(mask, grad_scores, 0.0)    # SDPA backward: zero out blocked grads
np.where(valid_mask, targets, 0)    # CE: replace PAD targets with safe value 0
```

`cond` is broadcast against `a` and `b`. The result has the broadcast shape.

### Boolean-array reductions

```python
(x > 0).astype(x.dtype)             # ReLU mask: 1.0 where x > 0, else 0.0
mask.sum()                          # count of True entries
mask.any()                          # is at least one True?
(safe_targets >= V).any()           # any out-of-range index? (used for validation)
```

### Fancy boolean indexing (read or write)

```python
x[x > 0] = 0                        # zero out positives -- not used in this project
                                    # but appears in many tutorials; mentioned for context
```

> **Used in:** `ScaledDotProductAttention.forward` (mask -> `-inf` before
> softmax so blocked positions get zero attention weight), `ReLU.forward`
> (build a 0/1 mask), `cross_entropy_with_label_smoothing` (PAD masking).

**Worked example -- masking attention scores with `-inf`:**

```python
scores = np.array([[2.0, 3.0, 1.0],
                   [0.5, 1.5, 2.5],
                   [1.0, 2.0, 3.0]])             # shape (3, 3)

mask = np.tril(np.ones((3, 3), dtype=bool))      # causal mask
# array([[ True, False, False],
#        [ True,  True, False],
#        [ True,  True,  True]])

masked = np.where(mask, scores, -np.inf)
# array([[2.0, -inf, -inf],
#        [0.5,  1.5, -inf],
#        [1.0,  2.0,  3.0]])
```

After softmax along `axis=-1`, the `-inf` entries become `exp(-inf) = 0`,
so blocked positions get exactly zero attention weight and the surviving
entries renormalise.

**Worked example -- ReLU as a mask:**

```python
x = np.array([-1.0, 0.0, 2.0, -3.0, 4.0])

mask = (x > 0).astype(x.dtype)
# array([0., 0., 1., 0., 1.])                    # 1 where positive

x * mask
# array([0., 0., 2., 0., 4.])                    # ReLU output
```

Storing the mask (rather than the original `x`) is the trick that makes
`ReLU.backward` cheap: `grad_input = grad_output * mask`.

---

## 11. Fancy indexing and scatter (embedding lookup)

### Reading: index an array with another array

```python
weight = np.random.randn(vocab_size, d_model)         # (V, d_model)
token_ids = np.array([[3, 1, 4], [1, 5, 9]])          # (B, n) int

embeddings = weight[token_ids]                        # (B, n, d_model)
```

This is *the* embedding lookup. NumPy reads `weight[i]` for every `i` in
`token_ids` and assembles them into the output shape. `token_ids` can be
any integer-array shape; the output gains the trailing feature axes from
`weight`.

### The shape rule

The output is always **a single `ndarray`** -- never a dictionary, list,
or anything fancier -- whose shape is determined by a clean rule:

```text
If W has shape  (V, d_0, d_1, ..., d_n)
   idx has shape  S
then W[idx] has shape  S + (d_0, d_1, ..., d_n)
```

In words: **the index array's shape becomes the leading shape of the
output, and the indexed array's trailing axes (everything after the one
being indexed into) are carried through unchanged.** For the embedding
case, `W: (V, d_model)` and `idx: (B, n)` give `(B, n, d_model)`.

This is why a one-line `weight[token_ids]` replaces the two nested
Python loops you'd otherwise write:

```python
out = np.zeros((B, n, d_model))
for b in range(B):
    for t in range(n):
        out[b, t] = weight[token_ids[b, t]]            # equivalent, much slower
```

Same answer, but the fancy-indexing form pushes the loop into compiled C
code -- which is why embeddings are essentially free in our model.

**Worked example -- tiny embedding lookup, `V=4`, `d_model=2`:**

```python
weight = np.array([[10., 11.],     # row 0 -- embedding of token 0
                   [20., 21.],     # row 1 -- embedding of token 1
                   [30., 31.],     # row 2 -- embedding of token 2
                   [40., 41.]])    # row 3 -- embedding of token 3
                                   # shape (V=4, d_model=2)

token_ids = np.array([[2, 0, 3],
                      [1, 1, 2]])  # shape (B=2, n=3)

weight[token_ids]
# array([[[30., 31.],              # row 2
#         [10., 11.],              # row 0
#         [40., 41.]],             # row 3
#        [[20., 21.],              # row 1
#         [20., 21.],              # row 1 (repeated -- same row picked twice)
#         [30., 31.]]])            # row 2
                                   # shape (2, 3, 2)
```

The trailing `d_model` axis is appended automatically; the index array
shape `(2, 3)` becomes the leading shape of the output.

### Writing: `np.add.at(target, indices, source)` -- scatter-add

The naive approach is wrong:

```python
weight[token_ids] += grad_output                       # WRONG: duplicate indices clobber
```

If a token appears twice in the batch, only one update lands. NumPy's
unbuffered scatter-add does the right thing:

```python
np.add.at(self.grad_weight, token_ids, grad_output)    # accumulates duplicates
```

This is `Embedding.backward`. Without `np.add.at`, gradients for repeated
tokens (very common in real text) would silently lose information.

**Worked example -- why the naive `+=` is wrong:**

```python
grad_weight = np.zeros((4, 2))                      # (V=4, d_model=2)
token_ids   = np.array([1, 1, 2])                   # token 1 appears TWICE
grad_output = np.array([[1., 1.],
                        [2., 2.],
                        [3., 3.]])                  # gradient per occurrence

# WRONG: only ONE of the two updates to row 1 actually lands.
buf = np.zeros((4, 2))
buf[token_ids] += grad_output
# buf:
# array([[0., 0.],
#        [2., 2.],            # only the LAST write to row 1 survived (1+1 was lost)
#        [3., 3.],
#        [0., 0.]])

# RIGHT: scatter-add accumulates duplicates.
buf = np.zeros((4, 2))
np.add.at(buf, token_ids, grad_output)
# array([[0., 0.],
#        [3., 3.],            # 1+2 -- both gradients for token 1 included
#        [3., 3.],
#        [0., 0.]])
```

The "wrong" version is a real footgun: it fails silently when a token
appears more than once in a batch, which happens in *every* real
training batch.

### `np.put_along_axis` -- scatter with an axis

Used in label smoothing to set the truth-token probability to `1 - eps`:

```python
p_smooth = np.full(logits.shape, eps / (V - 1))
np.put_along_axis(p_smooth, safe_targets[..., None], 1.0 - eps, axis=-1)
```

`safe_targets` has shape `(B, n)`; we add a trailing axis and tell NumPy
to interpret each value as an index along the vocab axis.

**Worked example -- label-smoothed targets:**

`V=4`, `eps=0.1`, target token = `2`:

```python
V, eps = 4, 0.1
target = np.array([2])                              # shape (B=1,)

p_smooth = np.full((1, V), eps / (V - 1))           # 0.0333... everywhere
# array([[0.0333, 0.0333, 0.0333, 0.0333]])

np.put_along_axis(p_smooth, target[..., None], 1.0 - eps, axis=-1)
# array([[0.0333, 0.0333, 0.9, 0.0333]])
```

The truth class gets `0.9`; the rest split `0.1` evenly. Sums to `1.0`
exactly, by construction.

---

## 12. In-place operations

All of `+=`, `-=`, `*=`, `/=` modify the array in place (no new
allocation, no new identity). This matters for two things in this project:

### Optimizers update parameters in place

```python
p -= update                          # Adam.step
```

Because `p` is the same Python object as `self.W` inside the layer, the
layer "sees" the updated weight on the next forward pass without any
explicit re-assignment. If we wrote `p = p - update` (no `-=`) we'd just
rebind a local name and the layer's weight would never change.

**Worked example -- in-place vs rebinding:**

```python
class Layer:
    def __init__(self):
        self.W = np.array([1., 2., 3.])

layer = Layer()
W = layer.W                           # W and layer.W point at the SAME array

# In-place: mutates the underlying array
W -= 0.5
print(layer.W)                        # [0.5 1.5 2.5]   -- changed, as intended

# Rebinding: makes a NEW array, leaves layer.W alone
W = W - 0.5
print(layer.W)                        # [0.5 1.5 2.5]   -- unchanged from above!
print(W)                              # [0.0 1.0 2.0]   -- this new array is local

# Confirm with id():
print(id(W) == id(layer.W))           # False -- after rebinding they diverged
```

This is why `Adam.step` uses `p -= update`. The optimizer never holds a
reference to `self.W`; it walks `params_and_grads()` getting `(p, g, name)`
tuples. If it rebinds `p`, the layer's `self.W` keeps the old (untrained)
values forever.

### Adam's running moments are updated in place

```python
s['m'] *= self.beta1
s['m'] += (1.0 - self.beta1) * g
s['v'] *= self.beta2
s['v'] += (1.0 - self.beta2) * (g * g)
```

Equivalent but allocates two temporaries per step:

```python
s['m'] = self.beta1 * s['m'] + (1.0 - self.beta1) * g     # slower, more memory
```

### Watching for accidental in-place modification

`x_norm = (x - mean) * std_inv` allocates fresh memory; safe.
`x -= mean` would overwrite the caller's `x`; avoid unless you own it.

> **Used in:** `Adam.step`, `TiedLinear.backward`
> (`existing += grad_W_contribution`).

---

## 13. Random numbers

Two APIs coexist in NumPy. We use **both** for historical reasons.

### Legacy global API (used in layer init)

```python
np.random.randn(in_features, out_features)     # standard normal, shape
np.random.rand(*x.shape)                       # uniform [0, 1)
```

Used in `Linear.__init__` (Xavier-style init scaled by
`sqrt(2/(in+out))`), `Embedding.__init__` (small Gaussian, scale 0.02),
`Dropout.forward` (uniform-then-threshold).

### Modern Generator API (used in training)

```python
rng = np.random.default_rng(seed)              # seeded -> reproducible
batch_idx = rng.integers(0, n_train, size=batch_size)
```

Used in `train()` for batch sampling. Always prefer this for new code:
the generator state is local and explicit, so tests don't depend on
hidden global state.

### Seeding

| Form | Effect |
|------|--------|
| `np.random.seed(0)` | Sets the **global** legacy state (affects everything in the process) |
| `rng = np.random.default_rng(0)` | Local seeded generator; doesn't touch global state |

In this project, tests rely on `np.random.seed` to fix layer
initialisation, and `train()` uses its own `default_rng` for the batch
sampler.

---

## 14. Numerical stability patterns

These are the two tricks the codebase uses to avoid `inf` and `nan`.

### Softmax: subtract the max before exponentiating

`softmax(x) = exp(x) / sum(exp(x))`. If any `x_i` is large, `exp(x_i)`
overflows to `inf`. The fix: shift the inputs by their max (mathematically
a no-op since the constant cancels):

```python
shifted = x - x.max(axis=-1, keepdims=True)    # max element becomes 0
exp_x = np.exp(shifted)                        # largest is exp(0) = 1
out = exp_x / exp_x.sum(axis=-1, keepdims=True)
```

Now the largest exponent is always 1.0, and the smallest underflows to
0.0 -- which is fine because it would have been negligible anyway.

**Worked example -- the overflow that the shift prevents:**

```python
x = np.array([1000., 1001., 1002.])

# Naive:
np.exp(x)                                       # array([inf, inf, inf])
np.exp(x) / np.exp(x).sum()                     # array([nan, nan, nan])  -- broken

# Stable: shift by the max
shifted = x - x.max()                            # array([-2., -1.,  0.])
exp_x = np.exp(shifted)                          # array([0.135, 0.368, 1.0])
exp_x / exp_x.sum()
# array([0.0900, 0.2447, 0.6652])               # correct softmax probabilities
```

Both inputs are mathematically equivalent (softmax is shift-invariant),
but only the second one survives floating point.

> **Used in:** `Softmax.forward`,
> `ScaledDotProductAttention.forward`, `log_softmax`.

### Log-softmax: log-sum-exp

`log_softmax(x) = x - log(sum(exp(x)))`. Same shift trick to keep the
intermediate `exp` finite:

```python
m = logits.max(axis=-1, keepdims=True)
shifted = logits - m
log_sum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
return shifted - log_sum
```

This is `training.log_softmax`. Computing `log(softmax(x))` directly via
`np.log(softmax(x))` is numerically poor when probabilities are tiny.

### LayerNorm: `+ eps` inside the sqrt

```python
std_inv = 1.0 / np.sqrt(var + self.eps)
```

If a feature has zero variance (rare but possible early in training), the
naive `1 / sqrt(var)` is `inf`. `eps = 1e-5` keeps the denominator
finite. The same idea appears in Adam:

```python
update = lr * m_hat / (np.sqrt(v_hat) + self.eps)   # eps = 1e-9
```

### Masking with `-inf` before softmax

Setting blocked positions to `-inf` before the exponent makes them
contribute exactly 0 to the softmax (because `exp(-inf) = 0`):

```python
scores = np.where(mask, scores, -np.inf)
```

This is preferable to multiplying weights by 0 *after* the softmax,
which would leave the surviving weights unnormalised.

---

## 15. Common pitfalls

### Forgetting `keepdims=True`

```python
m = x.max(axis=-1)         # shape (B, n) -- axis dropped
x - m                      # ERROR if x is (B, n, V)
```

Always pair an `axis=-1` reduction with `keepdims=True` if you plan to
broadcast back.

### `axis=0` vs `axis=-1`

For a 2D `(B, d)` array these differ:
- `axis=0` collapses the batch (returns a `(d,)` per-feature stat)
- `axis=-1` collapses the features (returns a `(B,)` per-row stat)

Using the wrong one is the #1 LayerNorm bug. We want per-token
statistics, which is `axis=-1`.

### `.T` on >2D arrays

```python
x = np.zeros((32, 4, 5, 8))
x.T.shape                  # (8, 5, 4, 32) -- ALL axes reversed
```

That's almost never what you want. Use `np.swapaxes(x, -1, -2)` to
transpose just the matrix part of a batched array.

### `reshape` vs `transpose` -- not interchangeable

A reshape **relabels memory layout** (`d_model` -> `(h, d_k)`).
A transpose / swapaxes **changes which axis is which** in logical
indexing. You typically need both, in the right order:

```python
# Wrong: reshape alone won't put heads on the leading axis
x.reshape(B, n, h, d_k)              # (B, n, h, d_k) -- heads still middle
# Right: reshape THEN swap
np.swapaxes(x.reshape(B, n, h, d_k), -2, -3)   # (B, h, n, d_k)
```

### Views vs copies (and `=` vs `+=`)

```python
y = x[:, 0]                # view -- modifying y modifies x
y = x[:, 0].copy()         # copy

y += 1                     # if y is a view: changes x too. Often a bug.
y = y + 1                  # always a fresh array (rebinds y)
```

For optimizer code we *want* in-place; for layer caches we *don't*.

**Worked example -- the silent view bug:**

```python
x = np.array([[1., 2., 3.],
              [4., 5., 6.]])

y = x[0]                   # VIEW -- shares memory with row 0 of x
y[:] = 99.0                # mutates x as well!
print(x)
# [[99. 99. 99.]
#  [ 4.  5.  6.]]          # row 0 was clobbered

# Defensive copy when caching for backward:
self._cache['x'] = x.copy()         # safe: backward sees the input as it was
self._cache['x'] = x                # risky: caller may mutate x later
```

Our codebase mostly gets away with `self._cache['x'] = x` because callers
treat layer inputs as immutable, but the rule "if you cache it for
backward, the caller must not mutate it" is implicit -- worth knowing.

### Integer overflow in indices

`np.put_along_axis` and fancy indexing both expect signed integers. If
you build indices from `int8` token IDs they may overflow on big
vocabularies. The codebase casts to `np.intp` defensively:

```python
safe_targets.astype(np.intp)
```

### Bool arrays in arithmetic context

`(x > 0)` is a bool array. It usually behaves like a 0/1 float when
multiplied, but for clarity (and because some operations require a
matching dtype) the codebase explicitly casts:

```python
mask = (x > 0).astype(x.dtype)
```

---

## 16. Quick-reference cheat sheet

```python
# Creation
np.zeros(shape)               np.ones(shape)             np.zeros_like(a)
np.full(shape, value)         np.array([...])
np.arange(n)                  np.tril(np.ones((n, n), dtype=bool))
np.random.randn(*shape)       np.random.rand(*shape)
rng = np.random.default_rng(seed); rng.integers(lo, hi, size=...)

# Shape
a.shape   a.ndim   a.size   a.dtype   a.astype(dtype)
a.reshape(...)                # use -1 to infer one axis
np.swapaxes(a, i, j)          # swap two axes (matmul-friendly transpose)
a.T                           # full reversal -- usually only for 2D
a[..., None]   a[:, np.newaxis]   # add a length-1 axis

# Math (element-wise + broadcasting)
+ - * / **                    np.exp  np.log  np.sqrt  np.power
np.sin  np.cos                np.where(cond, a, b)

# Reductions (always consider keepdims=True)
a.sum(axis, keepdims)         a.mean(axis, keepdims)     a.var(axis, keepdims)
a.max(axis, keepdims)         a.argmax(axis)             a.min(axis, keepdims)

# Linear algebra
A @ B                         np.matmul(A, B)            # last 2 dims = matrix
A.T                                                       # 2D transpose

# Indexing
a[i]   a[i:j]   a[..., -1]    a[bool_mask]    a[idx_array]
np.add.at(a, idx, src)        # unbuffered scatter-add
np.put_along_axis(a, idx[..., None], values, axis=-1)

# In-place
a += b   a -= b   a *= b   a /= b
np.copyto(a, b)               # a[:] = b, controlled

# Stability
shifted = x - x.max(axis=-1, keepdims=True)               # softmax
1.0 / np.sqrt(var + eps)                                   # LayerNorm
np.where(mask, scores, -np.inf)                            # masked softmax
```

---

## Where to learn more

- **Official docs**: <https://numpy.org/doc/stable/> -- the *Reference* section is
  exhaustive; the *User Guide* has the best broadcasting explanation anywhere.
- **NumPy for MATLAB users / R users**: useful translation tables linked from the
  NumPy site if you have those mental models.
- **In this codebase**: every concept above is grep-able. If you're unsure how
  an idiom is used in practice, search `nn/numpy-transformer/src/` for the
  function name (`np.swapaxes`, `keepdims`, `np.add.at`, etc.) -- you'll find
  one or two real, commented examples within seconds.

---

[< Back to Overview](00_overview.md)
