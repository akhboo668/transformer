# Backward Computation in NumPy -- How It Actually Works

> **Scope**: This document explains the NumPy mechanics behind every `backward()` method
> in `numpy_transformer/layers.py`. It covers the specific NumPy operations, why they
> work, and common pitfalls.

---

## Table of Contents

1. [The Pattern: Cache, Reverse, Accumulate](#1-the-pattern)
2. [NumPy Operations You Must Know](#2-numpy-operations)
3. [Linear Backward](#3-linear)
4. [Softmax Backward](#4-softmax)
5. [ReLU Backward](#5-relu)
6. [LayerNorm Backward](#6-layernorm)
7. [Dropout Backward](#7-dropout)
8. [Embedding Backward](#8-embedding)
9. [Shape Cheat Sheet](#9-shape-cheat-sheet)

---

## 1. The Pattern

Every backward method in our code follows the same three-phase pattern:

```
forward(x):
    1. Save inputs/intermediates in self._cache
    2. Compute and return output

backward(grad_output):
    1. Retrieve cached values
    2. Compute parameter gradients  (dL/dW, dL/db, etc.)
    3. Compute and return input gradient  (dL/dx)
```

**Why cache?** NumPy doesn't build a computation graph like PyTorch. There's no autograd.
We must manually remember what we need for the backward pass. The chain rule tells us
*what* to compute; the cache gives us the *values* to compute it with.

**What is `grad_output`?** It's `dL/dy` -- the gradient of the loss with respect to this
layer's output. It always has the **same shape as the forward output**. The layer's job
is to produce `dL/dx` (same shape as the forward input) and, if it has learnable params,
`dL/dW`, `dL/db`, etc.

---

## 2. NumPy Operations You Must Know

### 2a. `@` (matmul) -- Broadcasting Behavior

`@` is `np.matmul`. For 2D arrays, it's plain matrix multiplication:

```python
A @ B          # (m, k) @ (k, n) -> (m, n)
```

For 3D+ arrays, `@` **broadcasts over leading dimensions** and multiplies the last two:

```python
A @ B          # (batch, m, k) @ (k, n) -> (batch, m, n)
               # NumPy treats (k, n) as if it were (1, k, n) and broadcasts
```

This is why `grad_output @ self.W.T` works directly on 3D tensors -- NumPy applies the
matrix multiply independently to each (seq_pos) in each (batch), without an explicit loop.

### 2b. `.reshape(-1, n)` -- Flattening Leading Dimensions

```python
x.reshape(-1, x.shape[-1])
```

This collapses all leading dimensions into one. The `-1` means "infer this size":

```
(32, 5, 64).reshape(-1, 64)  ->  (160, 64)
(4, 3, 2, 8).reshape(-1, 8)  ->  (24, 8)
(10, 64).reshape(-1, 64)     ->  (10, 64)    # already 2D, no-op
```

**Why we need it**: The formula `dL/dW = x^T @ grad` is defined for 2D matrices. When x
is 3D (batch, seq, features), we must flatten batch*seq into rows first.

**Critical insight**: Reshaping doesn't copy data -- it returns a *view* of the same memory
with different stride metadata. It's essentially free.

### 2c. `.sum(axis=...)` -- Reducing Dimensions

```python
grad.sum(axis=0)            # sum over batch dimension
grad.sum(axis=-1)           # sum over last dimension
grad.sum(axis=(0, 1))       # sum over batch AND seq dimensions
```

**keepdims=True** preserves the reduced axis as size-1, enabling broadcasting:

```python
x = np.array([[1, 2, 3],
              [4, 5, 6]])       # shape (2, 3)

x.sum(axis=-1)                  # array([6, 15])        shape (2,)
x.sum(axis=-1, keepdims=True)   # array([[6], [15]])     shape (2, 1)
```

The `keepdims=True` version can broadcast back against the original `(2, 3)` shape.
Without it, you'd get shape mismatches.

### 2d. `*` (element-wise multiply) -- Hadamard Product

```python
A * B     # same shape: element-by-element
A * B     # different shapes: NumPy broadcasts (see 2e)
```

This is NOT matrix multiplication. It's the tool for applying masks (ReLU, Dropout) and
for the chain rule when the forward was element-wise.

### 2e. Broadcasting Rules

When shapes differ, NumPy aligns dimensions from the right and expands size-1 axes:

```
(32, 5, 64) * (64,)        -> (32, 5, 64)   # (64,) treated as (1, 1, 64)
(32, 5, 64) * (32, 5, 1)   -> (32, 5, 64)   # the 1 expands to 64
(32, 5, 64) * (32, 1, 64)  -> (32, 5, 64)   # the 1 expands to 5
```

**Why this matters for backward**: When forward uses broadcasting (e.g., `+ self.b` where
b is `(64,)`), the backward must **sum over the broadcasted axes** to get the gradient
back to the parameter's original shape.

### 2f. `np.add.at` -- Scatter-Add (No Overwrite)

```python
np.add.at(target, indices, values)
```

Unlike `target[indices] += values`, this correctly **accumulates** when indices repeat:

```python
target = np.zeros(5)
indices = [2, 2, 3]
values = [1.0, 0.5, 2.0]

target[indices] += values       # WRONG: target[2] = 0.5 (second write overwrites first)
np.add.at(target, indices, values)  # RIGHT: target[2] = 1.5 (both additions applied)
```

This is essential for Embedding backward where the same token ID can appear multiple
times in a sequence.

---

## 3. Linear Backward

**Forward**: `y = x @ W + b`

**What we cache**: `x`

**Shapes in our Transformer** (example: `Linear(32, 64)` with batch=32, seq=5):

| Tensor | Shape | Description |
|--------|-------|-------------|
| x | (32, 5, 32) | input: 32 batches, 5 tokens, 32 features |
| W | (32, 64) | weight matrix |
| b | (64,) | bias vector |
| y | (32, 5, 64) | output |
| grad_output | (32, 5, 64) | dL/dy from the next layer |

### Step-by-step

**Problem**: `dL/dW = x^T @ grad_output` needs 2D operands, but x is 3D.

```python
# Step 1: Flatten batch and seq into one dimension
x_2d = x.reshape(-1, x.shape[-1])            # (32,5,32) -> (160, 32)
grad_2d = grad_output.reshape(-1, grad_output.shape[-1])  # (32,5,64) -> (160, 64)
```

Each of the 160 rows is one token's feature vector. Flattening is valid because every
token contributes independently to `dL/dW` -- position (batch=3, seq=2) contributes the
same way as (batch=0, seq=0). We're just summing up all contributions.

```python
# Step 2: Weight gradient -- each token pair (x_i, grad_i) contributes x_i^T @ grad_i
self.grad_W = x_2d.T @ grad_2d               # (32, 160) @ (160, 64) = (32, 64)
```

This single matrix multiply computes the sum of 160 outer products. Expanding:

```
dL/dW = sum over all tokens t:  x_t^T @ grad_t
      = [x_0; x_1; ...; x_159]^T @ [grad_0; grad_1; ...; grad_159]
      = x_2d.T @ grad_2d
```

```python
# Step 3: Bias gradient -- bias was broadcast-added to every position
self.grad_b = grad_2d.sum(axis=0)             # (160, 64) -> (64,)
```

Since `b` was added to all 160 positions (via broadcasting), its gradient is the sum of
all 160 gradient rows. This is the reverse of broadcasting: **sum over the axes that were
broadcast**.

```python
# Step 4: Input gradient -- pass gradient back through the linear transform
grad_input = grad_output @ self.W.T           # (32,5,64) @ (64,32) = (32,5,32)
```

No reshape needed here. NumPy's `@` broadcasts the 2D weight `(64, 32)` across all
(batch, seq) positions automatically. Each token's gradient is independently multiplied
by `W.T`.

---

## 4. Softmax Backward

**Forward**: `s_i = e^{x_i} / sum(e^{x_j})`

**What we cache**: `out` (the softmax probabilities `s`)

**The math** (derived in `nn/docs/01_math_primitives.md`):

```
dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
```

**NumPy implementation**:

```python
s = self._cache['out']                                    # (batch, seq, vocab)
dot = (grad_output * s).sum(axis=-1, keepdims=True)       # (batch, seq, 1)
return s * (grad_output - dot)                            # (batch, seq, vocab)
```

### Why `keepdims=True` is essential

The `dot` product sums over the last axis (vocab). Without `keepdims`, its shape would be
`(batch, seq)` -- which can't broadcast against `grad_output`'s `(batch, seq, vocab)`.
With `keepdims=True`, dot is `(batch, seq, 1)`, and the subtraction `grad_output - dot`
broadcasts the single value across all vocab positions.

### What each operation does

| Line | Operation | Shape | Why |
|------|-----------|-------|-----|
| `grad_output * s` | element-wise multiply | (B, S, V) | weight each gradient by its probability |
| `.sum(axis=-1, keepdims=True)` | reduce vocab axis | (B, S, 1) | total "pull" from all outputs |
| `grad_output - dot` | subtract scalar per position | (B, S, V) | redistribution: outputs compete |
| `s * (...)` | scale by probability | (B, S, V) | larger probabilities get larger gradients |

**Intuition**: Softmax outputs compete -- if one goes up, others must go down (they sum
to 1). The `- dot` term captures this competition. It's why softmax gradients always sum
to zero along the vocab axis.

---

## 5. ReLU Backward

**Forward**: `y = max(0, x)`

**What we cache**: `mask` -- a binary array, 1 where `x > 0`, 0 elsewhere.

```python
# Forward
self._cache['mask'] = (x > 0).astype(x.dtype)    # 1.0 or 0.0
return x * self._cache['mask']

# Backward
return grad_output * self._cache['mask']
```

### The simplest backward

ReLU is a gate: it passes gradients through where x was positive, blocks them where x was
zero or negative. The derivative of `max(0, x)` is:

```
dy/dx = 1  if x > 0
        0  if x <= 0
```

So the chain rule is just `dL/dx = dL/dy * (1 or 0)` = element-wise multiply by the mask.

**Why `.astype(x.dtype)`?** The comparison `x > 0` produces a boolean array. Multiplying
a float array by a boolean works in NumPy, but storing the mask as float avoids repeated
type casting in backward and makes the code explicit.

**No parameters** -- ReLU has nothing to learn, so `params_and_grads()` returns `[]`.

---

## 6. LayerNorm Backward

**Forward**: `y = gamma * (x - mean) / sqrt(var + eps) + beta`

**What we cache**: `x_norm` (the normalized values) and `std_inv` (1/sqrt(var+eps)).

This is the most mathematically complex backward. The difficulty comes from the fact that
`mean` and `var` are themselves *functions of x*, so the chain rule branches three ways.

### Parameter gradients

```python
sum_axes = tuple(range(grad_output.ndim - 1))     # e.g., (0, 1) for 3D input
self.grad_gamma = (grad_output * x_norm).sum(axis=sum_axes)    # (d_model,)
self.grad_beta = grad_output.sum(axis=sum_axes)                # (d_model,)
```

**`sum_axes`**: gamma and beta have shape `(d_model,)` -- one value per feature. But
`grad_output` has shape `(batch, seq, d_model)`. To get a gradient with shape `(d_model,)`,
we sum over all axes except the last. This is the reverse of broadcasting: gamma was
broadcast across batch and seq, so we sum those axes to collect all contributions.

**Why `tuple(range(ndim - 1))`?** This dynamically handles any number of leading
dimensions:
- 2D input `(batch, d_model)`: sum_axes = `(0,)`
- 3D input `(batch, seq, d_model)`: sum_axes = `(0, 1)`

### Input gradient

```python
dx_norm = grad_output * self.gamma                              # (B, S, D)

grad_input = (std_inv / n) * (
    n * dx_norm                                                 # term 1
    - dx_norm.sum(axis=-1, keepdims=True)                       # term 2
    - x_norm * (dx_norm * x_norm).sum(axis=-1, keepdims=True)   # term 3
)
```

The three terms come from differentiating through the normalization formula. Each one
addresses a different dependency path:

| Term | Expression | What it represents |
|------|------------|-------------------|
| 1 | `n * dx_norm` | Direct path: gradient through the numerator `(x - mean)` |
| 2 | `- dx_norm.sum(axis=-1, keepdims=True)` | Mean path: x affects the mean, which affects all outputs |
| 3 | `- x_norm * (dx_norm * x_norm).sum(...)` | Variance path: x affects the variance, which affects all outputs |

### NumPy broadcasting in action

Both terms 2 and 3 use `.sum(axis=-1, keepdims=True)`, which produces shape `(B, S, 1)`.
This is then broadcast back to `(B, S, D)` during the subtraction and multiplication.

The `std_inv` has shape `(B, S, 1)` from the forward pass (computed with `keepdims=True`),
so `std_inv / n` broadcasts across all D features.

The entire computation is vectorized -- no loops over batch, seq, or features.

---

## 7. Dropout Backward

**Forward**: `y = x * mask / (1 - p)` during training, `y = x` during inference.

**What we cache**: `mask` -- the random binary array.

```python
# Backward (training)
return grad_output * self._cache['mask'] / (1.0 - self.p)

# Backward (inference)
return grad_output
```

### Why the same mask and same scale?

The forward scales survivors by `1/(1-p)` (inverted dropout). The backward must apply
the **exact same mask and scale** because:

1. If a neuron was zeroed in forward, it contributed nothing to the loss, so its gradient
   must be zero too.
2. If a neuron survived, its output was scaled by `1/(1-p)`, so its gradient must be
   scaled by the same factor (chain rule: `dy/dx = mask / (1-p)`).

**No parameters** -- Dropout is pure regularization with nothing to learn.

---

## 8. Embedding Backward

**Forward**: `y = self.weight[token_ids]` -- fancy indexing (row lookup).

**What we cache**: `token_ids` -- the integer indices.

```python
token_ids = self._cache['token_ids']
self.grad_weight = np.zeros_like(self.weight)       # (vocab_size, d_model) of zeros
np.add.at(self.grad_weight, token_ids, grad_output) # scatter gradients back
return None                                          # no input gradient (integers have no gradient)
```

### Why `np.add.at` instead of `+=`?

Consider token_ids = `[3, 1, 3]` (token 3 appears twice):

```python
# WRONG: the second assignment to row 3 overwrites the first
self.grad_weight[token_ids] += grad_output
# grad_weight[3] = grad_output[2]  (only the last write sticks)

# RIGHT: both contributions accumulate
np.add.at(self.grad_weight, token_ids, grad_output)
# grad_weight[3] = grad_output[0] + grad_output[2]  (both writes applied)
```

The `+=` operator uses **buffered** indexing -- it evaluates the right side, then assigns.
When the same index appears twice, the second assignment overwrites the first.
`np.add.at` uses **unbuffered** indexing -- each operation is applied immediately, so
repeated indices accumulate correctly.

### Why `return None`?

The input to Embedding is integer token IDs. You can't take the derivative of a loss with
respect to an integer -- gradients only exist for continuous values. The gradient chain
stops here. The optimizer updates `self.weight` using `self.grad_weight`, but there's no
`dL/dx` to pass upstream.

### Sparsity

Most rows of `grad_weight` stay zero. Only rows corresponding to token IDs that appeared
in the batch receive gradients. For our vocabulary of 13 tokens and a batch of 32 sequences
of length 5 (= 160 tokens), at most 160 out of 13 rows get updated -- in practice all 13
rows get hit because every digit 0-9 plus special tokens appear frequently. For real models
with 50K+ vocabularies, the sparsity is massive: a batch might touch only a few thousand
out of 50,000 rows.

---

## 9. Shape Cheat Sheet

Quick reference for all backward shapes in our Transformer config
(batch=32, seq=5, d_model=32, d_ff=64, vocab=13):

| Layer | Input Shape | grad_output Shape | Parameter Gradient Shapes | grad_input Shape |
|-------|-------------|-------------------|--------------------------|------------------|
| Linear(32,64) | (32, 5, 32) | (32, 5, 64) | W: (32,64), b: (64,) | (32, 5, 32) |
| Softmax | (32, 5, 13) | (32, 5, 13) | -- | (32, 5, 13) |
| ReLU | (32, 5, 64) | (32, 5, 64) | -- | (32, 5, 64) |
| LayerNorm(32) | (32, 5, 32) | (32, 5, 32) | gamma: (32,), beta: (32,) | (32, 5, 32) |
| Dropout(0.1) | (32, 5, 32) | (32, 5, 32) | -- | (32, 5, 32) |
| Embedding(13,32) | (32, 5) ints | (32, 5, 32) | weight: (13,32) | None |

**Rule of thumb**: `grad_input` always matches the shape of the forward input, and
`grad_output` always matches the shape of the forward output. If they don't, you have a bug.

---

## Key Takeaways

1. **reshape(-1, n)** flattens leading dimensions for matrix formulas that expect 2D.
   It's a free view operation -- no data copy.

2. **keepdims=True** is essential after `.sum()` when you need the result to broadcast
   back against the original shape.

3. **Sum over broadcast axes** to get parameter gradients: if a parameter was broadcast
   during forward, sum over those axes during backward.

4. **np.add.at** for scatter operations with duplicate indices. Regular `+=` silently
   drops contributions.

5. **Element-wise mask multiply** is the backward for any element-wise gating operation
   (ReLU, Dropout).

6. **Cache only what you need** -- every cached array costs memory proportional to
   batch_size * seq_len. Our layers cache the minimum required.
