# Step 2: Positional Encoding

> **Status**: Complete (notes + `SinusoidalPositionalEncoding` + tests + walkthrough)
> **Paper section**: Section 3.5
> **Depends on**: Step 1 (Embedding)

[< Math Primitives](01_math_primitives.md) | [Overview](00_overview.md) | [Next: Attention >](03_scaled_dot_product_attention.md)

---

## What We Build

Sinusoidal positional encoding: a **fixed** vector per sequence index, **added** to the token
embedding so the model knows **order**. The Transformer has no recurrence, so without this (or
an equivalent), permuting tokens would largely permute the internal representation in a
symmetric way.

---

## What Positional Encoding Is (and Is Not)

**Token embeddings** map each vocabulary id to a vector. By themselves they say *which token*,
not *where it sits* in the sentence.

**Attention** (the next step) mixes vectors **across positions** using data-dependent weights.
It does not replace positional information: you still need each position to carry a distinct
**positional** signal before or inside the stack so that "token A at place 3" differs from "token
A at place 9".

Positional encoding supplies that signal. In the original paper it is **not learned**; it is
an analytic function of the integer position.

---

## Key Equations (Paper Section 3.5)

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

- **`pos`**: index of the token along the sequence for this forward pass (first token, second,
  ...). It comes from **enumeration of the ordered token list**, not from the embedding table
  and not from the token id.

- **`seq_len`**: number of positions in that sequence slice (after tokenization, including any
  special tokens you insert). In code it is the length dimension of the tensor row. When
  batching, tensors are often **padded** to a common length; then `seq_len` may be the padded
  length, with a mask marking real vs padding positions. The model also has a **maximum context
  length** enforced by architecture and training.

- **`i`**: **frequency band index**, not the raw dimension index. It runs

  `i = 0, 1, 2, …, d_model/2 - 1`

  (assuming `d_model` is even). For each `i`, two embedding dimensions are filled:

  - dimension **`2i`** uses `sin`
  - dimension **`2i + 1`** uses `cos`

  So the **vector coordinates** run `0 … d_model - 1`, produced in **pairs** `(0,1), (2,3), …`
  as `i` steps by **one**.

- **`10000`**: a scale hyperparameter controlling how fast the angles change with `pos` across
  the dimension pairs. It is chosen so some dimensions change slowly with position (coarse
  cues) and others change quickly (fine local cues).

---

## Why It Matters

Without positional encoding, the stack tends to treat the input as a **bag of tokens**: it has
no built-in notion that one token comes **before** another. Order-sensitive tasks (syntax,
narrative, time, code) need that structure.

The sinusoidal form gives each position a **distinct** pattern across dimensions. The paper
also argues it may help **relative** structure: shifting `pos` by a fixed offset acts on each
sin/cos pair like a fixed linear transformation, which may be easy for subsequent linear layers
and attention to exploit.

**No learnable parameters** in the original definition: no gradient through PE itself (though
the rest of the model still learns how to use PE).

---

## Logic Behind Sin and Cos

**Role of `pos`**: it enters only inside the angle `pos / div(i)`. Moving along the sequence
changes the phase of the sinusoids.

**Role of `div(i) = 10000^(2i / d_model)`**: as `i` grows, the divisor grows, so the angle grows
**more slowly** with `pos` for large `i`. Small `i` → **shorter effective wavelength** along the
sequence (sensitive to nearby positions). Large `i` → **longer wavelength** (coarse position
signal).

**Why sin on even dimensions and cos on odd**: at a given frequency, the pair `(sin θ, cos θ)`
lies on a circle as `θ` changes. Two channels per frequency give the model a richer phase
representation than a single sinusoid per frequency.

**Why not raw integers**: raw position numbers are awkward magnitudes for a neural net to decode.
A **spectrum** of periodic features at different scales is a hand-designed basis that often
generalizes and needs no training step for PE itself.

---

## Worked Example: `d_model = 4`

Use **radians**. Then `i ∈ {0, 1}` only.

**Divisors**

- `i = 0`: `div(0) = 10000^(0/4) = 1`
- `i = 1`: `div(1) = 10000^(2/4) = sqrt(10000) = 100`

### `pos = 0`

Angles are `0` for both bands.

| dim | formula   | value |
|----:|-----------|------:|
| 0   | sin(0)    | 0     |
| 1   | cos(0)    | 1     |
| 2   | sin(0)    | 0     |
| 3   | cos(0)    | 1     |

`PE(0) = [0, 1, 0, 1]`.

### `pos = 1`

- Band `i = 0`: angle `1 / 1 = 1` rad
- Band `i = 1`: angle `1 / 100 = 0.01` rad

| dim | formula      | value (approx.) |
|----:|--------------|----------------:|
| 0   | sin(1)       | 0.8415          |
| 1   | cos(1)       | 0.5403          |
| 2   | sin(0.01)    | 0.0100          |
| 3   | cos(0.01)    | 0.99995         |

`PE(1) ≈ [0.8415, 0.5403, 0.0100, 1.0000]`.

**What to notice**: dimensions **0–1** (small `i`, divisor 1) change a lot between adjacent
positions; dimensions **2–3** (larger `i`, divisor 100) barely move between `pos = 0` and `pos =
1`. At much larger `pos` (try `100` vs `101` with the same `d_model`), the slow band moves more
noticeably while the fast band has already gone through many cycles. That multi-scale behavior
is the point.

---

## Connection to the Next Step

After adding PE to embeddings, each position has a vector that depends on **both** identity
(token) and **place** (order). **Scaled dot-product attention** then builds **contextual**
representations by mixing positions; see
[03_scaled_dot_product_attention.md](03_scaled_dot_product_attention.md).

---

## Implementation Notes (when we code)

- Shape: broadcast or add `PE` of shape compatible with `(batch, seq_len, d_model)` to
  embeddings of the same shape.
- Same PE table is reused for every batch row; only `seq_len` and masking differ per example.

---

## Alternatives (for later)

Learned positional embeddings (one vector per index up to `max_len`), rotary embeddings (RoPE),
ALiBi, etc. The sinusoidal form remains the historical baseline from the paper.
