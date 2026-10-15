# Step 4: Multi-Head Attention

> **Status**: In progress (intuition + math + worked example landed; reshape
> trick, backward pass, three usage modes, and code/tests still ahead)
> **Paper section**: Section 3.2.2, Section 3.2.3, Figure 2 (right)
> **Depends on**: Step 1 (Linear), Step 3 (Scaled Dot-Product Attention)

[< Scaled Attention](03_scaled_dot_product_attention.md) | [Overview](00_overview.md) | [Next: FFN >](05_feedforward_network.md)

---

## What we build

Instead of running scaled dot-product attention once, run it `h = 4` times in
parallel -- each "head" gets its own learned projections of the input into
queries, keys, and values, produces its own attention pattern, and the heads'
outputs are concatenated and re-projected. The result preserves the
`(batch, n, d_model)` shape so it can drop into a residual stream.

Paper equations (Section 3.2.2):

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

`Attention(...)` is exactly the operator from Step 3 -- multi-head adds the
learnable projections (`W_i^Q, W_i^K, W_i^V, W^O`) around it.

This doc is built **one concept at a time**, mirroring Step 3's structure. A
chronological Discussion Log lives at the bottom; everything we work through
in chat ends up there.

---

## 1. Where we stand after Step 3

After Step 3, we have a parameter-free operator:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

It takes already-projected `Q, K, V` and returns a soft, content-based mixture
of values. For each query position, it produces **one** distribution over
keys -- one set of attention weights, one mixture.

What's missing:

1. **Where do `Q, K, V` come from?** In Step 3 we just handed them in. In a
   real model the same input vector `X` gets projected three different ways
   -- once into a "query" role, once into a "key" role, once into a "value"
   role -- by learned matrices. Those matrices live in Step 4.
2. **One attention pattern is not enough.** A single softmax has to commit to
   one mixture per query, but real relationships between tokens are
   multi-faceted (more on this in section 2).

Multi-head attention adds the learnable projections **and** runs the operator
`h` times in parallel.

---

## 2. Why multiple heads?

### The single-head bottleneck

A single attention head produces, for each query position `i`, exactly one
row of weights:

```
weights[i, :] = softmax(Q[i] . K^T / sqrt(d_k))    # n probabilities, sum to 1
```

Because softmax is normalized, putting probability mass on token A means
taking mass away from token B. **One query = one attention pattern.**

But the relationships a token cares about are rarely one-dimensional. Even
in our toy sort task, position 2 (holding value `4`) might want to track
several things at once:

- "Which tokens have a **smaller** value than me?" -- the relative-order signal.
- "Which tokens have the **same** value as me?" -- duplicates need stable order.
- "How close am I to the boundary of the sequence?" -- positional patterns.
- "Who are my immediate neighbors?" -- local smoothing.

A single softmax can answer only one of those at a time. If we want
position 2 to attend strongly to "smaller values" **and** "duplicates"
simultaneously, we need multiple parallel attention computations.

### Panel-of-experts mental model

Multi-head attention = a panel of `h` independent experts:

- Each head has its own `W_i^Q, W_i^K, W_i^V` projections.
- Each head produces its own attention pattern over the same input.
- Their outputs are concatenated and re-projected with `W^O`.

We do **not** program these roles. The projections are initialized randomly
and the gradient signal during training causes heads to specialize.
Empirically they do (there is a whole sub-literature visualizing what
individual heads learn).

### Why split `d_model` into `h` chunks instead of running `h` full-width heads?

The paper sets `d_k = d_v = d_model / h`. There are actually three options to
compare, not two:

| Option                              | Per-head dim   | # heads | # patterns | Total Q K^T cost      |
|-------------------------------------|----------------|---------|------------|-----------------------|
| A. Single full-width head           | `d_model`      | 1       | 1          | `n^2 . d_model`       |
| **B. `h` narrow heads (the paper)** | `d_model / h`  | `h`     | `h`        | `n^2 . d_model`       |
| C. `h` full-width heads             | `d_model`      | `h`     | `h`        | `n^2 . h . d_model`   |

Option B is "free" relative to A: same FLOPs, but `h` different attention
patterns instead of one. That is the headline trade.

But the more interesting comparison is **B vs C**: why not give each head full
width? Three reasons, in increasing subtlety:

**1. Cost goes up `h`x.** In our config (`d_model = 32, h = 4`), B has 4096
total MHA parameters; C would have 16384 (~4x more), and ~4x more attention
FLOPs. For real models with `d_model = 512` and `h = 8`, this scales badly.

**2. The softmax bottleneck means width does not buy you more patterns.**
This is the deepest reason multi-head exists at all. Each head produces
**one** softmax distribution per query, so **one head = one attention
pattern, no matter how wide the head is**. Making `d_k` bigger sharpens the
similarity scoring, but the output of `softmax(Q K^T / sqrt(d_k))` is still
one row of probabilities summing to 1. If your attention pattern needs to
be bimodal ("smaller-valued tokens AND duplicate tokens"), a single softmax
can blend the two but cannot have two independent magnitudes -- mass
conservation forbids it. So:

- A (1 wide head): cheap, but 1 pattern. Loses to B on diversity.
- C (`h` wide heads): `h` patterns, but `h`x cost. Loses to B on cost.
- B (`h` narrow heads): `h` patterns at the cost of A. Strictly dominates.

**3. Narrow heads regularize toward specialization.** In option C, each
`W_i^Q` is full `(d_model, d_model)` -- it can map the input into anywhere in
`R^{d_model}`. Nothing structurally forces head 1 to look at different
things than head 2. With random init and gradient descent there is a real
risk of **head collapse**: heads learning redundant patterns. In option B,
each `W_i^Q` is `(d_model, d_model / h)` -- a **compressing** projection
into a smaller subspace. If two heads picked overlapping subspaces they
would waste their already-limited capacity. The bottleneck acts as a
regularizer that pushes heads toward different subspaces -- the same idea
that powers dropout, autoencoder bottlenecks, and random projections.
Empirical work (e.g. Voita et al. 2019) finds that narrow-head models have
more interpretable, specialized heads than wide-head ablations.

**Bonus consequence:** with `d_v = d_model / h`, `h . d_v = d_model`, so
`W^O` is square `d_model x d_model` and the concat output drops straight
back into the residual stream. The paper's choice is doing three things at
once: (1) keeping total cost the same as single-head, (2) making `W^O`
square so the residual stream stays at `d_model`, and (3) regularizing heads
toward specialization via the per-head bottleneck.

### Is `d_k = d_model / h` actually required?

Short answer: **no, it is a convention, not a mathematical requirement.**
The math allows any `h`, any `d_k`, and any `d_v`. The constraint comes
from implementation convenience and the three benefits above, not from the
attention equations themselves.

**What the math actually requires:**

| Tensor   | Shape                  | What is constrained                                                                  |
|----------|------------------------|--------------------------------------------------------------------------------------|
| `W_i^Q`  | `(d_model, d_k)`       | `d_k` can be **any positive integer**                                                |
| `W_i^K`  | `(d_model, d_k)`       | Must use the **same** `d_k` as `W_i^Q` (so per-head `Q . K^T` is defined)            |
| `W_i^V`  | `(d_model, d_v)`       | `d_v` can be **any positive integer**, and **need not equal `d_k`**                  |
| `W^O`    | `(h . d_v, d_model)`   | Input dim must match concat width `h . d_v`; output must match residual `d_model`    |

Notice what is **not** in this list: any constraint between `h` and
`d_model`. You can pick `h = 3` with `d_model = 32` and `d_k = 10`. Then
`W^O` is `(30, 32)` -- rectangular -- and everything else still works.

**Three reasons everyone uses `d_k = d_model / h` anyway:**

1. **Cost parity with single-head.** Total attention compute is
   `h . n^2 . d_k`. Setting `d_k = d_model / h` makes this equal `n^2 . d_model`
   -- identical to a single full-width head. Pick any other `d_k` and you
   change the FLOP budget, usually for the worse.
2. **`W^O` becomes square.** `h . d_v = d_model` makes `W^O` a square
   `(d_model, d_model)` matrix. Slightly cleaner, slightly faster, and it
   matches the standard "Linear" layer initialization schemes.
3. **Enables the one-big-matmul-plus-reshape implementation trick.** When
   `h . d_k = d_model`, you can replace `h` separate small matmuls with one
   big square matmul plus a reshape (covered in section 4). Without that
   constraint, the trick still works but `W^Q` becomes rectangular -- and
   most production libraries (PyTorch's `nn.MultiheadAttention`, JAX/Flax,
   TensorFlow) just `assert d_model % h == 0` rather than handle the
   rectangular case. The constraint is **implementation-driven**, not
   math-driven.

**Real-world counterexamples** where the convention is deliberately broken:

- **Multi-Query Attention (MQA, Shazeer 2019)** -- `h` query heads but **one
  shared key/value head**. Used in PaLM, Falcon. Cuts the KV cache by `h`x
  at inference time. Definitely not `d_k = d_model / h` in the symmetric
  sense.
- **Grouped-Query Attention (GQA, Ainslie et al. 2023)** -- in between MHA
  and MQA: `h_q` query heads, `h_kv < h_q` key/value heads, with each KV
  head shared by `h_q / h_kv` queries. Used in LLaMA-2 70B
  (`h_q = 64, h_kv = 8`).
- **Asymmetric `d_k` vs `d_v`** -- rare but legal. Some sparse-attention
  variants project keys to a small dim (just enough for similarity scoring)
  and values to a larger dim (more capacity per value).

**Concrete example.** If you really wanted `h = 3` with our `d_model = 32`,
four ways to make it work:

```
Option 1:  d_k = d_v = 11,  h * d_k = 33   -- W^O is (33, 32), rectangular. Works.
Option 2:  d_k = d_v = 10,  h * d_k = 30   -- W^O is (30, 32), rectangular. Works.
Option 3:  bump d_model to 33 (or 30)      -- restores h | d_model. Works.
Option 4:  pick h = 2 or 4 instead of 3    -- restores h | d_model with d_model = 32.
```

Most code defaults to Option 4 (pick `h` to divide `d_model`) because it is
the path of least resistance. Nothing fundamental forces it.

**What we do for this build.** Keep the convention `d_k = d_v = d_model / h`
because (a) it matches the paper, (b) the reshape trick keeps the code
clean and fast, and (c) it matches 99% of real-world Transformer code, so
you will recognize the pattern when reading other implementations. But
when we fill in section 4, we will explicitly call out **"the trick (not
the math) is what requires `h . d_k = d_model`"** so the conceptual model
does not get accidentally locked into the convention.

---

## 3. The paper equations, in shapes

### Equations (Section 3.2.2)

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i             = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

### Learnable parameters

| Symbol      | Shape (general)         | Shape (our toy)  | Count (toy) |
|-------------|-------------------------|------------------|-------------|
| `W_i^Q`     | `(d_model, d_k)`        | `(32, 8)`        | 4 . 256 = 1024 |
| `W_i^K`     | `(d_model, d_k)`        | `(32, 8)`        | 4 . 256 = 1024 |
| `W_i^V`     | `(d_model, d_v)`        | `(32, 8)`        | 4 . 256 = 1024 |
| `W^O`       | `(h . d_v, d_model)`    | `(32, 32)`       | 1024 |
| **Total**   |                         |                  | **4096 weights** per MHA layer |

(Plus optional biases on each projection. The original paper omits biases on
the attention projections; many later implementations include them. We will
follow the paper unless we find a reason not to.)

Note `h . d_v = 32 = d_model`, so `W^O` is square. Not a coincidence -- the
choice `d_v = d_model / h` is what makes the concat output drop straight back
into the residual stream.

### Forward-pass shapes

Starting from `X` of shape `(B, n, d_model) = (B, 5, 32)`:

```
Q = X @ W^Q                 -> (B, 5, 32)
K = X @ W^K                 -> (B, 5, 32)
V = X @ W^V                 -> (B, 5, 32)

split-heads (reshape + transpose):
    (B, 5, 32) -> (B, 5, 4, 8) -> (B, 4, 5, 8)

per-head attention (head dim acts like an extra batch dim):
    scores  = Q . K^T / sqrt(8)   -> (B, 4, 5, 5)
    weights = softmax(scores)     -> (B, 4, 5, 5)
    head    = weights . V         -> (B, 4, 5, 8)

merge-heads (transpose + reshape):
    (B, 4, 5, 8) -> (B, 5, 4, 8) -> (B, 5, 32)

output = merged @ W^O           -> (B, 5, 32)
```

Input shape and output shape both equal `(B, n, d_model)`. That is what lets
MHA sit inside a residual block: `output = x + MHA(LayerNorm(x))`.

---

## 4. The split-heads reshape trick

### The naive way (do not do this)

A literal reading of `head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)`
suggests we maintain `h` separate matrices for each role -- `h` matrices for
`Q`, `h` for `K`, `h` for `V`. For our toy config that is `h = 4` matrices
of shape `(32, 8)` per role. The forward pass would loop `h` times:

```
for i in range(h):
    Q_i = X @ W_i^Q       # (B, n, 32) @ (32, 8) -> (B, n, 8)
    K_i = X @ W_i^K
    V_i = X @ W_i^V
    head_i = Attention(Q_i, K_i, V_i)   # (B, n, 8)
heads = stack along last dim          # (B, n, 32)
out = heads @ W^O
```

This works, but it is `h` separate small matmuls per role. Real hardware
(BLAS, cuBLAS) is much happier with one big matmul than `h` small ones.

### The trick: one big matmul + a reshape

Observation: stacking `W_1^Q, ..., W_h^Q` side-by-side along the **output**
dimension produces a single matrix of shape `(d_model, h . d_k)`. When
`h . d_k = d_model` (the convention from section 2), that single matrix is
square `(d_model, d_model)`. Call it `W^Q` (no subscript).

```
       d_k         d_k                  d_k                h . d_k = d_model
    +-------+   +-------+            +-------+         +-------------------+
W^Q | W_1^Q | + | W_2^Q |  + ... +  | W_h^Q |   =     | W^Q (one big)     |
    |       |   |       |            |       |         |                   |
    +-------+   +-------+            +-------+         +-------------------+
   d_model       d_model              d_model              d_model
```

So we do one matmul:

```
Q = X @ W^Q       # (B, n, d_model) @ (d_model, d_model) -> (B, n, d_model)
```

The result is the per-head queries **concatenated** along the last axis. To
recover them as separate heads we just **reshape** that last axis from
`d_model` into `(h, d_k)` and then **transpose** so the head dim sits where
SDPA expects a batch dim.

### Forward: split-heads (the reshape, step by step)

Starting from `Q ∈ R^{(..., n, d_model)}`:

```
Step 1: reshape last axis  d_model -> (h, d_k)
        (..., n, d_model)  ->  (..., n, h, d_k)

Step 2: swap axes -2 and -3 (move h before n)
        (..., n, h, d_k)   ->  (..., h, n, d_k)
```

Step 1 is a no-op in memory -- it just relabels the last `d_model`
contiguous floats as an `h`x`d_k` grid. Step 2 is a transpose, which makes
the array non-contiguous as a view, but `numpy.matmul` (which SDPA uses)
handles non-contiguous inputs fine.

After both steps `h` is the second-to-last "batch" dim. SDPA sees:

```
Q_split:  (B, h, n,   d_k)     # the "h" is just another batch dim to SDPA
K_split:  (B, h, n,   d_k)
V_split:  (B, h, n,   d_v)
output:   (B, h, n_q, d_v)
```

### Why this transpose ordering?

We could have left the layout as `(..., n, h, d_k)` and modified SDPA to
contract over a different axis, but the chosen ordering has two
properties:

1. **`h` is in the leading "batch" position.** SDPA's contract is "do
   attention along the last two dims, treat everything to the left as
   batch." Putting `h` to the left of `n` means SDPA does `h` attention
   computations in parallel -- exactly the per-head semantics we want --
   without any code change to SDPA.
2. **The per-head Q/K/V slices are contiguous along `d_k`.** Their dot
   products `Q_i . K_i^T` work without any further reshape.

### Forward: merge-heads (undo the split)

The output of SDPA is `(..., h, n_q, d_v)`. We need to reverse the split
exactly:

```
Step 1: swap axes -2 and -3 (move n back before h)
        (..., h, n_q, d_v)  ->  (..., n_q, h, d_v)

Step 2: reshape last two axes  (h, d_v) -> h . d_v = d_model
        (..., n_q, h, d_v)  ->  (..., n_q, d_model)
```

Step 2 typically requires a copy (the data is non-contiguous after
step 1), but `numpy.reshape` handles that automatically.

The merged tensor of shape `(..., n_q, d_model)` is what `W^O` projects:

```
output = merged @ W^O          # (..., n_q, d_model)
```

### Required: `h . d_k = d_model`

The trick requires `W^Q` to be one big matrix whose output dim equals the
concat width `h . d_k`. With the convention `d_k = d_model / h`, that
output dim is exactly `d_model` and `W^Q` is square -- which is also what
makes the matmul reuse the residual-stream width. Drop the convention
(say `h = 3, d_model = 32, d_k = 10`) and the trick still works, but `W^Q`
becomes `(32, 30)` (rectangular), the post-matmul tensor is
`(..., n, 30)`, and the reshape becomes `30 -> (3, 10)`. See section 2 for
why we keep the convention anyway.

### Backward (preview)

The backward pass through this layer travels through three composed
operations: `W^O` projection, the merge-heads reshape+transpose, SDPA, and
finally the split-heads transpose+reshape into `W^Q`/`W^K`/`W^V`. The
reshape and the transpose are both linear and have trivial Jacobians (the
gradient is the **inverse** transpose, then the **inverse** reshape), so
the math is straightforward but worth deriving step by step. We do that in
section 8.

---

## 5. A tiny worked example you can trace by hand

Smallest non-trivial case: `n = 2` tokens, `d_model = 4`, `h = 2`, so
`d_k = d_v = 2`.

### Setup

```
X = [[1, 0, 1, 0],          shape (2, 4) -- two tokens, each a 4-vector
     [0, 1, 0, 1]]
```

Pick projections that make the arithmetic visible:

```
W^Q = I_4                            -> Q = X
W^K = I_4                            -> K = X
W^V = diag(1, 1, 2, 2)               -> V = [[1, 0, 2, 0],
                                              [0, 1, 0, 2]]
```

### Split into heads

Reshape `Q` from `(n=2, d_model=4)` to `(n=2, h=2, d_k=2)`, then transpose to
`(h=2, n=2, d_k=2)`. Same for `K` and `V`:

```
                head 0                       head 1
        +-------------------+        +-------------------+
Q  ->   | Q_0 = [[1, 0],    |        | Q_1 = [[1, 0],    |
        |        [0, 1]]    |        |        [0, 1]]    |
K  ->   | K_0 = [[1, 0],    |        | K_1 = [[1, 0],    |
        |        [0, 1]]    |        |        [0, 1]]    |
V  ->   | V_0 = [[1, 0],    |        | V_1 = [[2, 0],    |
        |        [0, 1]]    |        |        [0, 2]]    |
        +-------------------+        +-------------------+
```

`Q` and `K` split identically because `W^Q = W^K = I`. `V` splits asymmetrically
because of the `diag(1, 1, 2, 2)`: head 0 gets the first two columns
(scaled by 1), head 1 gets the last two (scaled by 2).

### Per-head scaled dot-product attention

`d_k = 2`, so the scale is `sqrt(2) ~ 1.414`.

**Head 0:**

```
Q_0 . K_0^T = I_2

scores_0 = (1 / sqrt(2)) . I_2 = [[0.707, 0    ],
                                   [0    , 0.707]]

softmax(row 0 of scores_0) = softmax([0.707, 0])     ~ [0.670, 0.330]
softmax(row 1 of scores_0) = softmax([0    , 0.707]) ~ [0.330, 0.670]

weights_0 = [[0.670, 0.330],
             [0.330, 0.670]]

out_0 = weights_0 . V_0 = [[0.670, 0.330],
                           [0.330, 0.670]]
```

**Head 1:** identical `weights` (since `Q_1, K_1` match `Q_0, K_0`), but
different `V_1`:

```
out_1 = weights_0 . V_1 = [[1.340, 0.660],
                           [0.660, 1.340]]
```

### Concat + project

Concatenate `out_0` and `out_1` along the feature dimension:

```
concat = [[0.670, 0.330, 1.340, 0.660],
          [0.330, 0.670, 0.660, 1.340]]    shape (2, 4)
```

Apply `W^O = I_4` for clarity -- final output equals `concat`.

### What this trace teaches

- Both heads in this rigged example produced the **same** attention pattern,
  because `W^Q` and `W^K` were identity (and shared between heads). In a
  real model the projections differ, so each head's `weights` matrix is
  different.
- The two heads produced **different outputs** purely because their value
  projections differed (`V_0` vs `V_1`). This is the smallest demonstration
  that "head specialization" can come from any of the four projections.
- Shapes line up: input `(2, 4)` -> output `(2, 4)`. MHA preserves the
  residual-stream width `d_model`.

**This trace is now the first unit test for the layer:**
`test_mha_section5_trace` in `tests/test_layers.py` reproduces these exact
numbers within `1e-3` tolerance. If the test ever drifts from these
numbers, this section is the canonical reference -- update one or the
other so they stay in sync.

---

## 6. Three usage modes (preview)

Multi-head attention is used in **three** different positions in the
Transformer, with different inputs and masking. Same class, different
arguments.

| Mode                     | Q from   | K, V from | Mask                      | Where in the architecture |
|--------------------------|----------|-----------|---------------------------|---------------------------|
| Encoder self-attention   | encoder  | encoder   | (optional padding mask)   | Each encoder layer        |
| Decoder masked self-attn | decoder  | decoder   | causal (no future tokens) | Each decoder layer        |
| Encoder-decoder cross-attn | decoder | encoder  | (optional padding mask)   | Each decoder layer        |

We will dwell on this when we get to encoder (Step 6) and decoder (Step 7).
**TODO: expand this section once we wire MHA into the encoder.**

---

## 7. Forward pass in code (landed)

The forward pass lives in
[`numpy_transformer.layers.MultiHeadAttention`](../numpy-transformer/src/numpy_transformer/layers.py)
and follows the layout from sections 3 and 4 directly.

### Class shape

```python
class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        # d_model % num_heads == 0 enforced (split-heads trick)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Four learnable projections via Linear (Glorot init, bias = 0)
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        self._sdpa = ScaledDotProductAttention()    # reused engine

    def forward(self, Q_in, K_in, V_in, mask=None):
        Q = self.W_q.forward(Q_in)
        K = self.W_k.forward(K_in)
        V = self.W_v.forward(V_in)

        Q_h = self._split_heads(Q)         # (..., h, n,   d_k)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        head_out = self._sdpa.forward(Q_h, K_h, V_h, mask=mask)   # (..., h, n_q, d_k)

        merged = self._merge_heads(head_out)    # (..., n_q, d_model)
        return self.W_o.forward(merged)
```

### Design choices we made

- **Reuse `ScaledDotProductAttention` from Step 3 unchanged.** Its existing
  contract -- "attend along last two dims, treat everything else as batch"
  -- is exactly what multi-head needs once `h` is the second-to-last
  leading dim. No code duplication, no inline copy of the SDPA logic.
- **Linear sub-layers for the four projections.** Each `Linear` already
  handles 3D inputs `(B, n, d_model)` via the reshape trick documented in
  step 1, so `MultiHeadAttention.forward` does not need a 2D-vs-3D branch.
  Biases are initialized to zero, so the section-5 trace test (which
  overrides `W` and leaves `b = 0`) reproduces the hand-traced numbers
  exactly.
- **Bias-or-no-bias.** The paper omits projection biases; PyTorch's
  `nn.MultiheadAttention` includes them. We keep the biases (zero-init)
  to match the more common modern convention. The deviation is a
  documented design call, not an oversight; future versions of `Linear`
  may add a `bias=False` flag if we want to align with the paper exactly.
- **Splitting via reshape + `np.swapaxes`.** Step 1 of `_split_heads` is a
  no-op in memory; step 2 returns a non-contiguous view that
  `numpy.matmul` (used inside SDPA) handles natively. The merge does the
  inverse and lets `numpy.reshape` copy when needed.
- **`last_weights` is forwarded from SDPA** for inspection (shape
  `(..., h, n_q, n_k)`). Useful for visualization later (Step 10).
- **Backward composes sub-layer backwards.** Each sub-`Linear` takes care
  of its own `grad_W` and `grad_b`; SDPA returns its own `(grad_Q,
  grad_K, grad_V)` 3-tuple; the reshape/transpose ops invert through
  `_split_heads` <-> `_merge_heads`. See section 8.

### Public surface

| Name | Type | Purpose |
|------|------|---------|
| `MultiHeadAttention(d_model, num_heads)` | constructor | Validates divisibility; allocates 4 `Linear`s |
| `.forward(Q_in, K_in, V_in, mask=None)` | method | Returns `(..., n_q, d_model)` |
| `.last_weights` | attribute | `(..., h, n_q, n_k)` from the last forward call |
| `.params_and_grads()` | method | Concatenates the four `Linear`s' params, prefixed with `W_q.`, `W_k.`, `W_v.`, `W_o.` |
| `.W_q`, `.W_k`, `.W_v`, `.W_o` | attribute | The underlying `Linear` sub-layers (override `.W` and `.b` to inject specific weights in tests) |

### What is *not* in the implementation yet

- Optional dropout on the attention weights -- the paper's section 5.4
  applies dropout to the softmax output in addition to the residual
  dropout. We will add this when wiring MHA into the encoder/decoder
  blocks, alongside the dropout layers there.
- Auto-insertion of a head dim for 3D `(B, n_q, n_k)` masks. Today the
  caller is responsible; tests use 2D masks where broadcasting is
  unambiguous.

---

## 8. Backward pass

The good news: this layer's backward is **almost entirely sub-layer
composition**. We have already derived the gradient through every
non-trivial piece in earlier steps -- `Linear` (Step 1), softmax (Step 1
+ Step 3), and `ScaledDotProductAttention` (Step 3). All that remains is
composing them and threading two **shape-only** operations: reshape and
`np.swapaxes`.

### The key observation: reshape and transpose are linear

A reshape just relabels the same memory under a different shape; it is
the identity map on values, only the index lookup changes. A
transpose / `np.swapaxes` is a permutation of indices. Both are linear
operations whose Jacobian is a permutation matrix `P` (a square 0/1
matrix with exactly one 1 per row and column).

For a permutation, `P^T = P^{-1}`. So if the forward applies `y = P x`,
the backward of a downstream loss `L` with respect to `x` is:

```
dL/dx = P^T . dL/dy = P^{-1} . dL/dy
```

In words: **the gradient through a permutation is the inverse permutation
applied to the upstream gradient**. For shape ops this means:

- `forward = reshape A -> B`  =>  `backward = reshape B -> A` on the gradient
- `forward = swapaxes(i, j)`  =>  `backward = swapaxes(i, j)` on the gradient
  (`swapaxes` is its own inverse -- swapping the same two axes twice gets
  you back where you started)

### Backward of `_split_heads` is `_merge_heads` (and vice versa)

`_split_heads(x)` does:

```
(..., n, d_model) -- reshape   --> (..., n, h, d_k)
                  -- swapaxes(-2, -3) --> (..., h, n, d_k)
```

Reversing each step in order, with the inverse Jacobian for each:

```
(..., h, n, d_k) -- swapaxes(-2, -3) --> (..., n, h, d_k)
                 -- reshape         --> (..., n, d_model)
```

That is **exactly** what `_merge_heads` does. So:

```
backward_of_split_heads(g) = _merge_heads(g)
backward_of_merge_heads(g) = _split_heads(g)
```

The forward and backward swap roles. Two helper methods, four uses --
forward calls each once, backward calls each once.

### The full backward chain

Forward, written compactly:

```
Q       = W_q(Q_in)              # Linear
K       = W_k(K_in)              # Linear
V       = W_v(V_in)              # Linear
Q_h     = split_heads(Q)         # reshape + swapaxes
K_h     = split_heads(K)
V_h     = split_heads(V)
head    = SDPA(Q_h, K_h, V_h)    # Step 3
merged  = merge_heads(head)      # swapaxes + reshape
output  = W_o(merged)            # Linear
```

Backward, run the chain in reverse with the matching backward of each op:

```
grad_merged   = W_o.backward(grad_output)               # Linear backward (also fills W_o's grad_W, grad_b)
grad_head     = split_heads(grad_merged)                # backward of merge_heads
grad_Q_h, grad_K_h, grad_V_h = SDPA.backward(grad_head) # 3-tuple from Step 3
grad_Q        = merge_heads(grad_Q_h)                   # backward of split_heads
grad_K        = merge_heads(grad_K_h)
grad_V        = merge_heads(grad_V_h)
grad_Q_in     = W_q.backward(grad_Q)                    # also fills W_q's grad_W, grad_b
grad_K_in     = W_k.backward(grad_K)
grad_V_in     = W_v.backward(grad_V)

return (grad_Q_in, grad_K_in, grad_V_in)                # 3-tuple, like SDPA
```

Three things worth noticing:

1. **`Linear.backward` does double duty.** Each call (a) computes
   `grad_input` to thread back through the chain, and (b) populates that
   sub-layer's `grad_W` and `grad_b` for the optimizer to consume later.
   We get parameter gradients "for free" as a side effect of computing
   input gradients.
2. **No new derivatives.** All the calculus was already done in Steps 1
   and 3. This step is bookkeeping.
3. **3-tuple return.** Same convention as SDPA: a layer with three
   inputs returns a 3-tuple. The three may be the same tensor (in
   self-attention `Q_in == K_in == V_in == X`) -- in that case the caller
   sums the three to get `grad_X`. MHA does **not** do that summation
   itself, because it does not know whether `Q_in, K_in, V_in` are the
   same tensor or different ones (e.g. cross-attention: `Q_in` is the
   decoder, `K_in == V_in` is the encoder output).

### Why we cache so little

Look at the actual cache in `forward` -- almost nothing. That is because
`Linear` and `SDPA` cache their own intermediates. `MultiHeadAttention`
just calls their `forward` (which fills their caches) and `backward`
(which reads them). The reshape / swapaxes operations are stateless --
their Jacobian depends only on shape, which the gradient already carries.

This is a general lesson: **when you compose sub-layers that each
maintain their own cache, the parent layer's cache stays small**. The
parent's job is to plumb gradients through; the sub-layers carry their
own state.

### Self-attention and gradient summation

When the same input `X` feeds all three roles (`Q_in == K_in == V_in
== X`), the chain rule says `grad_X = grad_Q_in + grad_K_in + grad_V_in`.
This is just multivariate chain rule: a single variable that affects the
loss through three distinct paths contributes the **sum** of the three
partial gradients.

The caller is responsible for that sum. For `test_mha_backward_gradient_check`
we test the three returned gradients separately by feeding `Q_in, K_in,
V_in` as three distinct random tensors. For self-attention tests the
caller (or the encoder layer in Step 6) does the summation.

### A beautiful sanity-check fact: `grad_W_k.b` is *exactly* zero

While writing the gradient-check test we discovered that the analytical
gradient for the bias of `W^K` came out at machine-precision noise
(`~1e-17`) while the central-difference numerical gradient came out at
`~1e-11`. Both are essentially zero -- but the relative-error metric
`|a - n| / |n|` exploded to `0.22%` because the denominator was tiny.

Why is `grad_b_k` *exactly* zero (not just small)? It is a structural
property of attention with key biases:

```
K_full[i, j]    = K_orig[i, j] + b_k[j]                         # add bias
scores[i, k]    = sum_j Q[i, j] . K_full[k, j]
                = sum_j Q[i, j] . K_orig[k, j]  +  sum_j Q[i, j] . b_k[j]
                = scores_orig[i, k]             +  (Q b_k)[i]
                              ^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^
                              real attention      depends only on i,
                              signal              not on k
```

The added term `(Q b_k)[i]` depends only on the **query index `i`**, not
the **key index `k`**. So `b_k` shifts every entry in row `i` of the
score matrix by the *same constant*. And softmax is invariant to per-row
constant shifts (we use that fact deliberately for numerical stability
in the softmax forward -- subtract the row max before exponentiating).

Conclusion: `b_k` cannot influence the attention weights, cannot
influence the output, and therefore has zero gradient -- regardless of
input. The `Linear.backward` formula `grad_b = grad.sum(axis=0)` does
return zero here, because the upstream `grad_K` (after summing over
`n_k`) is itself zero.

**Same argument does NOT apply to the other biases:**

| Bias  | Effect on attention                                      | Generally non-zero gradient? |
|-------|----------------------------------------------------------|------------------------------|
| `b_q` | Shift each *column* of scores by `(K b_q)[k]` -- per-key, not per-query, so softmax does see it | Yes |
| `b_k` | Shift each *row* of scores by `(Q b_k)[i]` -- per-query, softmax invariant   | **No -- always zero** |
| `b_v` | Shifts the output by `b_v` directly (since `weights @ b_v_broadcast = b_v`); the shift then flows through `W_o` | Yes |
| `b_o` | Output bias -- shifts the final output by `b_o` directly | Yes |

This is the kind of zero-gradient that can mask real bugs in a gradient
check if you only look at relative error. We handle it in the test by
combining absolute and relative tolerances:

```
ok = (abs_err < 1e-8) | (rel_err < 1e-4)
```

And we add a dedicated `test_mha_W_k_bias_gradient_is_exactly_zero` test
that asserts the structural fact directly, so the property is preserved
even if the gradient-check tolerances ever change.

(Practical implication: most real implementations of MHA omit the K-bias
entirely -- e.g. the original Transformer paper has no projection biases
at all. PyTorch's `nn.MultiheadAttention` retains it for code uniformity
even though it is mathematically dead weight, and so do we.)

---

## 9. Sanity checks and tests (landed)

Ten tests in
[`tests/test_layers.py`](../numpy-transformer/tests/test_layers.py),
all green.

**Forward / shape / config (sections 1-7):**

| Test | What it asserts |
|------|-----------------|
| `test_mha_section5_trace` | Reproduces the hand-traced numbers from section 5 (`n=2, d_model=4, h=2`, identity Q/K/O projections, diag V projection) within `1e-3`. Also checks `last_weights.shape == (h, n, n)` and per-row sum to 1. |
| `test_mha_sort_task_shapes` | Sort-task config (`B=32, n=5, d_model=32, h=4`): `forward(X, X, X)` returns `(32, 5, 32)`, `last_weights` is `(32, 4, 5, 5)`, per-row sums are 1. |
| `test_mha_d_model_not_divisible_raises` | Constructor rejects `d_model=32, num_heads=3` with a `ValueError` mentioning divisibility. |
| `test_mha_input_dim_mismatch_raises` | `forward` rejects inputs whose last dim is not `d_model`. |
| `test_mha_param_count_matches_doc` | Confirms the parameter count (4096 weights + 128 biases) matches section 3's table. |
| `test_mha_cross_attention_rectangular` | Cross-attention shape: `Q_in: (2, 3, 16), K_in/V_in: (2, 5, 16)` -> output `(2, 3, 16)`, `last_weights: (2, 4, 3, 5)`. |

**Backward and structural properties (section 8):**

| Test | What it asserts |
|------|-----------------|
| `test_mha_backward_gradient_check` | Numerical gradient check on a tiny cross-attention config (`d_model=4, h=2`, `n_q=2`, `n_k=3`). Compares analytical vs central-difference finite differences for **all 11 quantities**: `Q_in, K_in, V_in` and the `W` and `b` of `W_q, W_k, W_v, W_o`. Robust check: passes if abs error < `1e-8` OR rel error < `1e-4` (handles the structurally-zero `grad_b_k` -- see section 8 sub-section). |
| `test_mha_W_k_bias_gradient_is_exactly_zero` | Asserts the structural fact: `grad_W_k.b` is identically zero regardless of inputs, because softmax row-shift invariance kills the K-bias signal. Independent of any tolerance choice. |
| `test_mha_single_head_reduces_to_sdpa` | With `h=1`, `MHA(X) == W_o(SDPA(W_q(X), W_k(X), W_v(X)))` exactly (atol `1e-12`). Sanity-checks that the split/merge head reshapes are a no-op for `h=1`. |
| `test_mha_head_permutation_invariance` | Permuting head slices in `W_q/W_k/W_v` (and matching rows of `W_o`) leaves the output unchanged. Empirically confirms heads are interchangeable -- there is no preferred "first" head. |

---

## Discussion Log

A chronological log of every conceptual exchange. Append, do not edit.

### 2026-04-25 -- Step 4 kickoff (intuition + math + first worked example)

**Decision: cover Step 4 in slices, not in one go.** User requested
"all_in_order: intuition -> math -> trace example -> doc skeleton" with no
implementation code in this turn. This doc captures all four; the code
(forward, backward, tests) is parked in TODO sections 7-9.

**Why multiple heads (intuition section 2):**
- Single-head softmax produces one mixture per query; mass is conserved, so
  attending strongly to A means not attending strongly to B.
- Linguistic / structural relationships are multi-faceted -- in our sort
  task alone a token might want to track "smaller values", "duplicates",
  "boundary distance", and "neighbors" simultaneously.
- Multi-head = a panel of `h` independent experts, each with its own
  projections. Heads specialize via gradient descent, not via hand-coding.

**Why `d_k = d_v = d_model / h` (intuition section 2, and shapes section 3):**
- Total per-head compute scales as `n^2 . d_k`, so `h` heads of dim
  `d_model / h` cost `n^2 . d_model` total -- identical to one full-width
  head. Multi-head is "free" relative to single-head; you trade one big
  attention pattern for `h` smaller ones at no extra FLOPs.
- Side effect: `h . d_v = d_model`, so `W^O` is square `d_model x d_model`
  and the concat output drops straight back into the residual stream.

**Worked example (section 5):** smallest non-trivial config -- `n=2`,
`d_model=4`, `h=2`, `d_k=2` -- with `W^Q = W^K = I_4`,
`W^V = diag(1, 1, 2, 2)`, `W^O = I_4`. Both heads ended up with the same
attention weights `[[0.670, 0.330], [0.330, 0.670]]` because `Q, K` matched
across heads, but the head outputs differed because head 1 saw values 2x
those of head 0. Demonstrates that head specialization can flow from any of
the four projections, not just `W^Q / W^K`. These numbers will become the
first unit test when we implement the layer.

**Open items deferred to next turns:**
- Section 4 (split-heads reshape) -- math is sketched; needs a careful
  index-by-index derivation and the gradient through the reshape /
  transpose.
- Section 6 (three usage modes) -- short table only; full discussion lives
  with encoder (Step 6) and decoder (Step 7).
- Sections 7-9 (forward in code, backward, tests) -- empty placeholders.

### 2026-04-25 -- Q&A: what would go wrong with `h` full-width heads?

**Question (user):** "What would go wrong if `h` heads of `d_model`?" --
i.e., if instead of the paper's `d_k = d_model / h` we ran `h` heads each
with `d_k = d_model` (full width).

**Answer (folded into section 2 above):** Three things, in order:

1. **Cost goes up `h`x.** With our toy config, total MHA parameters jump
   from 4096 to 16384 and FLOPs jump similarly. For real models this is a
   showstopper.

2. **The softmax bottleneck means width does not buy more patterns.** Each
   head emits exactly one softmax distribution per query -- that is one
   attention pattern, and it is a per-head property, not a per-`d_k`
   property. Wider `d_k` gives sharper similarity scoring but still one
   pattern. So compared to single full-width head you are paying `h`x for
   `h` patterns; compared to the paper's narrow heads you are paying `h`x
   for nothing extra in terms of pattern count.

3. **Narrow heads regularize toward specialization.** A `(d_model, d_model
   / h)` projection is a compressing projection -- it picks a subspace.
   Multiple heads cannot all pick the same subspace without wasting
   capacity, so they are pushed apart in subspace-land. Full-width heads
   have no such pressure, so head collapse (multiple heads learning the
   same thing) becomes a real risk. This is the same regularization
   intuition behind dropout, autoencoder bottlenecks, and random
   projections. Empirical work (Voita et al. 2019, "Analyzing Multi-Head
   Self-Attention") confirms narrow-head models develop more
   interpretable, specialized heads than wide-head ablations.

**Synthesis:** the paper's choice `d_k = d_v = d_model / h` is doing three
jobs at once -- (1) cost parity with single-head, (2) making `W^O` square,
(3) regularizing toward specialization via the bottleneck. It is not
arbitrary, and "make heads wider" is not a free upgrade.

**Cross-reference:** the comparison table in section 2 was extended from a
two-row "A vs B" to a three-row "A vs B vs C" as a result of this
discussion.

### 2026-04-25 -- Q&A: must `h` divide `d_model`?

**Question (user, near-verbatim):** "should the number of heads always be
`d_model / h`, or can it be an arbitrary number (say 2, 3, or 4) without any
correlation with `d_model`?"

**Answer (folded into the new "Is `d_k = d_model / h` actually required?"
subsection in section 2):**

- **Math says no, it is a convention.** Attention works for any `h`, any
  `d_k`, and any `d_v`. The only hard constraint is that `W_i^Q` and
  `W_i^K` share `d_k` (so `Q . K^T` is defined per head). `d_v` can differ
  from `d_k`. `W^O` does not have to be square -- its input dim is
  `h . d_v` and its output dim is `d_model`, so when `h . d_v != d_model` it
  is just rectangular.
- **Three reasons the convention exists:** (1) cost parity with
  single-head (`h . d_k = d_model` keeps total attention FLOPs at
  `n^2 . d_model`); (2) `W^O` becomes square; (3) the one-big-matmul-plus-
  reshape implementation trick requires `h . d_k = d_model` to keep `W^Q`
  square. The third reason is what most production libraries actually
  enforce -- PyTorch / JAX / TensorFlow assert `d_model % h == 0` rather
  than handle the rectangular case.
- **Real-world counterexamples:** Multi-Query Attention (MQA, Shazeer 2019,
  used in PaLM and Falcon) shares one KV head across `h` query heads;
  Grouped-Query Attention (GQA, Ainslie et al. 2023, used in LLaMA-2 70B)
  has `h_kv < h_q`; some sparse-attention variants use asymmetric
  `d_k != d_v`. All of these break the simple `d_k = d_model / h` rule.
- **Concrete example for `h = 3, d_model = 32`:** four working choices --
  pick `d_k = 11` (rectangular `W^O`), pick `d_k = 10` (rectangular `W^O`),
  bump `d_model` to 33 or 30, or pick a different `h`. Most code defaults
  to "pick a different `h`" simply because it is the path of least
  resistance.
- **What we do for our build:** keep the convention `d_k = d_v = d_model / h`
  (so `h = 4, d_model = 32, d_k = 8`) because it matches the paper, keeps
  the reshape trick clean, and matches 99% of real Transformer code. But
  when section 4 is filled in, we will explicitly flag that the trick (not
  the math) is what depends on `h . d_k = d_model` -- the math itself is
  more permissive.

**Cross-reference:** new subsection "Is `d_k = d_model / h` actually
required?" added to the end of section 2, with the constraint table, three
convenience reasons, MQA/GQA counterexamples, and the four-options
concrete example. A forward-looking note was also added to the section 4
placeholder so we do not forget the trick-vs-math distinction when we get
there.

### 2026-04-25 -- Section 4 + forward implementation landed

**Decision:** user picked "ready_to_implement" once the conceptual
scaffolding for sections 1-3 + 5 was solid. Implemented the forward pass
plus first tests in this turn; backward is parked for the next turn (per
the workspace rule "one step at a time").

**Section 4 filled in** (the split-heads reshape trick):
- The naive way (`h` separate `(d_model, d_k)` matmuls per role) is
  contrasted with the trick (one big `(d_model, d_model)` matmul plus a
  reshape).
- Stacking the `h` per-head matrices side-by-side along the output dim
  *is* the big matrix -- mathematical equivalence shown explicitly with
  an ASCII diagram.
- Forward split: `(..., n, d_model)` -> reshape -> `(..., n, h, d_k)` ->
  swapaxes(-2, -3) -> `(..., h, n, d_k)`. Step 1 is a no-op in memory;
  step 2 is a non-contiguous view that `numpy.matmul` handles fine.
- Forward merge: inverse pair (swapaxes then reshape).
- Why this transpose ordering: putting `h` to the left of `n` lets SDPA
  do `h` parallel attention computations without changing SDPA's code at
  all -- the head dim acts as just another batch dim.
- Trick requires `h . d_k = d_model`; math itself is more permissive
  (cross-reference to the section 2 subsection from the previous Q&A).

**Implementation choices** (covered in section 7 of the doc):
- Reuse `ScaledDotProductAttention` from Step 3 unchanged. No code
  duplication; the test `test_sdpa_shape_contracts_unbatched_and_batched`
  already validated the `(B, h, n, d_k)` leading-dim pattern that MHA
  needs.
- Use `Linear` sub-layers for the four projections rather than raw
  `np.matmul`. Pros: 3D-input handling already done by `Linear`,
  parameter management already done, easier backward via sub-layer
  composition. Cons: `Linear` always has a bias, which deviates slightly
  from the paper. Resolution: keep biases (zero-init), match PyTorch's
  default, document the choice.
- Bias zero-init means the section-5 trace test (which overrides `W` and
  leaves `b = 0`) reproduces the hand-traced numbers exactly. Verified.
- `backward()` is a stub raising `NotImplementedError` with a pointer to
  section 8.

**Tests landed (7, all green):**
- `test_mha_section5_trace` -- the headline test. Reproduces the doc's
  hand-traced numbers within `1e-3`. Confirms doc and code agree.
- `test_mha_sort_task_shapes` -- sort-task config (`B=32, n=5, d_model=32,
  h=4`) round-trips shape `(B, n, d_model)`, `last_weights` is
  `(B, h, n, n)`, per-row sums are 1.
- `test_mha_d_model_not_divisible_raises` -- constructor validation.
- `test_mha_input_dim_mismatch_raises` -- forward validation.
- `test_mha_param_count_matches_doc` -- 4096 weight params + 128 bias
  params, matching the section 3 table.
- `test_mha_cross_attention_rectangular` -- `Q_in` from decoder
  (`n_q=3`), `K_in/V_in` from encoder (`n_k=5`), output is `(2, 3, 16)`,
  weights are `(2, h, 3, 5)`. Previews the encoder-decoder cross-attention
  usage mode (section 6) without needing the encoder/decoder yet.
- `test_mha_backward_not_yet_implemented` -- stub raises with a doc
  pointer.

**Full test suite green:** 28 tests pass total (21 pre-existing + 7
new).

**Open items:**
- Backward pass + numerical gradient check (section 8) -- next turn.
- Single-head reduces-to-SDPA-plus-projections test -- with backward.
- Head-permutation invariance test -- with backward.
- Section 6 (three usage modes) -- still preview-only; will expand when
  wiring MHA into encoder (Step 6) and decoder (Step 7).
- Optional dropout on attention weights -- defer to encoder/decoder
  blocks.

### 2026-04-25 -- Section 8 + backward implementation landed

**Mental model going in:** the backward through MHA reduces almost
entirely to **sub-layer composition**. We had already done the calculus
work in Steps 1 (Linear, softmax) and 3 (SDPA). The new piece is just
threading two shape-only operations -- reshape and `np.swapaxes` -- and
the observation that **their gradient is the inverse permutation**.

**The clean result:** `backward of _split_heads = _merge_heads`, and
vice versa. Forward and backward swap roles. Two helper methods, four
uses -- forward calls each once, backward calls each once. No new
calculus.

**Implementation:** ~10 lines in `MultiHeadAttention.backward`. Composes
`W_o.backward -> _split_heads -> SDPA.backward -> _merge_heads (x3) ->
W_q/k/v.backward`. Returns a 3-tuple `(grad_Q_in, grad_K_in, grad_V_in)`,
matching SDPA's convention. Caller is responsible for the
self-attention sum `grad_X = grad_Q_in + grad_K_in + grad_V_in` (we do
NOT do this in MHA because cross-attention has different sources for
`Q_in` vs `K_in == V_in`).

**Discovery during gradient check (this is worth memorizing):**
the analytical gradient of `W_k.b` was at `~1e-17` (pure FP noise) while
the numerical gradient was `~1e-11`. Both essentially zero -- but the
relative-error metric blew up to `0.22%`. This was not a bug; it is a
**structural property of attention with key biases**:

> Adding `b_k` to every row of `K` shifts every score in row `i` of the
> score matrix by the same constant `(Q b_k)[i]`, depending on the query
> index `i` but not the key index `k`. Softmax is invariant to per-row
> constant shifts (the same property we exploit for numerical stability
> by subtracting the row max). Therefore `b_k` cannot affect attention
> weights, cannot affect the output, and has zero gradient -- regardless
> of input.

This is the "key bias is structurally dead weight" result. It applies
ONLY to `b_k` -- the same argument fails for `b_q` (its shift is
per-key, which softmax is sensitive to) and for `b_v` (V participates
*after* softmax, so any shift propagates). Most published Transformer
implementations omit projection biases entirely (the original paper
does); PyTorch retains them only for code uniformity, and so do we.

**How we handle this in tests:**
1. The general gradient check uses a robust comparator
   `(abs_err < 1e-8) | (rel_err < 1e-4)` -- so the structurally-zero
   case passes via the absolute branch.
2. A dedicated test `test_mha_W_k_bias_gradient_is_exactly_zero`
   asserts the property directly with `atol=1e-12`, so the structural
   fact is preserved even if tolerances elsewhere change.

**Three deferred tests landed:**
- `test_mha_backward_gradient_check` -- numerical check for all 11
  quantities (3 inputs + 4 weights + 4 biases). Cross-attention shapes
  to keep paths distinct.
- `test_mha_single_head_reduces_to_sdpa` -- with `h=1`, MHA is exactly
  `W_o(SDPA(W_q(X), W_k(X), W_v(X)))` (atol `1e-12`).
- `test_mha_head_permutation_invariance` -- swap head 0 and 1 in the
  column-slices of `W_q/W_k/W_v` (and biases) AND in the row-slices of
  `W_o`; output is unchanged. Demonstrates the head-permutation symmetry
  in parameter space (`h!` equivalent reflections of every minimum).

**Full test suite green:** 31 tests pass total (21 pre-existing + 10
MHA).

**Open items:**
- Section 6 (three usage modes) -- still preview-only; expand in Step 6
  (encoder) and Step 7 (decoder).
- Optional dropout on attention weights -- defer to encoder/decoder
  blocks.
- Step 4 itself is now **complete**: forward + backward + 4 forward tests
  + 4 backward tests + 2 structural-property tests, doc fully written.
