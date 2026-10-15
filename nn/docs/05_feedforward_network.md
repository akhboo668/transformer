# Step 5: Position-wise Feed-Forward Network

> **Status**: Complete (forward + backward + 6 tests landed)
> **Paper section**: Section 3.3, Equation 2
> **Depends on**: Step 1 (`Linear`, `ReLU`)

[< Multi-Head Attention](04_multi_head_attention.md) | [Overview](00_overview.md) | [Next: Encoder >](06_encoder.md)

---

## 1. What we build

A two-layer fully-connected neural network applied **independently and
identically to every position** in the sequence.

Looking at the encoder block, the FFN sits **after attention**:

```
        +----------------+
        |   Attention    |   <- mixes information across positions
        +----------------+
                |
                v
        +----------------+
        |     FFN        |   <- mixes information within each position
        +----------------+
                |
                v
            output
```

The whole thing is **three operations** we already have from Step 1:

```
Linear(d_model -> d_ff)  ->  ReLU  ->  Linear(d_ff -> d_model)
```

So the implementation will be ~10 lines. The interesting part is the
**why**, not the how -- this doc is mostly about that.

---

## 2. Why FFN is needed: the nonlinearity argument

This is the deepest question in this step, and arguably one of the most
important "why" questions in the whole Transformer. Read this once
slowly even if it feels obvious -- it changes how you read the rest of
the architecture.

### The setup

Consider what attention actually computes. For a single self-attention
head with input `X` (shape `(n, d_model)`):

```
Q, K, V    = X W^Q, X W^K, X W^V                # linear in X
scores     = Q K^T / sqrt(d_k)                  # bilinear in X
weights    = softmax(scores)                    # depends on X (nonlinear)
output[i]  = sum_j weights[i, j] . V[j]         # weighted sum of V's rows
final      = output @ W^O                       # linear
```

The attention **weights** depend nonlinearly on `X` (via softmax). But
once the weights are computed, the **output** is a linear combination of
the rows of `V`, which is itself a linear function of `X`. So the layer
output is, schematically:

```
Attention(X) = (softmax-weighted-mix-of-X)
             = M(X) . X . W^V . W^O
```

where `M(X)` is a data-dependent matrix of attention weights. The key
observation: **the output lives in the column space of `X . W^V . W^O`**.
You cannot escape the linear span of the (projected) input via attention
alone, no matter how clever the weights are.

### Why this is a problem: stacking attention is (almost) just one attention

Now consider stacking two attention layers with no FFN between them:

```
Y = Attention_1(X)            # in span of X projections
Z = Attention_2(Y)            # in span of Y projections, which are in span of X projections
```

The composition `Z = Attention_2(Attention_1(X))` is still essentially a
"weighted-mean-of-linear-projections-of-X" operation. The weights are
more complicated (they depend on `Y` which depends on `X`), but the
**class of functions you can express has not really expanded**. You cannot
compute things like:

- "Is feature A larger than feature B?" (requires a comparison /
  threshold -- nonlinear)
- "XOR of two binary features" (requires a non-affine boundary)
- "Project feature A onto the unit cube" (requires clamping)

These all need a **non-affine, element-wise** operation. Attention does
not provide one. The softmax is element-wise nonlinear *within* the
weight computation, but the weights then act linearly on `V`.

### The role of FFN

The FFN is the **only** source of element-wise nonlinearity in the
Transformer:

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
                ^^^^^^^^^^^^
                here -- ReLU acts element-wise on the hidden state
```

That `max(0, .)` is what gives the Transformer its **universal
approximation** property (in the limit of enough parameters). Without
FFN, a stack of N attention layers cannot approximate arbitrary
functions of the input. With FFN, two attention layers + FFN already
can (in principle, given enough width).

**One-line summary**: attention moves information *between* positions;
FFN does **the actual computation** at each position. They are not
redundant -- they are complementary.

### A concrete check

Try removing the FFNs from a Transformer and training it. The
expressivity collapse is dramatic -- the model essentially behaves like
a single-layer attention network no matter how deep you stack. Several
papers have explored "attention-only" Transformers (e.g. linear
attention models) and they consistently underperform until something
nonlinear (often a gating mechanism, sometimes literally an FFN) is
added back in.

This is also why there's been so much research on the **FFN component
specifically**: it accounts for ~2/3 of the parameters in a typical
Transformer (see section 5), and it's where most of the model's
"knowledge" lives (see section 6).

---

## 3. The math

### Equation (paper section 3.3, equation 2)

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

### Shapes

The FFN is applied **per position**. Conceptually `x` is a single
`(d_model,)` vector. In practice we apply it batched over `(batch_size,
seq_len, d_model)`, but the math is the same -- `Linear` already
broadcasts over batch and sequence dims (Step 1).

| Symbol | Shape | Role |
|--------|-------|------|
| `x`    | `(d_model,)` per position | input vector |
| `W_1`  | `(d_model, d_ff)` | first projection: expand to wider hidden |
| `b_1`  | `(d_ff,)` | first bias |
| hidden | `(d_ff,)` | wider intermediate, after ReLU |
| `W_2`  | `(d_ff, d_model)` | second projection: contract back |
| `b_2`  | `(d_model,)` | second bias |
| output | `(d_model,)` | same shape as input |

The dimension expansion-contraction pattern (`d_model -> d_ff -> d_model`
with `d_ff > d_model`) is the **bottleneck-in-reverse**: the network is
*wider* in the middle, not narrower. This is sometimes called an
**inverted bottleneck**.

### Parameter count

For our build (`d_model = 32`, `d_ff = 64`):

| Param  | Shape         | Count |
|--------|---------------|-------|
| `W_1`  | `(32, 64)`    | 2048  |
| `b_1`  | `(64,)`       | 64    |
| `W_2`  | `(64, 32)`    | 2048  |
| `b_2`  | `(32,)`       | 32    |
| **Total**         |    | **4192** |

For comparison: a single `MultiHeadAttention(d_model=32, num_heads=4)`
has 4096 weights + 128 biases = 4224 params (from Step 4's table). So
**FFN and MHA have roughly the same parameter count per encoder block**
in our small config.

In the paper's full-size config (`d_model = 512`, `d_ff = 2048`):

- FFN: `512 . 2048 . 2 + 2048 + 512 = 2,099,712 ~= 2.1M` params per FFN.
- MHA: `512 . 512 . 4 + 512 . 4 = 1,050,624 ~= 1.05M` params per MHA.

So in the paper's config the **FFN has ~2x as many parameters as the
attention layer** in the same block. Across the encoder, FFN dominates.

---

## 4. "Position-wise": the decoupled mixer

### What the term means

"Position-wise" is the paper's way of saying:

> The same `(W_1, b_1, W_2, b_2)` are applied independently to **every
> position** in the sequence, with no mixing between positions.

In NumPy, `Linear.forward` already does this for free: `x @ W` broadcasts
over the leading `(batch, seq_len)` dims. There is no special code; it
falls out of how matrix multiplication broadcasts.

Two clarifications:

1. **Same weights across positions, within a layer.** The position 0
   and position 7 of the same layer use the *exact same* `W_1, b_1,
   W_2, b_2`. Position is irrelevant to the FFN's computation -- it
   sees a `d_model`-dim vector and produces a `d_model`-dim vector,
   without knowing or caring where in the sequence that vector came
   from.
2. **Different weights from layer to layer.** Encoder layer 1's FFN has
   different parameters from encoder layer 2's FFN. So the network can
   apply different "thinking steps" at different depths.

### The decoupled-mixer mental model

This is the most useful way to think about Transformer blocks once you
have both attention and FFN in your head:

| Component     | Mixes across | Mixes across |
|---------------|--------------|--------------|
|               | **positions** | **features (d_model)** |
| Attention     | Yes          | Limited (only via `W^V`, `W^O`) |
| FFN           | **No**       | **Yes** (the whole point) |

Attention is good at moving information **between tokens** but is
relatively weak at within-token feature mixing -- the `V`-side projection
is a bottleneck (often `d_v < d_model`). FFN is good at **within-token
feature mixing** but cannot move information between tokens at all.
Together they form a complete mixer.

This was made explicit by **MLP-Mixer (Tolstikhin et al. 2021)**, which
showed you can replace attention entirely with a "token-mixing MLP" and
get comparable results -- evidence that the FFN is doing more of the
heavy lifting than people initially thought, and that the
attention-FFN split is closer to a two-axis convolution than a
fundamentally bespoke design.

**Mnemonic:**

- **Attention = communication phase.** "What did the others say?"
- **FFN = computation phase.** "Now let me think about it on my own."

---

## 5. Design choices

### Why `d_ff = 4 . d_model`?

The paper picked this ratio without justification. It has stuck around
because empirically:

1. **Capacity sweet spot.** Less than 4x (e.g., 1x or 2x) noticeably
   reduces model quality; more than 4x mostly burns compute without
   proportional gains.
2. **Most of the model's "knowledge" lives in FFN.** With `d_ff = 4 .
   d_model`, the FFN parameters are `2 . d_model . d_ff = 8 . d_model^2`
   per layer, while attention is `4 . d_model^2` (the four `d_model x
   d_model` projections). So **FFN is already 2/3 of the encoder
   block**. Going wider (8x) would push that to 4/5 -- diminishing
   returns vs. the cost.
3. **Round number tradition.** Going from 4x to 4.7x or 3.5x is
   unprincipled; 4x is a clean choice that the field hasn't seriously
   challenged.

**Our build deviates: `d_ff = 64 = 2 . d_model`.** This is documented in
[`00_overview.md`](00_overview.md) as a deliberate trade-off -- our toy
config is small enough that 4x would inflate FFN parameters
disproportionately to the rest, and 2x is sufficient for the sort task
we use as a regression test in Step 8.

**Modern variants** (Llama, Mistral, PaLM) use `d_ff ~= 8/3 . d_model`,
which sounds smaller but actually keeps total parameter count similar
because they use **GLU-style FFN** with three matrices instead of two
(see "ReLU vs modern alternatives" below).

### ReLU vs modern alternatives

| Activation | Year   | Used by                          | Formula                          |
|------------|--------|----------------------------------|----------------------------------|
| ReLU       | 2010   | "Attention Is All You Need" (2017), us | `max(0, x)`                |
| GELU       | 2016   | BERT, GPT-2, ViT, T5             | `x . Phi(x)` (Phi = standard normal CDF) |
| Swish/SiLU | 2017   | EfficientNet, some LLMs          | `x . sigmoid(x)`                 |
| SwiGLU     | 2020   | PaLM, Llama, Mistral             | `Linear_3((Swish(Linear_1(x)) * Linear_2(x)))` |
| GeGLU      | 2020   | Newer T5 variants                | Same as SwiGLU but GELU instead of Swish |

**Why we use ReLU**: matches the paper exactly, simpler backward, no
new code needed (already implemented in Step 1). The differences across
activations are real but small at our toy scale -- they matter for very
large models.

**The GLU family is interesting**: it splits the first projection into
two parallel projections and **gates** one with the other:

```
GLU-FFN(x) = (Swish(x W_a) * (x W_b)) W_3
```

Three matrices instead of two, so to keep the parameter count fixed
they typically use `d_ff = 8/3 . d_model`. The element-wise product
gives the model a multiplicative interaction it couldn't easily get
from a flat ReLU/GELU. This is the FFN variant in modern open-weights
LLMs.

We may revisit this when extending the build, but it is **out of scope
for the paper-faithful Step 5**.

### The "1x1 conv equivalence"

The paper notes that the FFN is mathematically identical to **two 1x1
convolutions** applied along the sequence dimension.

To see this, take the sequence as a 1D "image" of shape `(seq_len,
d_model)` -- with `seq_len` as the spatial axis and `d_model` as the
channel axis. A 1D convolution with kernel size 1 looks at one position
at a time and applies a learned linear mix across channels:

```
Conv1x1(x)[pos, c_out] = sum_{c_in} W[c_out, c_in] . x[pos, c_in]
```

That is exactly `Linear(d_model -> d_ff)` applied at each position
independently. The FFN's two matmuls thus correspond to two 1x1 convs
with a ReLU between.

**Why this matters**: it's a conceptual bridge to CNN literature, where
1x1 convs are well understood as **pointwise channel mixers** (Inception
modules, ResNets, MobileNet). The FFN is doing the same job in a
sequence model. It does not change the math we write -- just helps unify
the mental model.

For our NumPy build we never call it a conv; we just use `Linear`. But
when reading transformer code in PyTorch, you may occasionally see
`nn.Conv1d(d_model, d_ff, kernel_size=1)` instead of `nn.Linear(d_model,
d_ff)`. They are the same operation, with slightly different memory
layouts.

---

## 6. The key-value memory interpretation

This is one of the most beautiful ways to think about FFN, due to **Geva
et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories"**.

### The framing

Look at the FFN forward again:

```
hidden  = ReLU(x W_1 + b_1)        # hidden[j] = how strongly key j matches x
output  = hidden W_2 + b_2          # output = sum over j of hidden[j] . W_2[j, :]
```

Now read this as a **soft lookup table**:

- **`W_1`'s columns are keys.** Each of the `d_ff` columns of `W_1` is a
  pattern (a `d_model`-dim vector) that the FFN looks for in its input.
- **`hidden[j] = ReLU(x . W_1[:, j] + b_1[j])`** is the **match score**:
  how strongly the j-th key matches the input. Negative matches are
  zeroed out by ReLU.
- **`W_2`'s rows are values.** Each of the `d_ff` rows of `W_2` is a
  vector to add to the output when its corresponding key matches.
- **The output is a sum of values, weighted by match strengths.**

In equation form:

```
FFN(x) = sum over j in {0, ..., d_ff-1}  of  ReLU(x . k_j + b_1[j]) . v_j  +  b_2
                                              ^^^^^^^^^^^^^^^^^^^   ^^^
                                              key match score       value
```

where `k_j = W_1[:, j]` and `v_j = W_2[j, :]`.

### Why this is striking

It's the **same structure as attention**:

| Component | "Soft lookup" form                              |
|-----------|-------------------------------------------------|
| Attention | `output = sum_j softmax_j(Q . K^T) . V_j`       |
| FFN       | `output = sum_j ReLU(x . k_j + b_j) . v_j`      |

Two big differences:

1. **Attention's keys/values are computed from data** (the keys are
   projections of the actual tokens in the sequence). **FFN's
   keys/values are learned parameters** -- the same set every time.
2. **Attention normalizes match scores via softmax** (so the result is
   a convex combination of values). **FFN does not normalize** (it just
   sums weighted values; ReLU only zeroes negative matches).

So the FFN is a **trained lookup table**: a fixed bank of `d_ff`
patterns the model looks for, and a fixed value to add for each match.
Attention is a **dynamic lookup table** built fresh from the input
sequence each time.

### Interpretability evidence

Geva et al. examined trained Transformers and found that:

- Many keys correspond to **interpretable input patterns** -- e.g.
  "input ends in a specific suffix", "input mentions a specific named
  entity", "input is in a particular topic".
- The corresponding values often shift the output **toward specific
  tokens** -- e.g. when key "ends in -ing" matches, the value pushes
  the next-token distribution toward verbs.
- The FFN is therefore where most of the **factual / lexical
  knowledge** of LLMs lives. Editing FFN weights ("ROME", "MEMIT") can
  surgically modify factual associations like "the Eiffel Tower is in
  ___".

For our build the takeaway is simply: **FFN is not an opaque
nonlinearity -- it's a soft lookup table**. Section 7 below makes this
concrete.

---

## 7. Worked numerical example

A two-token, two-feature example we can trace by hand. Configuration:

- `d_model = 2`, `d_ff = 4`
- Input: two tokens, `x_1 = [1, -1]` and `x_2 = [-1, 2]`
- Biases all zero (for clarity)

### Weights

```
       key 0   key 1   key 2   key 3
W_1 = [[ 1,     0,     -1,      1   ],     <- d_model row 0
       [ 0,     1,      1,     -1   ]]     <- d_model row 1
       
       (so key j = W_1[:, j], a d_model=2 vector each)

key_0 = [ 1,  0]    "feature 0 is positive"
key_1 = [ 0,  1]    "feature 1 is positive"
key_2 = [-1,  1]    "feature 0 negative AND feature 1 positive"
key_3 = [ 1, -1]    "feature 0 positive AND feature 1 negative"

       value -> output dim
W_2 = [[ 1,  0],     <- value 0 = [ 1,  0]
       [ 0,  1],     <- value 1 = [ 0,  1]
       [-1,  0],     <- value 2 = [-1,  0]
       [ 0, -1]]     <- value 3 = [ 0, -1]
```

### Forward for token 1: `x_1 = [1, -1]`

**Match scores** (`x_1 . W_1`):

```
match[0] = [1, -1] . [1,  0] =  1
match[1] = [1, -1] . [0,  1] = -1
match[2] = [1, -1] . [-1, 1] = -2
match[3] = [1, -1] . [1, -1] =  2
```

**ReLU** (zero out negative matches):

```
hidden = [1, 0, 0, 2]      <- only keys 0 and 3 fired
```

**Sum of weighted values**:

```
output = 1 . [1, 0]  +  0 . [0, 1]  +  0 . [-1, 0]  +  2 . [0, -1]
       = [1, 0]  +  [0, -2]
       = [1, -2]
```

Reading this as a story:

- Key 0 ("feature 0 is positive") fired with strength 1; it added value
  0 = `[1, 0]` to the output.
- Key 3 ("feature 0 positive AND feature 1 negative") fired with
  strength 2; it added `2 . [0, -1] = [0, -2]` to the output.
- Result: `[1, -2]`.

### Forward for token 2: `x_2 = [-1, 2]`

**Match scores**:

```
match[0] = [-1, 2] . [1,  0] = -1
match[1] = [-1, 2] . [0,  1] =  2
match[2] = [-1, 2] . [-1, 1] =  3
match[3] = [-1, 2] . [1, -1] = -3
```

**ReLU**:

```
hidden = [0, 2, 3, 0]      <- keys 1 and 2 fired
```

**Output**:

```
output = 0 . [1, 0]  +  2 . [0, 1]  +  3 . [-1, 0]  +  0 . [0, -1]
       = [0, 2]  +  [-3, 0]
       = [-3, 2]
```

### Things to notice from this example

1. **Position-wise**: `x_1` and `x_2` were processed completely
   independently. Same weights, no interaction. The sequence has no
   role in the FFN -- only the per-position vectors matter.
2. **Different keys fire for different inputs**. `x_1` activated keys
   {0, 3}; `x_2` activated keys {1, 2}. The FFN is acting as a *soft
   classifier*: it routes different inputs through different
   "knowledge slots".
3. **Sparsity from ReLU**. Half the hidden activations are zero at each
   token. This is typical -- empirically, real FFNs in trained
   Transformers also have ~50% sparsity.
4. **No softmax**. Unlike attention, the matches are not normalized.
   Strong matches dominate; weak matches simply do not contribute.

This example will be turned into `test_ffn_section7_trace` in slice 2.

---

## 8. Forward pass in code

The implementation lives in
[`numpy-transformer/src/numpy_transformer/layers.py`](../numpy-transformer/src/numpy_transformer/layers.py)
as `class FeedForward(Layer)`. ~30 lines including docstrings -- pure
composition of pieces from Step 1.

### Structure

```python
class FeedForward(Layer):
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        self.W_1 = Linear(d_model, d_ff)
        self._relu = ReLU()
        self.W_2 = Linear(d_ff, d_model)
        self._cache = {}

    def forward(self, x):
        # validate x.shape[-1] == d_model
        h_pre = self.W_1.forward(x)
        h     = self._relu.forward(h_pre)
        return  self.W_2.forward(h)
```

### Design choices

- **Reuse `Linear` from Step 1.** Already handles the 3D-input
  `(batch, seq, d_model)` reshape trick under the hood (see Step 1
  doc). Biases are zero-init -- which means they have no effect on the
  first forward pass and the `test_ffn_section7_trace` reproduces the
  doc's hand-traced numbers exactly.
- **Reuse `ReLU` from Step 1.** It already caches its mask for the
  backward pass; we don't need to do anything special.
- **No new caching at the FFN level.** `_cache = {}` is kept as a stub
  for symmetry with other layers, but `Linear` and `ReLU` carry all the
  state we need. Same lesson as MHA section 8: when sub-layers manage
  their own caches, the parent's cache is empty.
- **Bias on both `Linear`s.** The paper's equation 2 has biases. Our
  `Linear` constructor always builds a bias, so we get them by default.
- **`params_and_grads()` aggregates** the two `Linear`s' param tuples
  with `W_1.` / `W_2.` prefixes, matching the convention used by `MHA`
  and `LayerNorm` for namespaced sub-layer parameters.

### Public surface

| Name | Type | Purpose |
|------|------|---------|
| `FeedForward(d_model, d_ff)` | constructor | Builds two `Linear`s and one `ReLU`. |
| `.forward(x)` | method | Returns `(..., d_model)`-shape output. |
| `.backward(grad_output)` | method | Returns `grad_x`; populates `W_1.grad_W`, `W_1.grad_b`, `W_2.grad_W`, `W_2.grad_b` as side effects. |
| `.params_and_grads()` | method | Concatenates the two `Linear`s' params, prefixed `W_1.` / `W_2.`. |
| `.W_1`, `.W_2` | attribute | Underlying `Linear` sub-layers (override `.W` and `.b` in tests to inject specific weights). |

---

## 9. Backward pass

Like `MultiHeadAttention.backward` (Step 4), the FFN backward is **pure
sub-layer composition** -- nothing new derivative-wise. Forward chain is
`Linear -> ReLU -> Linear`; backward chain is the exact reverse:

```
Forward:    x ---W_1---> h_pre ---ReLU---> h ---W_2---> output

Backward:   grad_x <-W_1- grad_h_pre <-ReLU- grad_h <-W_2- grad_output
```

In code:

```python
def backward(self, grad_output):
    grad_h     = self.W_2.backward(grad_output)
    grad_h_pre = self._relu.backward(grad_h)
    grad_x     = self.W_1.backward(grad_h_pre)
    return grad_x
```

Three observations:

1. **Each `Linear.backward` does double duty.** It returns `grad_input`
   to thread back through the chain, AND populates that sub-layer's
   `grad_W` and `grad_b` as a side effect. Parameter gradients fall out
   for free.
2. **`ReLU.backward` is just multiplication by the activation mask.**
   `grad_h_pre[i, j] = grad_h[i, j] . (1 if h_pre[i, j] > 0 else 0)`.
   This is what introduces the asymmetry between forward and backward
   in this layer -- it's the only place where information about the
   forward state (the sign of `h_pre`) gates the gradient flow.
3. **No new calculus.** All derivatives were derived in Step 1 (Linear,
   ReLU). This step is bookkeeping.

A subtle gotcha: at exactly `h_pre = 0`, ReLU's gradient is undefined
(the subgradient is anywhere in `[0, 1]`). Our implementation picks
`(h_pre > 0).astype(...)` as the mask, treating zero as a "not active"
point. With random initialization, the probability of hitting exactly
zero is essentially zero, so this never matters in practice. The
numerical gradient check would catch the issue if a perturbation of
`eps = 1e-5` happened to straddle zero -- which would manifest as a
spike in absolute error for one parameter -- but in our test seed it
does not.

---

## 10. Sanity checks and tests (landed)

Six tests in
[`tests/test_layers.py`](../numpy-transformer/tests/test_layers.py),
all green.

| Test | What it asserts |
|------|-----------------|
| `test_ffn_section7_trace` | Reproduces the hand-traced numbers from section 7 (`d_model=2, d_ff=4`, two tokens, biases zero, interpretable keys/values) exactly to `1e-12`. Doc and code agree. |
| `test_ffn_position_wise_independence` | For `d_model=8, d_ff=16`, the batched forward of `(2, 5, 8)` matches the stack of 10 single-token forwards exactly to `1e-12`. Empirical statement of "no mixing across positions" (section 4). |
| `test_ffn_input_dim_mismatch_raises` | `forward` rejects inputs whose last dim is not `d_model` with a `ValueError` mentioning `d_model`. |
| `test_ffn_param_count_matches_doc` | Confirms the parameter count (4096 weights + 96 biases = 4192 total) matches section 3's table for `d_model=32, d_ff=64`. |
| `test_ffn_backward_gradient_check` | Numerical gradient check for **all 5 quantities** (`x, W_1, b_1, W_2, b_2`) on `d_model=4, d_ff=8`. Robust comparator: `(abs_err < 1e-8) | (rel_err < 1e-4)`. |
| `test_ffn_relu_kills_all_hidden` | With weights chosen so every hidden pre-activation is negative, ReLU zeros the entire hidden state and `output == b_2` exactly to `1e-12`. Confirms ReLU is actually firing (not silently passing values through). |

**Full test suite green:** 37 tests pass total (31 pre-existing + 6
new FFN).

---

## Discussion Log

A chronological log of every conceptual exchange. Append, do not edit.

### 2026-04-25 -- Step 5 kickoff (slice 1: conceptual exploration only)

**Context.** Stepping into FFN immediately after closing out Step 4
(MHA backward + structural-property tests). User picked the "slice 1 =
conceptual + doc, no code" path so we could properly chew on the WHY
questions before building what is otherwise a 10-line layer.

**Five questions in scope for this slice:**

1. What does FFN do that attention can't?
2. Why `d_ff = 4 . d_model`?
3. What does "position-wise" mean and why does it matter?
4. Why ReLU and not GELU/SwiGLU?
5. The 1x1 conv equivalence -- what is it actually saying?

**Headline finding (section 2): FFN is the only source of element-wise
nonlinearity in the Transformer.** Attention's output is a linear
function of `V` (modulated by softmax-derived weights). Stacking
attention layers without FFN does *not* meaningfully expand the class
of representable functions -- you stay in the linear span of (projected)
inputs. The ReLU inside FFN is what gives the model its universal
approximation property. Removing FFN collapses expressivity dramatically;
this matches empirical observations from "attention-only" models.

**Decoupled mixer view (section 4):** the cleanest way to think about an
encoder block. Attention mixes across positions but is weak at
within-token feature mixing (V-projection bottleneck). FFN mixes across
features within each position but cannot move information between
positions. Together, complete mixer. MLP-Mixer (2021) made this
explicit by showing you can swap attention out for a "token-mixing MLP"
and get comparable results.

**The `4x` ratio (section 5):**

- It's a balance: less is too little capacity, more is diminishing
  returns.
- With 4x, FFN already accounts for **2/3 of encoder block parameters**.
  (`8 d_model^2` for FFN vs `4 d_model^2` for attention.) Going wider
  pushes that to dominate even more.
- Modern LLMs use `8/3 ~= 2.67x` only because they switched to GLU-style
  FFN with **three** matrices, which restores total parameter count.

**Our build deviation:** `d_ff = 2 . d_model` (not 4x), already
documented in `00_overview.md`. The toy config doesn't need the extra
capacity for the sort task; 2x keeps the FFN footprint reasonable.

**Activation choices (section 5):** ReLU (2017 default, paper, us) ->
GELU (BERT, GPT-2 era) -> SwiGLU/GeGLU (Llama, PaLM, Mistral). We stick
with ReLU. Different activations matter at scale; at our toy scale they
are roughly interchangeable.

**1x1 conv equivalence (section 5):** if the sequence is a 1D image
with positions as pixels and features as channels, then "Linear applied
to each position independently" is exactly a 1x1 conv. Conceptual bridge
to CNN literature; doesn't change anything we write. PyTorch idiom you
may see: `nn.Conv1d(d_model, d_ff, kernel_size=1)` in place of
`nn.Linear(d_model, d_ff)`.

**Beautiful interpretation (section 6, Geva et al. 2021): FFN is a
soft key-value memory.** `W_1`'s columns are keys (patterns the FFN
looks for), `W_2`'s rows are values (vectors to add when keys match).
ReLU(x . k_j + b_j) is the match score; output is sum of values
weighted by match scores. **Same structure as attention, but with
learned (static) keys/values instead of data-derived ones, and no
softmax normalization.** This framing motivates the FFN as the model's
"long-term memory": much of an LLM's factual knowledge can be located
to specific FFN layers and even specific neurons.

**Worked example (section 7):** `d_model=2, d_ff=4`, two tokens
`[1, -1]` and `[-1, 2]`, traced by hand all the way through. Outputs:
`[1, -2]` and `[-3, 2]`. Annotated with the key-value framing -- which
keys "fired" for which tokens, what each value contribution looked
like. Will be a unit test in slice 2.

**Open items going into slice 2:**

- Implement `FeedForward(d_model, d_ff)` (forward + backward together,
  since the layer is small).
- Tests: section 7 trace, position-wise independence, parameter count,
  backward gradient check, ReLU-firing sanity.
- Update `STATUS.md` row for Step 5 to **Complete** once tests pass.
- No deviations from paper section 3.3 except `d_ff = 2 . d_model`
  (already pre-existing convention from `00_overview.md`).

### 2026-04-25 -- Slice 2: implementation + tests landed

**Implementation:** ~30 lines for `FeedForward` in `layers.py` --
constructor builds two `Linear`s and one `ReLU`, forward chains them,
backward chains them in reverse, `params_and_grads` aggregates with
`W_1.` / `W_2.` prefixes. No new calculus, no new caching at the FFN
level (sub-layers carry their own state -- same lesson as MHA section 8).

**Six tests landed (all green, 37 total in test suite):**
- `test_ffn_section7_trace` -- the headline test, reproduces the doc's
  hand-traced numbers to `1e-12`. Confirms doc + code agree.
- `test_ffn_position_wise_independence` -- batched `(2, 5, 8)` forward
  exactly equals the stack of 10 single-token forwards. Empirical
  statement of "no mixing across positions" from section 4.
- `test_ffn_input_dim_mismatch_raises` -- shape validation.
- `test_ffn_param_count_matches_doc` -- 4096 weights + 96 biases
  = 4192 total for `d_model=32, d_ff=64`.
- `test_ffn_backward_gradient_check` -- all 5 quantities (`x` and the
  4 parameters) numerically match analytical gradients. Robust
  `(abs_err | rel_err)` comparator borrowed from MHA's gradient check.
- `test_ffn_relu_kills_all_hidden` -- with weights forced negative
  (all-`-1`s in `W_1` and `b_1`), ReLU kills the entire hidden state
  and the output collapses to `b_2`. Confirms ReLU is firing rather
  than silently passing values through.

**Notes for future:**

- We saw none of the asymmetric / structurally-zero gradient surprises
  that came up with MHA (no equivalent of softmax row-shift invariance
  here). FFN gradients are well-behaved across the board.
- ReLU-at-zero gotcha (subgradient anywhere in `[0, 1]`): noted in
  section 9. Our implementation picks "zero is not active" via
  `(x > 0).astype(...)`. A finite-difference perturbation that happens
  to straddle zero would produce a spike in absolute error -- the test
  seed avoids this; if we ever change seeds and see a single-element
  spike in the gradient check, this is the first place to look.
- `d_ff = 2 . d_model` (vs paper's 4x) was the only deviation; already
  documented in `00_overview.md` and section 5.

**Step 5 is now Complete.** All four primitives needed for the encoder
block exist as standalone, tested layers:
- `MultiHeadAttention` (Step 4)
- `FeedForward` (Step 5, this step)
- `LayerNorm` (Step 1)
- residual connections (will be `x + sublayer(x)` -- no new layer
  needed, just composition in Step 6)

Next: **Step 6 -- Encoder Layer** -- assemble these into the encoder
block and stack `N=2` of them.
