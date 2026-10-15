# Step 6: Encoder

> **Status**: Complete
> **Paper section**: Section 3.1 (Encoder), Section 3.4 (Embeddings), Section 5.4 (Dropout)
> **Depends on**: Steps 1, 2, 4, 5 (LayerNorm, Positional Encoding, Multi-Head Attention, FFN)

[< FFN](05_feedforward_network.md) | [Overview](00_overview.md) | [Next: Decoder >](07_decoder.md)

---

## 1. What we build

The encoder consumes the full input sequence and produces, for every input
position, a single `d_model`-dimensional vector that encodes "this token in
the context of the whole sequence". The decoder later attends into this
output to produce its own predictions.

Two object levels:

- **`EncoderLayer`** -- one repeating block. Internally: a self-attention
  sub-layer, a feed-forward sub-layer, each wrapped in a residual
  connection and a LayerNorm.
- **`Encoder`** -- the full input pipeline. Embedding -> scale by
  `sqrt(d_model)` -> add positional encoding -> dropout -> stack of `N`
  `EncoderLayer`s.

### One layer at a glance (Post-LN, paper-faithful)

```
x_in
 |
 +---------> MultiHeadSelfAttention -> Dropout -+
 |                                              |
 +----------------- + (residual) ---------------+
                                                |
                              LayerNorm <-------+
                                  |
                                  v
                            ----------------
 +---------------- (the same vector enters the FFN sub-layer) -----+
                                                                   |
                                                                   v
 x_mid                                                          x_mid
   |                                                              |
   +--------> FeedForward -> Dropout --+                          |
   |                                   |                          |
   +------------- + (residual) --------+                          |
                                       |                          |
                       LayerNorm <-----+                          |
                           |                                      |
                           v                                      |
                        x_out                                     |
```

In one line of math, the **paper formulation (Post-LN)** is:

```
x_mid = LayerNorm(x_in  + Dropout(MHA(x_in)))
x_out = LayerNorm(x_mid + Dropout(FFN(x_mid)))
```

We will see in section 5 why modern code mostly uses a different placement
of `LayerNorm` (Pre-LN), and how that choice trades off correctness vs
training stability.

### Full encoder at a glance

```
token_ids  --[Embedding]-->  E       (B, n, d_model)
E          --[* sqrt(d_model)]-->  E_scaled
E_scaled   --[+ PE]-->        E_pos  (broadcast PE over batch)
E_pos      --[Dropout]-->     x_0    (paper section 5.4)
x_0        --[EncoderLayer x N]-->  x_N
x_N is the encoder output, fed to every cross-attention block in the decoder.
```

This step builds **everything in that diagram except the decoder**. The
decoder's cross-attention is Step 7.

---

## 2. Sizing the encoder: rules and conventions

This section answers a question that came up between Steps 5 and 6:
**what determines the size of the encoder?** It is captured here
(rather than the Discussion Log alone) because the answer is lasting
theory that anyone reading this doc should see.

### The single hard mathematical rule

```
d_model % num_heads == 0
```

This is the only encoder dimension constraint enforced by mathematics
(specifically, by the split-heads reshape trick from Step 4 section 4 --
`d_model` must be divisible into `num_heads` equal-sized chunks of width
`d_k = d_model / num_heads`). Our `MultiHeadAttention.__init__`
fail-fasts on violations. The math itself is technically more
permissive -- ragged-width heads work in principle -- but no mainstream
implementation does this, and neither do we.

**Everything else about encoder shape is empirical convention or
scaling-law guidance, not proof.**

### The four orthogonal axes that size an encoder

| Axis | Symbol | Paper (base) | Our build | What it controls |
|------|--------|--------------|-----------|------------------|
| **Depth** | `N` | 6 | **2** | Number of encoder layers stacked -> compositional capacity |
| **Width** | `d_model` | 512 | **32** | Per-token vector dimension -> representational capacity |
| **Parallel heads** | `h` | 8 | **4** | Number of attention heads per MHA block -> relational specialization |
| **FFN expansion** | `d_ff` | 2048 (`4 . d_model`) | **64** (`2 . d_model`) | FFN inner width -> "key-value memory" capacity |

These axes are **not fully independent**:

- `d_k = d_model / h` is forced (a definition, not a choice).
- `d_ff` is conventionally `~4 . d_model` (paper) or `~8/3 . d_model`
  (modern GLU variants -- see Step 5 section 5).
- `N` and `d_model` trade off against each other for a fixed parameter
  budget (deeper-and-narrower vs shallower-and-wider).

### Empirical conventions

#### 1. `d_k = 64` is a near-universal "magic constant"

Almost every Transformer since the original paper -- BERT, GPT-2, T5,
ViT, Llama-2 -- uses `d_k = 64`. Larger models keep `d_k = 64` constant
and **scale by adding more heads** rather than widening per-head
dimension.

Why 64?

- Smaller (`d_k = 16, 32`): each head is too narrow to capture useful
  relational patterns. Heads collapse to redundancy or to representing
  uninteresting things like "match position".
- Larger (`d_k = 128, 256`): per-head expressivity goes up but you lose
  the benefit of having multiple specialised heads. Empirically not
  worth the trade until very large scale.

Llama-3 (2024) finally broke the convention with `d_k = 128`, but only
at >= 70B-parameter scale.

**Our build uses `d_k = 8`** -- deliberately well below the magic
constant -- because we want attention matrices small enough to print and
trace by hand in `test_mha_section5_trace`. This is a "learn-by-
building" trade-off; it would be a poor choice for any real model.

#### 2. `d_ff` follows two main conventions

- **Paper / classical**: `d_ff = 4 . d_model`. With 4x, FFN already
  accounts for ~2/3 of the parameters in an encoder block.
- **Modern (GLU-style)**: `d_ff = (8/3) . d_model = ~2.67 . d_model`,
  but with **three** matrices instead of two -- so total parameter
  count is similar. Used by PaLM, Llama, Mistral.

Our build uses `d_ff = 2 . d_model = 64`. At our scale, 4x would push
FFN to ~80% of all encoder-block params, which is disproportionate for
the toy sort task. See Step 5 section 5 for the full discussion.

#### 3. Depth grows slower than width

For a fixed parameter budget, there is an optimal depth/width
trade-off. The Levine et al. (2020) finding is roughly: **depth scales
as `~sqrt(d_model)`**.

In practice real models tend to be a bit *shallower* than the square-
root rule predicts. Examples:

| Model            | d_model | N    | sqrt(d_model) | Match? |
|------------------|---------|------|---------------|--------|
| Original (base)  | 512     | 6    | ~22.6         | model is shallower |
| BERT-base        | 768     | 12   | ~27.7         | still shallower |
| GPT-3 175B       | 12288   | 96   | ~110.8        | still shallower |
| Llama-3-405B     | 16384   | 126  | 128.0         | finally close |

**Why shallower?** Deeper models are progressively harder to train --
gradient flow becomes finicky, residual-stream variance grows
unbounded without scaling tricks. Each additional layer has
diminishing returns once `N >= ~6`, and the engineering cost of
training stability rises faster than the loss reduction.

#### 4. Modern scaling laws (Kaplan 2020 -> Chinchilla 2022)

These do not tell you encoder *shape*. They tell you how to allocate
*parameters vs data* given a compute budget:

- **Kaplan et al. 2020**: loss decreases as a power law in compute,
  parameters, and data. The paper recommended bigger models, less data.
- **Chinchilla (Hoffmann et al. 2022)**: corrected Kaplan. The
  compute-optimal ratio is roughly **`20 tokens per parameter`**. A
  7B-param model wants ~140B training tokens.
- **Llama-3 era**: production models often train *past* Chinchilla
  (e.g., 1500 tokens/param) because inference cost matters too --
  smaller models cost less to serve. "Compute-optimal" got redefined to
  include serving.

None of this affects our toy build. But it is **the rule that decides
encoder size in real production** today: pick a parameter budget that
fits your compute, then size data to ~20x that param count.

#### 5. Encoder vs decoder asymmetry

| Model class              | Encoder layers | Decoder layers |
|--------------------------|----------------|----------------|
| Original Transformer     | 6              | 6              |
| Translation systems (industry) | often 8-12 | 6              |
| BERT-class (encoder-only)| 12-24          | 0              |
| GPT-class (decoder-only) | 0              | 12-96          |

Pattern: encoders are **easier to deepen** than decoders -- no
autoregressive constraint, all positions process in parallel during
training. So production translation models often use deeper encoders
than decoders. The paper's symmetric `N=6/N=6` is a clean design
choice, not an optimum.

### What our build does, and why

```
d_model = 32       # toy size, 16x smaller than paper
N       = 2        # toy task (sort 5 numbers) doesn't need depth
h       = 4        # so d_k = 8 -- hand-traceable, well below magic 64
d_ff    = 64       # 2x d_model (vs paper's 4x); sufficient for the toy task
```

| Decision      | Driver |
|---------------|--------|
| `d_model = 32`| Small enough to print attention matrices in tests; large enough to learn the sorting task. |
| `N = 2`       | Sorting is single-hop reasoning. Layer 1 can encode "what's at each position", layer 2 can encode "where each position fits in sorted order". More layers buy nothing. |
| `h = 4`       | Forces `d_k = 32/4 = 8`. `d_k = 8` is small enough to hand-trace `softmax(QK^T / sqrt(8))`, large enough that `> 1` head can plausibly specialize. |
| `d_ff = 64`   | `2 . d_model` rather than `4x` to keep FFN from dominating parameter count at our scale. At `4x`, FFN would be ~80% of all encoder-block params. |

These are **deliberate trade-offs for learn-by-building**: pick the
smallest config that still demonstrates the architecture working, not
what you would choose for a real translation model.

---

## 3. Encoder layer: the sub-layer wrapper pattern

The encoder layer reduces to a single concept repeated twice:

> **Take a sub-layer that maps `(B, n, d_model) -> (B, n, d_model)`,
> wrap it in a residual connection and a LayerNorm.**

The two sub-layers in an encoder block are:

1. **Multi-head self-attention** (`MHA(x, x, x)` -- Q, K, V all from `x`).
2. **Position-wise feed-forward network** (`FFN(x)`).

Both have the same input/output shape, so they compose trivially. Both
benefit from the same wrapping pattern (residual + LN) for the same
reasons. The whole encoder block is therefore:

```
x_in  --(MHA wrapper)-->  x_mid  --(FFN wrapper)-->  x_out
```

Each wrapper does, in the paper's Post-LN form:

```
output = LayerNorm(x + Dropout(Sublayer(x)))
```

Three things happen in one line. We will spend the next three sections
unpacking them in order:

- **Section 4**: Why `x +`. (Residual connections.)
- **Section 5**: Why `LayerNorm(...)` -- and crucially, *where* the
  LayerNorm goes. This is the Pre-LN vs Post-LN fork.
- **Section 6**: Why `Dropout(...)` and where else dropout shows up in
  the encoder.

This wrapper pattern is the architectural backbone of **every Transformer
ever made**. Decoder layers (Step 7) use the exact same pattern with one
extra cross-attention sub-layer. Vision Transformers, BERT, GPT, T5,
Llama -- all of them are "MHA wrapper + FFN wrapper" repeated `N` times,
with variations only in *what* goes in the wrapper and *where* the LN
sits.

---

## 4. Residual connections: why they are vital

The "Add" in "Add & LayerNorm" is the addition `x + Sublayer(x)`. This
single design choice is what allows Transformers (and ResNets, which
inspired them) to be deep at all.

### The vanishing/exploding gradient problem, in one sentence

In a deep network without residuals, gradients flowing backward must
pass through every layer multiplicatively. If each layer's local
Jacobian has spectral radius `< 1`, gradients shrink exponentially with
depth. If `> 1`, they explode. Neither is recoverable.

### How a residual fixes it

Consider one wrapped sub-layer's forward pass (ignore LN/dropout for
now):

```
y = x + S(x)
```

The Jacobian with respect to `x` is:

```
dy/dx = I + dS/dx
```

There is an **identity term `I`**. The gradient flowing back is:

```
grad_x = grad_y + grad_y . dS/dx
```

The first term is the gradient flowing **directly through the residual
shortcut, unmodified by the sub-layer**. Even if `dS/dx` is tiny or
zero, gradients still propagate via the identity path. Stacking `N`
such layers gives:

```
grad_input = grad_output . prod_{i=1..N} (I + dS_i/dx)
```

Because every factor has an `I` term, the product never collapses to
zero -- there is always at least the identity contribution surviving.

### Initialization friendliness

A second, equally important benefit:

At initialization, sub-layers are roughly "small random functions".
Their output magnitude is much smaller than their input. So at init:

```
y = x + S(x)  approx=  x
```

The whole stack is approximately the identity function at init. This
is a **good starting point** for SGD: the model starts as "do nothing"
and learns to add useful perturbations. Without residuals, deep stacks
at init produce noise that is uncorrelated with the input, and SGD has
to climb out of that noise before any signal flows.

### Where residuals come from

Residuals were introduced in **He et al. 2016 (ResNet)**, cited as
reference `[11]` in the Transformer paper. ResNet showed you can train
a 152-layer image classifier with this one trick, where 30+ layer
networks had previously been untrainable. The Transformer paper adopts
the exact same idea.

### Shape requirement

For `x + S(x)` to even type-check, `S` must be **shape-preserving**:
input shape == output shape. This is why every sub-layer in an encoder
is `(B, n, d_model) -> (B, n, d_model)`. MHA is shape-preserving by
construction (the `W^O` projection brings it back to `d_model`). FFN is
shape-preserving by construction (the `W_2` projection brings it back
to `d_model`). This shape constraint is **the reason `W^O` and `W_2`
exist at all**.

---

## 5. LayerNorm placement: Pre-LN vs Post-LN

This is the deepest design choice in the encoder. There are two
formulations of "wrap a sub-layer in residual + LN", and they trade off
**training stability** against **paper-faithfulness**.

### The two variants

**Post-LN** (the paper, "Add & Norm"):

```
y = LayerNorm(x + Dropout(S(x)))
```

LN is applied **after** the residual add. The residual stream itself
gets normalized at every layer.

**Pre-LN** (modern default since ~2020):

```
y = x + Dropout(S(LayerNorm(x)))
```

LN is applied **only on the path entering the sub-layer**. The
residual stream itself **never gets normalized** -- it just keeps
accumulating sub-layer outputs.

A small ASCII diff:

```
POST-LN:                              PRE-LN:
  x ---+                                x --------+--+
       |                                          |  |
       +--> S --> Dropout                         v  |
       |     |                                    LN |
       +-----+                                    |  |
            +                                     v  |
            v                                     S  |
           LN                                     |  |
            |                                     v  |
            v                                  Dropout
           y                                     |  |
                                                 +--+
                                                  |
                                                  v
                                                  y
```

### Why the placement matters: gradient flow

In **Pre-LN**, the residual stream is untouched. Differentiate:

```
y = x + S(LN(x))
dy/dx = I + dS/dLN . dLN/dx
```

The gradient flowing back through the identity path is **literally
unchanged**. After `N` Pre-LN layers, gradients can still propagate
back to layer 1 with no normalization-induced loss.

In **Post-LN**, the residual stream is renormalized at every layer.
Differentiate:

```
y = LN(x + S(x))
dy/dx = dLN_in . (I + dS/dx)
```

Now `dLN_in` (LayerNorm's Jacobian) sits in the way of every gradient
that flows back. LN is *not* an isometry -- its Jacobian has a complex
structure that, at deep layers, mixes gradients in ways that make
training unstable unless you are very careful with the learning rate
schedule.

### The empirical fallout

The original paper used Post-LN and trained successfully -- but only
with an elaborate **learning rate warmup schedule** (4000 steps of
linear LR ramp before the cosine decay starts). Without that warmup,
Post-LN training famously diverges in the first few hundred steps.

**Xiong et al. 2020, "On Layer Normalization in the Transformer
Architecture"**, dissected this. Their headline findings:

| Setup            | Trains? | Needs warmup? | Scales to deep models? |
|------------------|---------|---------------|------------------------|
| Post-LN + warmup | yes     | **yes (critical)** | poorly |
| Post-LN no warmup| often diverges | n/a    | n/a                    |
| Pre-LN  + warmup | yes     | helpful but not strictly required | well |
| Pre-LN  no warmup| **yes** | -             | **well**               |

After this paper, essentially every new Transformer (GPT-3, T5, ViT,
PaLM, Llama, ...) switched to Pre-LN. Post-LN survives mostly in
encoder-only NLP models that were originally trained Post-LN
(BERT-base, the original Transformer Big translation model) and in
papers explicitly arguing it gives a slightly better final loss
**when** it converges.

### Final-loss vs trainability trade-off

Some recent work (Liu et al. 2020 "Understanding the Difficulty of
Training Transformers"; Shleifer et al. 2021 "NormFormer") argues that
when Post-LN does train, it gives marginally lower final loss because
its sub-layers can move the residual stream more freely without
worrying about variance blow-up at depth. Pre-LN, by contrast, has a
residual stream whose variance grows roughly linearly with depth --
which means deep Pre-LN models can have "lazy" later layers that
contribute very little because their relative magnitude vs the
residual is small.

In short:

- **Post-LN**: harder to train, slightly better final loss when it
  trains, *especially* sensitive to warmup.
- **Pre-LN**: easier to train, robust at scale, slightly worse final
  loss in the "when it trains" regime.

The community made the trade because **a model that does not train is
useless**, and Pre-LN scales effortlessly to 100+ layers.

### Decision for our build

Our `N = 2` encoder is shallow enough that Post-LN training stability
is **not a real concern**. We will follow the paper exactly:

```
y = LayerNorm(x + Dropout(Sublayer(x)))   # Post-LN, paper-faithful
```

This is a learn-the-paper project; deviating from the paper's
formulation here would defeat the point of the exercise. We do plan to
add a `pre_ln: bool = False` constructor flag in slice 2 so that
ablation tests can compare both formulations -- understanding the
*difference* is part of the point.

Concrete consequences for tests:

- The Post-LN encoder with random weights produces outputs whose
  per-token vectors are approximately unit-norm (because every layer
  ends with a LN).
- The Pre-LN encoder, *when we toggle the flag*, produces outputs
  whose magnitude grows with `N`, because the residual stream is
  un-normalized. We can write a sanity test for both behaviors.

---

## 6. Dropout: the four places it appears

The paper's section 5.4 says, quoted verbatim:

> "We apply dropout to the output of each sub-layer, before it is added
> to the sub-layer input and normalized. In addition, we apply dropout
> to the sums of the embeddings and the positional encodings in both
> the encoder and decoder stacks."

`P_drop = 0.1` for the base model. That paragraph specifies **two**
places. Real implementations also use **two more** that are not in
that paragraph but are in the original released code. So there are
four total places dropout can sit in an encoder:

### Place 1: After each sub-layer, before the residual add

This is the paper's primary dropout location. Inside the wrapper:

```
y = LayerNorm(x + Dropout(Sublayer(x)))
                  ^^^^^^^
                  paper section 5.4, location 1
```

It applies to **both** sub-layers (MHA and FFN). It noises the
sub-layer's contribution but leaves the residual identity path clean.
Mandatory in our build -- this is the canonical Transformer dropout.

### Place 2: After embedding + positional encoding, before layer 1

This is the paper's other explicit location:

```
x_0 = Dropout(Embedding(tokens) * sqrt(d_model) + PE)
                                  ^^^^^^^^^^^^^^^^^^^^^^
                                  this whole sum is dropout-noised
```

It noises the very first input to encoder layer 1. With `p = 0.1`,
roughly 1 in 10 components of every token+position vector gets zeroed
each step. This is a form of **input regularization**. Mandatory in
our build.

### Place 3: On attention weights (post-softmax)

The original Transformer code release applied dropout to the attention
weights *after* softmax, before they are multiplied by `V`:

```
A = softmax(QK^T / sqrt(d_k))
A = Dropout(A)
output = A @ V
```

The paper text does not mention this directly, but it is in the
official code and has been adopted by every major implementation
since. Conceptually it is "stochastic head pruning per token pair":
sometimes a query position will pretend a key position contributed
nothing.

**Decision for our build**: not in slice 2. We will leave a
`# TODO(step 6 future): attention dropout` comment in the MHA layer.
Adding it requires touching the MHA layer (which is finished and
tested), and it is not strictly needed for the toy task.

### Place 4: Inside FFN, after ReLU

Some implementations add a dropout between `W_1` and `W_2` inside the
FFN:

```
h = Dropout(ReLU(W_1 x + b_1))
out = W_2 h + b_2
```

This is **not** in the paper or the original code. It shows up in
some BERT and GPT-2 variants. Conceptually it is "noise the FFN's
internal hidden state" rather than "noise the FFN's output".

**Decision for our build**: not in slice 2. Place 1 (after the FFN
output, before the residual add) is the paper's canonical FFN dropout
and is sufficient.

### Summary of where we put dropout

| Place | In paper? | In our build? |
|-------|-----------|---------------|
| 1. After each sub-layer (MHA, FFN), before residual add | yes | **yes** |
| 2. After embedding + PE | yes | **yes** |
| 3. On attention weights post-softmax | no (in code) | not now |
| 4. Inside FFN after ReLU | no | not now |

Our `Dropout` layer (Step 1, layer 1e) supports a `training` flag that
disables it at inference. Section 8 of the build plan
(`08_training.md`) will set `training=False` on every dropout layer
when running validation/inference.

---

## 7. Embedding scaling: why multiply by `sqrt(d_model)`

Paper section 3.4, the relevant sentence:

> "In the embedding layers, we multiply those weights by sqrt(d_model)."

This is a one-line spec with no explanation. Three possible reasons
have been proposed in the literature; the dominant one is **variance
balance** with the positional encoding. All the explanations below are
consistent.

### The variance balance argument

Sinusoidal positional encoding has fixed magnitude. For position
`pos`, the L2 norm of `PE[pos, :]` is:

```
|PE[pos, :]|^2  =  sum_{i=0..d_model/2-1} (sin^2(arg_i) + cos^2(arg_i))
                =  sum_i 1
                =  d_model / 2
```

So `|PE[pos, :]| = sqrt(d_model / 2)`. For `d_model = 32` that is
about `4.0`. **PE rows have a fixed L2 norm of ~sqrt(d_model/2),
independent of position**.

Embeddings, by contrast, are typically initialized with **small per-
element variance**. There are two common conventions:

- **Xavier-style**: per-element std `~ 1 / sqrt(d_model)`. Then a row
  has expected L2 norm `sqrt(d_model) . 1/sqrt(d_model) = 1`. **Norm
  ~1**.
- **BERT/GPT-2 style** (what our `Embedding` class uses): per-element
  std `0.02`, fixed. Then a row has expected L2 norm
  `0.02 . sqrt(d_model)`. For `d_model=32`, that is `~0.113`.

If we naively add `Embedding + PE`, PE dominates by a factor of 4-35x
depending on init scheme. The token information is drowned in the
positional signal at the very start of training.

Multiplying embeddings by `sqrt(d_model)` boosts their L2 norm by a
factor `sqrt(d_model)`. The result:

| Init scheme       | Raw norm         | * sqrt(d_model)         | PE norm            | Ratio (after) |
|-------------------|------------------|-------------------------|--------------------|---------------|
| Xavier (1/sqrt(d_model)) | 1         | sqrt(d_model)          | sqrt(d_model/2)    | sqrt(2) ~ 1.4 |
| BERT-style (0.02) | 0.02 sqrt(d_model) | 0.02 d_model         | sqrt(d_model/2)    | depends on d_model |

For Xavier-style init the scaling **exactly balances** PE and the
embedding (within a factor of `sqrt(2)`). For BERT-style init it
helps but does not fully balance -- with `d_model=32`, embeddings
become `0.64`, PE is `4.0`, so PE is still ~6x larger. This
imbalance is **inherited from the init choice**, not a bug in the
scaling rule.

### Two other framings (consistent with the above)

**Weight tying.** Paper section 3.4 also notes that the input
embedding matrix is **shared with the output linear layer's weight
matrix** (`W_out = E^T`). The output linear is initialized with
variance `~1 / d_model`, which gives the output projection unit-norm-
per-column behavior. To make the embedding lookup behave like a
unit-norm vector emitter (matching what the output projection is
designed to consume), you scale up by `sqrt(d_model)`. The two
constraints (input-side and output-side) are not perfectly
compatible, and `sqrt(d_model)` is the compromise.

**Gradient compensation.** Embeddings receive gradients **only on the
tokens that appear in a given batch** -- they are sparsely updated.
Other parameters are dense. Scaling embeddings up by `sqrt(d_model)`
in the forward pass means their effective learning rate in the
backward pass is **scaled down** by `sqrt(d_model)`, partly
compensating for the sparse update pattern.

### Decision for our build

We follow the paper exactly: embeddings are multiplied by
`sqrt(d_model)` before adding PE. Our `Embedding` class already uses
BERT-style `0.02` init, so the variance balance with PE will be
imperfect at initialization -- which is fine for a toy build, and we
will simply note this in the doc. The model has plenty of capacity to
learn its way out of any initial imbalance during training.

In code (slice 2):

```python
e = self.embedding.forward(token_ids)        # (B, n, d_model)
e = e * np.sqrt(self.d_model)                 # the paper line
e = e + self.pos_enc.forward(seq_len)         # broadcast PE over batch
x_0 = self.dropout.forward(e)                 # paper section 5.4
```

---

## 8. Full encoder data flow

Putting all of sections 3-7 together:

```python
def encode(token_ids):
    # token_ids: (B, n) integer
    # 1. Embedding lookup -> (B, n, d_model)
    e = embedding(token_ids)

    # 2. Paper section 3.4: scale by sqrt(d_model) (section 7 above)
    e = e * sqrt(d_model)

    # 3. Add positional encoding (broadcast over batch)
    #    pos_enc(seq_len) -> (n, d_model), broadcasts to (B, n, d_model)
    e = e + pos_enc(seq_len=n)

    # 4. Paper section 5.4: dropout on (Embedding + PE) -- place 2
    x = dropout_input(e)

    # 5. Stack of N encoder layers, each is a "MHA wrapper -> FFN wrapper"
    for layer in encoder_layers:  # N=2 in our build
        x = layer(x)              # (B, n, d_model) -> (B, n, d_model)

    return x  # (B, n, d_model) -- input to decoder cross-attention
```

And inside one layer, Post-LN form:

```python
def encoder_layer(x):
    # MHA wrapper
    h = mha(x, x, x)              # self-attention: Q=K=V=x
    h = dropout_mha_out(h)        # paper section 5.4 -- place 1
    x = layernorm_1(x + h)        # Post-LN

    # FFN wrapper
    h = ffn(x)
    h = dropout_ffn_out(h)        # paper section 5.4 -- place 1 (again)
    x = layernorm_2(x + h)        # Post-LN

    return x
```

Every dropout layer has the same `p`. Every LayerNorm has the same
`d_model`. Every MHA has the same `(d_model, h)`. Every FFN has the
same `(d_model, d_ff)`. The encoder is configurationally tiny --
essentially just `(N, d_model, h, d_ff, p_drop, max_len, vocab_size)`.

---

## 9. Worked shape walk-through

Concrete trace through our build's encoder. Config:

```
B = 2          (batch size)
n = 3          (sequence length: tiny example)
vocab = 16     (toy vocab)
d_model = 32
N = 2
h = 4          (so d_k = 8)
d_ff = 64
p_drop = 0.1
```

Input: a batch of two sequences of three tokens each:

```
token_ids = [[5, 7, 11],
             [3, 9,  2]]            shape (2, 3)
```

### Step-by-step shapes

| # | Step                               | Output shape       | Notes |
|---|------------------------------------|--------------------|-------|
| 1 | `embedding[token_ids]`             | `(2, 3, 32)`       | lookup, no math |
| 2 | `* sqrt(d_model)`                  | `(2, 3, 32)`       | scalar broadcast |
| 3 | `+ PE[:3, :]`                      | `(2, 3, 32)`       | PE shape `(3, 32)` broadcasts over batch |
| 4 | `dropout(p=0.1)`                   | `(2, 3, 32)`       | mask shape `(2, 3, 32)` -- same as input |
|   | -- enter Layer 1 --                |                    |       |
| 5 | `MHA(x, x, x)` (self-attention)    | `(2, 3, 32)`       | internally: split heads -> `(2, 4, 3, 8)`; SDPA gives `(2, 4, 3, 8)`; merge heads back to `(2, 3, 32)`; W^O |
| 6 | `Dropout` (sub-layer output)       | `(2, 3, 32)`       |       |
| 7 | `+ residual (x_in from step 4)`    | `(2, 3, 32)`       | shape-preserving add |
| 8 | `LayerNorm`                        | `(2, 3, 32)`       | normalize over last dim |
| 9 | `FFN`                              | `(2, 3, 32)`       | internally: `(2, 3, 32) -> W_1 -> (2, 3, 64) -> ReLU -> (2, 3, 64) -> W_2 -> (2, 3, 32)` |
| 10| `Dropout`                          | `(2, 3, 32)`       |       |
| 11| `+ residual (x_in to FFN, from step 8)` | `(2, 3, 32)`  |       |
| 12| `LayerNorm`                        | `(2, 3, 32)`       | end of layer 1 |
|   | -- enter Layer 2 (same as 5-12) --  |                   |       |
| 13-20| Layer 2 mirror of layer 1       | `(2, 3, 32)`       |       |
|   | -- encoder output --                |                   |       |
| 21| Final encoder output               | `(2, 3, 32)`       | fed to decoder cross-attention |

**Invariant**: every line is `(2, 3, 32)`. The encoder is a sequence
of `(B, n, d_model) -> (B, n, d_model)` transforms. Internally, MHA
briefly visits `(B, h, n, d_k) = (2, 4, 3, 8)` and FFN briefly visits
`(B, n, d_ff) = (2, 3, 64)`, but each sub-layer restores the
canonical shape on exit. This is what makes `N` layers stackable.

### Parameter count for our build

Counting every learnable parameter in the encoder, per sub-component:

| Component                | Weights         | Biases       | Per-layer total |
|--------------------------|----------------:|-------------:|----------------:|
| MHA: `W^Q, W^K, W^V`     | `3 . 32 . 32 = 3072` | `3 . 32 = 96` | `3168`     |
| MHA: `W^O`               | `32 . 32 = 1024`     | `32`          | `1056`     |
| LayerNorm 1              | --              | `2 . 32 = 64` (`gamma`+`beta`) | `64` |
| FFN: `W_1`               | `32 . 64 = 2048`     | `64`          | `2112`     |
| FFN: `W_2`               | `64 . 32 = 2048`     | `32`          | `2080`     |
| LayerNorm 2              | --              | `2 . 32 = 64` | `64`            |
| **Per encoder layer**    | **`8192`**      | **`352`**    | **`8544`**     |

Per layer total: `3168 + 1056 + 64 + 2112 + 2080 + 64 = 8544`. Two layers: `2 . 8544 = 17088`.

Plus the input pipeline:

| Component                | Count |
|--------------------------|------:|
| Embedding (`vocab=16, d_model=32`) | `16 . 32 = 512` |
| Positional encoding      | `0` (fixed sinusoidal -- no learnable params) |
| Input dropout            | `0` (no learnable params) |

**Total encoder parameter count for our toy build: `17088 + 512 = 17600`**.

The test `test_encoder_full_param_count_matches_doc` re-derives this
number cell by cell and asserts the actual encoder reports it.

For comparison, the paper's base encoder has ~32M parameters (`6
layers x ~5.3M per layer`). Our encoder is **~1800x smaller**. Our
sort-5-numbers task expects this to still be enough capacity; if the
training step (Step 8) shows otherwise, we can revisit `d_ff` or
`d_model` first before adding layers.

---

## 10. Forward pass in code

Two classes, both in `numpy-transformer/src/numpy_transformer/layers.py`.

### `EncoderLayer`

One repeating block. Owns six sub-layers:

```
mha       : MultiHeadAttention(d_model, num_heads)
dropout_1 : Dropout(p_drop)
ln_1      : LayerNorm(d_model)
ffn       : FeedForward(d_model, d_ff)
dropout_2 : Dropout(p_drop)
ln_2      : LayerNorm(d_model)
```

Constructor: `EncoderLayer(d_model, num_heads, d_ff, p_drop=0.1, pre_ln=False)`.

The forward dispatches on `self.pre_ln` to one of two private methods.
**Post-LN** is the paper formulation:

```python
def _forward_post_ln(self, x):
    h_mha   = self.mha.forward(x, x, x)
    h_mha_d = self.dropout_1.forward(h_mha)
    y_mid   = self.ln_1.forward(x + h_mha_d)

    h_ffn   = self.ffn.forward(y_mid)
    h_ffn_d = self.dropout_2.forward(h_ffn)
    y_out   = self.ln_2.forward(y_mid + h_ffn_d)
    return y_out
```

**Pre-LN** is the modern formulation, exposed via `pre_ln=True`:

```python
def _forward_pre_ln(self, x):
    x_ln    = self.ln_1.forward(x)
    h_mha   = self.mha.forward(x_ln, x_ln, x_ln)
    h_mha_d = self.dropout_1.forward(h_mha)
    y_mid   = x + h_mha_d                # un-normalized residual stream

    y_ln    = self.ln_2.forward(y_mid)
    h_ffn   = self.ffn.forward(y_ln)
    h_ffn_d = self.dropout_2.forward(h_ffn)
    y_out   = y_mid + h_ffn_d            # un-normalized residual stream
    return y_out
```

### `Encoder`

Full input pipeline. Owns:

```
embedding     : Embedding(vocab_size, d_model)
pos_enc       : SinusoidalPositionalEncoding(d_model, max_len)
dropout_input : Dropout(p_drop)               # paper section 5.4 -- place 2
layers        : [EncoderLayer(...) for _ in range(num_layers)]
```

The constructor caches `self._embed_scale = sqrt(d_model)` once, since
it is used on every forward pass.

Constructor:
`Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, p_drop=0.1, max_len=5000, pre_ln=False)`.

Forward:

```python
def forward(self, token_ids):
    e        = self.embedding.forward(token_ids)
    e_scaled = e * self._embed_scale            # paper section 3.4
    e_pe     = self.pos_enc.forward(e_scaled)   # broadcast PE over batch
    x        = self.dropout_input.forward(e_pe) # paper section 5.4 place 2
    for layer in self.layers:
        x = layer.forward(x)
    return x
```

### Cache strategy

The encoder layer itself does **not** have a `_cache` dict. Every
sub-layer (`MultiHeadAttention`, `FeedForward`, `LayerNorm`, `Dropout`)
already caches whatever it needs for its own backward pass. The
encoder layer only needs to remember the `pre_ln` flag, which is a
constructor attribute.

This means `EncoderLayer.backward` does no manual unpacking -- it just
calls each sub-layer's backward in the reverse order of forward.

### Training-mode toggle

Both `EncoderLayer` and `Encoder` expose `train(mode=True)` and
`eval()` methods (PyTorch-style):

```python
encoder.eval()        # disables every Dropout in the encoder
encoder.train(True)   # re-enables them
```

These walk the dropouts recursively. They are mandatory for two
purposes:

- **Inference** (Step 9): we want deterministic output, so dropouts
  must be off.
- **Numerical gradient checks** in tests: the eps-perturbed re-forward
  must produce a deterministic output, otherwise the central-
  difference estimate becomes garbage.

---

## 11. Backward pass

Backward is "run forward in reverse, calling each sub-layer's backward
instead of forward". No new mathematical derivations -- the heavy
lifting already happened in steps 1, 2, 4, 5. Two structural rules
are worth stating explicitly because they are easy to get wrong.

### Rule 1 -- residual sum: gradient flows to BOTH branches

Forward: `y = a + b`. Backward: `grad_a = grad_y` AND `grad_b = grad_y`.
The same `grad_y` is delivered to both branches; no division, no
splitting. This is just the chain rule for a sum: `dy/da = 1`,
`dy/db = 1`.

In code, both wrappers (Post-LN and Pre-LN) have a residual sum, and
in both cases we end up adding a "via residual" gradient to a "via
sub-layer path" gradient at the layer's input.

### Rule 2 -- self-attention input gradient sums three branches

`MultiHeadAttention.backward(grad_output)` returns a 3-tuple
`(grad_Q_in, grad_K_in, grad_V_in)` because MHA has three logical
inputs. **In self-attention, all three are aliases to the same `x`.**

The chain rule for a function `f(x, x, x)` says

```
df/dx = (partial f / partial Q_in)|_{Q_in=x}
      + (partial f / partial K_in)|_{K_in=x}
      + (partial f / partial V_in)|_{V_in=x}
```

So the encoder layer must do `grad_x_via_mha = gQ + gK + gV`. MHA's
own backward documentation (section 8 of `04_multi_head_attention.md`)
deliberately leaves this sum to the caller, because **cross-attention
in the decoder** does *not* have all three inputs aliased -- there
the caller would NOT sum.

### Post-LN backward

Forward (recap):

```
y_mid = LN1(x + Drop1(MHA(x, x, x)))
y_out = LN2(y_mid + Drop2(FFN(y_mid)))
```

Backward (the implementation):

```python
def _backward_post_ln(self, grad_output):
    g_sum2          = self.ln_2.backward(grad_output)        # rule 1: this is grad of the sum (y_mid + h_ffn_d)
    grad_h_ffn      = self.dropout_2.backward(g_sum2)
    grad_y_mid_ffn  = self.ffn.backward(grad_h_ffn)
    grad_y_mid      = g_sum2 + grad_y_mid_ffn                # rule 1: residual + sub-layer paths

    g_sum1          = self.ln_1.backward(grad_y_mid)
    grad_h_mha      = self.dropout_1.backward(g_sum1)
    gQ, gK, gV      = self.mha.backward(grad_h_mha)
    grad_x_via_mha  = gQ + gK + gV                           # rule 2: self-attention sum
    grad_x          = g_sum1 + grad_x_via_mha                # rule 1
    return grad_x
```

### Pre-LN backward

Forward (recap):

```
y_mid = x + Drop1(MHA(LN1(x), LN1(x), LN1(x)))
y_out = y_mid + Drop2(FFN(LN2(y_mid)))
```

Backward (the implementation):

```python
def _backward_pre_ln(self, grad_output):
    grad_h_ffn      = self.dropout_2.backward(grad_output)   # grad_output reaches the residual untouched too
    grad_y_ln       = self.ffn.backward(grad_h_ffn)
    grad_y_mid_ffn  = self.ln_2.backward(grad_y_ln)
    grad_y_mid      = grad_output + grad_y_mid_ffn           # rule 1

    grad_h_mha      = self.dropout_1.backward(grad_y_mid)
    gQ, gK, gV      = self.mha.backward(grad_h_mha)
    grad_x_ln       = gQ + gK + gV                           # rule 2
    grad_x_mha_path = self.ln_1.backward(grad_x_ln)
    grad_x          = grad_y_mid + grad_x_mha_path           # rule 1
    return grad_x
```

The Pre-LN path runs `grad_output` to **two destinations**: through
the FFN path (Dropout2 -> FFN -> LN2), and **untouched** to the
residual addition `grad_y_mid`. The same pattern repeats for the MHA
wrapper.

### Encoder backward returns `None`

```python
def backward(self, grad_output):
    grad_x = grad_output
    for layer in reversed(self.layers):
        grad_x = layer.backward(grad_x)
    grad_e_pe     = self.dropout_input.backward(grad_x)
    grad_e_scaled = self.pos_enc.backward(grad_e_pe)
    grad_e        = grad_e_scaled * self._embed_scale
    self.embedding.backward(grad_e)
    return None
```

The `Embedding` layer takes integer token IDs as input. Integers have
no continuous gradient, so `Embedding.backward` returns `None` and
`Encoder.backward` propagates that. All learnable parameters'
gradients are still populated as a side effect on every sub-layer
(this is what `params_and_grads()` exposes for the optimizer in Step
8). The `* self._embed_scale` step on the way back is the chain-rule
mirror of the `e * sqrt(d_model)` scaling on the way forward.

`SinusoidalPositionalEncoding.backward` is the identity (PE has no
learnable parameters and is added, not multiplied), so the line is
mostly there for symmetry with the forward.

---

## 12. Sanity checks and tests

Eleven tests in `numpy-transformer/tests/test_layers.py`, all green.

### EncoderLayer-level

- **`test_encoder_layer_shape_preservation`** -- forward preserves
  shape `(B, n, d_model)` for both Post-LN and Pre-LN. The shape
  invariant from section 9 holds in both modes.
- **`test_encoder_layer_post_ln_outputs_unit_normalized`** -- with
  default `gamma=1, beta=0` LN init, every per-token output vector of
  a Post-LN layer has empirical mean 0, variance 1, and L2 norm
  exactly `sqrt(d_model)`. Structural property; independent of
  sub-layer weights and input scale.
- **`test_encoder_layer_pre_ln_residual_unbounded`** -- the
  complementary property for Pre-LN: scale the input 10x, the output
  L2 norm tracks the input L2 norm (within 50% tolerance) instead of
  collapsing to `sqrt(d_model)`. Demonstrates the un-normalized
  residual stream from section 5.
- **`test_encoder_layer_residual_passes_zero_sublayer_pre_ln`** --
  zero out the *last* projection in each sub-layer (`W^O` of MHA,
  `W_2` of FFN). Both sub-layer outputs become exactly zero, the
  Pre-LN block reduces to two residual additions of zero, and the
  output equals the input exactly (within `atol=1e-12`). Confirms
  the residual identity path is wired correctly.
- **`test_encoder_layer_backward_gradient_check_post_ln`** --
  numerical gradient check (eps=1e-5, central differences) on `x`
  and every parameter of the Post-LN layer. Robust comparator:
  `abs_err < 1e-7 OR rel_err < 1e-4` so structurally-zero gradients
  (like `W_k.b` from section 8 of Step 4) are not flagged as
  failing.
- **`test_encoder_layer_backward_gradient_check_pre_ln`** -- same
  check for the Pre-LN layer.

### Encoder-level

- **`test_encoder_section9_shape_trace`** -- the worked example from
  section 9: `vocab=16, d_model=32, h=4, d_ff=64, N=2`, token IDs
  `(2, 3)` -> output `(2, 3, 32)`.
- **`test_encoder_full_forward_shape`** -- generic
  `(B, n) -> (B, n, d_model)` shape contract.
- **`test_encoder_full_param_count_matches_doc`** -- builds the
  encoder with our config and asserts the total parameter count
  equals **17,600**, derived per cell from the table in section 9.
- **`test_encoder_dropout_off_at_eval_is_deterministic`** -- with
  `enc.eval()` set, two forward passes on the same input return
  bitwise-identical output. Confirms the recursive train/eval toggle
  reaches every dropout in the stack.
- **`test_encoder_full_backward_gradient_check`** -- end-to-end
  numerical gradient check through the full encoder. Token IDs are
  integer (no input gradient checked), but every learnable parameter
  -- embedding, both encoder layers' MHA/LN/FFN/LN -- is verified
  against central-difference numerical gradients.

Run command (workspace root):

```cmd
uv run --package numpy-transformer pytest numpy-transformer/tests/test_layers.py -k encoder
```

---

## Discussion Log

A chronological log of every conceptual exchange. Append, do not edit.

### 2026-04-25 -- Q&A: are there any rules deciding the size of the encoder?

**Question raised** while wrapping up Step 5 (FFN), before Step 6 itself
had started. Captured here because the answer applies to the whole
encoder build, not just one sub-component.

**Headline answer.** There is **exactly one hard mathematical rule** --
`d_model % num_heads == 0`, from the split-heads reshape trick in Step 4
section 4. Everything else about encoder shape (`N, d_model, h, d_ff`,
positional encoding choice, etc.) is **empirical convention or
scaling-law guidance, not proof**.

**Four-axis decomposition** (added as the doc's main "Sizing" section
above):

1. **Depth `N`** -- compositional layers.
2. **Width `d_model`** -- per-token vector dimension.
3. **Heads `h`** -- parallel attention experts.
4. **FFN expansion `d_ff`** -- inner width of the FFN.

Plus the constrained derivative `d_k = d_model / h`.

**Empirical conventions surveyed:**

- `d_k = 64` as a near-universal magic constant (BERT, GPT-2/3, T5, ViT,
  Llama-2 -- only Llama-3 finally went to 128 at very large scale).
- `d_ff = 4 . d_model` (paper) or `~8/3 . d_model` (modern GLU variants
  with three FFN matrices).
- Depth-vs-width: depth grows as `~sqrt(d_model)` per Levine et al.
  2020, but real models tend to be **a bit shallower than that** --
  diminishing returns + training-stability cost rise with depth.
- Encoder/decoder asymmetry: production translation systems often have
  deeper encoders than decoders (encoders parallelise more easily).

**Modern scaling laws (Kaplan 2020 -> Chinchilla 2022 -> Llama-3 era):**
not about encoder *shape* -- about allocating params vs data given a
compute budget. Chinchilla's `~20 tokens per parameter` is the modern
default; Llama-3 era trains 75x past Chinchilla because inference cost
matters too.

**Our build's encoder shape**, justified per axis:

- `d_model = 32`: toy small but big enough for the sort task.
- `N = 2`: sorting is single-hop, more layers buy nothing.
- `h = 4`: forces `d_k = 8`, hand-traceable in unit tests.
- `d_ff = 64 = 2 . d_model`: deviates from paper's 4x to keep FFN
  parameter share manageable at toy scale.

**Why this matters going into Step 6.** Once we start composing the
encoder block we will be looking at concrete tensor shapes -- for
example, `attention(d_model=32) -> residual+LN(d_model=32) ->
FFN(d_model=32, d_ff=64) -> residual+LN(d_model=32)`. Knowing **why
each constant has the value it does** (and which ones are forced vs
chosen) makes the data-flow walk-through in Step 6 a lot clearer than
just reading the numbers.

**Open thread for Step 6 itself:** Pre-LN vs Post-LN, where dropout
goes, and whether to follow the paper's `LayerNorm(x + sublayer(x))`
exactly or use the more modern `x + sublayer(LayerNorm(x))` (Pre-LN,
which is more stable to train). That decision belongs in Step 6
proper, not here.

### 2026-04-25 -- Step 6 kickoff (slice 1: conceptual deep-dive only)

**No code this turn.** Slice 1 of Step 6 is dedicated to fully
documenting the encoder's design before we touch `layers.py`. Slice 2
will land the `EncoderLayer` and `Encoder` classes plus tests.

**Three big questions resolved in this doc:**

1. **Pre-LN vs Post-LN.** Section 5. Headline: Pre-LN is the modern
   default (since Xiong et al. 2020) because gradients flow through
   the un-normalized residual stream cleanly, removing the need for
   the elaborate LR warmup that Post-LN requires. Post-LN survives
   in BERT-class models and gives marginally better final loss when
   it converges. **For this build we follow the paper (Post-LN) by
   default**, with `pre_ln: bool = False` as a constructor flag for
   ablation experiments. Our `N=2` is shallow enough that Post-LN
   training stability is not a real concern.

2. **Where dropout goes.** Section 6. Four candidate locations
   (after each sub-layer's output, on embedding+PE, on attention
   weights post-softmax, and inside FFN after ReLU). Paper section
   5.4 prescribes the first two. The other two are common in code
   but not in the paper. **For this build we ship places 1 and 2
   only**; the others get TODO comments.

3. **Why multiply embeddings by `sqrt(d_model)`.** Section 7. The
   variance-balance argument with PE is the canonical reason: PE
   rows have fixed L2 norm `sqrt(d_model/2)`, while raw embedding
   rows under any reasonable init have norm O(1) or smaller, so
   without scaling PE drowns out the token information at layer 1's
   input. Two consistent secondary framings (weight tying with the
   output projection, sparse-update gradient compensation) are
   noted. Our `0.02`-init embeddings will not balance PE perfectly
   even after scaling, but the model can learn around any initial
   imbalance.

**Things deliberately deferred:**

- Attention dropout (place 3) and FFN-internal dropout (place 4):
  documented but not implemented, with TODO comments planned for
  slice 2.
- The `Encoder` constructor signature and exactly how `pre_ln` flows
  through the layer stack: drafted in this doc as pseudocode but
  not finalised. Will be pinned down in slice 2.
- The full numerical hand-trace (versus just shapes): section 9
  gives the **shape** trace for `(B, n, d_model) = (2, 3, 32)`. The
  full *value* trace at section-7-of-Step-4 fidelity would require
  picking specific weight initializations and is more useful as a
  test fixture than a doc walk-through. Slice 2 may include it as
  `test_encoder_section9_shape_trace`.

**Worked example for section 9** uses the smallest non-trivial config
the build allows -- `B=2, n=3, h=4, d_model=32, d_ff=64`. The point
of section 9 is **not** to derive new math but to confirm by direct
inspection that every line of the data flow returns to the canonical
`(B, n, d_model)` shape. Internally, MHA briefly visits `(2, 4, 3,
8)` and FFN briefly visits `(2, 3, 64)`, but the canonical shape is
restored by `W^O` and `W_2` respectively. **This is the structural
reason `W^O` and `W_2` exist**: they make sub-layers stackable.

**Invariant to keep in mind for slice 2.** Every encoder sub-layer
must be **shape-preserving** `(B, n, d_model) -> (B, n, d_model)`,
because the residual `x + Sublayer(x)` requires it. Any future
change to MHA, FFN, or the layer wrapper must preserve this
invariant or the encoder breaks at the type level.

**Next turn (slice 2):** implement `EncoderLayer`, `Encoder`, the
`pre_ln` flag, and the test suite outlined in section 12. The doc's
sections 10-12 get filled in with the actual code structure and
final test list at that time.

### 2026-04-25 -- Slice 2: implementation + tests landed

**`EncoderLayer` and `Encoder` are in
`numpy-transformer/src/numpy_transformer/layers.py`** as classes 6 and
7 (after FFN at 5). Both follow the existing `Layer` interface
(`forward`, `backward`, `params_and_grads`) and pass numerical
gradient checks. Eleven new tests, all green; full `pytest` is 55
green.

**Things finalised this slice:**

- The `pre_ln: bool = False` flag is on **both** `EncoderLayer` and
  `Encoder`. It propagates from the `Encoder` constructor down to
  every `EncoderLayer` it builds. Default is Post-LN
  (paper-faithful), as planned in slice 1's section 5 decision.
- A `train(mode=True)` / `eval()` toggle (PyTorch-style) is exposed
  on both classes. It walks every `Dropout` in the block. This is
  required by the gradient-check tests (deterministic forward) and
  by the eventual training/inference scripts.
- `Encoder.backward()` returns `None`. The decision was forced by
  `Embedding.backward`, which itself returns `None` because integer
  token IDs have no continuous gradient. All learnable parameters'
  gradients are still populated as a side effect on each sub-layer.

**Doc fix-up made this slice (worth flagging because it would have
caused future confusion):**

The original Section 9 parameter-count table from slice 1 had an
arithmetic inconsistency -- the column header said "x N=2" but the
prose treated the values as per-layer. The corrected count is **8544
parameters per encoder layer** (3168 MHA + 1056 W^O + 64 LN_1 + 2112
FFN W_1 + 2080 FFN W_2 + 64 LN_2), giving **17,088 for two layers**
plus **512 embedding** for a **17,600 total**. The
`test_encoder_full_param_count_matches_doc` test re-derives this cell
by cell and locks it in.

**Backward chain notable subtleties (now in section 11):**

1. **Residual sum -> both branches receive `grad_y` unchanged.** Easy
   to forget; if you accidentally divide by 2 or split, the
   gradient check fails immediately.
2. **Self-attention gradient is `gQ + gK + gV`.** MHA's backward
   returns the 3-tuple deliberately so that cross-attention (where
   `Q_in` is distinct from `K_in == V_in`) can do the *correct*
   2-way sum instead. Encoder self-attention does the 3-way sum.
3. **`Encoder.backward` reverses through:** `[layers in reverse]
   -> dropout_input -> pos_enc -> * sqrt(d_model) -> embedding`.
   The PE backward is identity; the scaling backward is just the
   same scalar multiply. The only place a gradient is *consumed*
   without being passed back is at the embedding (integer input).

**Tests deliberately included for the slice 1 conceptual claims:**

- `test_encoder_layer_post_ln_outputs_unit_normalized` and
  `test_encoder_layer_pre_ln_residual_unbounded` together
  **demonstrate the Pre-LN vs Post-LN section 5 trade-off
  empirically**: with the same input, the same seed, and the same
  weights, Post-LN clamps every output token to L2 norm
  `sqrt(d_model)` exactly, while Pre-LN preserves the input's L2
  norm (within 50%). This is the structural reason gradient flow
  is so different between the two formulations.
- `test_encoder_layer_residual_passes_zero_sublayer_pre_ln` is a
  dedicated check that the residual *identity path* survives even
  when both sub-layers' last projection is zeroed out. Forces the
  Pre-LN block to be the identity exactly.

**Things deliberately NOT done this slice (and where they'll happen):**

- Attention-weight dropout (Place 3 from section 6) and FFN-internal
  dropout (Place 4) are still not implemented. Neither is
  paper-text. Documented as future work; we did not even add TODO
  comments in `layers.py` because the relevant locations are inside
  the existing MHA / FFN classes (already complete) and adding the
  comment now would be cosmetic.
- The FFN-side `dropout` placement *between* `W_1` and `W_2`
  internally would require modifying `FeedForward` (already
  finished). If we ever decide to add it, we'll do it as a
  constructor flag on `FeedForward` itself, mirroring `pre_ln`.
- A full *numerical* hand-trace of the encoder (in the spirit of
  `test_mha_section5_trace`) was deferred. Section 9 has only a
  shape trace. The reasoning from slice 1 stands: a full numerical
  trace requires fixing weights for every sub-layer, which is much
  more useful as a regression fixture than as documentation.

**On to Step 7 (Decoder).** The decoder reuses the same wrapper
pattern but adds a third sub-layer (cross-attention into the
encoder's output) and replaces self-attention with **masked**
self-attention. Both extensions are mechanical; the conceptual
content of Step 7 is "what does the cross-attention mask actually
do" and "how does the autoregressive constraint interact with
training-time parallelism" (teacher forcing).
