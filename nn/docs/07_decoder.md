# Step 7: Decoder

> **Status**: Complete
> **Paper section**: Section 3.1 (Decoder), Section 3.2.3 (Applications of attention -- masked & cross), Section 3.4 (Embeddings and Softmax)
> **Depends on**: Steps 1, 2, 4, 5, 6 (LayerNorm, Positional Encoding, MHA with masking, FFN, Encoder)

[< Encoder](06_encoder.md) | [Overview](00_overview.md) | [Next: Training >](08_training.md)

---

## 1. What we build

The decoder consumes two inputs and produces a probability distribution
over vocabulary tokens at every output position:

- **Input 1**: the (right-shifted) target token IDs `(B, n_q)`.
- **Input 2**: the encoder's output `(B, n_k, d_model)`. This is what the
  encoder built in Step 6.
- **Output**: logits `(B, n_q, vocab_size)`, ready to feed into a
  cross-entropy loss against the unshifted target.

Two object levels (mirroring the encoder):

- **`DecoderLayer`** -- one repeating block. Internally **three**
  sub-layers (vs the encoder's two): masked self-attention,
  cross-attention into the encoder output, and FFN. Each is wrapped
  in residual + LayerNorm.
- **`Decoder`** -- the full pipeline. Target embedding -> scale by
  `sqrt(d_model)` -> add positional encoding -> dropout -> stack of
  `N` `DecoderLayer`s -> output linear projection -> logits.

### One layer at a glance (Post-LN, paper-faithful)

```
x_dec_in  (target hidden state, (B, n_q, d_model))
   |
   +-----> MaskedSelfAttn(x, x, x, causal_mask) -> Dropout -+
   |                                                       |
   +-------------- + (residual) ---------------------------+
                                                           |
                                  LayerNorm <--------------+
                                       |
                                       v
                                    x_after_self
                                       |
   enc_out (encoder output) ------>   |
                                       v
   +-----> CrossAttn(Q=x_after_self, K=enc_out, V=enc_out) -> Dropout -+
   |                                                                  |
   +----------- + (residual on x_after_self) ------------------------ +
                                                                      |
                                       LayerNorm <--------------------+
                                            |
                                            v
                                         x_after_cross
                                            |
   +-----> FFN(x_after_cross) -> Dropout -+
   |                                      |
   +----- + (residual) ------------------+
                                          |
                       LayerNorm <--------+
                            |
                            v
                          x_dec_out
```

In one block of math (Post-LN form):

```
x_after_self  = LayerNorm(x_in        + Dropout(MaskedSelfAttn(x_in, x_in, x_in, causal_mask)))
x_after_cross = LayerNorm(x_after_self + Dropout(CrossAttn(x_after_self, enc_out, enc_out)))
x_dec_out     = LayerNorm(x_after_cross+ Dropout(FFN(x_after_cross)))
```

Note the second sub-layer's residual is on `x_after_self`, not on
`x_in`. Each wrapper is independent: input -> sub-layer -> dropout ->
add to input -> normalise.

### Full decoder at a glance

```
target_ids       --[Embedding]-->  E       (B, n_q, d_model)
E                --[* sqrt(d_model)]-->  E_scaled
E_scaled         --[+ PE]-->        E_pos
E_pos            --[Dropout]-->     x_0    (paper section 5.4)
x_0              --[DecoderLayer x N (with enc_out)]-->  x_N
x_N              --[OutputProj (W_out = embed.weight^T)]-->  logits  (B, n_q, vocab_size)
```

`enc_out` is threaded into every `DecoderLayer` (each one cross-
attends into the encoder output independently). It does NOT change
between layers; the same encoder output is used by all `N` decoder
layers.

`OutputProj` is, by default, **tied** to the input embedding (paper
section 3.4): `W_out = embedding.weight^T`. A flag exposes the
untied alternative.

---

## 2. Three sub-layers and the wrapper pattern

The decoder layer reuses the wrapper pattern from section 3 of
`06_encoder.md`:

> Take a sub-layer that maps `(B, n, d_model) -> (B, n, d_model)`,
> wrap it in residual connection and LayerNorm.

The only structural difference from the encoder: **three** sub-layers
stacked instead of two. Each sub-layer plays a distinct semantic role:

| # | Sub-layer | Role | What it answers |
|---|-----------|------|-----------------|
| 1 | Masked self-attention | Self-context | "What have I generated *so far*?" |
| 2 | Cross-attention       | Source-context | "What does the *source* say that's relevant here?" |
| 3 | FFN                   | Per-position integration | "Given both contexts, what's my next representation?" |

The encoder needs only sub-layer 1 (self-context) + sub-layer 3
(integration) and so has two sub-layers. The decoder additionally
needs sub-layer 2 (source-context, the cross-attention into the
encoder), giving three. This is the conceptual reason the decoder is
"thicker" per layer than the encoder.

Wrapping each sub-layer is identical to the encoder:

- **Post-LN** (paper, default): `y = LayerNorm(x + Dropout(Sublayer(x)))`
- **Pre-LN**  (modern, opt-in): `y = x + Dropout(Sublayer(LayerNorm(x)))`

Section 5 of `06_encoder.md` has the full Pre-LN vs Post-LN trade-off
discussion. The decoder follows the same default (Post-LN) and the
same `pre_ln=False` flag.

The shape invariant `(B, n_q, d_model) -> (B, n_q, d_model)` holds for
all three sub-layers, so the residuals type-check and the layers
stack cleanly.

---

## 3. The causal mask

The first decoder sub-layer is **self-attention with a causal mask**.
This is the single most important addition over the encoder.

### Why we need it

The decoder is autoregressive: at inference time, position `i`'s
output is generated *before* position `i+1` exists. Position `i`'s
representation can only depend on positions `0..i`.

But during **training** we want all positions processed in parallel
(it would be agonising to train one position at a time). Teacher
forcing (section 5) gives us the entire target sequence at once -- so
we feed it all in and process every position simultaneously.

Without any masking, position `i`'s self-attention would attend to
positions `j > i` -- it would *see future tokens*. The model would
trivially learn "to predict the token at position `i+1`, just look at
the input at position `i+1`". The training task collapses to a copy.
At inference, where future tokens don't exist, the model would fail
catastrophically.

The causal mask is what makes parallel training compatible with
autoregressive inference.

### How it works

A lower-triangular boolean matrix `M` of shape `(n_q, n_q)`. Convention
in our codebase (see `ScaledDotProductAttention.forward`): `True` =
allowed, `False` = blocked.

```
positions:  0   1   2   3   4
        0:  T   F   F   F   F
        1:  T   T   F   F   F
        2:  T   T   T   F   F
        3:  T   T   T   T   F
        4:  T   T   T   T   T
```

In NumPy: `np.tril(np.ones((n_q, n_q), dtype=bool))`.

The mask is applied to the *scores*, BEFORE softmax. In our SDPA
implementation:

```python
scores = Q @ K^T / sqrt(d_k)
if mask is not None:
    scores = np.where(mask, scores, -np.inf)
weights = softmax(scores)
```

`np.where` replaces blocked entries with `-inf`. Then softmax sends
`-inf` to `0.0` (because `exp(-inf) = 0`). Result: every blocked
attention weight is exactly zero. Position `i` is a weighted sum over
only positions `0..i`.

This works *per-row* of the attention weight matrix: each query
position's row sums to 1 over only the allowed key positions.
Mathematically, `softmax` is row-shift-invariant (Step 4 section 8),
so subtracting `max(scores)` for numerical stability still gives the
same weights -- the mask survives the stability shift.

### "Causal in attention" != "causal in operation"

The forward pass *itself* still runs in parallel for every position.
We compute Q, K, V for all `n_q` positions in one big matmul. The
causal property is a constraint on **what attends to what**, not on
the order of operations. Information flows forward in time within
attention; the implementation flows in parallel across time.

This is the entire trick. Autoregressivity is enforced by the *mask*,
not by the *control flow*.

### Cross-attention is NOT causally masked

The asymmetry is deliberate and important:

- **Self-attention in the decoder**: causally masked. A target token
  cannot peek at future target tokens.
- **Cross-attention** (decoder -> encoder): no mask. The decoder is
  allowed to look at the *entire* source sequence, freely.

Why? Because the source sequence is fully observed from the very
beginning of decoding. Looking at "future" source positions is not
cheating -- the model is allowed to see all of the input it was
given. There is no "future" on the source side; the source exists in
its entirety before the decoder starts.

This asymmetry (causal on self, full on cross) is the geometric heart
of the encoder-decoder architecture. It captures a key fact:

> Past target tokens *constrain* future target tokens (autoregressive),
> but the entire source sequence simultaneously *informs* every
> target position (cross-attention).

### Padding masks (deferred)

For batched training with variable-length sequences, we'd also need a
*padding mask* -- a boolean over key positions indicating which are
real vs padding. The padding mask combines (logical AND) with the
causal mask via `mask_combined = causal_mask & ~is_padding`.

Our toy sort task has fixed-length sequences and no padding, so we
**defer padding masks entirely**. Slice 2 will pass `mask=None` for
cross-attention and a pure causal mask for self-attention. Padding
support is straightforward to add later (it just changes the mask
construction) and is documented as a future extension in Section 12.

---

## 4. Cross-attention

The middle decoder sub-layer. Same `MultiHeadAttention` machinery
from Step 4, with three differences in how it is *called*:

1. **`Q_in` is the decoder's hidden state** (output of the previous
   sub-layer in the same decoder layer).
2. **`K_in = V_in = enc_out`**, the encoder's output. Both K and V
   come from the same source, but they go through *different*
   projection matrices `W^K, W^V` -- so they are no longer identical
   after projection.
3. **No causal mask**.

### The shape change: rectangular attention

Recall MHA's input shapes (Step 4):

```
Q_in: (..., n_q, d_model)
K_in: (..., n_k, d_model)
V_in: (..., n_k, d_model)
```

In encoder self-attention, `n_q == n_k == n` because all three come
from the same sequence. **In cross-attention they are independent**:

```
Q_in (from decoder):   (B, n_q, d_model)   -- target sequence length
K_in (from encoder):   (B, n_k, d_model)   -- source sequence length
V_in (from encoder):   (B, n_k, d_model)
```

For our toy sort task `n_q == n_k == 5`. For a translation task, a
4-word English source might decode into a 6-word French target;
`n_k = 4`, `n_q = 6`. Cross-attention handles this naturally because
`Q @ K^T` is rectangular `(n_q, n_k)`.

The attention weight matrix in cross-attention has shape
`(B, h, n_q, n_k)`: each *target* position has a softmax distribution
over *source* positions. Test
`test_sdpa_cross_attention_rectangular_weights` already exercises
this property at the SDPA level; at the MHA level, no new code is
needed -- we just feed the right tensors.

### The semantic meaning: soft alignment

For each decoder position `i` and each source position `j`, the
cross-attention weight `A[i, j]` answers:

> "How relevant is source position `j` to whatever I (the decoder)
> need to do at target position `i`?"

This is a **soft alignment** between target and source. In a
translation model trained on English -> French, you can plot
`A[i, j]` as a heatmap and see which English words each French word
attended to most strongly. The original paper's Figure 4 shows
exactly this for "le" attending to "the" while preserving phrase
structure, etc.

For our sort task, the expected pattern is sharper: target position
`i` (which holds the `i`-th-smallest number in the sorted output)
should attend most strongly to whichever source position holds that
number in the input. Whether the model actually learns this clean
alignment is something we will be able to inspect via
`mha.last_weights` (already exposed by our MHA implementation).

### How the decoder learns where to look

The query projection `W^Q` (in the cross-attention MHA) maps the
decoder hidden state into "search vectors". The key projection `W^K`
(same MHA, but acting on the encoder output) maps source positions
into "feature vectors". Their dot product determines which source
positions a given target position will attend to.

The two projections are learned end-to-end by SGD against the
cross-entropy loss. There is no explicit alignment supervision; the
alignment emerges because attending to the right source positions
makes the next-token prediction easier.

### Per-layer parameter cost

Each decoder layer has *two* MHA blocks (self + cross), each with
its own `W^Q, W^K, W^V, W^O`. So per decoder layer:

- 8 attention projection matrices (4 self + 4 cross)
- 2 FFN matrices (`W_1, W_2`)
- 3 LayerNorms (one per wrapper)
- 3 Dropouts (one per wrapper, no parameters)

Compared to an encoder layer's 4 attention + 2 FFN + 2 LN, the
decoder layer has roughly **1.5x the parameters** at the same
`d_model, h, d_ff`. Section 9 makes this concrete.

---

## 5. Teacher forcing

The training-time idea that resolves the apparent paradox "the
decoder is autoregressive, but training must be parallel".

### The naive picture

If we trained the decoder one position at a time:

1. Sample one position in the target.
2. Run decoder up to that position.
3. Compute loss for that one position.
4. Backprop, update.
5. Repeat.

This would work but be ~`n_q` times slower than necessary. We want to
process all positions of a target sequence in *one* forward + one
backward pass.

### The trick

The target tokens at positions `0..i-1` are exactly what position `i`
needs as decoder input (under autoregressive semantics). At training
time we *have* the ground-truth target. So we feed it as decoder
input.

Concretely, if the target sequence is `[a, b, c, d]` and `<BOS>` is a
special "begin of sequence" token, then:

```
decoder INPUT:     [<BOS>, a, b, c]      (right-shifted target)
decoder OUTPUT:    [a,    b, c, d]      (predict the next token at every position)
```

Position 0 sees `<BOS>` and is asked to predict `a`. Position 1 sees
`<BOS>, a` (with the causal mask hiding `b, c`) and is asked to
predict `b`. Etc.

The causal mask makes this safe: position `i`'s self-attention can
only see positions `0..i` of the input, which are exactly the
ground-truth target tokens up to (but not including) the one we're
predicting. Position `i` cannot peek at the answer.

This is **teacher forcing**: the decoder is forced to be conditioned
on the ground-truth previous tokens (the "teacher's" outputs) rather
than on its own predictions.

### Concrete trace for our sort task

Source = `[7, 3, 11, 5, 9]` (unsorted).
Target (sorted) = `[3, 5, 7, 9, 11]`.

Encoder input: `[7, 3, 11, 5, 9]`. Length 5.
Encoder output: `(1, 5, 32)`.

Decoder input (right-shifted target): `[<BOS>, 3, 5, 7, 9]`. Length 5.
Decoder target (what we predict at each position): `[3, 5, 7, 9, 11]`.

In one forward pass, the decoder produces logits of shape
`(1, 5, vocab_size)`. We compute cross-entropy at each of the 5
positions against the target. All 5 losses are summed (or averaged)
and backprop runs once.

### Exposure bias (a known limitation)

At inference time, position `i` does NOT see the ground-truth
previous tokens -- it sees the model's *own* previous predictions,
which may be wrong. This distribution mismatch between training
inputs (always correct) and inference inputs (potentially wrong) is
called **exposure bias**.

Mitigations exist (scheduled sampling, professor forcing, RL
fine-tuning, beam search at inference), but all are out of scope for
this build. We accept the bias and rely on the model being good
enough that early errors don't compound severely. For a deterministic
task like sorting, the gap should be small.

### Note on the BOS / start-of-decoder token

The decoder needs *some* input at position 0 -- there is no "previous
target token" for the first position. By convention, we use a
special `<BOS>` (begin-of-sequence) token reserved in the vocabulary.

Our toy build's sort task has an extra slot in `vocab_size` for this
purpose. Step 8 (Training) will pin down the exact convention.

---

## 6. Output projection and weight tying

After `N` decoder layers, the decoder's output is `(B, n_q, d_model)`.
To turn this into vocabulary logits we apply a linear projection:

```
logits = decoder_output @ W_out         # (B, n_q, d_model) @ (d_model, vocab_size)
```

Resulting shape: `(B, n_q, vocab_size)`.

The paper does NOT use a bias on this projection (section 3.4). We
follow that convention -- saves `vocab_size` parameters.

### Weight tying with the input embedding

Paper section 3.4, exact words:

> "We use the same weight matrix between the two embedding layers
> [encoder + decoder] and the pre-softmax linear transformation."

Three things share one matrix:

1. The encoder's input embedding (Step 6).
2. The decoder's input embedding.
3. The decoder's output projection's `W_out`.

The first two have shape `(vocab_size, d_model)` (token-id index
lookup). The third needs `(d_model, vocab_size)` -- which is just the
*transpose* of the first two.

So a single matrix `E ∈ R^(vocab_size × d_model)` plays three roles:

```
encoder.embedding[token_ids]    = E[token_ids]              # row lookup
decoder.embedding[token_ids]    = E[token_ids]              # row lookup
decoder.output_projection(x)    = x @ E.T                   # transpose multiply
```

### Why tie

- **Parameter savings**: `vocab_size * d_model` per shared role. For
  our toy build (`vocab=16, d_model=32`) the saving is small (`512`
  per role, `1024` total compared to fully untied). For real models
  (`vocab=50000, d_model=4096`) the saving is `~200M` parameters per
  role -- an enormous fraction of total parameter count.
- **Structural prior**: forces the same vector to represent a token
  whether it is being looked up at input or compared-against at
  output. Empirically improves perplexity; see Press & Wolf 2017
  ("Using the Output Embedding to Improve Language Models") and
  Inan et al. 2017 ("Tying Word Vectors and Word Classifiers").

### When (and why) modern models *don't* tie

Llama-3 and several other large models have started **un-tying** at
multi-billion-parameter scale. The argument is that at scale the
parameter savings are tiny relative to the model size, and the
constraint that "input vector == output vector" is mildly
suboptimal. For models below about 7B params, tying remains the
default.

Our toy build is far below that threshold; we tie by default.

### Decision for our build

`Decoder` constructor will take a `tie_embeddings: bool = True` flag.
When true:

- The decoder's embedding object's `.weight` is the *same* NumPy
  array as the output projection's `W_out.T` (more precisely: the
  output projection is implemented as `x @ embedding.weight.T`, with
  no separate `W_out` storage).
- Both the embedding side (token-lookup) and the output side
  (logits) accumulate gradients into the **same** `.grad_weight`
  array.
- We do NOT use a bias on the output projection.

When false:

- The output projection has its own `Linear(d_model, vocab_size,
  bias=False)`. Two extra `vocab_size * d_model` floats and a
  separate gradient buffer.
- This is the ablation handle for slice 2 tests.

### Three-way sharing across encoder and decoder

The paper's full sharing involves the *encoder embedding* too. In a
translation setting with separate source/target vocabularies (e.g.,
English source, French target), this would NOT be possible -- two
different vocabs need two different matrices. The paper trains on
WMT En->De/Fr where they used a *shared* sub-word vocabulary
specifically so that this sharing was possible.

Our build's sort task has the same vocabulary on the input and
output sides, so three-way sharing is natural. Slice 2 will provide
two architectures:

1. **Single shared embedding** (paper-faithful, default for our
   sort task): one `Embedding` object passed to both encoder and
   decoder.
2. **Independent embeddings** (translation-style): encoder builds
   its own embedding, decoder builds its own. Each tied separately
   with its own output side (or untied if `tie_embeddings=False`).

The slice 2 design will likely have the shared embedding constructed
*outside* the encoder and decoder and passed in -- a lightweight
"`Transformer`" wrapper class that owns the shared embedding and
plumbs it into both halves.

---

## 7. Pre-LN vs Post-LN

Same trade-off as the encoder. Section 5 of `06_encoder.md` is the
full discussion; the short version:

- **Post-LN** (`pre_ln=False`, default): `y = LayerNorm(x + Dropout(Sublayer(x)))`. Paper-faithful.
- **Pre-LN** (`pre_ln=True`): `y = x + Dropout(Sublayer(LayerNorm(x)))`. Modern, more stable to train.

One subtlety specific to the decoder: **with three sub-layers
instead of two, the un-normalised residual stream in Pre-LN
accumulates contributions from one extra sub-layer per decoder
block**. So Pre-LN decoder magnitudes grow slightly faster per layer
than Pre-LN encoder magnitudes at matching `N`. Not a real concern
at our `N=2`; relevant at `N=24+` and the kind of detail that
shipped real-world ablations chase.

The `pre_ln` flag will live on both `DecoderLayer` and `Decoder`
constructors and propagate the same way it does in the encoder.

---

## 8. Autoregressive inference at a glance

Full treatment in Step 9. Quick preview here so the training-time
picture in section 5 is properly in context:

```python
def generate(source_tokens, max_len=10):
    enc_out = encoder(source_tokens)            # encode once

    target = [BOS_ID]                           # start the output
    for _ in range(max_len):
        target_arr = np.array([target])         # (1, current_len)
        logits = decoder(target_arr, enc_out)   # (1, current_len, vocab_size)
        next_token = argmax(logits[0, -1])      # only the LAST position matters
        if next_token == EOS_ID:
            break
        target.append(int(next_token))
    return target
```

Three things to notice:

1. **Encode once**. The encoder's output does not change during
   generation; we cache it.
2. **Decode loop**. Each iteration of the loop re-runs the *whole*
   decoder forward pass on the growing target. Without caching,
   that is `O(n_q^2)` per step.
3. **Only the last position matters**. The decoder produces logits
   at every position, but during generation we throw away all but
   the last -- the others were predicting tokens we already have.

### KV caching teaser

Production transformers cache the K and V tensors of all previous
positions across iterations. Each step then only computes attention
between the *new* query position and *all cached* keys/values. This
turns generation from `O(n_q^2)` per token into `O(n_q)` per token.

Our toy build will skip KV caching for clarity in Step 9 and accept
the quadratic cost. It is a layer of optimisation on top of the
correct implementation, not a structural change.

---

## 9. Worked shape walk-through

Concrete trace through one decoder layer + the output projection,
plus the parameter count for our build.

### Config

```
B        = 2          (batch size)
n_q      = 3          (target sequence length, tiny example)
n_k      = 4          (source sequence length -- chosen != n_q to make
                       cross-attention shapes visible)
vocab    = 16         (toy vocab)
d_model  = 32
N        = 2
h        = 4          (so d_k = 8)
d_ff     = 64
p_drop   = 0.1
```

### Inputs

```
target_ids = [[3, 7, 11], [5, 9, 2]]            shape (2, 3)
enc_out    = encoder.forward(source_ids)        shape (2, 4, 32)
                                                  -- from Step 6's encoder
```

### Step-by-step shapes

| # | Step | Output shape | Notes |
|---|------|--------------|-------|
| 1 | `embedding[target_ids]` | `(2, 3, 32)` | row lookup |
| 2 | `* sqrt(d_model)` | `(2, 3, 32)` | scalar broadcast |
| 3 | `+ PE[:3, :]` | `(2, 3, 32)` | broadcasts over batch |
| 4 | `dropout(p=0.1)` | `(2, 3, 32)` | mask same shape |
|   | -- enter Layer 1 sub-layer 1 (masked self-attn) -- | | |
| 5 | `MHA(x, x, x, causal_mask=(3,3) lower-tri)` | `(2, 3, 32)` | internally splits to `(2, 4, 3, 8)`; weights `(2, 4, 3, 3)` are lower-triangular |
| 6 | Dropout | `(2, 3, 32)` | |
| 7 | + residual (x_in from step 4) | `(2, 3, 32)` | |
| 8 | LayerNorm 1 | `(2, 3, 32)` | end of sub-layer 1 |
|   | -- sub-layer 2 (cross-attn) -- | | |
| 9 | `MHA(Q=step8, K=enc_out, V=enc_out, mask=None)` | `(2, 3, 32)` | Q has `(2, 4, 3, 8)`, K and V have `(2, 4, 4, 8)`; weights `(2, 4, 3, 4)` -- rectangular |
| 10| Dropout | `(2, 3, 32)` | |
| 11| + residual (step 8 output) | `(2, 3, 32)` | |
| 12| LayerNorm 2 | `(2, 3, 32)` | end of sub-layer 2 |
|   | -- sub-layer 3 (FFN) -- | | |
| 13| FFN | `(2, 3, 32)` | internal `(2, 3, 64)` after `W_1` |
| 14| Dropout | `(2, 3, 32)` | |
| 15| + residual (step 12 output) | `(2, 3, 32)` | |
| 16| LayerNorm 3 | `(2, 3, 32)` | end of layer 1 |
|   | -- enter Layer 2 (steps 5-16 again, with same enc_out) -- | | |
| 17-28| Layer 2 mirror of layer 1 | `(2, 3, 32)` | |
|   | -- exit decoder stack -- | | |
| 29| Output projection: `x @ embedding.weight.T` | `(2, 3, 16)` | logits over vocab |

The `enc_out` tensor is **read** at step 9 in every layer, but never
modified. The decoder layers form a left-to-right (in this table
top-to-bottom) chain on the target side; the encoder's output is a
*broadcast* read across the whole stack.

### Parameter count for our build

Decoder-side parameters, per sub-component, per decoder layer:

| Component                | Weights              | Biases       | Per-layer total |
|--------------------------|---------------------:|-------------:|----------------:|
| Self-attn `W^Q, W^K, W^V`| `3 . 32 . 32 = 3072` | `3 . 32 = 96`| `3168`          |
| Self-attn `W^O`          | `1024`               | `32`         | `1056`          |
| LayerNorm 1              | --                   | `64`         | `64`            |
| Cross-attn `W^Q, W^K, W^V`| `3072`              | `96`         | `3168`          |
| Cross-attn `W^O`         | `1024`               | `32`         | `1056`          |
| LayerNorm 2              | --                   | `64`         | `64`            |
| FFN `W_1`                | `2048`               | `64`         | `2112`          |
| FFN `W_2`                | `2048`               | `32`         | `2080`          |
| LayerNorm 3              | --                   | `64`         | `64`            |
| **Per decoder layer**    | **`12288`**          | **`544`**    | **`12832`**     |

Two layers: `2 . 12832 = 25664`.

Plus the decoder-side input pipeline:

| Component                      | Count |
|--------------------------------|------:|
| Decoder embedding (tied with encoder embedding & output proj) | `0` *additional* (shared with encoder) |
| Positional encoding            | `0` (sinusoidal, no learnable params) |
| Input dropout                  | `0` |
| Output projection `W_out`      | `0` *additional* (tied to embedding) |
| Output projection bias         | `0` (paper convention: no bias) |

**Decoder-only learnable parameters: `25664`**.

When combined with the encoder's `17600` from Step 6 (which already
includes the shared embedding's `512`), the full Transformer has
`17600 + 25664 = 43264` parameters in the tied configuration.

Without weight tying (i.e. an extra `Linear(32, 16)` for the output
projection, no bias):

```
extra = vocab_size * d_model = 16 * 32 = 512
total_untied = 43264 + 512 = 43776
```

The relative impact of tying at our toy scale: `~1.2%` savings. At
real scale (`vocab=50000`, `d_model=4096`, `N=24`), tying saves
`~200M` params -- a few percent of the model.

### Sanity invariant

Every line in the table above is `(2, 3, 32)` until step 29, where
it becomes `(2, 3, 16)`. The decoder body is a sequence of
`(B, n_q, d_model) -> (B, n_q, d_model)` transforms; the output
projection is the *only* layer that changes the last dim. The
internal transient shapes -- `(2, 4, 3, 8)` for self-attention,
`(2, 4, 3, 4)` for cross-attention weights, `(2, 3, 64)` for the FFN
hidden -- are all sub-layer-internal and restored on exit, just like
in the encoder.

---

## 10. Forward pass in code

Three new classes plus a small change to one existing class. All in
`numpy_transformer/layers.py`.

### `DecoderLayer`

```python
class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, d_ff, p_drop=0.1, pre_ln=False):
        self.mha_self  = MultiHeadAttention(d_model, num_heads)
        self.dropout_1 = Dropout(p_drop)
        self.ln_1      = LayerNorm(d_model)

        self.mha_cross = MultiHeadAttention(d_model, num_heads)
        self.dropout_2 = Dropout(p_drop)
        self.ln_2      = LayerNorm(d_model)

        self.ffn       = FeedForward(d_model, d_ff)
        self.dropout_3 = Dropout(p_drop)
        self.ln_3      = LayerNorm(d_model)
```

`forward(x, enc_out, causal_mask)` dispatches to either
`_forward_post_ln` (paper, default) or `_forward_pre_ln` based on
the `pre_ln` flag. Post-LN body, faithful to section 1 of this doc:

```python
def _forward_post_ln(self, x, enc_out, causal_mask):
    h_self    = self.mha_self.forward(x, x, x, mask=causal_mask)
    h_self_d  = self.dropout_1.forward(h_self)
    y1        = self.ln_1.forward(x + h_self_d)

    h_cross   = self.mha_cross.forward(y1, enc_out, enc_out, mask=None)
    h_cross_d = self.dropout_2.forward(h_cross)
    y2        = self.ln_2.forward(y1 + h_cross_d)

    h_ffn     = self.ffn.forward(y2)
    h_ffn_d   = self.dropout_3.forward(h_ffn)
    y3        = self.ln_3.forward(y2 + h_ffn_d)
    return y3
```

The Pre-LN variant only re-arranges where `LayerNorm` sits relative
to the residual additions; see the matching `_forward_pre_ln` in the
code.

The DecoderLayer does NOT cache anything itself -- each sub-layer
already caches what it needs for its own backward. The
`DecoderLayer` just orchestrates the call sequence.

### `Decoder`

```python
class Decoder(Layer):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 p_drop=0.1, max_len=5000, pre_ln=False,
                 shared_embedding=None, tie_embeddings=True):
        if shared_embedding is None:
            self.embedding = Embedding(vocab_size, d_model)
            self._owns_embedding = True
        else:
            self.embedding = shared_embedding
            self._owns_embedding = False

        self.pos_enc       = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout_input = Dropout(p_drop)
        self.layers        = [DecoderLayer(...) for _ in range(num_layers)]

        if tie_embeddings:
            self.output_projection = TiedLinear(self.embedding)
        else:
            self.output_projection = Linear(d_model, vocab_size)

        self._embed_scale     = np.sqrt(d_model)
        self._cache_token_ids = None
```

The `forward(target_ids, enc_out)` body:

```python
def forward(self, target_ids, enc_out):
    n_q = target_ids.shape[-1]
    self._cache_token_ids = target_ids

    e        = self.embedding.forward(target_ids)
    e_scaled = e * self._embed_scale
    e_pe     = self.pos_enc.forward(e_scaled)
    x        = self.dropout_input.forward(e_pe)

    causal_mask = np.tril(np.ones((n_q, n_q), dtype=bool))
    self._cache['enc_out_shape'] = enc_out.shape

    for layer in self.layers:
        x = layer.forward(x, enc_out, causal_mask)

    logits = self.output_projection.forward(x)
    return logits
```

Three things worth pausing on:

1. **Why we cache `target_ids` ourselves rather than relying on
   `embedding._cache['token_ids']`.** When the embedding is *shared*
   (with an encoder, via the `Transformer` wrapper), the encoder's
   forward populates the embedding's cache first; the decoder's
   forward then *overwrites* it. The encoder would later see the
   wrong IDs in its own `backward`. We dodge this by caching our
   own `target_ids` and passing them explicitly to
   `embedding.backward(grad_e, token_ids=...)`. The `Encoder` does
   the same.

2. **`np.tril` is rebuilt every forward call.** It is `O(n_q^2)`
   memory and arithmetic, dwarfed by anything else in the layer.
   We could cache by `n_q` if profiling ever surfaced it; not worth
   the complexity now.

3. **`self._cache['enc_out_shape']`** is the only thing the
   `Decoder` itself needs for backward -- so `Decoder.backward`
   knows what shape `grad_enc_out` accumulator to allocate. Each
   `DecoderLayer` then contributes to that accumulator.

### `TiedLinear`

A small, self-contained class for the tied output projection
(section 6).

```python
class TiedLinear(Layer):
    def __init__(self, shared_owner, weight_attr='weight',
                 grad_attr='grad_weight'):
        self.shared_owner = shared_owner
        self.weight_attr  = weight_attr
        self.grad_attr    = grad_attr

    def forward(self, x):
        W = getattr(self.shared_owner, self.weight_attr)
        return x @ W.T          # (..., d_model) @ (d_model, vocab) -> (..., vocab)

    def backward(self, grad_output):
        x = self._cache['x']
        W = getattr(self.shared_owner, self.weight_attr)
        grad_input = grad_output @ W

        x_2d   = x.reshape(-1, x.shape[-1])
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_W_contribution = grad_2d.T @ x_2d   # shape: (vocab, d_model)

        existing = getattr(self.shared_owner, self.grad_attr)
        if existing is None:
            existing = np.zeros_like(W)
            setattr(self.shared_owner, self.grad_attr, existing)
        existing += grad_W_contribution

        return grad_input

    def params_and_grads(self):
        return []        # owner exposes its own weight; do not double-count
```

Bias-free by construction (paper section 3.4). The accumulation
into the shared owner's gradient buffer is what makes tied training
work; the `Embedding` class was updated to match (next section).

### Embedding contract change

To make tied weights work cleanly, `Embedding.backward` now
**accumulates** rather than resets. The new contract:

- First call after construction (or after `grad_weight = None`):
  initializes a fresh zero buffer, then accumulates into it.
- Subsequent calls: accumulate into the existing buffer.
- A new optional argument `token_ids=None`: if provided, used
  instead of the cached value. This is what `Encoder.backward` and
  `Decoder.backward` now pass, defending against the shared-cache
  race described above.

This is the PyTorch convention. For training loops, the caller
zeros gradients between iterations
(`embedding.grad_weight = None`).

The existing `test_embedding` test was updated to:

- Insert an explicit `layer.grad_weight = None` between two
  unrelated backward calls, so the within-call duplicate test still
  reads `1.5` not `2.5`.
- Add a *cross-call accumulation* assertion verifying the new
  contract (`2.5` after no reset).
- Add an *explicit `token_ids` override* assertion verifying the
  shared-cache defense.

### `Transformer`

The wrapper class that owns the shared `Embedding`:

```python
class Transformer(Layer):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_encoder_layers, num_decoder_layers,
                 p_drop=0.1, max_len=5000,
                 pre_ln=False, tie_embeddings=True):
        self.embedding = Embedding(vocab_size, d_model)
        self.encoder   = Encoder(
            ..., shared_embedding=self.embedding,
        )
        self.decoder   = Decoder(
            ..., shared_embedding=self.embedding,
            tie_embeddings=tie_embeddings,
        )

    def forward(self, source_ids, target_ids):
        enc_out = self.encoder.forward(source_ids)
        logits  = self.decoder.forward(target_ids, enc_out)
        return logits

    def params_and_grads(self):
        # ONLY the wrapper exposes the shared embedding's params;
        # encoder and decoder do NOT.
        out = [(p, g, f'embedding.{n}')
               for (p, g, n) in self.embedding.params_and_grads()]
        out.extend((p, g, f'encoder.{n}')
                   for (p, g, n) in self.encoder.params_and_grads())
        out.extend((p, g, f'decoder.{n}')
                   for (p, g, n) in self.decoder.params_and_grads())
        return out
```

The `_owns_embedding` flag inside `Encoder` and `Decoder` controls
whether each one returns the embedding's parameter from its own
`params_and_grads()`. When the `Transformer` owns it, both inner
modules return `False` for that flag and skip the embedding row, so
the shared matrix appears exactly once in the optimizer's parameter
list.

---

## 11. Backward pass

No new derivative work. Every component already had its own backward
in earlier steps; the decoder is composition. But the residual-sum
rule, the self-attention-sum rule, and a NEW cross-attention-split
rule all matter and need to be wired correctly.

### Three rules

1. **Residual sum (encoder section 11)**: gradient flowing into
   `y = a + b` flows *unchanged* to both `a` and `b`. So
   `grad_a = grad_y`, `grad_b = grad_y`.

2. **Self-attention input sum (encoder section 11)**: in
   self-attention, `Q_in = K_in = V_in = x`. `MHA.backward` returns
   `(grad_Q_in, grad_K_in, grad_V_in)`. The actual gradient into
   `x` is the sum: `grad_x = grad_Q_in + grad_K_in + grad_V_in`.

3. **Cross-attention input split (NEW for decoder)**: in
   cross-attention, `Q_in = decoder_hidden_state`, `K_in = V_in =
   enc_out`. `MHA.backward` still returns
   `(grad_Q_in, grad_K_in, grad_V_in)`, but now they go to
   *different upstream tensors*:
   - `grad_decoder_hidden_state += grad_Q_in`
   - `grad_enc_out_for_this_layer = grad_K_in + grad_V_in`

   Both K and V are derived from the *same* `enc_out`, so their
   gradients sum (rule 2 again, but applied to enc_out instead of
   the decoder hidden state).

### `DecoderLayer.backward` (Post-LN)

```python
def _backward_post_ln(self, grad_output):
    g_sum3                = self.ln_3.backward(grad_output)
    grad_h_ffn            = self.dropout_3.backward(g_sum3)
    grad_y2_via_ffn       = self.ffn.backward(grad_h_ffn)
    grad_y2               = g_sum3 + grad_y2_via_ffn         # rule 1

    g_sum2                = self.ln_2.backward(grad_y2)
    grad_h_cross          = self.dropout_2.backward(g_sum2)
    gQ_c, gK_c, gV_c      = self.mha_cross.backward(grad_h_cross)
    grad_y1_via_cross     = gQ_c                             # rule 3 (Q side)
    grad_enc_out          = gK_c + gV_c                      # rule 3 (K+V side)
    grad_y1               = g_sum2 + grad_y1_via_cross       # rule 1

    g_sum1                = self.ln_1.backward(grad_y1)
    grad_h_self           = self.dropout_1.backward(g_sum1)
    gQ_s, gK_s, gV_s      = self.mha_self.backward(grad_h_self)
    grad_x_via_self       = gQ_s + gK_s + gV_s               # rule 2
    grad_x                = g_sum1 + grad_x_via_self         # rule 1

    return grad_x, grad_enc_out
```

The Pre-LN variant interleaves `LayerNorm.backward` differently but
applies the same three rules. Returns the same 2-tuple shape:
`(grad_x, grad_enc_out)`. Code is in
`DecoderLayer._backward_pre_ln`.

### `Decoder.backward`

Only the Decoder knows the `enc_out` shape, so it allocates the
accumulator and threads it through every layer:

```python
def backward(self, grad_logits):
    grad_x = self.output_projection.backward(grad_logits)
    # If tie_embeddings, the TiedLinear has already added its
    # contribution to embedding.grad_weight (via accumulate-on-backward).

    grad_enc_out = np.zeros(self._cache['enc_out_shape'])
    for layer in reversed(self.layers):
        grad_x, grad_enc_out_layer = layer.backward(grad_x)
        grad_enc_out += grad_enc_out_layer

    grad_e_pe     = self.dropout_input.backward(grad_x)
    grad_e_scaled = self.pos_enc.backward(grad_e_pe)
    grad_e        = grad_e_scaled * self._embed_scale
    self.embedding.backward(grad_e, token_ids=self._cache_token_ids)
    # Accumulates the input-side contribution on top of the
    # already-present tied output-side contribution (when tied).

    return grad_enc_out
```

### `Transformer.backward`

The decoder runs first (gradient flows from logits backwards), then
the encoder takes `grad_enc_out` as its grad_output. Both call
`embedding.backward`, both contributions land in the same
`grad_weight` thanks to the new accumulate contract:

```python
def backward(self, grad_logits):
    grad_enc_out = self.decoder.backward(grad_logits)
    self.encoder.backward(grad_enc_out)
    return None
```

Three contributions accumulate into `embedding.grad_weight` in this
order (when `tie_embeddings=True`):

1. From the Decoder's tied output projection (every row of the
   embedding matrix gets a piece, because every output position
   produces logits over every vocabulary token).
2. From the Decoder's input-side embedding lookup (only the rows
   corresponding to `target_ids` get a piece).
3. From the Encoder's input-side embedding lookup (only the rows
   corresponding to `source_ids` get a piece).

When `tie_embeddings=False`, contribution 1 lives in a separate
`Linear` instead and the embedding only has contributions 2 + 3.

---

## 12. Sanity checks and tests

Twenty new tests in `tests/test_layers.py`. All pass. Run with:

```
cd nn && uv run --package numpy-transformer pytest \
   numpy-transformer/tests/test_layers.py -k "decoder or transformer or embedding"
```

### Existing test updates

- **`test_embedding`** -- updated for the new accumulate-on-backward
  contract. Adds the within-call duplicate test (with explicit
  `grad_weight = None` reset between unrelated backward calls), a
  cross-call accumulation assertion, and an explicit `token_ids`
  override assertion.

### `DecoderLayer` tests (one block)

- **`test_decoder_layer_shape_preservation`** --
  `(B, n_q, d_model) -> (B, n_q, d_model)`, with `n_q != n_k`,
  Post-LN and Pre-LN.
- **`test_decoder_layer_causal_mask_blocks_future`** -- inspects
  `mha_self.last_weights`. Upper-triangular entries are exactly
  zero; lower-triangular rows sum to 1.
- **`test_decoder_layer_cross_attn_rectangular_unmasked`** --
  with `n_q=3, n_k=5`, cross-attn weights have shape
  `(B, h, 3, 5)`, every row sums to 1, every weight is strictly
  positive.
- **`test_decoder_layer_residual_passes_zero_sublayer_pre_ln`** --
  zero out `W_o` of both MHAs and `W_2` of FFN; the Pre-LN layer
  becomes the identity regardless of `enc_out`.
- **`test_decoder_layer_backward_gradient_check_post_ln`** --
  numerical check on `grad_x`, `grad_enc_out`, and every
  parameter.
- **`test_decoder_layer_backward_gradient_check_pre_ln`** -- same
  for Pre-LN.

### `Decoder` tests (full pipeline)

- **`test_decoder_full_forward_shape`** --
  `(B, n_q) + (B, n_k, d_model) -> (B, n_q, vocab_size)`.
- **`test_decoder_section9_shape_trace`** -- exact shapes from
  Section 9: self-attn weights `(2, 4, 3, 3)`, cross-attn weights
  `(2, 4, 3, 4)` (rectangular).
- **`test_decoder_standalone_param_count_matches_doc`** -- full
  per-component arithmetic; standalone Decoder = `26176`.
- **`test_decoder_tied_vs_untied_param_count`** -- untied path adds
  exactly `vocab_size * d_model + vocab_size = 528` params.
- **`test_decoder_tied_output_uses_shared_embedding_weight`** --
  `output_projection.shared_owner is decoder.embedding` and
  mutating the embedding zeroes the output logits.
- **`test_decoder_tied_gradient_reaches_untouched_rows`** --
  structural test of the tied gradient path: vocab rows that are
  NOT touched by `target_ids` still receive a nonzero gradient via
  the output projection. (Without tying, those rows would be
  exactly zero.)
- **`test_decoder_dropout_off_at_eval_is_deterministic`** --
  identical output across two forward calls at `eval()`.
- **`test_decoder_full_backward_gradient_check`** -- end-to-end
  numerical check (tied) including `grad_enc_out` and the
  embedding's combined input + output gradient.
- **`test_decoder_full_backward_gradient_check_untied`** -- same
  with `tie_embeddings=False`, exercising the untied Linear output
  projection path.

### `Transformer` tests (encoder + decoder)

- **`test_transformer_forward_shape`** --
  `(B, n_k) + (B, n_q) -> (B, n_q, vocab_size)`.
- **`test_transformer_shared_embedding_three_way`** -- three `is`
  checks (`encoder.embedding is decoder.embedding is
  transformer.embedding`), the tied output projection's
  `shared_owner` is the same object, mutation propagates,
  `params_and_grads()` listings show the embedding row only on the
  Transformer (not on the encoder or decoder).
- **`test_transformer_full_param_count_matches_doc`** --
  `17600 + 25664 = 43264`.
- **`test_transformer_full_backward_gradient_check`** --
  end-to-end through encoder + decoder. This is the test that
  certifies the encoder-decoder coupling: if the gradient flowing
  back from logits through cross-attention into `enc_out` and then
  back through the encoder is wrong, this test catches it. *This
  is the single most important test in slice 2.*
- **`test_transformer_eval_deterministic`** -- both encoder and
  decoder dropouts disabled at `eval()`.

### Test count summary

- 48 tests pre-Step-7 (Step 1 through Step 6).
- 1 updated (`test_embedding`).
- 20 new (Step 7).
- **Total: 68 tests, all passing.**

---

## Discussion Log

A chronological log of every conceptual exchange. Append, do not edit.

### 2026-04-25 -- Step 7 kickoff (slice 1: conceptual deep-dive only)

**No code this turn.** Slice 1 is dedicated to the doc; slice 2 lands
the `DecoderLayer`, `Decoder`, the causal-mask helper, the tied-
weight output projection, and the test suite outlined in Section 12.

**Five conceptually new topics resolved here:**

1. **Three sub-layers, not two.** Section 2. The encoder needs only
   self-context + integration (2 sub-layers). The decoder also needs
   *source-context* (cross-attention into the encoder output), so it
   gets 3 sub-layers. Wrapper pattern is identical to the encoder.

2. **The causal mask.** Section 3. A lower-triangular boolean over
   self-attention scores. `True` allowed, `False` blocked.
   Implementation: `np.tril(np.ones((n_q, n_q), bool))`. Effect:
   `softmax(-inf) = 0`, so attention weights for future positions
   are zero; the decoder cannot peek at tokens it is supposed to
   predict. Critical insight: causality is enforced *in attention*,
   not *in the order of operations*. The forward pass is still
   parallel across positions; only the *information flow within
   attention* is restricted to past+self.

3. **Cross-attention is rectangular and unmasked.** Section 4.
   `Q` from decoder (length `n_q`), `K=V` from encoder (length
   `n_k`). For our sort task `n_q == n_k == 5`, but the machinery
   handles arbitrary asymmetry. The asymmetry "causal on self, full
   on cross" is the geometric heart of the encoder-decoder split:
   past target tokens *constrain* future target tokens, but the
   entire source *informs* every target position.

4. **Teacher forcing.** Section 5. The training-time trick that
   resolves the autoregressive-but-parallel paradox. Decoder INPUT
   is the right-shifted ground-truth target (`[<BOS>, a, b, c]`);
   decoder OUTPUT predicts each next token (`[a, b, c, d]`). The
   causal mask makes this safe -- no peeking. Exposure bias
   (training-vs-inference distribution mismatch) is a known
   limitation we accept.

5. **Output projection + weight tying.** Section 6. The output is a
   linear projection from `d_model` to `vocab_size`, no bias (paper
   convention). Tied by default with the input embedding (paper
   section 3.4): one matrix `E ∈ R^(vocab_size × d_model)` plays
   three roles -- encoder embedding, decoder embedding,
   `W_out = E^T`. Saves params and imposes a useful structural
   prior. We use `tie_embeddings: bool = True` as the default with
   a flag for ablation.

**Two design decisions made for slice 2:**

- **`tie_embeddings=True` by default**, with the flag as the
  ablation handle. Confirmed by user before writing this doc.
- **Single shared embedding across encoder + decoder + output
  projection** for our sort task (same vocab on both sides). The
  shared embedding will be constructed *outside* the encoder and
  decoder, in a lightweight wrapper, and passed in. Enables the
  three-way sharing the paper specifies without forcing one of
  encoder/decoder to "own" the matrix.

**Cross-references made in this doc:**

- Wrapper pattern: section 3 of `06_encoder.md`.
- Pre-LN vs Post-LN trade-off: section 5 of `06_encoder.md`.
- `sqrt(d_model)` embedding scaling: section 7 of `06_encoder.md`.
- Dropout placement: section 6 of `06_encoder.md` (we add no new
  dropout placements over the encoder; one per wrapper, plus input).
- MHA mask handling: section 4 of `04_multi_head_attention.md`,
  test `test_sdpa_causal_mask_triangular` in `tests/test_layers.py`.
- Soft alignment in cross-attention: paper Figure 4.

**Things deliberately deferred from this slice (for slice 2 or
later):**

- **Padding masks.** Our toy build has fixed-length sequences with
  no padding. Cross-attention will use `mask=None`; self-attention
  will use a pure causal mask. Padding can be added later by
  combining the causal mask with a per-batch-and-key padding
  boolean.
- **KV caching at inference.** Only relevant at inference (Step 9).
  We will skip it for clarity and accept the `O(n_q^2)`-per-token
  cost.
- **Beam search / sampling strategies.** Step 9. Greedy `argmax` is
  enough for the sort task at training time.
- **Exposure-bias mitigations** (scheduled sampling, RL fine-tuning):
  out of scope for this build. Documented in section 5 for context.
- **Numerical hand-trace** (in the spirit of
  `test_mha_section5_trace`): section 9 has a *shape* trace.
  Producing a full numerical trace would require pinning down weights
  for every sub-layer including the encoder; more useful as a slice-2
  fixture than as documentation.

**Decision on shared-embedding ownership (made at end of slice 1):**

A new lightweight **`Transformer`** class will own the shared
`Embedding` matrix and pass it into:

- `Encoder(shared_embedding=...)`
- `Decoder(shared_embedding=...)`

The decoder will additionally use that same embedding's `weight^T`
as its output projection when `tie_embeddings=True`.

Why this shape over "decoder takes a reference to the encoder's
embedding":

- Cleanest separation -- neither the encoder nor the decoder *owns*
  the embedding; they both borrow it. Symmetric and obvious.
- Sets up Step 8 (training loop) nicely: the `Transformer` is
  already the right object to call `forward(source, target)` on,
  collect parameters from, and step the optimiser.
- Avoids a circular dependency in tests where decoder construction
  needs an encoder instance.

The `Transformer` class's constructor will look roughly like:

```python
Transformer(
    vocab_size, d_model, num_heads, d_ff,
    num_encoder_layers, num_decoder_layers,
    p_drop=0.1, max_len=5000,
    pre_ln=False, tie_embeddings=True,
)
```

It builds the shared `Embedding` once and constructs the encoder
and decoder with references to it. The `tie_embeddings` flag flows
into the decoder, controlling whether the output projection re-uses
the same weight or gets its own `Linear`.

**Next turn (slice 2):** implement `DecoderLayer`, `Decoder`, the
causal-mask helper, the tied output projection, the `Transformer`
wrapper class, and the test suite from Section 12.

### 2026-04-25 -- Step 7 slice 2 (code + tests)

**Goal:** translate slice 1's design into code. Land
`DecoderLayer`, `Decoder`, `TiedLinear`, `Transformer`, and the
20-test suite called out in Section 12. Make the gradient through
the encoder-decoder coupling work end-to-end.

**What landed (`numpy_transformer/layers.py`):**

- **`TiedLinear`** -- bias-free linear layer; weight is a reference
  to a shared owner's matrix (typically `Embedding.weight`).
  Forward: `y = x @ W.T`. Backward accumulates `dL/dW` into the
  shared owner's `grad_weight`.
- **`DecoderLayer`** -- three sub-layers (masked self-attn,
  cross-attn, FFN) each wrapped in residual + LN. Post-LN and
  Pre-LN supported. `backward` returns a 2-tuple
  `(grad_x, grad_enc_out_for_this_layer)`.
- **`Decoder`** -- target embedding -> scale -> PE -> dropout -> N
  decoder layers -> output projection. Accepts
  `shared_embedding=None` (standalone owns its own) and
  `tie_embeddings=True` (default). Builds the causal mask once per
  forward via `np.tril`.
- **`Transformer`** -- lightweight wrapper that owns one shared
  `Embedding`, constructs the `Encoder` and `Decoder` around it,
  and exposes `forward(source_ids, target_ids)` and
  `backward(grad_logits)`.
- **`Embedding` contract change** -- `backward` now ACCUMULATES
  rather than resets, and accepts an optional explicit
  `token_ids=` argument to defend against the shared-cache race
  between encoder and decoder forward calls. PyTorch convention.
- **`Encoder` updated** -- accepts `shared_embedding=`. When
  shared, does not expose the embedding via `params_and_grads()`.
  Caches its own `token_ids` and passes them explicitly to
  `embedding.backward`.

**The big realisation while wiring the backward:**

The cleanest way to handle the shared-`Embedding` cache race
between the encoder and decoder forwards is for *each* outer
module (`Encoder`, `Decoder`) to cache its own `token_ids` and
pass them explicitly to `embedding.backward(grad_e, token_ids=...)`.
The `Embedding`'s internal cache becomes a fallback for the
standalone case (existing tests untouched). Tried "have the
`Embedding` keep a stack of caches" first; abandoned as more
mechanism than the fix needed.

The second big realisation: the `Embedding.backward` reset
semantics had to change. With tied weights, the same matrix
receives gradient from *two* sources in a single backward pass
(input-side lookup + output-side projection). If the second call
reset the buffer, the first call's contribution would vanish.
PyTorch's convention (accumulate, zero between iterations) is the
right answer. The existing `test_embedding` was updated to
explicitly reset between unrelated backward calls, plus new
assertions for the cross-call accumulate path and the explicit
`token_ids` override.

**Tricky bit during gradient checks:**

For the standalone-Decoder gradient check, perturbing
`embedding.weight[i, j]` changes the loss through TWO paths:
(1) the input embedding lookup (only when row `i` is in
`target_ids`) and (2) the output projection (always). The numerical
gradient correctly sums both. This forced us to verify that the
analytical gradient also sums both -- which is exactly what the
accumulate-on-backward contract gives us. The
`test_decoder_full_backward_gradient_check` (tied) is the test that
certifies this works.

**Pre-LN cross-attention sees an un-normalized `enc_out`.** This is
a known minor quirk -- in a fully Pre-LN architecture, the encoder
should end with a final `LayerNorm` so its output is normalized
before being consumed by cross-attention. We chose not to add that
final LN to the encoder in this slice (would be invasive), and
instead documented the imperfection in section 9 of
`DecoderLayer._forward_pre_ln`. Revisit if Step 8 training is
unstable in Pre-LN mode.

**Untied output projection retains a bias.** The paper says no bias
on the pre-softmax linear (section 3.4). The tied path follows the
paper exactly (`TiedLinear` is bias-free by construction). The
untied path uses our existing `Linear` class which has a bias --
adds `vocab_size` extra params (`16` for our build). We accepted
this rather than building a `LinearNoBias` subclass for this slice;
the tied path is the default and the paper-faithful one.
`test_decoder_tied_vs_untied_param_count` documents the
discrepancy.

**Numerical sanity:** all 68 tests pass (48 carried over from
Steps 1-6, 1 updated, 19 new for Step 7) including the two most
demanding ones:

1. `test_decoder_full_backward_gradient_check` -- end-to-end
   through the standalone Decoder with tied weights.
2. `test_transformer_full_backward_gradient_check` -- end-to-end
   through the full Transformer, including the gradient that
   couples decoder training back to encoder training. *This is the
   test that makes Step 8 (joint training) possible.*

**Cross-references baked into the new code's docstrings:**

- `TiedLinear` -- section 6 of this doc.
- `DecoderLayer` -- sections 2-4 of this doc; sections 5-6 of
  `06_encoder.md` for the wrapper-pattern lineage.
- `Decoder` -- sections 1, 6, 8 of this doc.
- `Transformer` -- section 6 of this doc.
- `Embedding.backward` -- section 6 of this doc (why accumulate).

**What is now ready for Step 8:**

- `Transformer.forward(source_ids, target_ids)` -> logits.
- `Transformer.backward(grad_logits)` -> populates every gradient.
- `Transformer.params_and_grads()` -> exactly the right list (no
  duplicates from the shared embedding).
- `Transformer.train()` / `.eval()` -- toggles every dropout in
  the encoder + decoder.

Step 8 (training loop) only needs to add: cross-entropy loss
(reduces logits + label IDs to a scalar), an optimizer (SGD or
Adam), the right-shifted teacher-forcing input construction, and a
training schedule.

**Things deferred to Step 8 or later (no change from slice 1):**

- Padding masks.
- KV caching for inference.
- Beam search / sampling.
- Final LN on the encoder for Pre-LN architectures.
- Bias-free untied output projection.

**Bottom line:** the encoder-decoder skeleton is wired end-to-end
and gradient-correct. Step 7 is complete.
