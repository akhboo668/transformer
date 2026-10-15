# Step 12: PyTorch Port

> **Status**: Design phase -- this doc was written before any code, on
> 2026-04-26, to lock in the design decisions we agreed on in chat.
>
> **Not a paper concept.** Like Step 11 (Persistence), this is pure
> engineering. We treat it as a numbered step because it earns the same
> documentation rigor as the math steps, and because the lottery
> experiment in Step 13 depends on it.
>
> **Depends on**: Steps 1-11 of the NumPy build (we will validate the
> PyTorch port against the NumPy reference implementation, especially
> the saved model from Step 11).
>
> **Enables**: Step 13 (lottery floor experiment) and any future
> cross-implementation experiments.

[< Step 11 (Persistence)](11_model_persistence.md) | [Overview](00_overview.md) | [Reference: Capabilities and Limits](reference_capabilities_and_limits.md)

---

## 1. Why this step exists

The NumPy implementation (Steps 1-11) is the *learning* implementation.
Every layer's forward and backward pass was hand-derived from first
principles. Hand-rolled backward propagation is excellent for
understanding gradient flow but creates two pain points:

- **Slow iteration**: every architectural change requires redoing the
  backward derivation by hand.
- **No GPU**: NumPy doesn't easily target accelerator hardware.

For Step 13 (the lottery floor experiment), we want to:

- Run more experiments faster (autograd handles all gradients automatically).
- Have a known-good *cross-implementation* check on the NumPy reference
  (if both implementations produce the same results, both are likely
  correct).
- Build the muscle memory of working in a real ML framework (PyTorch
  is the de-facto standard for research code).

Step 12 is a faithful PyTorch port of the same architecture, validated
against the NumPy reference, without changing any of the math.

### A practical motivation

After the lottery experiment, we'll likely want to scale up
(longer contexts, larger vocabularies, more layers). The NumPy
implementation tops out at "tiny models that fit on CPU in seconds."
PyTorch removes that ceiling. Step 12 is the bridge.

---

## 2. Scope

### In scope

- A complete PyTorch implementation of the Transformer architecture in
  `nn/pytorch-transformer/`, paralleling `nn/numpy-transformer/`.
- All layers re-implemented as `torch.nn.Module` subclasses
  (educational port -- see Section 3).
- Training loop with `torch.optim.Adam`, Noam LR schedule, label smoothing,
  cross-entropy with PAD masking.
- Validation on the same digit-sort task that Steps 8-9 used, achieving
  comparable accuracy (target: >=95%, NumPy reference is 100%).
- Persistence module (PyTorch flavor of Step 11).
- Tests mirroring the NumPy test suite plus cross-implementation tests.

### Out of scope (deferred to later steps)

- The lottery dataset and Powerball-format generator (Step 13).
- The lottery floor experiment itself, including Experiments A
  (prediction) and B (memorization) (Step 13).
- GPU optimization (CPU is fine for our model sizes).
- Distributed training, mixed precision, gradient checkpointing
  (overkill for tiny models).

### Specifically not changing from the NumPy version

- Architecture (encoder-decoder, MHA, FFN, LayerNorm wrappers,
  residual structure, sinusoidal positional encoding).
- Hyperparameters for the digit-sort validation
  (`d_model=32, num_heads=4, d_ff=64, num_layers=2`).
- Training procedure (Adam betas, Noam warmup curve, label smoothing
  rate, batch size, step count).
- Tokenization conventions (PAD/SOS/EOS, single-pool vocabularies for
  the sort task).

---

## 3. The educational port philosophy

We are doing an **educational port**, not a "use the framework to
shortcut the work" port. The distinction matters.

### What an educational port means

We re-implement each architectural component ourselves as a
`nn.Module`, using PyTorch's *primitive* operations (`nn.Linear`,
`nn.LayerNorm`, `nn.Embedding`, `nn.Dropout`) but **not** its high-level
abstractions (`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`,
`nn.Transformer`).

| Component | NumPy | PyTorch (educational port) | PyTorch (framework version, NOT what we're doing) |
|---|---|---|---|
| Linear projection | `Linear` (custom) | `nn.Linear` | `nn.Linear` |
| LayerNorm | `LayerNorm` (custom) | `nn.LayerNorm` | `nn.LayerNorm` |
| Embedding | `Embedding` (custom) | `nn.Embedding` (with tied-weight handling on top) | `nn.Embedding` |
| Dropout | `Dropout` (custom) | `nn.Dropout` | `nn.Dropout` |
| Positional encoding | `SinusoidalPositionalEncoding` (custom) | `SinusoidalPositionalEncoding` (custom, mirrors NumPy) | Often skipped or replaced with learned embeddings |
| Scaled dot-product attention | `ScaledDotProductAttention` (custom) | `ScaledDotProductAttention` (custom, mirrors NumPy) | `F.scaled_dot_product_attention` |
| Multi-head attention | `MultiHeadAttention` (custom) | `MultiHeadAttention` (custom, mirrors NumPy) | `nn.MultiheadAttention` |
| Encoder layer | `EncoderLayer` (custom) | `EncoderLayer` (custom, mirrors NumPy) | `nn.TransformerEncoderLayer` |
| Decoder layer | `DecoderLayer` (custom) | `DecoderLayer` (custom, mirrors NumPy) | `nn.TransformerDecoderLayer` |
| Encoder | `Encoder` (custom) | `Encoder` (custom, mirrors NumPy) | `nn.TransformerEncoder` |
| Decoder | `Decoder` (custom) | `Decoder` (custom, mirrors NumPy) | `nn.TransformerDecoder` |
| Transformer | `Transformer` (custom) | `Transformer` (custom, mirrors NumPy) | `nn.Transformer` |

### Why educational, not framework

Three reasons:

1. **Learning value**. The framework version is ~50 lines and teaches
   you nothing about transformers; you'd just be wrapping a black box.
   The educational port lets you see *exactly* which operations
   PyTorch's autograd is doing for you that we hand-rolled in NumPy.
2. **Direct comparability**. With the same architecture in both
   implementations, we can write tests that load NumPy weights into
   the PyTorch model (or vice versa) and assert numerical equivalence.
   The framework version's internal layout (heads packed, parameter
   names, etc.) doesn't match ours, so this comparison would require
   significant adapter code.
3. **Workspace philosophy**. The workspace's learn-by-building rule
   says: build understanding layer by layer with testable components.
   The educational port is the natural fit.

### What we DO get from PyTorch

The shorter line counts in PyTorch come from three places, and each is
a real lesson:

1. **`autograd` replaces hand-rolled backward**. Our NumPy
   `MultiHeadAttention.backward` is roughly 80 lines of careful
   gradient bookkeeping. PyTorch's version is 0 lines -- autograd
   tracks the forward computation graph and computes backward
   automatically.
2. **`nn.Module` organizes parameters**. We don't have to maintain
   `params_and_grads()` ourselves; `model.parameters()` walks the
   module tree. (We will still implement an equivalent for naming, to
   make persistence work cleanly -- see Section 8.)
3. **`torch.optim.Adam` replaces hand-rolled optimizer**. Our NumPy
   `Adam` class with its `m`, `v`, and bias-correction logic is
   replaced by `torch.optim.Adam(model.parameters(), lr=...)`. This is
   one of the cleaner wins.

The forward passes stay almost identical to NumPy in structure -- same
matrix multiplies, same masks, same residuals. Only the backward and
optimizer machinery moves into the framework.

---

## 4. What changes, what stays the same

### Forward pass: nearly identical to NumPy

The math doesn't change. Scaled dot-product attention is still:

```python
# Both NumPy and PyTorch
scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
if mask is not None:
    scores = scores.masked_fill(mask, -inf)  # PyTorch syntax
weights = softmax(scores, dim=-1)
output = weights @ V
```

The only differences are:

- `np.matmul` becomes `torch.matmul` (or `@`).
- `np.where` becomes `torch.where` or `tensor.masked_fill`.
- `np.exp` / `np.softmax` become `torch.exp` / `F.softmax`.

These are 1-to-1 substitutions.

### Backward pass: handled by autograd

In NumPy, every layer has a `forward()` and a `backward()`. In PyTorch,
we only write `forward()`. Autograd records the operations and
computes gradients automatically when we call `loss.backward()`.

What this means concretely:

| NumPy | PyTorch |
|---|---|
| `Linear.forward(x)` and `Linear.backward(grad)` | `nn.Linear.__call__(x)` -- backward is automatic |
| `MultiHeadAttention.backward()` does ~80 lines of gradient routing across heads | We write only the forward; backward is generated |
| Manual gradient accumulation (`existing += contribution`) | Tensors automatically accumulate in `.grad` attributes |
| `zero_grads(model)` walks `params_and_grads()` and sets each to zero | `optimizer.zero_grad()` does the equivalent |

This means our NumPy backward pass code (`backward()` methods,
gradient bookkeeping, `params_and_grads()` walking) is **all replaced
by autograd**. The corresponding lessons (chain rule, manual gradient
flow) are encoded in the NumPy reference and don't need to be repeated.

### Parameter management: `nn.Module` does it

In NumPy, we manually defined `params_and_grads()` on every module to
expose `(param_array, grad_array, name)` triples. In PyTorch:

- Anything assigned to `self.something = nn.Parameter(...)` or
  `self.something = nn.Module(...)` is automatically registered.
- `model.parameters()` walks the module tree and yields all
  `nn.Parameter` tensors.
- `model.named_parameters()` yields `(name, param)` with hierarchical
  names like `"encoder.layers.0.attention.W_q.weight"`.

This is exactly the dotted-name convention we manually built in NumPy.
For persistence, we can save/load using `state_dict()` -- a built-in
PyTorch convention -- but we'll wrap it in our own `save_transformer`
/ `load_transformer` to maintain the same metadata-versioning
discipline as Step 11.

### Optimizer: built-in

Replace ~80 lines of NumPy `Adam` with:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0,                      # set per-step by the Noam schedule
    betas=(0.9, 0.98),
    eps=1e-9,
)
```

Per-step LR update remains our hand-rolled `noam_lr` function. The
training loop becomes a tight ~5 lines:

```python
loss = compute_loss(model, batch)
optimizer.zero_grad()
loss.backward()
for g in optimizer.param_groups:
    g['lr'] = noam_lr(step, d_model, warmup)
optimizer.step()
```

### Tied embeddings: requires care

The NumPy version of tied embeddings works by having `TiedLinear` hold a
reference to the master `Embedding` and dereference its `weight` lazily.
The aliasing is transparent: encoder-embedding, decoder-embedding, and
the output projection all reference the same numpy array.

In PyTorch, the equivalent pattern is:

```python
# Inside Transformer.__init__:
self.embedding = nn.Embedding(vocab_size, d_model)
self.encoder = Encoder(..., shared_embedding=self.embedding)
self.decoder = Decoder(..., shared_embedding=self.embedding)
# Inside Decoder, when tie_embeddings=True:
# don't create a separate output_projection; instead apply
# F.linear(x, weight=self.embedding.weight) at output time.
```

Tested by:

- `model.encoder.embedding is model.decoder.embedding` -> True (same `nn.Module`)
- `model.embedding.weight.data_ptr() == model.encoder.embedding.weight.data_ptr()` -> True (same memory)
- After `optimizer.step()` updates `self.embedding.weight`, the decoder's
  output projection uses the updated values automatically (because the
  output projection reads `self.embedding.weight` lazily).

Same aliasing-preservation discipline as Step 11, with PyTorch tooling.

---

## 5. Project layout

```
nn/
  numpy-transformer/             <-- existing, stays as-is
  pytorch-transformer/           <-- NEW, parallel structure
    pyproject.toml
    src/
      pytorch_transformer/
        __init__.py
        layers.py                  (Transformer + components)
        training.py                (Adam/Noam wiring, training step, loss helpers)
        inference.py               (greedy_decode etc.)
        sort_data.py               (mirrors numpy_transformer.sort_data)
        persistence.py             (save_transformer / load_transformer for PyTorch)
        # lottery_data.py is added in Step 13, not here
    train_input/                   (sort dataset fixtures, may symlink to numpy version)
    train_output/                  (loss curves)
    train_model/                   (saved .pt files)
    tests/
      test_layers.py               (unit tests + cross-impl tests vs NumPy)
      test_training.py
      test_inference.py
      test_persistence.py
      test_cross_implementation.py (load NumPy model, port weights, compare logits)
    scripts/
      train_sort.py                (validation: digit sort task)
      # train_lottery.py is added in Step 13
```

Sibling structure to `numpy-transformer/`. Independent uv-managed
package. Tests can cross-reference the NumPy version for cross-impl
validation.

---

## 6. Layer-by-layer port plan

We port in dependency order, with a unit test for each layer that
includes a cross-implementation check against NumPy.

### Port order

1. **`Linear`-like primitives**: not needed -- use `nn.Linear` directly.
2. **`Embedding`**: use `nn.Embedding`, no wrapper needed (we pass it
   around for tied-weight sharing, see below).
3. **`SinusoidalPositionalEncoding`**: hand-coded. Identical math to
   NumPy. No learnable parameters.
4. **`ScaledDotProductAttention`**: hand-coded. Forward only.
5. **`MultiHeadAttention`**: hand-coded. Manages 4 `nn.Linear`
   projections (Q, K, V, O), reshapes, head splitting/merging.
6. **`FeedForward`**: 2 `nn.Linear`s + ReLU. Trivial.
7. **`LayerNormWrapper` / `SublayerWrapper`**: residual + LayerNorm
   + dropout pattern (paper's Post-LN by default, optional Pre-LN).
8. **`EncoderLayer`**: composes MHA + FFN with sublayer wrappers.
9. **`DecoderLayer`**: composes masked self-attn + cross-attn + FFN
   with sublayer wrappers.
10. **`Encoder`**: stacks N encoder layers, owns the shared embedding.
11. **`Decoder`**: stacks N decoder layers, output projection (tied
    or untied).
12. **`Transformer`**: orchestrates encoder + decoder, owns the master
    `Embedding`, distributes it to children when `tie_embeddings=True`.

### Unit test contract for each layer

For a given NumPy class `Foo`, the PyTorch test does:

```python
def test_pytorch_foo_matches_numpy():
    # 1. Build both classes with the same hyperparameters and same RNG.
    np_foo = NumPyFoo(...)
    pt_foo = PyTorchFoo(...)

    # 2. Copy NumPy weights into the PyTorch module.
    pt_foo.load_from_numpy(np_foo)

    # 3. Run the same input through both.
    x_np = np.random.randn(2, 5, 32)
    x_pt = torch.from_numpy(x_np).to(torch.float64)  # float64 for fidelity

    out_np = np_foo.forward(x_np)
    out_pt = pt_foo(x_pt).detach().numpy()

    # 4. Assert numerical equivalence.
    np.testing.assert_allclose(out_np, out_pt, atol=1e-7)
```

The `load_from_numpy` helper is module-specific. For `nn.Linear`, it
copies the NumPy `weights` into `pt_module.weight` (note PyTorch
convention is `out_features x in_features`, NumPy convention may vary).
For `MultiHeadAttention`, it copies all four `W_*` matrices.

### Why float64

PyTorch defaults to float32. NumPy defaults to float64. To make the
cross-implementation tests succeed at `atol=1e-7` (the same threshold
we used for NumPy gradient checks), we run PyTorch in float64 mode for
these tests:

```python
torch.set_default_dtype(torch.float64)
```

Production training will use float32 (faster, less memory). This is a
testing-only choice.

---

## 7. Validation strategy

Three levels of validation, run in sequence.

### Level 1: Layer-by-layer numerical equivalence

For every layer (Section 6), we have a unit test that loads NumPy
weights into the PyTorch module and asserts the forward outputs match
within `atol=1e-7` on a random input. This catches any "I ported the
math wrong" bug.

### Level 2: End-to-end forward equivalence

A single integration test: load the entire Step 11 trained model
(`numpy-transformer/train_model/sort_model.npz`) into a PyTorch
`Transformer`, run greedy decode on the held-out test set, and
**assert 100% accuracy match** (the same 1000/1000 the NumPy version
achieves). Bit-perfect logit equivalence is too strict (float32 vs
float64), but the *decisions* (argmax over logits) should be identical.

### Level 3: Train from scratch in PyTorch

Run `train_sort.py` end-to-end in PyTorch. Achieve `>=95%` test
accuracy (NumPy reference is 100%; we allow 5pp slack for floating-
point and RNG differences). This validates the whole pipeline:
training loop, optimizer, loss, autograd, parameter updates.

If all three levels pass, the port is correct, and Step 13 can
proceed safely. If any level fails, we have a localized bug to fix
before moving on.

---

## 8. Persistence (PyTorch flavor of Step 11)

Same design discipline as Step 11, different mechanics:

| Step 11 (NumPy) | Step 12 (PyTorch) |
|---|---|
| `np.savez_compressed` to `.npz` | `torch.save` to `.pt` (uses pickle internally) |
| Manual key conventions (`meta.*`, `params.*`) | Use `state_dict()` for params, separate dict for `meta` |
| Manual array dtype/shape validation on load | PyTorch validates tensor compatibility on `load_state_dict()` |
| `meta.format_version` field | Same: keep our own format version, independent of PyTorch's |
| `Storage` Protocol with `LocalStorage` and `FsspecStorage` | Same protocol, same backends -- PyTorch's `torch.save` accepts a file-like object |
| In-place `param[...] = value` to preserve tied-embedding aliasing | Use `module.load_state_dict(..., strict=True, assign=False)` (the default). PyTorch handles aliasing via shared parameter references in module hierarchy. |

The metadata schema mirrors Step 11:

```python
{
    "format_version": 1,
    "framework": "pytorch",            # NEW: distinguishes from NumPy .npz files
    "torch_version": torch.__version__,
    "meta": {
        "vocab_size": 13,
        "d_model": 32,
        "num_heads": 4,
        "d_ff": 64,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "max_len": 5000,
        "p_drop": 0.1,
        "tie_embeddings": True,
        "pre_ln": False,
    },
    "state_dict": model.state_dict(),  # PyTorch tensors
}
```

`save_transformer` and `load_transformer` are free functions in
`pytorch_transformer.persistence`, mirroring the API of
`numpy_transformer.persistence`. Tests follow the same 17-test
pattern: round-trip, version mismatch, missing/extra keys, etc.

---

## 9. What this enables: the lottery experiment (Step 13)

This doc does NOT cover the lottery experiment. That's Step 13's
scope. But for context:

Once Step 12 is complete, Step 13 will:

1. Add a `lottery_data.py` module to generate synthetic i.i.d.
   Powerball-format draws (5 white balls from 1-69 + 1 Powerball
   from 1-26).
2. Use Option B vocabulary: `vocab_size=99` with white balls as tokens
   1-69 and Powerballs as tokens 70-95.
3. Run two experiments:
   - **Experiment A (prediction)**: train on 990 draws, evaluate on
     held-out draws 991-1000. Predicted outcome: loss plateaus at
     `log(69)` for white-ball positions and `log(26)` for the
     Powerball position. Exact-match: 0/10.
   - **Experiment B (memorization)**: train on all 1000 draws,
     evaluate on the same 991-1000. Predicted outcome: loss drops
     toward 0 if capacity is sufficient. Exact-match: ~10/10.

The contrast is the lesson: **memorization is not prediction**. Same
model, same data, same architecture -- different result, depending
solely on whether the answer was in the training set.

See [`reference_capabilities_and_limits.md`](reference_capabilities_and_limits.md)
Section 6 for the theoretical argument that motivates the experiment.

---

## 10. Build plan

Each slice ends with a stop point per the workspace's "one step at a
time" rule.

### Slice 1: Project scaffolding -- ~30 min

- Create `nn/pytorch-transformer/` with parallel structure.
- `pyproject.toml` declaring `torch` as a dependency.
- Empty `__init__.py`, empty test file.
- `uv sync` works; `uv run pytest` runs (zero tests passed is fine).

### Slice 2: Layer-by-layer port -- ~2-3 hours

In dependency order (Section 6.1):

- `SinusoidalPositionalEncoding`
- `ScaledDotProductAttention`
- `MultiHeadAttention`
- `FeedForward`
- `LayerNormWrapper`
- `EncoderLayer`, `DecoderLayer`
- `Encoder`, `Decoder`, `Transformer`

Each with a unit test that asserts cross-implementation equivalence
(Section 6.2 contract).

### Slice 3: Training loop -- ~1 hour

- `noam_lr` (mirrors NumPy)
- `cross_entropy_with_label_smoothing` (use `F.cross_entropy` with
  `label_smoothing=0.1`)
- `train()` -- the actual training loop
- `train_sort.py` script that mirrors the NumPy CLI

### Slice 4: Validation on digit sort -- ~30 min

- Run `train_sort.py` end-to-end.
- Assert `>=95%` test accuracy.
- Run cross-impl test: load `numpy-transformer/train_model/sort_model.npz`,
  port weights into PyTorch, run greedy decode, assert 100% accuracy
  match against NumPy.

### Slice 5: Persistence -- ~1 hour

- `save_transformer` / `load_transformer` for PyTorch.
- 15-17 tests mirroring `test_persistence.py` from Step 11.
- Update `train_sort.py` to save the trained model.

### Total estimated effort: 5-7 hours

After Slice 5 ships, Step 12 is complete and we move to Step 13
(lottery experiment).

---

## 11. Open questions

These are deliberately deferred to be revisited during implementation:

- **PyTorch version pinning**: target torch>=2.0 (for the modern API)
  but verify our usage works on the latest stable. Decide during
  Slice 1.
- **CPU vs accelerator**: default to CPU. If anyone has a GPU/MPS handy,
  we'd add `device='cuda'` / `device='mps'` support behind a CLI flag
  in a follow-up, not blocking Step 13.
- **Test fixtures**: do we symlink `train_input/` to the NumPy version
  (saves disk + ensures both implementations train on the same data),
  or duplicate? Decide in Slice 1.
- **Cross-impl test thresholds**: `atol=1e-7` for forward equivalence
  is the NumPy gradient-check threshold. May need to relax for some
  layers due to PyTorch's internal precision choices. Calibrate in
  Slice 2.

---

## Discussion Log

### 2026-04-26: Design conversation

The Step 12 design emerged from a multi-turn chat between user and
agent. Captured here to preserve the reasoning.

**Trigger**: After Step 11 (Persistence) shipped on 2026-04-26 and
the model was empirically tested on out-of-distribution inputs
(captured in `reference_capabilities_and_limits.md`), the user asked
about extending to lottery numbers. The conversation revealed that
to do the lottery experiment cleanly, we needed (a) a PyTorch
implementation for faster iteration, and (b) a clean separation
between the framework port (Step 12) and the experimental design
(Step 13).

**Key design decisions, with rationales**:

1. **Educational port, not framework port**. The user wanted to *learn*
   PyTorch, not just use it. The framework port (using
   `nn.Transformer`) would be ~50 lines and teach nothing about
   transformers. The educational port preserves the layer-by-layer
   correspondence with NumPy that makes both implementations directly
   comparable.

2. **Validate on digit sort BEFORE switching to lottery**. The user
   initially proposed combining the framework switch and the dataset
   switch into one move. The agent strongly recommended port-first,
   citing differential-diagnosis reasoning: if a combined run failed,
   we wouldn't know whether the bug was in the port, the dataset, or
   the architecture. Port-first eliminates option 1 from the
   diagnosis space at the cost of ~10 seconds of training time.

3. **Powerball format with Option B vocabulary**. The user chose
   Powerball (5 from 1-69 + 1 from 1-26) over single-pool lotteries.
   The agent walked through three vocabulary designs (shared, separate
   token spaces, separator-token) and the user selected Option B
   (separate token spaces). Rationale: cleanest semantics -- token 12
   (white ball) and token 81 (Powerball #12) are different things and
   should be different tokens. `vocab_size=99`.

4. **Two experiments: A (prediction) and B (memorization)**. The
   conversation revealed an ambiguity in "test if the model can
   produce the last 10 draws correctly": did the user mean held-out
   prediction (impossible) or in-sample memorization (possible with
   sufficient capacity)? The user confirmed both. The contrast is the
   lesson.

5. **Synthetic i.i.d. data over real historical data**. For the
   floor-demonstration to be definitive, we need to *know* the data is
   uniform. Real historical data would give the same result (lotteries
   are audited to be fair) but introduces a "what if there's hidden
   bias?" doubt that synthetic data eliminates. Real data is a
   follow-up experiment if desired.

6. **Two step docs, not one**. Step 12 = framework mechanics. Step 13
   = empirical demonstration. They are conceptually distinct and
   deserve separate writeups. Step 12 references Step 13 as the
   downstream consumer; Step 13 will reference Step 12 as the
   prerequisite.

7. **Context size: 10 draws, not 1000**. The user initially said
   "1000 previous draws". The agent flagged that 1000 draws =
   6000 tokens of context, prohibitive on CPU. The agent recommended
   10 draws (60 tokens), arguing the principle is identical at any
   context size and the loss floor will be the same. User confirmed.

### Predicted outcomes for Step 13 (recorded BEFORE the experiment)

Per workspace discipline, predictions go in writing before the
experiment runs. The Step 13 doc will include these and we'll see how
well the predictions match reality.

**For Experiment A (prediction)**:
- Loss plateaus at `log(69) ~= 4.234` for white-ball positions and
  `log(26) ~= 3.258` for the Powerball position.
- Per-position accuracy lands at `5/69 ~= 7.25%` for any white ball
  position and `1/26 ~= 3.85%` for the Powerball.
- Exact-match draws: 0/10 (probability `1 / 292,201,338` per draw).

**For Experiment B (memorization)**:
- Loss on the memorized draws drops from `~4` toward `0` as training
  proceeds.
- Exact-match draws: `>=9/10` if model capacity is sufficient. Our
  model with `d_model=64, num_heads=4, d_ff=128, num_layers=2` has
  ~80K parameters; this is more than enough to memorize
  `1000 * (5*log2(69) + log2(26)) ~= 36,000` bits of information.

If Experiment A's loss drops *below* the predicted floors, that's a
bug -- either in our synthetic data generator (introducing
non-uniformity) or in the training pipeline (data leakage between
train and test). If Experiment B fails to memorize, our model
is undersized -- expected to be very unlikely with these
hyperparameters.

---

[< Step 11 (Persistence)](11_model_persistence.md) | [Overview](00_overview.md) | [Reference: Capabilities and Limits](reference_capabilities_and_limits.md)
