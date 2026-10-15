# Step 8: Training

> **Status**: **Complete** -- slice 1 (conceptual deep-dive) and slice 2 (code, tests, runnable training script) both landed.
> **Paper section**: 5 (Training Hardware and Schedule), 5.3 (Optimizer), 5.4 (Regularization)
> **Depends on**: Step 7 (a complete `Transformer` with forward / backward / `params_and_grads`)

[< Decoder](07_decoder.md) | [Overview](00_overview.md) | [Next: Inference >](09_inference.md)

---

## 1. What we build

The training pipeline takes a randomly initialized `Transformer` and turns it
into one that *actually sorts*.

Inputs available going into Step 8:

- A `Transformer(vocab_size=13, ...)` with `forward(source, target)`,
  `backward(grad_logits)`, and `params_and_grads()` from Step 7.
- Tensors `(encoder_input, decoder_input, decoder_target)` produced by
  `sort_data.py` from Step 0/8 setup. Already on disk in
  `numpy-transformer/train_input/sort_{train,test}.npz`.
- Vocabulary: digits `0..9`, `PAD=10`, `SOS=11`, `EOS=12` -- size 13.

Outputs at the end of Step 8:

- A trained `Transformer` whose embedding and layer weights have been
  updated by many gradient steps.
- A `loss_history` array (and optionally `lr_history`) suitable for
  plotting a loss curve.

The five new ingredients we add in this step:

1. **Cross-entropy loss** -- the scalar that gradient-descent tries to
   minimize. Reduces `(B, n_dec, V)` logits and `(B, n_dec)` labels to one
   number, with a known gradient back to logits.
2. **Label smoothing** (paper section 5.4) -- soften the target
   distribution from a one-hot spike to `(1-eps)` on truth + uniform `eps`
   elsewhere. Regularizer.
3. **Adam optimizer** (paper section 5.3) -- per-parameter adaptive
   learning rate using running first and second moments of the gradient.
4. **Noam learning rate schedule** (paper equation 3) -- linear warmup
   then inverse-square-root decay. Critical for stability with the
   embedding scaling and Adam settings the paper picked.
5. **The training loop** -- the `for` that ties forward + loss + backward
   + optimizer-step + zero-grad together; also tracks loss and lr for
   monitoring and convergence diagnosis.

Every other piece is already built. Step 8 is *almost entirely* about
adding these five and gluing them to the existing parts.

---

## 2. The training step at a glance

Five operations per step:

```
1. batch  : (enc_input, dec_input, dec_target) = next_batch()
2. forward: logits = transformer.forward(enc_input, dec_input)        # (B, n_dec, V)
3. loss   : loss, grad_logits = ce_with_label_smoothing(logits, dec_target, eps=0.1)
4. zero   : zero_grads(transformer.params_and_grads())                # critical: see 7.3
5. back   : transformer.backward(grad_logits)                         # populates grads
6. step   : optimizer.step(noam_lr(step), transformer.params_and_grads())
7. log    : loss_history.append(float(loss))
```

Six lines if you count batch fetching and logging. Steps 2 and 5 are
already done (Step 7); the rest is what we add here.

A few observations to bookmark before the deep-dive:

- The **gradient of cross-entropy w.r.t. logits is beautifully simple**:
  `softmax(logits) - one_hot(labels)`. This is the only "math" that
  composes with Step 7's backward. Everything else (`backward` of every
  layer, accumulation into shared embedding, etc.) is already wired.
- The optimizer **does not see the model's structure**. It just receives
  a flat list `[(p, g, name), ...]` from `params_and_grads()` and updates
  each `p` based on its own `g`. This is why `params_and_grads()` is the
  contract that holds Step 7 and Step 8 together.
- **Zeroing gradients between steps is mandatory.** Embedding gradients
  accumulate by design (Step 7); without explicit zeroing, every step
  would carry the previous step's gradient on top.

---

## 3. Cross-entropy loss

### 3.1 Definition

For one prediction position with logits `z ∈ R^V` and target class
`y ∈ {0, 1, ..., V-1}`:

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
log_softmax(z)_i = z_i - log(sum_j exp(z_j))
loss = -log_softmax(z)_y                       # negative log-likelihood
```

For a batch of predictions across `B` sequences and `n_dec` positions, the
default reduction we use is the **mean over all valid (non-PAD) tokens**:

```
total = 0; count = 0
for b in 0..B:
    for t in 0..n_dec:
        if y[b, t] != PAD:
            total += -log_softmax(z[b, t])[y[b, t]]
            count += 1
loss = total / count
```

For the sort task, every position is valid (no padding), so
`count = B * n_dec`. We keep the PAD-aware form anyway because the data
layer reserves `PAD=10` and we may need it later.

### 3.2 Why "cross-entropy", and why this is what gradient descent actually wants

The label `y` is a class index but conceptually it represents a one-hot
target distribution `p_target`. The model produces a predicted distribution
`q_pred = softmax(z)`. Cross-entropy between them:

```
H(p_target, q_pred) = -sum_i p_target[i] * log(q_pred[i])
                    = -log(q_pred[y])         (because p_target is one-hot)
                    = -log_softmax(z)[y]
```

So our `loss` is exactly `H(target, prediction)`, summed over positions.
Gradient descent on this is equivalent to **maximum likelihood**: we are
maximizing the probability the model assigns to the true target sequence.

### 3.3 The combined-softmax-log-NLL trick (numerical stability)

A naive implementation:

```
prob = softmax(z)                # exp can overflow for large z
loss = -log(prob[y])             # log(0) = -inf if any prob underflowed
```

is unstable. The standard fix uses the **log-sum-exp trick**:

```
m = max_i z_i                    # subtract this to avoid exp overflow
log_softmax(z) = z - m - log(sum_j exp(z_j - m))
loss = -log_softmax(z)[y]
```

Subtracting `m` is mathematically a no-op (softmax is shift-invariant) but
keeps the largest exponent at `exp(0) = 1`. This is what every framework
does under the hood (`F.log_softmax` in PyTorch, `tf.nn.log_softmax`).

We will implement `log_softmax` once, use it for both the loss value and
the gradient, and never call `np.log(softmax(z))` separately.

### 3.4 Gradient back to logits (the part we need for backward)

This is the magic that makes cross-entropy + softmax pair so well. With
`q_i = softmax(z)_i` and `loss = -log(q_y)`:

```
dloss / dq_i = -1/q_y * delta(i, y)             # only nonzero at i = y
dq_i / dz_j  = q_i * (delta(i, j) - q_j)        # softmax Jacobian
dloss / dz_j = sum_i (dloss/dq_i) * (dq_i/dz_j) = q_j - delta(j, y)
```

In words:

> **The gradient of cross-entropy at logit `j` is the predicted probability
> of class `j`, minus 1 if `j` is the correct class.**

That is:

```
grad_logits[b, t, :] = softmax(z[b, t, :])
grad_logits[b, t, y[b, t]] -= 1.0
```

So `grad_logits` looks like the predicted softmax distribution with `1`
subtracted at the true-class position. If the model is confident on the
right class (`q_y ≈ 1`), the gradient is near zero (good; nothing to
fix). If it's wrong, the gradient pushes `z_y` up and the rest down.

This identity is **why softmax + cross-entropy go together**: the
softmax derivative cancels the `-1/q_y` factor, leaving the clean
`q - one_hot(y)` form. It is the entire reason we apply softmax inside
the loss rather than as a separate output layer.

### 3.5 Reduction: divide by what?

Three plausible choices for batched cross-entropy:

| Reduction        | Loss value       | LR sensitivity                                   |
|------------------|------------------|--------------------------------------------------|
| `sum`            | total NLL        | LR scales with `B * n_dec` -- fragile            |
| `mean over batch`| `total / B`      | LR scales with `n_dec` -- still fragile          |
| `mean over tokens` (default) | `total / count` | LR is invariant to batch size and seq length |

Most modern frameworks default to **mean over tokens**. We do too, and
`count` excludes PAD positions if any.

### 3.6 Worked example

Tiny case: `B=1, n_dec=3, V=4`. Logits and targets:

```
logits = [[ 2.0, -1.0,  0.5,  3.0],          # position 0, target = 3
          [ 1.0,  1.5, -2.0,  0.0],          # position 1, target = 1
          [ 0.5,  0.5,  0.5,  0.5]]          # position 2, target = 2

targets = [3, 1, 2]
```

Position 0:
```
m = 3.0
shifted = [-1.0, -4.0, -2.5, 0.0]
exp     = [0.3679, 0.0183, 0.0821, 1.0000]
sum_exp = 1.4683
log_sum_exp = log(1.4683) = 0.3839
log_softmax = shifted - log_sum_exp = [-1.3839, -4.3839, -2.8839, -0.3839]
loss_0 = -log_softmax[3] = 0.3839
```

Position 1: `target = 1`. Same recipe gives `loss_1 ≈ 0.6748`.
Position 2: uniform logits, `target = 2`, `loss_2 = log(4) ≈ 1.3863`.

Total / 3: `(0.3839 + 0.6748 + 1.3863) / 3 ≈ 0.815`.

Gradient at position 0:
```
softmax(z) = exp / 1.4683 = [0.2506, 0.0125, 0.0559, 0.6810]
one_hot(3) = [0, 0, 0, 1]
grad_logits[0] = softmax - one_hot = [0.2506, 0.0125, 0.0559, -0.3190]
```

Sanity: components on non-target classes are positive (push them down),
component on target is negative (push it up). All elements of `grad_logits`
are between -1 and 1. The mean-over-tokens reduction means the per-token
gradient contributions are scaled by `1/count`:

```
grad_logits = (softmax(z) - one_hot(y)) / count
```

(For `B=1, n_dec=3`, `count = 3`.) This division is what keeps optimizer
hyperparameters batch-size-invariant.

---

## 4. Label smoothing (paper section 5.4)

### 4.1 The problem

Cross-entropy with one-hot targets pushes `softmax(z)_y → 1` and every
other class `→ 0`. To do that, the network must drive `z_y → +∞` and
`z_other → -∞`. The model never converges in *probability* (always wants
to be more confident), only in *gradient magnitude*.

Two practical consequences:

1. **Weight magnitudes grow without bound** as logits push toward
   infinity. Training becomes unstable.
2. **The model is overconfident at inference**: it assigns very high
   probability to its top guess, which is wrong about as often as it is
   right.

### 4.2 The fix (paper equation cited from "Rethinking the Inception Architecture", Szegedy et al. 2016)

Replace the one-hot target with a **smoothed** distribution:

```
p_smooth[i] = (1 - eps)  if i == y
              eps / (V - 1)  otherwise
```

(`V` = vocab size, `eps` = paper's `0.1`.) The target distribution now
puts `0.9` on the truth and spreads `0.1` uniformly across the `V-1`
non-target classes.

For `V=13` (sort task) and `eps=0.1`:

```
p_smooth[y]   = 0.9
p_smooth[!=y] = 0.1 / 12 ≈ 0.00833
```

Note: some implementations use `eps / V` uniformly (so the truth gets
`(1 - eps) + eps/V` rather than exactly `1 - eps`). Both choices are
called "label smoothing" in the literature; we use the paper's
**`eps / (V-1)` over non-target classes** form.

### 4.3 Effect on the loss

With smoothed targets:

```
loss = -sum_i p_smooth[i] * log_softmax(z)[i]
```

Re-arrange to show what changes vs. one-hot:

```
loss = (1 - eps) * (-log_softmax(z)[y])              # truth term
     + (eps / (V-1)) * sum_{i != y} (-log_softmax(z)[i])
     = (1 - eps) * loss_one_hot
     + (eps / (V-1)) * (-sum_i log_softmax(z)[i] + log_softmax(z)[y])
```

This means **even a perfect model has nonzero loss** under label
smoothing: the irreducible piece is the cross-entropy between the
smooth target and itself. We will adjust loss-curve expectations
accordingly.

### 4.4 Effect on the gradient

```
dloss / dz_j = softmax(z)_j - p_smooth[j]
             = softmax(z)_j - (1 - eps)         if j == y
             = softmax(z)_j - eps/(V-1)         otherwise
```

The `softmax(z) - target_distribution` form survives. The only change is
that `target_distribution` is smoothed instead of one-hot. We just pass
`p_smooth` instead of `one_hot(y)` and the rest of backward is unchanged.

### 4.5 When it helps and when it doesn't

| Helps                                       | Hurts / Doesn't help                  |
|--------------------------------------------|---------------------------------------|
| Multi-class classification with V > ~10    | Binary classification (V = 2)         |
| Targets are sometimes ambiguous            | Targets are always exact (small toy?) |
| Calibration matters at inference           | You only care about top-1 accuracy    |

For the sort task, `V=13` and the targets are exact (sort is
deterministic). Label smoothing **mildly hurts top-1 accuracy** but
**helps numerical stability** during training. The paper turns it on for
all NMT experiments and reports `BLEU` improves slightly. We default it
on with `eps=0.1` and document the off-by-default ablation in the
Discussion Log.

---

## 5. Teacher forcing in concrete shapes

The mechanic was introduced in section 5 of `07_decoder.md`. Step 8 is
where it touches actual data.

`sort_data.py` already produces three tensors per example:

```
encoder_input  : (n_train, 5)   raw unsorted digit ids 0..9
decoder_input  : (n_train, 6)   [SOS, sorted_d0, sorted_d1, ..., sorted_d4]
decoder_target : (n_train, 6)   [sorted_d0, sorted_d1, ..., sorted_d4, EOS]
```

`decoder_input` is `decoder_target` shifted right by one position with
`SOS` prepended and `EOS` dropped. Same in reverse: target is
`decoder_input` shifted left by one with `EOS` appended.

The causal mask inside the decoder ensures position `t` of `decoder_input`
attends only to positions `0..t`, i.e. tokens up to and including
`decoder_input[t-1]`. So when the model predicts `decoder_target[t]`, it
has seen `[SOS, sorted_d0, ..., sorted_d{t-1}]`, exactly what an
autoregressive decoder would have at inference at the same step.

The forward call is:

```
logits = transformer.forward(encoder_input, decoder_input)
# logits.shape == (B, 6, 13)
```

Loss compares `logits` against `decoder_target`:

```
loss = ce(logits.reshape(-1, V), decoder_target.reshape(-1), eps=0.1)
```

(After flatten, both arrays have leading dim `B * 6`.)

Three quirks worth noting:

- **`SOS` only appears as input, never as a target.** The model does not
  predict `SOS` at any position; it predicts `sorted_d0, ..., sorted_d4,
  EOS`.
- **`EOS` only appears as target, never as input.** Symmetric.
- **`PAD` is never used** for fixed-length sort. We keep the slot in the
  vocab so we can do variable-length later without a vocab change.

---

## 6. Adam optimizer

### 6.1 Why adaptive: motivation

Plain SGD: `param ← param - lr * grad`. Single global `lr` for every
parameter. Two well-known weaknesses for deep networks:

1. **Different parameters need different effective learning rates.** A
   layer-norm `gamma` and an attention `W_q` see gradients of wildly
   different magnitudes; one global `lr` is either too aggressive for
   one or too conservative for the other.
2. **Gradient noise in mini-batch SGD.** With one mini-batch you only
   see a noisy estimate of the true gradient. SGD pushes proportional to
   the noise; sustained momentum smooths it out, but momentum alone
   doesn't solve (1).

Adam (Kingma & Ba 2015) fixes both:

- **Per-parameter scale** by dividing each step by an estimate of the
  gradient's standard deviation (the second moment).
- **Smoothing** by using running averages instead of the raw gradient
  (the first moment).

### 6.2 The math (paper Algorithm 1 of Kingma & Ba 2015, hyperparams from
"Attention Is All You Need" section 5.3)

For each parameter `p` with gradient `g` at step `t`:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g                # 1st moment (mean of gradient)
v_t = beta2 * v_{t-1} + (1 - beta2) * g**2             # 2nd moment (variance, elementwise)

m_hat = m_t / (1 - beta1**t)                           # bias-corrected
v_hat = v_t / (1 - beta2**t)

p   ← p - lr * m_hat / (sqrt(v_hat) + epsilon)
```

Initial values: `m_0 = 0`, `v_0 = 0`. The bias correction undoes the
zero initialization's pull-toward-zero in the first few steps.

Paper hyperparameters (section 5.3):

```
beta1   = 0.9
beta2   = 0.98              <-- NOT the Adam default 0.999!
epsilon = 1e-9              <-- NOT the Adam default 1e-8!
```

The paper uses `beta2=0.98` (more responsive 2nd moment than the Adam
default) because the Noam schedule warmup means the early gradients are
informative-but-not-typical, and a sluggish 2nd moment stretches the
warmup into instability. We use the paper values.

### 6.3 Per-parameter or per-tensor?

`m_t` and `v_t` have the **same shape as `p`** -- one running stat per
scalar parameter. So a 32x32 weight matrix has 1024 entries in `m_t` and
1024 in `v_t`. The optimizer's state size is `2x` the model's parameter
count. For our 43k-param toy Transformer: 86k extra floats. Negligible.

### 6.4 What "bias correction" actually does

`m_t` and `v_t` start at zero. Without correction, the first step's
`m_1 = 0.1 * g_1` is a 10x underestimate of `g_1`. The correction
factor `1 / (1 - beta1^t)` is exactly the right scale to undo this:

```
t=1: 1/(1 - 0.9^1)   = 1/0.1   = 10.0
t=2: 1/(1 - 0.9^2)   = 1/0.19  ≈ 5.26
t=3: 1/(1 - 0.9^3)   = 1/0.271 ≈ 3.69
...
t=∞: 1/(1 - 0)       = 1.0
```

So early steps are scaled up; once `m_t` reaches steady-state the
correction is a no-op.

### 6.5 Worked Adam step

Tiny example: one scalar parameter `p = 1.0`, gradient `g = 0.5` at
`t = 1`, `lr = 0.001`, paper hyperparams.

```
m_1 = 0.9 * 0   + 0.1  * 0.5    = 0.05
v_1 = 0.98 * 0  + 0.02 * 0.5**2 = 0.005

m_hat = 0.05  / (1 - 0.9)   = 0.5      # = g, the bias correction recovers the raw gradient
v_hat = 0.005 / (1 - 0.98)  = 0.25

step  = lr * m_hat / (sqrt(v_hat) + eps)
      = 0.001 * 0.5 / (0.5 + 1e-9)
      ≈ 0.001

p ← p - 0.001 = 0.999
```

At step 1 with bias correction, the effective step size is **exactly**
`lr * sign(g)` regardless of `g`'s magnitude. This is Adam's
"normalize-by-gradient-magnitude" property -- big gradients don't lead
to big steps, only small but consistent ones.

---

## 7. Noam learning rate schedule (paper equation 3)

### 7.1 Equation 3

```
lr(step) = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

Two pieces glued at `step = warmup_steps`:

- **Warmup phase** (`step < warmup_steps`): linear in `step`.
  `lr ∝ step / warmup_steps^(1.5)` rises from `0` at step 1 to
  `1 / warmup_steps^0.5` at `step = warmup_steps`.
- **Decay phase** (`step >= warmup_steps`): inverse-square-root.
  `lr ∝ 1 / sqrt(step)` decays toward zero.

The transition at `step = warmup_steps` is continuous: both pieces
evaluate to `warmup_steps^(-0.5)` there.

The `d_model^(-0.5)` factor scales the entire curve. Bigger models
(higher `d_model`) get smaller learning rates. With our `d_model = 32`,
`d_model^(-0.5) = 1 / sqrt(32) ≈ 0.177`.

### 7.2 Why warmup, why decay

Two well-documented phenomena:

1. **Without warmup, large adaptive optimizers diverge early.** Adam's
   first-step normalization (section 6.4 above) makes early steps
   effectively `lr * sign(g)`. With a "normal" `lr`, this is a huge
   step in random directions while the model is freshly initialized.
   Warmup linearly ramps `lr` up so the early steps are tiny.
2. **Without decay, training plateaus or diverges.** As the model
   converges, gradient noise dominates the signal; smaller steps prevent
   the parameters from random-walking away from a good optimum.
   Inverse-square-root decay is empirically the right rate for
   transformers.

The paper found this schedule was *required* for stable training of the
big Transformer with the chosen Adam hyperparameters and the
`sqrt(d_model)` embedding scaling. Modern practice uses essentially the
same shape (linear warmup + cosine or inverse-sqrt decay).

### 7.3 The shape, in ASCII

```
lr
^
|        peak ≈ 1 / sqrt(warmup_steps * d_model)
|       _.--._
|      /      `--._
|     /            `-..___
|    /                    `------........_____
|   /
|  /
| /
|/_______________________________________________________> step
0     warmup_steps                            many * warmup_steps
```

For `d_model=32, warmup_steps=400`:

```
peak_lr   = 1 / sqrt(400 * 32) ≈ 1 / sqrt(12800) ≈ 0.00884
lr(step=1)= 1 / 32^0.5 * 1 / 400^1.5 ≈ 0.177 * 1 / 8000 ≈ 2.2e-5
lr(step=4000) = 1 / sqrt(32) * 1/sqrt(4000) ≈ 0.177 * 0.0158 ≈ 0.0028
```

The peak is ~0.009; it's a small lr by general-deep-learning standards
(Adam default is `1e-3 = 0.001`), which works because Adam's
normalization makes per-step magnitudes comparable across optimizers.

### 7.4 `warmup_steps` choice for our toy build

Paper uses `warmup_steps=4000` for the big Transformer trained on WMT.
We have 10,000 training samples, batch size we'll likely pick is 32, so
one epoch is `~312` steps. `warmup_steps=4000` would warm up over
~13 epochs -- way too long for a 13-vocab toy task that should converge
in 5-10 epochs.

Sensible choices for our build (TBD in slice 2):

| `warmup_steps` | Warmup duration       | Notes                                  |
|---------------|----------------------|----------------------------------------|
| `400`         | ~1.3 epochs          | Reasonable default                    |
| `100`         | ~0.3 epochs          | Aggressive; risk of early instability |
| `1000`        | ~3.2 epochs          | Conservative; slower start            |

Slice 2 will set `warmup_steps = 400` as the default and verify
convergence; revisit only if loss plateaus.

### 7.5 Constant LR as ablation (also covered)

A constant `lr` (no schedule) is the simplest possible alternative:

```
lr(step) = lr0    for all step
```

With `lr0 = peak_lr / 2` (e.g. `0.005` for our build), this gives a
useful baseline. We will document it in the Discussion Log of slice 2
as an ablation -- not the default, but a one-liner alternative for
debugging.

---

## 8. Putting it all together: the full training loop

Pseudo-code we will implement in slice 2:

```
def train(transformer, data, n_steps,
          batch_size=32, eps=0.1, warmup_steps=400, log_every=50, seed=0):
    rng = np.random.default_rng(seed)
    optimizer = Adam(beta1=0.9, beta2=0.98, eps=1e-9)

    transformer.train()
    loss_history = []
    lr_history = []

    for step in range(1, n_steps + 1):
        idx = rng.integers(0, n_train, size=batch_size)
        enc_in = data['encoder_input'][idx]
        dec_in = data['decoder_input'][idx]
        dec_tgt = data['decoder_target'][idx]

        logits = transformer.forward(enc_in, dec_in)
        loss, grad_logits = ce_with_label_smoothing(
            logits, dec_tgt, eps=eps, vocab_size=V, pad_id=PAD_ID
        )

        zero_grads(transformer.params_and_grads())
        transformer.backward(grad_logits)

        lr = noam_lr(step, d_model=32, warmup_steps=warmup_steps)
        optimizer.step(lr, transformer.params_and_grads())

        loss_history.append(float(loss))
        lr_history.append(float(lr))

        if step % log_every == 0:
            print(f"step {step}: loss={loss:.4f}, lr={lr:.5f}")

    return loss_history, lr_history
```

Tracing the dependencies between modules:

- `transformer.forward` and `transformer.backward` are Step 7.
- `params_and_grads` is Step 7.
- `ce_with_label_smoothing`, `noam_lr`, `Adam`, and `zero_grads` are Step 8.
- The data side (`encoder_input`, `decoder_input`, `decoder_target`) is
  Step 0/8 setup (already in `sort_data.py`).

Five new functions/classes that will live in
`numpy_transformer/training.py`:

1. `cross_entropy_with_label_smoothing(logits, targets, eps, V, pad_id) -> (loss, grad_logits)`
2. `noam_lr(step, d_model, warmup_steps) -> float`
3. `class Adam` with `__init__` and `step(lr, params_and_grads)`
4. `zero_grads(params_and_grads)` -- helper that sets every grad to None
   (or zeros) so the next backward starts fresh
5. `train(transformer, data, ...)` -- the loop above

---

## 9. Worked end-to-end numerical example (one mini-step)

Tiny config: `B=1, n_dec=2, V=4, d_model=4`. We'll trace one full
training step assuming all weights and gradients are scalar-shaped for
clarity.

### 9.1 Setup

```
logits  = [[ 1.0,  2.0, -1.0,  0.5],          # position 0, target = 1
           [ 0.5,  0.0,  3.0,  1.0]]          # position 1, target = 2
targets = [1, 2]
eps = 0.1, V = 4
```

Smoothed target distributions:

```
p_smooth[0] = [0.0333, 0.9, 0.0333, 0.0333]   # eps/(V-1) = 0.1/3 ≈ 0.0333
p_smooth[1] = [0.0333, 0.0333, 0.9, 0.0333]
```

### 9.2 Forward (loss)

```
softmax(logits[0]) = [0.2042, 0.5550, 0.0276, 0.1238]   # max-shifted then normalized
softmax(logits[1]) = [0.0617, 0.0374, 0.7521, 0.1488]

loss[0] = -(0.0333 * log(0.2042) + 0.9 * log(0.5550) + 0.0333 * log(0.0276) + 0.0333 * log(0.1238))
        = -(0.0333 * -1.589 + 0.9 * -0.589 + 0.0333 * -3.589 + 0.0333 * -2.089)
        = -(-0.053 + -0.530 + -0.120 + -0.070)
        = 0.773
loss[1] ≈ 0.434  (similar arithmetic with target=2 and lower-loss logits)

total_loss = (0.773 + 0.434) / 2 ≈ 0.604
```

### 9.3 Gradient back to logits

```
grad_logits[0] = (softmax(logits[0]) - p_smooth[0]) / count
              = ([0.2042, 0.5550, 0.0276, 0.1238] - [0.0333, 0.9, 0.0333, 0.0333]) / 2
              = [0.0855, -0.1725, -0.0029, 0.0453]

grad_logits[1] = (softmax(logits[1]) - p_smooth[1]) / count
              = ([0.0617, 0.0374, 0.7521, 0.1488] - [0.0333, 0.0333, 0.9, 0.0333]) / 2
              = [0.0142, 0.0021, -0.0740, 0.0578]
```

These flow through `transformer.backward(grad_logits)` (Step 7) and
eventually populate `(p, g, name)` triples for every parameter via
`params_and_grads()`.

### 9.4 Adam update on a single parameter

Suppose one parameter `W[i, j]` has analytical grad `g = 0.0855` at
step 1 (the first entry of `grad_logits[0]`, hypothetically backpropagated
to a particular weight).

```
m_1 = 0.9 * 0 + 0.1 * 0.0855  = 0.00855
v_1 = 0.98 * 0 + 0.02 * 0.0855^2 = 1.46e-4

m_hat = m_1 / (1 - 0.9)  = 0.0855       # bias-corrected: matches g
v_hat = v_1 / (1 - 0.98) = 7.31e-3
sqrt(v_hat) = 0.0855

lr at step=1: 1 / sqrt(32) * 1 / 400^1.5 ≈ 2.2e-5  (assuming d_model=32, warmup=400)
delta = lr * m_hat / (sqrt(v_hat) + 1e-9)
      = 2.2e-5 * 0.0855 / 0.0855
      ≈ 2.2e-5
W[i, j] ← W[i, j] - 2.2e-5
```

The effective per-step magnitude at step 1 is essentially `lr` (because
of bias correction normalizing the step). A 2e-5 update is small --
that's what warmup is doing. By step ~400 the lr is at its peak ~0.009
and steps are 100x larger.

### 9.5 Where the gradient `g` came from

The trace above assumed `g = 0.0855` for some specific weight. In
reality, `g` is the result of running `grad_logits` backward through:

```
grad_logits -> output_projection.backward
            -> through N decoder layers (each: LN, FFN, cross-attn, LN, self-attn, LN)
            -> through dropout, PE, embedding scaling
            -> embedding (input side)
            -> through encoder layers
            -> embedding (encoder input side)
```

This is the entire chain Step 7's `Transformer.backward` already does. We
do not re-derive it; we just kick it off with `grad_logits` and read off
the `(p, g)` pairs.

---

## 10. Sort-task specifics

### 10.1 Vocabulary

```
ID  0..9  : digits
ID  10    : PAD       (unused for fixed-length sort)
ID  11    : SOS
ID  12    : EOS
V   = 13
```

### 10.2 Shapes

```
encoder_input  : (B, 5)         e.g. [[3, 1, 4, 1, 5]]
decoder_input  : (B, 6)              [[SOS=11, 1, 1, 3, 4, 5]]
decoder_target : (B, 6)              [[1, 1, 3, 4, 5, EOS=12]]
logits         : (B, 6, 13)
grad_logits    : (B, 6, 13)
```

### 10.3 Expected behavior

The task is deterministic: given the multiset `{3, 1, 4, 1, 5}`, the
sort is the unique sequence `[1, 1, 3, 4, 5, EOS]`. Cross-entropy can
in principle reach the irreducible label-smoothing floor (close to
`H(p_smooth, p_smooth)` per token).

For one position with `V=13` and `eps=0.1`:

```
H(p_smooth, p_smooth) = -((1-eps) * log(1-eps) + (V-1) * (eps/(V-1)) * log(eps/(V-1)))
                      = -(0.9 * log(0.9) + 0.1 * log(0.1/12))
                      = -(0.9 * -0.1054 + 0.1 * -4.787)
                      = -( -0.0948 - 0.4787 )
                      ≈ 0.574
```

So a perfect model trained with `eps=0.1` on `V=13` will plateau at
**loss ≈ 0.574 per token**. We should not expect loss to go below this.
A common bug is to be confused by this floor; we'll annotate it on the
loss curve.

### 10.4 Convergence expectations

Empirically, the toy sort task with this architecture:

- **First 100 steps**: loss drops fast from initial ~`log(13) ≈ 2.56`
  toward ~`1.5`. Model learns "output digits, not random tokens".
- **100-500 steps**: loss drops from ~1.5 to ~1.0. Model learns "output
  digits in non-decreasing order".
- **500+ steps**: loss approaches the label-smoothing floor (~0.6).
  Model is correctly sorting almost every example.

A run of 2000-3000 steps with `batch_size=32` should be more than
enough. Slice 2's training script will be tuned to terminate when loss
plateaus.

---

## 11. Slice 2 plan

What lands in slice 2:

1. **`numpy_transformer/training.py`** with:
   - `log_softmax(logits)` -- numerically stable.
   - `cross_entropy_with_label_smoothing(logits, targets, eps, vocab_size, pad_id) -> (loss, grad_logits)`.
   - `class Adam` with `__init__(beta1, beta2, eps)`, `step(lr, params_and_grads)`,
     and a small per-parameter state dict keyed on parameter `id()`.
   - `zero_grads(params_and_grads)` -- sets each `g` to zeros (or `None`
     where the layer expects it; matches Step 7's accumulate contract).
   - `noam_lr(step, d_model, warmup_steps) -> float`.
   - `train(transformer, data, n_steps, batch_size, ...) -> (loss_history, lr_history)`.

2. **`tests/test_training.py`** with:
   - Unit tests for `log_softmax` (numerical stability, shape).
   - Unit test for `cross_entropy_with_label_smoothing` against the
     worked example in Section 9.
   - **Numerical gradient check** of CE+LS against finite differences on
     `logits` -- the most important correctness test.
   - Unit test for `noam_lr` shape: warmup linearly increasing,
     transition at `warmup_steps`, post-warmup decreasing.
   - Unit test for `Adam` against a known closed-form (e.g. constant
     gradient -> closed-form parameter trajectory).
   - **Integration test**: train the Transformer on a tiny dataset
     (e.g. 16 examples) for 200 steps; assert final loss is below a
     threshold (e.g. 1.5). This proves the full Step 7 + Step 8 stack
     can actually learn something. Marked `@pytest.mark.slow` if it
     exceeds ~10 seconds.

3. **`scripts/train_sort.py`** (or a notebook) that loads
   `train_input/sort_train.npz`, instantiates the Transformer, runs the
   training loop for a few thousand steps, and saves the loss curve to
   `train_output/loss_curve.csv` (or `.png`). This is the artifact that
   demonstrates "the project works".

4. **`docs/08_training.md` sections 12 (forward in code), 13 (backward
   verification), 14 (sanity tests)** -- filled in slice 2 with the
   actual code excerpts and the test catalog, mirroring Step 7's
   sections 10/11/12.

5. **Discussion Log update** -- slice 2 entry covering what was easy and
   what was tricky.

6. **`STATUS.md`** -- Step 8 row to "Complete".

---

## 12. Forward pass in code

The full module lives at
`numpy-transformer/src/numpy_transformer/training.py`. Five public
entry points; nothing in this module touches the layer internals from
Step 7 -- it interacts only through the `Transformer.forward`,
`Transformer.backward`, and `Transformer.params_and_grads` contract.

### 12.1 `log_softmax`

```python
def log_softmax(logits):
    m = logits.max(axis=-1, keepdims=True)
    shifted = logits - m
    log_sum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_sum
```

Three lines. The max-shift is the log-sum-exp trick (Section 3.3) that
keeps `exp` from overflowing for large logits. We use it both to compute
the loss value and to compute the softmax probabilities (`np.exp(log_softmax(z))`).

### 12.2 `cross_entropy_with_label_smoothing`

```python
def cross_entropy_with_label_smoothing(logits, targets, eps=0.1,
                                       vocab_size=None, pad_id=None):
    if vocab_size is None:
        vocab_size = logits.shape[-1]

    log_probs = log_softmax(logits)
    probs = np.exp(log_probs)

    valid_mask = (targets != pad_id) if pad_id is not None else np.ones_like(targets, bool)
    count = int(valid_mask.sum())
    if count == 0:
        return 0.0, np.zeros_like(logits)

    # Build smoothed target: (1-eps) at truth, eps/(V-1) elsewhere.
    safe_targets = np.where(valid_mask, targets, 0).astype(np.intp)
    p_smooth = np.full(logits.shape, eps / (vocab_size - 1))
    np.put_along_axis(p_smooth, safe_targets[..., None], 1.0 - eps, axis=-1)

    per_token_loss = -(p_smooth * log_probs).sum(axis=-1) * valid_mask
    loss = per_token_loss.sum() / count

    grad_logits = (probs - p_smooth) * valid_mask[..., None] / count
    return float(loss), grad_logits
```

Key implementation notes:

- **`safe_targets`**: PAD positions may carry sentinel target values
  outside `[0, V)` (e.g., a data layer may use a PAD ID that doesn't
  index into the logits). We clip to a safe placeholder index 0 there;
  the contribution at those positions is zeroed by `valid_mask` later.
- **`probs - p_smooth`**: this *is* the gradient at logits. Section 4.4
  derived it; the implementation is one line. Divided by `count` to
  match the mean-over-tokens reduction in the forward.
- **The `* valid_mask[..., None]` broadcast** zeros the gradient at PAD
  positions so the rest of backward propagates exactly zero through them
  -- the tests verify this.

### 12.3 `noam_lr`

```python
def noam_lr(step, d_model, warmup_steps):
    if step < 1: step = 1
    arg1 = step ** (-0.5)
    arg2 = step * (warmup_steps ** (-1.5))
    return (d_model ** (-0.5)) * float(min(arg1, arg2))
```

Direct transliteration of paper equation 3. The `step < 1` guard
prevents a divide-by-zero on the very first call before the loop has
incremented its counter (defensive only; the train loop is 1-indexed).

### 12.4 `Adam`

State per parameter, keyed by `id(param)`:

```python
class Adam:
    def __init__(self, beta1=0.9, beta2=0.98, eps=1e-9):
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.state = {}   # id(p) -> {'m': ..., 'v': ..., 't': int, 'name': str}

    def step(self, lr, params_and_grads):
        for p, g, name in params_and_grads:
            if g is None:
                continue
            key = id(p)
            s = self.state.get(key)
            if s is None or s['m'].shape != p.shape:
                s = {'m': np.zeros_like(p), 'v': np.zeros_like(p),
                     't': 0, 'name': name}
                self.state[key] = s

            s['t'] += 1
            t = s['t']
            s['m'] *= self.beta1
            s['m'] += (1.0 - self.beta1) * g
            s['v'] *= self.beta2
            s['v'] += (1.0 - self.beta2) * (g * g)

            m_hat = s['m'] / (1.0 - self.beta1 ** t)
            v_hat = s['v'] / (1.0 - self.beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p -= update
```

Three implementation choices worth flagging:

- **State keyed by `id(p)`**, not by name. Python's `id()` is stable for
  the lifetime of an object; as long as the layer keeps its weight
  reference alive (it always does), `id(p)` is a valid key. Using names
  would require each call to `params_and_grads` produce identical name
  strings; not a hard requirement, but `id()` avoids the assumption.
- **In-place update via `p -= update`**. `p` is the same NumPy array
  that the layer stored as `self.W` (or whatever). Updating in place
  means the layer's next forward sees the new values without any extra
  bookkeeping. **This is the only non-trivial Pythonic trick in the
  whole optimizer.**
- **`s['m'] *= beta1; s['m'] += (1 - beta1) * g`** is also in place,
  rather than `s['m'] = beta1 * s['m'] + (1 - beta1) * g` which would
  allocate two temporaries per parameter per step. Saves allocations on
  hot path.

### 12.5 `zero_grads` and the training loop

```python
def zero_grads(transformer):
    if hasattr(transformer, 'embedding'):
        transformer.embedding.grad_weight = None
```

Why this is so short: the only layer in our codebase that **accumulates**
across calls is `Embedding` (Step 7's PyTorch-style change). Every other
layer overwrites in `backward()`, so nothing else needs zeroing. We rely
on `Embedding.backward` allocating a fresh zero buffer when
`grad_weight is None`, which it does.

The loop:

```python
def train(transformer, data, n_steps, batch_size=32, eps=0.1,
          vocab_size=None, pad_id=None, d_model=None, warmup_steps=400,
          log_every=50, seed=0, print_fn=print):
    ...
    rng = np.random.default_rng(seed)
    optimizer = Adam(beta1=0.9, beta2=0.98, eps=1e-9)

    transformer.train()    # enable dropout

    loss_history, lr_history = [], []
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, n_train, size=batch_size)
        enc_in  = data['encoder_input'][idx]
        dec_in  = data['decoder_input'][idx]
        dec_tgt = data['decoder_target'][idx]

        logits = transformer.forward(enc_in, dec_in)
        loss, grad_logits = cross_entropy_with_label_smoothing(
            logits, dec_tgt, eps=eps, vocab_size=vocab_size, pad_id=pad_id)

        zero_grads(transformer)
        transformer.backward(grad_logits)

        lr = noam_lr(step, d_model=d_model, warmup_steps=warmup_steps)
        optimizer.step(lr, transformer.params_and_grads())

        loss_history.append(float(loss))
        lr_history.append(float(lr))

        if log_every and (step % log_every == 0 or step == 1):
            print_fn(f"step {step:6d}  loss={loss:.4f}  lr={lr:.6f}")
    return loss_history, lr_history
```

Six lines per step (batch fetch, forward, loss, zero, backward,
optimizer step), exactly matching the schematic in Section 2.

### 12.6 Convenience: `label_smoothing_loss_floor`

A small helper that returns `H(p_smooth, p_smooth)` per token; useful
for annotating loss curves so we don't get confused when training
plateaus above zero (Section 10.3). For our build (`V=13, eps=0.1`):

```python
>>> label_smoothing_loss_floor(0.1, 13)
0.5739...
```

The `train_sort.py` script prints this floor at startup and again at the
end so the convergence claim is calibrated.

---

## 13. Backward pass

There is no new backward code to write in Step 8. The whole point of
the Step 7 `Transformer.backward(grad_logits)` contract is that Step 8
just provides `grad_logits` and lets Step 7's code run.

The data flow on a single training step:

```
batch_enc, batch_dec_in, batch_dec_tgt
          v
   transformer.forward                       <-- Step 7
          v
        logits  (B, n_dec, V)
          v
   cross_entropy_with_label_smoothing        <-- Step 8 (this module)
          |
          +--> loss   (scalar, monitored)
          +--> grad_logits  (B, n_dec, V)
                  v
   zero_grads(transformer)                   <-- Step 8 (this module)
                  v
   transformer.backward(grad_logits)         <-- Step 7
   (populates *.grad_W, *.grad_b, etc., and
    accumulates into transformer.embedding.grad_weight
    via three paths: encoder lookup, decoder lookup,
    tied output projection)
                  v
   optimizer.step(lr, transformer.params_and_grads())  <-- Step 8
   (in-place update of every parameter)
```

The single new gradient identity is the one in Section 12.2:

```
grad_logits = (softmax(logits) - p_smooth) / count
            -- masked at PAD positions
```

Everything downstream is composition of `backward` methods we already
verified gradient-check-correct in steps 1-7.

### 13.1 Why `zero_grads` is mandatory

Step 7 changed `Embedding.backward` from "overwrite" to "accumulate"
specifically to support shared/tied weights. Without `zero_grads`, the
embedding gradient at step 2 would equal step-1's contribution + step-2's
contribution, growing without bound. The test
`test_zero_grads_prevents_grad_accumulation_across_steps` proves this:
running two backward passes without zeroing yields exactly `2x` the
gradient, and inserting `zero_grads(tf)` between them recovers the
single-pass value.

### 13.2 Why `Linear` and friends do not need `zero_grad`

`Linear.backward`:

```python
self.grad_W = x_2d.T @ grad_2d
self.grad_b = grad_2d.sum(axis=0)
```

This is *assignment*, not in-place addition. Each new backward
overwrites the previous result. Same for `LayerNorm`, `MultiHeadAttention`,
`FeedForward`. So `zero_grads` only needs to handle the one accumulating
layer.

---

## 14. Sanity checks and tests

All tests are in `nn/numpy-transformer/tests/test_training.py`. Run from
the workspace root:

```
cd nn && uv run --package numpy-transformer pytest numpy-transformer/tests/test_training.py -v
```

**25 tests, all passing.** The full project suite is now **100 tests
passing in ~4.3s** (75 from steps 1-7, 25 new for Step 8).

### 14.1 Unit tests: `log_softmax`

| Test | What it certifies |
|------|-------------------|
| `test_log_softmax_matches_naive_for_small_inputs` | Output equals the textbook formula `z - log(sum(exp(z)))` for ordinary inputs |
| `test_log_softmax_numerically_stable_for_large_logits` | `log_softmax([1000, 1000, 1001])` is finite and `exp(out).sum() == 1.0` |
| `test_log_softmax_shape_and_lastaxis` | Operates on the last axis only; rows independently normalize |

### 14.2 Unit tests: `cross_entropy_with_label_smoothing`

| Test | What it certifies |
|------|-------------------|
| `test_ce_no_smoothing_matches_negative_log_likelihood` | With `eps=0`, reduces to standard NLL |
| `test_ce_label_smoothing_floor_for_perfect_prediction` | When `softmax(logits) == p_smooth`, loss equals `H(p_smooth, p_smooth)` (the floor) |
| `test_ce_loss_floor_helper_value` | The published per-token floor of ~0.574 for `(V=13, eps=0.1)` (Section 10.3) is correct |
| `test_ce_gradient_matches_softmax_minus_smoothed_target` | Analytical gradient equals `(softmax - p_smooth) / count` exactly |
| `test_ce_pad_positions_have_zero_grad_and_excluded_from_count` | PAD positions contribute zero gradient and are excluded from the loss denominator |
| `test_ce_gradient_finite_difference` | Numerical-finite-difference gradient check; abs/rel error within `1e-7 / 1e-5` |
| `test_ce_count_normalization_makes_loss_batch_invariant` | Loss is identical under `(B, n)` and `(2B, n)` for duplicated data, confirming mean-over-tokens reduction |

The **gradient finite-difference test** is the most important
correctness signal: it rules out off-by-one bugs in the smoothing math
and the count normalization simultaneously.

### 14.3 Unit tests: `noam_lr`

| Test | What it certifies |
|------|-------------------|
| `test_noam_warmup_is_linear_in_step` | `lr(10*step) / lr(step) == 10.0` during warmup |
| `test_noam_continuous_at_warmup_boundary` | `lr(warmup_steps) == d_model^-0.5 * warmup_steps^-0.5` (both pieces agree) |
| `test_noam_decays_after_warmup` | Monotone decrease post-warmup; `lr(2 * warmup) / lr(warmup) == sqrt(1/2)` |
| `test_noam_d_model_scaling` | Scaling `d_model` by `4x` scales lr by `1/sqrt(4)` |
| `test_noam_rejects_invalid_warmup` | `warmup_steps=0` raises `ValueError` |

### 14.4 Unit tests: `Adam`

| Test | What it certifies |
|------|-------------------|
| `test_adam_first_step_is_lr_times_sign_with_bias_correction` | At step 1, `\|param_change\| ≈ lr` regardless of `\|g\|` (Adam's signature) |
| `test_adam_constant_gradient_long_run_approaches_signed_lr` | After 2000 steps with constant `g`, total displacement ≈ `2000 * lr * sign(g)` |
| `test_adam_per_parameter_state_is_independent` | Two parameters with very different gradients receive nearly equal step-1 updates |
| `test_adam_skips_none_gradient` | `grad=None` → no update |
| `test_adam_in_place_update_preserves_array_identity` | `id(param)` unchanged across calls; layers see updated values |
| `test_adam_bias_correction_at_step_1_recovers_raw_gradient` | `m_hat = g` exactly at step 1 |

### 14.5 Unit tests: `zero_grads`

| Test | What it certifies |
|------|-------------------|
| `test_zero_grads_resets_embedding_grad_buffer` | `transformer.embedding.grad_weight is None` after `zero_grads` |
| `test_zero_grads_prevents_grad_accumulation_across_steps` | Without zeroing, two passes give `2 * grad`; with zeroing, the second pass equals the first |

### 14.6 Integration tests

| Test | What it certifies |
|------|-------------------|
| `test_train_loop_actually_learns` | **Headline test for Step 8.** Train on 128 samples for 300 steps; assert avg loss drops by `> 0.3` and stays above the smoothing floor. Wires up forward + loss + backward + Adam + Noam end-to-end. |
| `test_train_loss_curve_at_step_1_matches_uniform_prior` | At step 1 a freshly initialized model produces loss within ±25% of `log(V) ≈ 2.56`, the entropy of the uniform prior |

### 14.7 Runnable training script

`scripts/train_sort.py` provides a CLI to train the toy Transformer on
the full 10k-sample sort-task dataset. With defaults
(`d_model=32, h=4, d_ff=64, N=2 each, batch=32, warmup=400`),
2000 steps converge in ~9s on CPU:

```
step      1  loss=2.5876  lr=0.000022      <-- ~log(13) = 2.5649 (uniform prior)
step    250  loss=0.8091  lr=0.005524
step    500  loss=0.6981  lr=0.007906
step    750  loss=0.6335  lr=0.006455
step   1000  loss=0.6076  lr=0.005590
step   1500  loss=0.5925  lr=0.004564
step   2000  loss=0.5799  lr=0.003953      <-- floor = 0.5736
```

The script writes a per-step `loss_curve.csv` to the data directory.
The output is the canonical "Step 8 was successful" artifact.

---

## Discussion Log

A chronological log of every conceptual exchange. Append, do not edit.

### 2026-04-25 -- Step 8 kickoff (slice 1: conceptual deep-dive only)

**No code this turn.** Slice 1 is dedicated to the doc; slice 2 lands
the loss, optimizer, schedule, training loop, tests, and a runnable
training script.

**Five conceptually new topics resolved here:**

1. **Cross-entropy loss.** Section 3. Standard NLL with the log-sum-exp
   trick for numerical stability. The crucial identity:
   `dL/dlogits = softmax(logits) - one_hot(label)`. This is the only
   "math" we add on top of Step 7's backward; everything else is
   composition.

2. **Label smoothing.** Section 4. Replace one-hot target with
   `(1-eps)` on truth and `eps/(V-1)` uniform on rest. Same gradient
   shape (`softmax - target_distribution`), just a softened target.
   Paper section 5.4. We default it on with `eps=0.1`.

3. **Adam optimizer.** Section 6. Per-parameter adaptive lr using running
   first and second moments with bias correction. Paper hyperparams
   `beta1=0.9, beta2=0.98 (NOT 0.999), eps=1e-9 (NOT 1e-8)`.

4. **Noam LR schedule.** Section 7. `lr = d_model^-0.5 * min(step^-0.5,
   step * warmup^-1.5)`. Linear warmup (essential for Adam stability
   under transformer initial conditions) then inverse-sqrt decay.

5. **The full training loop.** Section 8. Six lines per step. The
   `params_and_grads()` contract from Step 7 makes the optimizer
   structure-agnostic.

**Five design decisions made before slice 1 was written, confirmed by
user:**

- **Optimizer = Adam (paper-faithful).** Doc covers the math in detail
  with the paper's specific hyperparameters.
- **LR schedule = Noam.** Constant LR is documented as an ablation in
  this Discussion Log (below).
- **Label smoothing = on by default with `eps=0.1`.** Flag for
  ablation.
- **Loss reduction = mean over valid (non-PAD) tokens.** Makes lr
  invariant to batch size and seq length. Consistent with PyTorch's
  `F.cross_entropy(reduction='mean')` default.
- **Slice 1 scope = full conceptual deep-dive in one doc.** Mirrors
  steps 6/7 slice 1.

**Cross-references made in this doc:**

- Teacher-forcing mechanic: section 5 of `07_decoder.md`. We re-state
  it here in concrete shapes (`(B, 5)`, `(B, 6)`, `(B, 6, 13)`).
- The `Transformer` interface: sections 10-11 of `07_decoder.md`. We
  rely on `forward(source, target)`, `backward(grad_logits)`,
  `params_and_grads()`.
- Embedding accumulate-on-backward contract: section 6 of
  `07_decoder.md`. The `zero_grads` step in the training loop
  exists *because* of this contract.
- Sort-task data layout: `numpy-transformer/src/numpy_transformer/sort_data.py`.

**Things deliberately deferred to later:**

- **Padding masks.** Sort task is fixed-length; no padding. The CE
  function will accept `pad_id` for forward-compatibility.
- **Validation loop.** Slice 2 will track loss on the train split only.
  A separate val pass on `sort_test.npz` is a one-liner add to slice 2
  if it doesn't crowd out the unit tests.
- **Gradient clipping.** Paper does not use it; we won't either. If
  training is unstable, revisit.
- **Mixed precision / checkpointing.** Out of scope.

**Constant-LR ablation (documented, not implemented):**

A simple alternative to Noam is `lr(step) = lr0` for all `step`. With
`lr0 = peak_noam_lr / 2 ≈ 0.005` for our build, this gives a useful
baseline. The user chose to keep Noam as the default and have constant
LR documented but not implemented. If slice 2 training is unstable with
Noam, we will add a `--lr-schedule constant --lr 0.005` flag as a
debugging aid.

**Key insight worth flagging for slice 2:**

The label-smoothing **loss floor** is non-zero. A perfect model trained
with `eps=0.1, V=13` will plateau at `loss ≈ 0.574` per token (Section
10.3 derivation). We must annotate this on the loss curve in slice 2 so
the convergence claim is calibrated correctly.

**Open questions for slice 2 (to discuss before code):**

- **`warmup_steps` value.** Paper uses 4000 for the big Transformer.
  Section 7.4 proposes 400 for our toy. Confirmed default unless slice
  2 training shows instability.
- **Adam state dict keying.** Two reasonable choices: keyed by parameter
  `id()` (Pythonic, requires no extra bookkeeping) or by parameter name
  from `params_and_grads()` (clearer for debugging, but names need to
  be unique). Slice 2 will go with `id()`; revisit if confusing.
- **Test threshold for the integration test.** "After 200 steps on 16
  examples, loss < 1.5" is a guess. Calibrate empirically once slice 2
  trains.

**Next turn (slice 2):** implement `log_softmax`,
`cross_entropy_with_label_smoothing`, `Adam`, `zero_grads`, `noam_lr`,
`train`, the test suite from Section 11, and the
`scripts/train_sort.py` runnable. Fill in sections 12, 13, 14.

### 2026-04-25 -- Step 8 slice 2 (code, tests, runnable)

**What landed:**

- `nn/numpy-transformer/src/numpy_transformer/training.py` (new) -- 314
  lines covering `log_softmax`, `cross_entropy_with_label_smoothing`,
  `noam_lr`, `Adam`, `zero_grads`, `train`, and a
  `label_smoothing_loss_floor` helper. Module docstring annotates each.
- `nn/numpy-transformer/tests/test_training.py` (new) -- 25 tests
  covering each public function plus the integration test that proves
  forward + loss + backward + Adam wire up correctly.
- `nn/numpy-transformer/scripts/train_sort.py` (new) -- CLI runnable
  that trains the canonical toy build (43,168 parameters) on the full
  10k-sample sort dataset and writes `train_output/loss_curve.csv`.
- `nn/numpy-transformer/src/numpy_transformer/__init__.py` -- exports
  the new training utilities.
- Doc sections 12 (forward in code), 13 (backward), and 14 (sanity
  tests) filled in with code excerpts and a 25-row test catalog.

**Test result:** 100/100 passing in ~4.3s. Step 8's 25 tests run in
~0.6s, dominated by the 300-step integration test.

**Convergence demo (2000 steps, defaults):**

```
step      1  loss=2.5876   <-- ~log(13) = 2.5649, the uniform-prior baseline
step    750  loss=0.6335
step   2000  loss=0.5799   <-- floor = 0.5736
```

The model reaches the label-smoothing floor in ~9s on CPU. Loss drops
~75% in the first 250 steps, then sub-linearly toward the floor for the
remaining 1750 steps. By step ~750 we are within `~0.06` of the floor.
This matches the Section 10.4 expectation almost exactly.

**Three implementation realisations during slice 2:**

1. **PAD-aware loss has to clip indices, not just mask.**
   `np.put_along_axis` raises `IndexError` if any index is out of range,
   even at masked positions. Solution: replace PAD positions with a
   placeholder index `0` before scattering, then mask the resulting
   loss/gradient via `valid_mask`. The placeholder's contribution is
   zeroed before it can influence anything. The unit test
   `test_ce_pad_positions_have_zero_grad_and_excluded_from_count`
   surfaced this almost immediately on first run -- caught a real bug
   in a single iteration.

2. **`Adam` state must NOT close over the parameter array's value.**
   The optimizer mutates `param` in place via `param -= update`. If
   `Adam.state` had stored a reference to a *snapshot* of the
   parameter (e.g. `s['p_initial'] = p.copy()`), it would silently
   diverge from the layer's current state. We key by `id(p)` and store
   only `m, v, t` -- no parameter value -- so the optimizer is a pure
   stateless transformation per call. The test
   `test_adam_in_place_update_preserves_array_identity` guards against
   accidental array re-allocation.

3. **`zero_grads` is shorter than expected.** Originally I sketched a
   recursive walker that visits every layer and resets every `grad_*`
   attribute. After a careful audit, the only accumulating layer in the
   codebase is `Embedding`. Everything else overwrites in `backward`,
   so a recursive walker would do nothing useful for them. The final
   `zero_grads(transformer)` is a 2-line function:
   `transformer.embedding.grad_weight = None`. The
   `test_zero_grads_prevents_grad_accumulation_across_steps` test
   pinned this contract.

**Trade-offs accepted:**

- **No validation pass during training.** As discussed in slice 1's
  Discussion Log, slice 2 keeps the loop lean. A validation forward over
  `sort_test.npz` is a one-liner that the next iteration (or Step 10)
  can add without changing any signatures.
- **`label_smoothing_loss_floor` exposed publicly.** A small helper
  rather than an internal calculation, because the script and tests
  both want it. Worth a public symbol for clarity even though it's
  arithmetic.
- **`Adam` does not implement weight decay.** Paper section 5.4 lists
  three regularizers: dropout, label smoothing, and the schedule
  itself. Weight decay (AdamW) is not in the paper. We follow the
  paper.
- **No early stopping.** The training loop runs `n_steps` regardless.
  For the toy task, the user sees the loss curve and decides when to
  stop. Adding patience-based early stopping is a future extension.

**Open questions resolved by slice 2 results:**

- **`warmup_steps=400` is correct** for our toy build. Loss curve shape
  is healthy, no instability observed in any test run.
- **Adam state-dict keying by `id()`** has caused zero issues across
  100 tests. Confirmed.
- **Integration-test loss threshold = 0.3 drop in 300 steps** is
  comfortably hit (initial avg ~2.55, final avg ~1.0 in slice 2's
  smoke run; integration test config is even smaller and consistently
  drops by `>= 0.3`).

**Validated by slice 2's experimental runs:**

- Step-1 loss is exactly `~log(V)` for a Xavier-initialized model
  with eps=0 -- this is the uniform-prior baseline, and confirms the
  loss is correctly normalized.
- Loss approaches but does not cross the label-smoothing floor of
  `~0.574` for `(V=13, eps=0.1)` -- the floor is mathematically
  irreducible, and it being approached but not crossed is a *positive*
  signal (proves the floor calculation is correct and the loss is
  numerically stable near the optimum).
- Sort-task convergence in `~750 steps` to within `0.06` of the floor.
  No need for >2000 steps.

**Next step (Step 9: Inference):** implement greedy / sampling /
beam-search decoding. The trained Transformer is ready; we just need a
loop that, instead of teacher-forcing the ground truth, feeds back the
model's own predictions one token at a time.

### 2026-04-28 -- Train/test split design (Q&A captured from chat)

User noticed during a debug-session walk-through that
`sort_train.npz` and `sort_test.npz` "look the same, just different
sizes" and asked whether that was a bug. It is not -- it is the
defining property of a held-out test set. The lasting strategy
material on splitting/sizing already lives in
`numpy-transformer/train_input/README.md` (sections "Strategy:
splitting and sizing training data" and "For this sort task
specifically"); this entry preserves the conversational framing for
future readers who hit the same intuition.

**The two requirements that pull in opposite directions.**
A held-out test set must satisfy both at once:

1. **Same distribution as training.** Test inputs must be drawn from
   the same family of problems we trained on. Otherwise a low test
   score could mean either "model failed to learn the task" or "we
   asked it a different task" -- ambiguity we cannot afford. So the
   shape, vocab, and statistics of the two splits must match.
2. **Different concrete samples.** We cannot reuse training rows --
   that would just measure how well the model memorized them. Test
   rows must be fresh draws from the same distribution.

The recipe "same generator, different seed" in
`write_default_datasets()` (`seed_train=42`, `seed_test=4242`) is
precisely how to satisfy both at once. Same generator => same
distribution. Different seeds => independent samples.

**Why the sizes differ (`n_train=10_000` vs `n_test=1_000`).**
Train rows are *reused thousands of times* across epochs of gradient
descent -- each row contributes signal at every batch it appears in,
so we want a lot of variety. Test rows are *seen exactly once* at
evaluation time and only need to be enough for a stable accuracy
estimate. With `n_test=1_000`, a measured accuracy of `97.5%` has a
standard error of about
`sqrt(0.975 * 0.025 / 1000) ≈ 0.5%` -- plenty tight for a toy
benchmark. Going to 10k test rows would mostly slow evaluation
without telling us anything new. The split (10k / 1k = 90/10) sits in
the standard 80/10-90/10 train/test rule-of-thumb band.

**Train-test overlap caveat (acknowledged, not corrected).**
The space of possible 5-digit sequences is only `10^5 = 100_000`. We
draw `10_000 + 1_000` independently with no deduplication, so the
expected number of test rows that happen to also appear in training
is roughly
`(10_000 * 1_000) / 100_000 = 100` (give or take). For a serious
benchmark we would dedupe; here we deliberately do not, because the
sort function is so structurally regular that we want the model to
learn the *rule*, not memorize a lookup table -- and even with ~100
accidental overlaps, the other ~900 test rows are genuinely novel and
the model gets no special hint about which is which. Recorded so the
trade-off is explicit.

**Asymmetric field usage between splits (separate but related point).**

The `.npz` files store three arrays each: `encoder_input`,
`decoder_input`, `decoder_target`. Train and test consume them
asymmetrically:

| field          | train (in `train()`)    | test (in `run_evaluation`) |
|----------------|-------------------------|---------------------------|
| encoder_input  | fed to encoder          | fed to encoder via `greedy_decode` |
| decoder_input  | teacher-forced into decoder | **not used** (only its shape gets printed) |
| decoder_target | compared with logits to compute loss | answer key for accuracy comparison |

So `test_data['decoder_input']` is technically dead weight in this
codebase -- referenced once for a `print(...shape)` banner and never
fed to the model. We keep it on disk for two reasons: (a) symmetry
(both splits go through the same `generate_sort_split()` function so
they share schema, simpler code), and (b) optional future
teacher-forced test-loss measurement, which would let us track the
generalization gap (train CE vs test CE) alongside accuracy. The
field is there if Step 10 or future verification work wants it.
Cross-references the deeper point that *the model never sees the
target at test time* -- see
`09_inference.md` for the full information-barrier argument.

**Cross-references:**

- Lasting strategy / sizing rules of thumb:
  `numpy-transformer/train_input/README.md` sections "Strategy:
  splitting and sizing training data" and "For this sort task
  specifically".
- Generator implementation: `generate_sort_split()` and
  `write_default_datasets()` in
  `numpy-transformer/src/numpy_transformer/sort_data.py`.
- Train-time consumption of all three fields: section 5 of this doc
  ("Teacher forcing") and `train()` in `training.py`.
- Test-time consumption of only encoder_input + decoder_target:
  `run_evaluation()` in `scripts/train_sort.py` and
  `09_inference.md`.
