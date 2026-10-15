"""
Step 8: Training -- loss, optimizer, learning-rate schedule, training loop.

Conceptual map: see nn/docs/08_training.md.

Five new pieces in this module:

    log_softmax(logits)
        Numerically stable log-softmax along the last axis.

    cross_entropy_with_label_smoothing(logits, targets, eps, vocab_size, pad_id)
        Forward + gradient. The only "new math" added on top of Step 7's
        backward chain. Returns (scalar_loss, grad_logits) where
        grad_logits = (softmax(logits) - p_smooth) / count, with PAD positions
        zeroed.

    noam_lr(step, d_model, warmup_steps)
        Paper equation 3: d_model^-0.5 * min(step^-0.5, step * warmup^-1.5).

    class Adam
        Per-parameter adaptive optimizer with bias-corrected first and
        second moments. Paper section 5.3 hyperparams (beta1=0.9,
        beta2=0.98, eps=1e-9). State keyed by id(param).

    zero_grads(transformer)
        Reset accumulating gradients before each backward pass. Only the
        Embedding's grad_weight accumulates (PyTorch convention introduced
        in Step 7); every other layer overwrites in its backward. So
        zeroing reduces to nulling out the shared embedding's grad buffer.

    train(transformer, data, n_steps, ...)
        The glue: batch -> forward -> loss -> zero_grads -> backward ->
        optimizer step. Returns (loss_history, lr_history).

Paper: "Attention Is All You Need" (Vaswani et al., 2017), section 5.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Numerically stable log-softmax
# ---------------------------------------------------------------------------

def log_softmax(logits: np.ndarray) -> np.ndarray:
    """
    log_softmax along the last axis.

    Uses the log-sum-exp trick: subtract max(logits) before exponentiating
    so the largest exponent is exp(0) = 1. Mathematically a no-op (softmax
    is shift-invariant); numerically essential for large logits.

    Returns an array with the same shape as `logits`.
    """
    m = logits.max(axis=-1, keepdims=True)
    shifted = logits - m
    log_sum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_sum


# ---------------------------------------------------------------------------
# 2. Cross-entropy with label smoothing
# ---------------------------------------------------------------------------

def cross_entropy_with_label_smoothing(
    logits: np.ndarray,
    targets: np.ndarray,
    eps: float = 0.1,
    vocab_size: int | None = None,
    pad_id: int | None = None,
) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy loss with label smoothing.

    Parameters
    ----------
    logits : (..., V) float array
        Raw, unnormalised scores. Typical shape (B, n_dec, V).
    targets : (...) int array
        Class indices. Typical shape (B, n_dec). Must be in [0, V).
    eps : float
        Label-smoothing factor (paper section 5.4 uses 0.1). When 0,
        reduces to plain cross-entropy with hard one-hot targets.
    vocab_size : int, optional
        Defaults to logits.shape[-1].
    pad_id : int, optional
        If given, positions where `targets == pad_id` are excluded from
        the loss and contribute zero gradient.

    Returns
    -------
    loss : float
        Mean loss over valid (non-PAD) tokens.
    grad_logits : ndarray with same shape as `logits`
        d(loss) / d(logits). Equals (softmax(logits) - p_smooth) / count
        at valid positions and 0 at PAD positions.
    """
    if vocab_size is None:
        vocab_size = logits.shape[-1]
    if vocab_size <= 1:
        raise ValueError("vocab_size must be >= 2 for label smoothing")
    if eps < 0.0 or eps >= 1.0:
        raise ValueError(f"eps must be in [0, 1), got {eps}")

    log_probs = log_softmax(logits)
    probs = np.exp(log_probs)

    if pad_id is not None:
        valid_mask = (targets != pad_id)
    else:
        valid_mask = np.ones_like(targets, dtype=bool)

    count = int(valid_mask.sum())
    if count == 0:
        return 0.0, np.zeros_like(logits)

    # Build smoothed target distribution p_smooth in the same shape as logits.
    # p_smooth[truth] = 1 - eps; p_smooth[non-truth] = eps / (V - 1)
    #
    # PAD positions may carry sentinel target values outside [0, V) (e.g.
    # data layers sometimes reuse a vocab ID that is not represented in the
    # logits dimension). We clip indices to a safe range before scattering;
    # the contribution at PAD positions is zeroed below by `valid_mask`.
    safe_targets = np.where(valid_mask, targets, 0).astype(np.intp)
    if (safe_targets < 0).any() or (safe_targets >= vocab_size).any():
        raise ValueError(
            "Non-PAD target index out of range [0, vocab_size); "
            "did you forget to pass pad_id?"
        )
    p_smooth = np.full(logits.shape, eps / (vocab_size - 1), dtype=logits.dtype)
    np.put_along_axis(p_smooth, safe_targets[..., None], 1.0 - eps, axis=-1)

    # Per-token loss: -sum_i p_smooth[i] * log_probs[i]
    per_token_loss = -(p_smooth * log_probs).sum(axis=-1)

    # Mask PAD positions before averaging.
    per_token_loss = per_token_loss * valid_mask
    loss = per_token_loss.sum() / count

    # Gradient: dloss/dlogits = (softmax - p_smooth) / count, masked.
    grad_logits = (probs - p_smooth) * valid_mask[..., None]
    grad_logits = grad_logits / count

    return float(loss), grad_logits


# ---------------------------------------------------------------------------
# 3. Noam learning-rate schedule (paper equation 3)
# ---------------------------------------------------------------------------

def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """
    lr(step) = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)

    `step` is 1-indexed; `noam_lr(1, ...)` is the smallest lr produced.
    Linear warmup until step == warmup_steps, then inverse-sqrt decay.
    Both pieces evaluate to warmup_steps^-0.5 at the boundary.
    """
    if step < 1:
        step = 1
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")
    arg1 = step ** (-0.5)
    arg2 = step * (warmup_steps ** (-1.5))
    return (d_model ** (-0.5)) * float(min(arg1, arg2))


# ---------------------------------------------------------------------------
# 4. Adam optimizer (Kingma & Ba 2015; paper section 5.3 hyperparameters)
# ---------------------------------------------------------------------------

class Adam:
    """
    Adam with bias correction.

        m_t = beta1 * m_{t-1} + (1 - beta1) * g
        v_t = beta2 * v_{t-1} + (1 - beta2) * g**2
        m_hat = m_t / (1 - beta1**t)
        v_hat = v_t / (1 - beta2**t)
        p   <- p - lr * m_hat / (sqrt(v_hat) + epsilon)

    Paper hyperparameters (section 5.3):
        beta1   = 0.9
        beta2   = 0.98          (NOT the Adam default 0.999)
        epsilon = 1e-9          (NOT the Adam default 1e-8)

    State (m, v, t) is per parameter, keyed by id(param). The optimizer
    is parameter-agnostic: `step(lr, params_and_grads)` walks an iterable
    of (param, grad, name) tuples and updates each `param` in place.

    Notes
    -----
    - Updates `param` IN PLACE via `param -= update`. The caller's layer
      keeps its weight reference unchanged, so the new values are visible
      on the next forward.
    - If `grad is None` (parameter never received a gradient), the update
      is silently skipped.
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.98, eps: float = 1e-9):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # state[id(p)] = {'m': ndarray, 'v': ndarray, 't': int, 'name': str}
        self.state: dict[int, dict] = {}

    def step(
        self,
        lr: float,
        params_and_grads: Iterable[Tuple[np.ndarray, np.ndarray, str]],
    ) -> None:
        for p, g, name in params_and_grads:
            if g is None:
                continue
            key = id(p)
            s = self.state.get(key)
            if s is None or s['m'].shape != p.shape:
                s = {
                    'm': np.zeros_like(p),
                    'v': np.zeros_like(p),
                    't': 0,
                    'name': name,
                }
                self.state[key] = s

            s['t'] += 1
            t = s['t']
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g
            s['m'] *= self.beta1
            s['m'] += (1.0 - self.beta1) * g
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            s['v'] *= self.beta2
            s['v'] += (1.0 - self.beta2) * (g * g)

            bias1 = 1.0 - self.beta1 ** t
            bias2 = 1.0 - self.beta2 ** t
            m_hat = s['m'] / bias1
            v_hat = s['v'] / bias2

            update = lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p -= update


# ---------------------------------------------------------------------------
# 5. zero_grads -- reset accumulating buffers between training steps
# ---------------------------------------------------------------------------

def zero_grads(transformer) -> None:
    """
    Reset accumulating gradient buffers before the next backward pass.

    In our codebase only `Embedding.grad_weight` accumulates (PyTorch-style;
    introduced in Step 7 to support tied weights and shared embeddings).
    `Linear`, `LayerNorm`, `MultiHeadAttention`, and `FeedForward` all
    overwrite their grads in `backward`, so explicit zeroing is a no-op
    for those layers and can be skipped.

    We accept the full Transformer (or any object exposing
    `.embedding.grad_weight`) and null out its embedding grad. Setting to
    None lets the next `Embedding.backward` call allocate a fresh zero
    buffer of the right shape.
    """
    if hasattr(transformer, 'embedding'):
        transformer.embedding.grad_weight = None


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------

def train(
    transformer,
    data: dict,
    n_steps: int,
    batch_size: int = 32,
    eps: float = 0.1,
    vocab_size: int | None = None,
    pad_id: int | None = None,
    d_model: int | None = None,
    warmup_steps: int = 400,
    log_every: int = 50,
    seed: int = 0,
    print_fn=print,
) -> Tuple[List[float], List[float]]:
    """
    Train `transformer` on `data` for `n_steps` mini-batch steps.

    Parameters
    ----------
    transformer : Transformer
        Must expose `forward(enc_in, dec_in)`, `backward(grad_logits)`,
        `params_and_grads()`, `train(mode=True)`, and `.embedding`.
    data : dict
        Must contain 'encoder_input', 'decoder_input', 'decoder_target'
        as 2D int arrays with matching first dimension.
    n_steps : int
        Number of (forward, loss, backward, update) iterations.
    batch_size : int
    eps : float
        Label-smoothing factor.
    vocab_size, pad_id :
        Forwarded to the loss. Defaults inferred from the model and data.
    d_model : int, optional
        Used by the Noam schedule. If None, taken from `transformer.d_model`.
    warmup_steps : int
        Noam warmup length.
    log_every : int
        Print every N steps. Set to 0 to silence.
    seed : int
        Seeds the batch sampler (numpy default_rng).
    print_fn :
        Logging hook; replace for tests.

    Returns
    -------
    loss_history : list of float
    lr_history   : list of float
    """
    enc_in_all = data['encoder_input']
    dec_in_all = data['decoder_input']
    dec_tgt_all = data['decoder_target']
    n_train = enc_in_all.shape[0]
    if dec_in_all.shape[0] != n_train or dec_tgt_all.shape[0] != n_train:
        raise ValueError("encoder/decoder inputs/targets must agree on first axis")

    if vocab_size is None:
        vocab_size = getattr(transformer, 'vocab_size', None)
        if vocab_size is None:
            vocab_size = int(dec_tgt_all.max()) + 1

    if d_model is None:
        d_model = getattr(transformer, 'd_model', None)
        if d_model is None:
            raise ValueError(
                "d_model not provided and not discoverable on transformer")

    rng = np.random.default_rng(seed)
    optimizer = Adam(beta1=0.9, beta2=0.98, eps=1e-9)

    transformer.train()

    loss_history: List[float] = []
    lr_history: List[float] = []

    for step in range(1, n_steps + 1):
        idx = rng.integers(0, n_train, size=batch_size)
        enc_in = enc_in_all[idx]
        dec_in = dec_in_all[idx]
        dec_tgt = dec_tgt_all[idx]

        logits = transformer.forward(enc_in, dec_in)
        loss, grad_logits = cross_entropy_with_label_smoothing(
            logits, dec_tgt, eps=eps, vocab_size=vocab_size, pad_id=pad_id
        )

        zero_grads(transformer)
        transformer.backward(grad_logits)

        lr = noam_lr(step, d_model=d_model, warmup_steps=warmup_steps)
        optimizer.step(lr, transformer.params_and_grads())

        loss_history.append(float(loss))
        lr_history.append(float(lr))

        if log_every and (step % log_every == 0 or step == 1):
            print_fn(f"step {step:6d}  loss={loss:.4f}  lr={lr:.6f}")

    return loss_history, lr_history


# ---------------------------------------------------------------------------
# 7. Convenience: theoretical label-smoothing loss floor (per token)
# ---------------------------------------------------------------------------

def label_smoothing_loss_floor(eps: float, vocab_size: int) -> float:
    """
    Per-token cross-entropy a *perfect* model achieves under label
    smoothing. Useful for annotating loss curves.

    H(p_smooth, p_smooth)
        = -((1-eps) log(1-eps) + (V-1) * (eps/(V-1)) * log(eps/(V-1)))
        = -((1-eps) log(1-eps) + eps * log(eps/(V-1)))
    """
    if eps == 0.0:
        return 0.0
    return -(
        (1.0 - eps) * math.log(1.0 - eps)
        + eps * math.log(eps / (vocab_size - 1))
    )
