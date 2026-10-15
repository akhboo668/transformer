"""
Step 12: Training -- loss, LR schedule, optimizer wiring, training loop.

What replaces the NumPy equivalents:
  - cross_entropy_with_label_smoothing  ->  F.cross_entropy(ignore_index, label_smoothing)
  - Adam class + zero_grads             ->  torch.optim.Adam + optimizer.zero_grad()
  - Manual lr assignment per step       ->  LambdaLR scheduler
  - Raw array batching                  ->  DataLoader (cycling)
"""

from __future__ import annotations

import math
from itertools import cycle, islice

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from pytorch_transformer.layers import Transformer


# ---------------------------------------------------------------------------
# Noam LR schedule (paper equation 3)
# lr(step) = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
# ---------------------------------------------------------------------------

def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Paper equation 3. 1-indexed; linear warmup then inverse-sqrt decay."""
    step = max(1, step)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


def label_smoothing_loss_floor(eps: float, vocab_size: int) -> float:
    """Minimum achievable loss under label smoothing for a perfect model."""
    if eps == 0.0:
        return 0.0
    return -(
        (1.0 - eps) * math.log(1.0 - eps)
        + eps * math.log(eps / (vocab_size - 1))
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: Transformer,
    dataloader: DataLoader,
    n_steps: int,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.1,
    warmup_steps: int = 400,
    log_every: int = 100,
    device: str | torch.device = "cpu",
    print_fn=print,
) -> tuple[list[float], list[float]]:
    """Train model for exactly n_steps gradient steps, cycling the dataloader.

    Parameters
    ----------
    model:
        Transformer to train. Modified in place.
    dataloader:
        Yields (encoder_input, decoder_input, decoder_target) int64 batches.
        Cycled automatically if n_steps exceeds one epoch.
    n_steps:
        Total number of (forward, loss, backward, update) iterations.
    ignore_index:
        Token id excluded from the loss (PAD). Forwarded to F.cross_entropy.
    label_smoothing:
        Paper section 5.4 uses 0.1.
    warmup_steps:
        Noam schedule warmup length.
    log_every:
        Print a loss/lr line every N steps. 0 = silent.
    device:
        Where to move batches. Model must already be on this device.

    Returns
    -------
    loss_history, lr_history : lists of floats, one entry per step.
    """
    device = torch.device(device)
    d_model = model.d_model

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_lr(step + 1, d_model, warmup_steps),
    )

    model.train()
    loss_history: list[float] = []
    lr_history: list[float] = []

    for step, (enc, dec_in, dec_tgt) in enumerate(
        islice(cycle(dataloader), n_steps), start=1
    ):
        enc    = enc.to(device)
        dec_in = dec_in.to(device)
        dec_tgt = dec_tgt.to(device)

        logits = model(enc, dec_in)                    # (B, T, V)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),           # (B*T, V)
            dec_tgt.view(-1),                           # (B*T,)
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        lr_val   = scheduler.get_last_lr()[0]
        loss_history.append(loss_val)
        lr_history.append(lr_val)

        if log_every and (step % log_every == 0 or step == 1):
            print_fn(f"step {step:6d}  loss={loss_val:.4f}  lr={lr_val:.6f}")

    return loss_history, lr_history
