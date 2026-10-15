"""Tests for training loop, loss schedule, and LR schedule."""

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformer.layers import Transformer
from pytorch_transformer.training import (
    label_smoothing_loss_floor,
    noam_lr,
    train,
)

VOCAB = 13
PAD_ID = 10


# ---------------------------------------------------------------------------
# noam_lr
# ---------------------------------------------------------------------------

def test_noam_lr_warmup_is_linear():
    """During warmup the schedule is linear (step * warmup^-1.5 dominates)."""
    warmup = 400
    lrs = [noam_lr(s, d_model=32, warmup_steps=warmup) for s in range(1, warmup + 1)]
    # Each step should be larger than the previous during warmup.
    assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1))


def test_noam_lr_peaks_at_warmup():
    """LR at warmup == LR just after warmup (boundary is the peak)."""
    d, w = 32, 400
    at_warmup  = noam_lr(w,     d, w)
    just_after = noam_lr(w + 1, d, w)
    assert at_warmup >= just_after


def test_noam_lr_decays_after_warmup():
    """After warmup, every subsequent step has a lower LR."""
    warmup = 400
    post = [noam_lr(s, 32, warmup) for s in range(warmup, warmup + 200)]
    assert all(post[i] > post[i + 1] for i in range(len(post) - 1))


def test_noam_lr_step_zero_clamped():
    """Step <= 0 is clamped to 1, not NaN/inf."""
    lr = noam_lr(0, 32, 400)
    assert math.isfinite(lr) and lr > 0


# ---------------------------------------------------------------------------
# label_smoothing_loss_floor
# ---------------------------------------------------------------------------

def test_label_smoothing_floor_zero_eps():
    assert label_smoothing_loss_floor(0.0, 13) == 0.0


def test_label_smoothing_floor_positive():
    floor = label_smoothing_loss_floor(0.1, 13)
    assert floor > 0


def test_label_smoothing_floor_formula():
    eps, V = 0.1, 13
    expected = -((1 - eps) * math.log(1 - eps) + eps * math.log(eps / (V - 1)))
    assert abs(label_smoothing_loss_floor(eps, V) - expected) < 1e-12


# ---------------------------------------------------------------------------
# train() integration
# ---------------------------------------------------------------------------

def _tiny_sort_dataset(n: int = 256, seq: int = 5) -> TensorDataset:
    """Synthetic digit-sort dataset (no file I/O)."""
    rng = np.random.default_rng(0)
    enc     = rng.integers(0, 10, (n, seq), dtype=np.int64)
    sorted_ = np.sort(enc, axis=1)
    dec_in  = np.concatenate([np.full((n, 1), 11), sorted_], axis=1).astype(np.int64)
    dec_tgt = np.concatenate([sorted_, np.full((n, 1), 12)],  axis=1).astype(np.int64)
    return TensorDataset(
        torch.from_numpy(enc),
        torch.from_numpy(dec_in),
        torch.from_numpy(dec_tgt),
    )


def test_train_loop_loss_decreases():
    """Loss after 300 steps must be lower than the initial loss."""
    ds = _tiny_sort_dataset()
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = Transformer(VOCAB, d_model=16, num_heads=2, d_ff=32,
                        num_encoder_layers=1, num_decoder_layers=1)
    losses, _ = train(model, dl, n_steps=300, ignore_index=PAD_ID,
                      warmup_steps=100, log_every=0)
    assert losses[-1] < losses[0], "Loss did not decrease"


def test_train_loop_returns_correct_length():
    ds = _tiny_sort_dataset(n=64)
    dl = DataLoader(ds, batch_size=32)
    model = Transformer(VOCAB, d_model=16, num_heads=2, d_ff=32,
                        num_encoder_layers=1, num_decoder_layers=1)
    losses, lrs = train(model, dl, n_steps=50, log_every=0)
    assert len(losses) == 50
    assert len(lrs)    == 50


def test_train_loop_lr_warms_up():
    """LR must increase during the warmup phase."""
    ds = _tiny_sort_dataset(n=64)
    dl = DataLoader(ds, batch_size=32)
    model = Transformer(VOCAB, d_model=16, num_heads=2, d_ff=32,
                        num_encoder_layers=1, num_decoder_layers=1)
    _, lrs = train(model, dl, n_steps=100, warmup_steps=200, log_every=0)
    assert lrs[-1] > lrs[0], "LR did not increase during warmup"


def test_train_loop_cycles_dataloader():
    """n_steps can exceed one epoch -- dataloader cycles automatically."""
    ds = _tiny_sort_dataset(n=32)
    dl = DataLoader(ds, batch_size=32)          # only 1 batch per epoch
    model = Transformer(VOCAB, d_model=16, num_heads=2, d_ff=32,
                        num_encoder_layers=1, num_decoder_layers=1)
    losses, _ = train(model, dl, n_steps=20, log_every=0)   # 20 > 1 epoch
    assert len(losses) == 20


def test_train_produces_finite_losses():
    ds = _tiny_sort_dataset()
    dl = DataLoader(ds, batch_size=32)
    model = Transformer(VOCAB, d_model=16, num_heads=2, d_ff=32,
                        num_encoder_layers=1, num_decoder_layers=1)
    losses, _ = train(model, dl, n_steps=50, log_every=0)
    assert all(math.isfinite(l) for l in losses)
