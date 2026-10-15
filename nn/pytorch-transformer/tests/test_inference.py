"""Tests for greedy_decode and evaluate_accuracy."""

import numpy as np
import torch
import torch.nn as nn

from pytorch_transformer.inference import evaluate_accuracy, greedy_decode
from pytorch_transformer.layers import Transformer

VOCAB   = 13
SOS_ID  = 11
EOS_ID  = 12
PAD_ID  = 10


def _small_model() -> Transformer:
    return Transformer(VOCAB, d_model=16, num_heads=2, d_ff=32,
                       num_encoder_layers=1, num_decoder_layers=1)


# ---------------------------------------------------------------------------
# greedy_decode
# ---------------------------------------------------------------------------

def test_greedy_decode_output_is_1d():
    model = _small_model()
    src   = torch.randint(0, 10, (5,))
    out   = greedy_decode(model, src, SOS_ID, EOS_ID, max_len=10)
    assert out.dim() == 1


def test_greedy_decode_accepts_2d_input():
    model = _small_model()
    src   = torch.randint(0, 10, (1, 5))
    out   = greedy_decode(model, src, SOS_ID, EOS_ID, max_len=10)
    assert out.dim() == 1


def test_greedy_decode_stops_at_eos():
    """A model that always predicts EOS should return a length-1 sequence."""
    model = _small_model()
    # Patch output so argmax is always EOS_ID.
    with torch.no_grad():
        model.embedding.weight.zero_()
    # Force the decoder to emit EOS by zeroing all logits except EOS.
    original_forward = model.decoder.forward
    def forced_eos(target_ids, enc_out):
        logits = torch.full((1, target_ids.size(1), VOCAB), -1e9)
        logits[:, :, EOS_ID] = 1e9
        return logits
    model.decoder.forward = forced_eos

    src = torch.randint(0, 10, (5,))
    out = greedy_decode(model, src, SOS_ID, EOS_ID, max_len=20)
    assert out.tolist() == [EOS_ID]


def test_greedy_decode_respects_max_len():
    """If EOS is never emitted the sequence length == max_len."""
    model = _small_model()
    # Force decoder to always predict token 0 (not EOS).
    def never_eos(target_ids, enc_out):
        logits = torch.full((1, target_ids.size(1), VOCAB), -1e9)
        logits[:, :, 0] = 1e9
        return logits
    model.decoder.forward = never_eos

    src = torch.randint(0, 10, (5,))
    out = greedy_decode(model, src, SOS_ID, EOS_ID, max_len=7)
    assert len(out) == 7


def test_greedy_decode_deterministic_in_eval():
    """Two calls on the same input must return identical sequences."""
    model = _small_model()
    src   = torch.randint(0, 10, (5,))
    out1  = greedy_decode(model, src, SOS_ID, EOS_ID)
    out2  = greedy_decode(model, src, SOS_ID, EOS_ID)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# evaluate_accuracy
# ---------------------------------------------------------------------------

def test_evaluate_accuracy_perfect():
    """Accuracy must be 1.0 when the model always generates the target."""
    model = _small_model()
    # Build 4 simple examples where target = [t, EOS]
    target_tok = 3
    enc  = torch.zeros(4, 5, dtype=torch.long)
    tgt  = torch.tensor([[target_tok, EOS_ID]] * 4)

    def perfect_decoder(target_ids, enc_out):
        logits = torch.full((1, target_ids.size(1), VOCAB), -1e9)
        # emit target_tok at every step except after EOS
        pos = target_ids.size(1) - 1
        emit = target_tok if pos == 0 else EOS_ID
        logits[:, :, emit] = 1e9
        return logits
    model.decoder.forward = perfect_decoder

    acc = evaluate_accuracy(model, enc, tgt, SOS_ID, EOS_ID, max_len=5)
    assert acc == 1.0


def test_evaluate_accuracy_zero():
    """Accuracy is 0 when the model always emits the wrong token."""
    model = _small_model()
    enc = torch.zeros(4, 5, dtype=torch.long)
    tgt = torch.tensor([[3, EOS_ID]] * 4)

    def wrong_decoder(target_ids, enc_out):
        logits = torch.full((1, target_ids.size(1), VOCAB), -1e9)
        logits[:, :, 5] = 1e9        # always predict token 5, not 3
        return logits
    model.decoder.forward = wrong_decoder

    acc = evaluate_accuracy(model, enc, tgt, SOS_ID, EOS_ID, max_len=5)
    assert acc == 0.0


def test_evaluate_accuracy_returns_float():
    model = _small_model()
    enc = torch.randint(0, 10, (10, 5))
    tgt = torch.randint(0, 10, (10, 6))
    acc = evaluate_accuracy(model, enc, tgt, SOS_ID, EOS_ID, max_len=8)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
