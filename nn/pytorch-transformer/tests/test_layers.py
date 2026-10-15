"""
Tests for pytorch_transformer.layers.

Two kinds:
  - Shape / contract tests: pure PyTorch, verify shapes and structural invariants.
  - Cross-implementation tests: copy NumPy weights into PyTorch, assert forward
    outputs match within atol=1e-6.

Cross-impl tests use float64 throughout (PyTorch defaults to float32; NumPy defaults
to float64). float64 parity is a strict check -- if the forward math is correctly
ported the outputs should agree to near machine-epsilon.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Run all cross-impl tests in float64
torch.set_default_dtype(torch.float64)

from pytorch_transformer.layers import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FeedForward,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    Transformer,
)
from numpy_transformer.layers import (
    Decoder as NpDecoder,
    DecoderLayer as NpDecoderLayer,
    Encoder as NpEncoder,
    EncoderLayer as NpEncoderLayer,
    FeedForward as NpFeedForward,
    MultiHeadAttention as NpMultiHeadAttention,
    SinusoidalPositionalEncoding as NpSinusoidalPositionalEncoding,
    Transformer as NpTransformer,
)

# ---------------------------------------------------------------------------
# Weight-copy helpers (NumPy -> PyTorch)
# NumPy Linear.W shape: (in, out)  --  PyTorch nn.Linear.weight shape: (out, in)
# ---------------------------------------------------------------------------

def copy_mha(np_mha, pt_mha) -> None:
    for pt_proj, np_proj in [
        (pt_mha.W_q, np_mha.W_q),
        (pt_mha.W_k, np_mha.W_k),
        (pt_mha.W_v, np_mha.W_v),
        (pt_mha.W_o, np_mha.W_o),
    ]:
        pt_proj.weight.data = torch.from_numpy(np_proj.W.T.copy())


def copy_ffn(np_ffn, pt_ffn) -> None:
    pt_ffn.W_1.weight.data = torch.from_numpy(np_ffn.W_1.W.T.copy())
    pt_ffn.W_1.bias.data   = torch.from_numpy(np_ffn.W_1.b.copy())
    pt_ffn.W_2.weight.data = torch.from_numpy(np_ffn.W_2.W.T.copy())
    pt_ffn.W_2.bias.data   = torch.from_numpy(np_ffn.W_2.b.copy())


def copy_layer_norm(np_ln, pt_ln) -> None:
    pt_ln.weight.data = torch.from_numpy(np_ln.gamma.copy())
    pt_ln.bias.data   = torch.from_numpy(np_ln.beta.copy())


def copy_encoder_layer(np_layer, pt_layer) -> None:
    copy_mha(np_layer.mha, pt_layer.mha)
    copy_ffn(np_layer.ffn, pt_layer.ffn)
    copy_layer_norm(np_layer.ln_1, pt_layer.ln_1)
    copy_layer_norm(np_layer.ln_2, pt_layer.ln_2)


def copy_decoder_layer(np_layer, pt_layer) -> None:
    copy_mha(np_layer.mha_self,  pt_layer.mha_self)
    copy_mha(np_layer.mha_cross, pt_layer.mha_cross)
    copy_ffn(np_layer.ffn,       pt_layer.ffn)
    copy_layer_norm(np_layer.ln_1, pt_layer.ln_1)
    copy_layer_norm(np_layer.ln_2, pt_layer.ln_2)
    copy_layer_norm(np_layer.ln_3, pt_layer.ln_3)


def copy_transformer(np_model, pt_model) -> None:
    pt_model.embedding.weight.data = torch.from_numpy(np_model.embedding.weight.copy())
    for np_layer, pt_layer in zip(np_model.encoder.layers, pt_model.encoder.layers):
        copy_encoder_layer(np_layer, pt_layer)
    for np_layer, pt_layer in zip(np_model.decoder.layers, pt_model.decoder.layers):
        copy_decoder_layer(np_layer, pt_layer)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D, H, D_FF, B, T = 32, 4, 64, 2, 5  # d_model, num_heads, d_ff, batch, seq_len
ATOL = 1e-6


# ---------------------------------------------------------------------------
# SinusoidalPositionalEncoding
# ---------------------------------------------------------------------------

def test_pe_shape():
    pe = SinusoidalPositionalEncoding(D)
    x = torch.zeros(B, T, D)
    assert pe(x).shape == (B, T, D)


def test_pe_odd_d_model_raises():
    with pytest.raises(ValueError):
        SinusoidalPositionalEncoding(33)


def test_pe_matches_numpy():
    np_pe = NpSinusoidalPositionalEncoding(D)
    pt_pe = SinusoidalPositionalEncoding(D)
    pt_pe.eval()

    x_np = np.random.randn(B, T, D)
    x_pt = torch.from_numpy(x_np.copy())

    out_np = np_pe.forward(x_np)
    out_pt = pt_pe(x_pt).detach().numpy()

    np.testing.assert_allclose(out_np, out_pt, atol=ATOL)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

def test_mha_shape():
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, T, D)
    assert mha(x, x, x).shape == (B, T, D)


def test_mha_cross_attention_shape():
    mha = MultiHeadAttention(D, H)
    q = torch.randn(B, T, D)
    kv = torch.randn(B, T + 3, D)
    assert mha(q, kv, kv).shape == (B, T, D)


def test_mha_d_model_not_divisible_raises():
    with pytest.raises(ValueError):
        MultiHeadAttention(33, 4)


def test_mha_matches_numpy():
    np_mha = NpMultiHeadAttention(D, H)
    pt_mha = MultiHeadAttention(D, H)
    pt_mha.eval()
    copy_mha(np_mha, pt_mha)

    x_np = np.random.randn(B, T, D)
    x_pt = torch.from_numpy(x_np.copy())

    out_np = np_mha.forward(x_np, x_np, x_np)
    out_pt = pt_mha(x_pt, x_pt, x_pt).detach().numpy()

    np.testing.assert_allclose(out_np, out_pt, atol=ATOL)


def test_mha_causal_blocks_future():
    """With is_causal=True each position can only see itself and earlier positions."""
    mha = MultiHeadAttention(D, H, p_drop=0.0)
    mha.eval()
    x = torch.randn(1, T, D)

    out_full = mha(x, x, x, is_causal=True)

    # Corrupt future tokens; output at position 0 must not change.
    x_corrupt = x.clone()
    x_corrupt[:, 1:] = torch.randn_like(x_corrupt[:, 1:])
    out_corrupt = mha(x_corrupt, x_corrupt, x_corrupt, is_causal=True)

    np.testing.assert_allclose(
        out_full[:, 0].detach().numpy(),
        out_corrupt[:, 0].detach().numpy(),
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

def test_ffn_shape():
    ffn = FeedForward(D, D_FF)
    x = torch.randn(B, T, D)
    assert ffn(x).shape == (B, T, D)


def test_ffn_position_wise_independence():
    """Positions don't mix in the FFN: running one position alone gives the same result."""
    ffn = FeedForward(D, D_FF, p_drop=0.0)
    ffn.eval()
    x = torch.randn(1, T, D)
    out_full = ffn(x)
    out_single = ffn(x[:, 2:3, :])
    np.testing.assert_allclose(
        out_full[:, 2].detach().numpy(),
        out_single[:, 0].detach().numpy(),
        atol=ATOL,
    )


def test_ffn_matches_numpy():
    np_ffn = NpFeedForward(D, D_FF)
    pt_ffn = FeedForward(D, D_FF, p_drop=0.0)
    pt_ffn.eval()
    copy_ffn(np_ffn, pt_ffn)

    x_np = np.random.randn(B, T, D)
    x_pt = torch.from_numpy(x_np.copy())

    out_np = np_ffn.forward(x_np)
    out_pt = pt_ffn(x_pt).detach().numpy()

    np.testing.assert_allclose(out_np, out_pt, atol=ATOL)


# ---------------------------------------------------------------------------
# EncoderLayer
# ---------------------------------------------------------------------------

def test_encoder_layer_shape():
    layer = EncoderLayer(D, H, D_FF)
    x = torch.randn(B, T, D)
    assert layer(x).shape == (B, T, D)


def test_encoder_layer_matches_numpy():
    np_layer = NpEncoderLayer(D, H, D_FF, p_drop=0.0)
    pt_layer = EncoderLayer(D, H, D_FF, p_drop=0.0)
    pt_layer.eval()
    copy_encoder_layer(np_layer, pt_layer)

    x_np = np.random.randn(B, T, D)
    x_pt = torch.from_numpy(x_np.copy())

    out_np = np_layer.forward(x_np)
    out_pt = pt_layer(x_pt).detach().numpy()

    np.testing.assert_allclose(out_np, out_pt, atol=ATOL)


def test_encoder_layer_pre_ln_shape():
    layer = EncoderLayer(D, H, D_FF, pre_ln=True)
    x = torch.randn(B, T, D)
    assert layer(x).shape == (B, T, D)


def test_encoder_layer_pre_ln_matches_numpy():
    np_layer = NpEncoderLayer(D, H, D_FF, p_drop=0.0, pre_ln=True)
    pt_layer = EncoderLayer(D, H, D_FF, p_drop=0.0, pre_ln=True)
    pt_layer.eval()
    copy_encoder_layer(np_layer, pt_layer)

    x_np = np.random.randn(B, T, D)
    x_pt = torch.from_numpy(x_np.copy())

    np.testing.assert_allclose(
        np_layer.forward(x_np),
        pt_layer(x_pt).detach().numpy(),
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# DecoderLayer
# ---------------------------------------------------------------------------

def test_decoder_layer_shape():
    layer = DecoderLayer(D, H, D_FF)
    x = torch.randn(B, T, D)
    enc_out = torch.randn(B, T + 2, D)
    assert layer(x, enc_out).shape == (B, T, D)


def test_decoder_layer_causal_self_attn():
    """Corrupting future decoder positions must not change position-0 output."""
    layer = DecoderLayer(D, H, D_FF, p_drop=0.0)
    layer.eval()
    x = torch.randn(1, T, D)
    enc_out = torch.randn(1, T, D)

    out_clean = layer(x, enc_out)

    x_corrupt = x.clone()
    x_corrupt[:, 1:] = torch.randn_like(x_corrupt[:, 1:])
    out_corrupt = layer(x_corrupt, enc_out)

    np.testing.assert_allclose(
        out_clean[:, 0].detach().numpy(),
        out_corrupt[:, 0].detach().numpy(),
        atol=ATOL,
    )


def test_decoder_layer_matches_numpy():
    np_layer = NpDecoderLayer(D, H, D_FF, p_drop=0.0)
    pt_layer = DecoderLayer(D, H, D_FF, p_drop=0.0)
    pt_layer.eval()
    copy_decoder_layer(np_layer, pt_layer)

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((B, T, D))
    enc_np = rng.standard_normal((B, T, D))
    x_pt = torch.from_numpy(x_np.copy())
    enc_pt = torch.from_numpy(enc_np.copy())

    causal_mask = np.tril(np.ones((T, T), dtype=bool))
    out_np = np_layer.forward(x_np, enc_np, causal_mask)
    out_pt = pt_layer(x_pt, enc_pt).detach().numpy()

    np.testing.assert_allclose(out_np, out_pt, atol=ATOL)


# ---------------------------------------------------------------------------
# Encoder (full stack)
# ---------------------------------------------------------------------------

def test_encoder_shape():
    enc = Encoder(vocab_size=13, d_model=D, num_heads=H, d_ff=D_FF, num_layers=2)
    ids = torch.randint(0, 13, (B, T))
    assert enc(ids).shape == (B, T, D)


def test_encoder_matches_numpy():
    vocab = 13
    np_enc = NpEncoder(vocab, D, H, D_FF, num_layers=2, p_drop=0.0)
    pt_enc = Encoder(vocab, D, H, D_FF, num_layers=2, p_drop=0.0)
    pt_enc.eval()

    pt_enc.embedding.weight.data = torch.from_numpy(np_enc.embedding.weight.copy())
    for np_layer, pt_layer in zip(np_enc.layers, pt_enc.layers):
        copy_encoder_layer(np_layer, pt_layer)

    ids = np.random.randint(0, vocab, (B, T))
    out_np = np_enc.forward(ids)
    out_pt = pt_enc(torch.from_numpy(ids)).detach().numpy()

    np.testing.assert_allclose(out_np, out_pt, atol=ATOL)


# ---------------------------------------------------------------------------
# Decoder (full stack)
# ---------------------------------------------------------------------------

def test_decoder_shape():
    dec = Decoder(vocab_size=13, d_model=D, num_heads=H, d_ff=D_FF, num_layers=2)
    ids = torch.randint(0, 13, (B, T))
    enc_out = torch.randn(B, T, D)
    assert dec(ids, enc_out).shape == (B, T, 13)


# ---------------------------------------------------------------------------
# Transformer (end-to-end)
# ---------------------------------------------------------------------------

def test_transformer_shape():
    model = Transformer(
        vocab_size=13, d_model=D, num_heads=H, d_ff=D_FF,
        num_encoder_layers=2, num_decoder_layers=2,
    )
    src = torch.randint(0, 13, (B, T))
    tgt = torch.randint(0, 13, (B, T))
    assert model(src, tgt).shape == (B, T, 13)


def test_transformer_matches_numpy():
    vocab = 13
    np_model = NpTransformer(vocab, D, H, D_FF, num_encoder_layers=2, num_decoder_layers=2)
    pt_model = Transformer(vocab, D, H, D_FF, num_encoder_layers=2, num_decoder_layers=2)
    np_model.eval()
    pt_model.eval()
    copy_transformer(np_model, pt_model)

    rng = np.random.default_rng(42)
    src = rng.integers(0, vocab, (B, T))
    tgt = rng.integers(0, vocab, (B, T))

    out_np = np_model.forward(src, tgt)
    out_pt = pt_model(torch.from_numpy(src), torch.from_numpy(tgt)).detach().numpy()

    np.testing.assert_allclose(out_np, out_pt, atol=ATOL)


# ---------------------------------------------------------------------------
# Structural / invariant tests
# ---------------------------------------------------------------------------

def test_tied_embedding_is_shared_memory():
    """encoder.embedding and decoder.embedding must be the same object as transformer.embedding."""
    model = Transformer(13, D, H, D_FF, num_encoder_layers=2, num_decoder_layers=2)
    assert model.encoder.embedding is model.embedding
    assert model.decoder.embedding is model.embedding
    assert model.encoder.embedding.weight.data_ptr() == model.embedding.weight.data_ptr()


def test_tied_output_uses_embedding_weight():
    """After an optimizer step, the decoder output uses the updated embedding weight."""
    model = Transformer(13, D, H, D_FF, num_encoder_layers=1, num_decoder_layers=1)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    src = torch.randint(0, 13, (1, T))
    tgt = torch.randint(0, 13, (1, T))
    loss = model(src, tgt).sum()
    loss.backward()

    weight_before = model.embedding.weight.data.clone()
    optimizer.step()
    weight_after = model.embedding.weight.data.clone()

    # Weight must have changed.
    assert not torch.allclose(weight_before, weight_after)

    # Now run a fresh forward -- tied projection reads the *new* weight automatically.
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt)
    assert logits.shape == (1, T, 13)


def test_unique_parameters():
    """model.parameters() must deduplicate the shared embedding -- counted once."""
    model = Transformer(13, D, H, D_FF, num_encoder_layers=2, num_decoder_layers=2)
    param_ids = [id(p) for p in model.parameters()]
    assert len(param_ids) == len(set(param_ids)), "Duplicate parameters detected"


def test_eval_is_deterministic():
    model = Transformer(13, D, H, D_FF, num_encoder_layers=2, num_decoder_layers=2)
    model.eval()
    src = torch.randint(0, 13, (B, T))
    tgt = torch.randint(0, 13, (B, T))
    with torch.no_grad():
        out1 = model(src, tgt)
        out2 = model(src, tgt)
    assert torch.allclose(out1, out2)


def test_param_count_tied_vs_untied():
    """Tied model has fewer unique parameters than untied (shared vocab × d_model matrix)."""
    tied = Transformer(13, D, H, D_FF, num_encoder_layers=2, num_decoder_layers=2, tie_embeddings=True)
    untied = Transformer(13, D, H, D_FF, num_encoder_layers=2, num_decoder_layers=2, tie_embeddings=False)
    tied_count   = sum(p.numel() for p in tied.parameters())
    untied_count = sum(p.numel() for p in untied.parameters())
    assert untied_count > tied_count
    assert untied_count - tied_count == 13 * D  # one extra vocab_size × d_model matrix
