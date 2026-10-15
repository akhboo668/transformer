"""
Tests for nn/numpy-transformer/src/numpy_transformer/inference.py
(Steps 9 / 10: autoregressive decoding and sort-task accuracy).

The headline test is `test_trained_model_actually_sorts`, which trains
a small Transformer on real sort data and asserts that greedy decoding
recovers the correct sorted sequence on most held-out examples. This
certifies that Steps 1-8's full forward + backward + training pipeline
produces a model that can actually sort.
"""

import numpy as np

from numpy_transformer.inference import evaluate_sort_accuracy, greedy_decode
from numpy_transformer.layers import Transformer
from numpy_transformer.sort_data import EOS_ID, SOS_ID, VOCAB_SIZE, generate_sort_split
from numpy_transformer.training import train


def _tiny_untrained_model(seed: int = 0) -> Transformer:
    np.random.seed(seed)
    return Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        p_drop=0.0,
    )


def test_greedy_decode_returns_int_array_with_max_len_cap():
    """An untrained model probably never emits EOS quickly, so the
    output should be exactly `max_len` tokens unless EOS shows up first.
    All emitted tokens are valid vocab ids."""
    model = _tiny_untrained_model(seed=11)
    source = np.array([3, 1, 4, 1, 5], dtype=np.int64)
    out = greedy_decode(model, source, sos_id=SOS_ID, eos_id=EOS_ID, max_len=8)

    assert out.dtype == np.int64
    assert out.ndim == 1
    assert 1 <= out.shape[0] <= 8
    assert ((out >= 0) & (out < VOCAB_SIZE)).all()


def test_greedy_decode_stops_immediately_when_eos_dominates():
    """Force argmax to emit EOS at every step by zeroing every output
    logit except EOS. Decoder should produce exactly [EOS] and stop."""
    model = _tiny_untrained_model(seed=12)
    source = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    # Monkey-patch the decoder to always return logits where EOS_ID
    # is the argmax. The simplest way is to wrap the existing forward.
    real_forward = model.decoder.forward

    def stub_forward(target_ids, enc_out):
        logits = real_forward(target_ids, enc_out)
        logits[..., :] = -1e9
        logits[..., EOS_ID] = 0.0
        return logits

    model.decoder.forward = stub_forward
    try:
        out = greedy_decode(model, source, sos_id=SOS_ID, eos_id=EOS_ID, max_len=10)
    finally:
        model.decoder.forward = real_forward

    np.testing.assert_array_equal(out, np.array([EOS_ID], dtype=np.int64))


def test_greedy_decode_respects_max_len_when_eos_never_emitted():
    """Force EOS to never be the argmax; decoder must hit max_len."""
    model = _tiny_untrained_model(seed=13)
    source = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    real_forward = model.decoder.forward

    def stub_forward(target_ids, enc_out):
        logits = real_forward(target_ids, enc_out)
        logits[..., EOS_ID] = -1e9
        return logits

    model.decoder.forward = stub_forward
    try:
        out = greedy_decode(model, source, sos_id=SOS_ID, eos_id=EOS_ID, max_len=7)
    finally:
        model.decoder.forward = real_forward

    assert out.shape == (7,)
    assert (out != EOS_ID).all()


def test_greedy_decode_accepts_2d_source_with_batch_one():
    """source_ids may already be (1, n); shouldn't be reshaped twice."""
    model = _tiny_untrained_model(seed=14)
    source_2d = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    out = greedy_decode(model, source_2d, sos_id=SOS_ID, eos_id=EOS_ID, max_len=4)
    assert out.shape[0] >= 1


def test_greedy_decode_rejects_batch_size_above_one():
    """We support single-sequence decoding only (slice scope)."""
    model = _tiny_untrained_model(seed=15)
    source = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    try:
        greedy_decode(model, source, sos_id=SOS_ID, eos_id=EOS_ID, max_len=4)
    except ValueError as e:
        assert "single-sequence" in str(e)
    else:
        raise AssertionError("expected ValueError for batch_size > 1")


def test_evaluate_sort_accuracy_handles_zero_correct():
    """Untrained model: accuracy is approximately zero."""
    model = _tiny_untrained_model(seed=16)
    data = generate_sort_split(n=8, seq_len=5, seed=16)
    acc, examples = evaluate_sort_accuracy(
        model,
        data['encoder_input'],
        data['decoder_target'],
        sos_id=SOS_ID,
        eos_id=EOS_ID,
        max_len=8,
    )
    assert 0.0 <= acc <= 1.0
    assert acc < 0.5, f"Untrained model should be far from solving sort, got {acc}"
    assert examples == []


def test_evaluate_sort_accuracy_returns_examples_when_requested():
    model = _tiny_untrained_model(seed=17)
    data = generate_sort_split(n=4, seq_len=5, seed=17)
    acc, examples = evaluate_sort_accuracy(
        model,
        data['encoder_input'],
        data['decoder_target'],
        sos_id=SOS_ID,
        eos_id=EOS_ID,
        max_len=8,
        return_examples=2,
    )
    assert len(examples) == 2
    for src, pred, expected in examples:
        assert src.shape == (5,)
        assert pred.ndim == 1 and pred.shape[0] >= 1
        assert expected.shape == (6,)


def test_trained_model_actually_sorts():
    """Headline test for Steps 9 / 10.

    Train a small Transformer for ~1500 steps on real sort data; assert
    that greedy decoding sorts >70% of held-out examples exactly.

    This is the test that ties the entire 8-step pipeline together:
    embeddings, attention, encoder, decoder, training, AND inference all
    have to be correct for the model to actually sort.
    """
    np.random.seed(101)
    train_data = generate_sort_split(n=2000, seq_len=5, seed=101)
    test_data = generate_sort_split(n=64, seq_len=5, seed=202)

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        p_drop=0.0,
        tie_embeddings=True,
    )

    train(
        model,
        train_data,
        n_steps=1500,
        batch_size=32,
        eps=0.1,
        warmup_steps=200,
        log_every=0,
        seed=101,
    )

    acc, _ = evaluate_sort_accuracy(
        model,
        test_data['encoder_input'],
        test_data['decoder_target'],
        sos_id=SOS_ID,
        eos_id=EOS_ID,
        max_len=8,
    )
    assert acc > 0.70, (
        f"Trained Transformer should sort >70% of held-out 5-digit sequences; "
        f"got accuracy={acc:.3f}. Either training is broken, "
        f"inference is broken, or the model is too small."
    )


def test_trained_model_outputs_eos_within_max_len():
    """A trained model should learn to emit EOS roughly when expected
    (after seq_len = 5 sorted digits). Don't require exact correctness
    here -- just that EOS appears."""
    np.random.seed(102)
    train_data = generate_sort_split(n=512, seq_len=5, seed=102)
    test_data = generate_sort_split(n=8, seq_len=5, seed=303)

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        p_drop=0.0,
    )
    train(
        model, train_data, n_steps=600, batch_size=16, eps=0.1,
        warmup_steps=80, log_every=0, seed=102,
    )

    eos_emitted = 0
    for src in test_data['encoder_input']:
        out = greedy_decode(model, src, sos_id=SOS_ID, eos_id=EOS_ID, max_len=10)
        if out[-1] == EOS_ID and out.shape[0] <= 8:
            eos_emitted += 1
    assert eos_emitted >= 6, (
        f"Trained model should emit EOS within max_len for most sequences; "
        f"only {eos_emitted}/8 did"
    )
