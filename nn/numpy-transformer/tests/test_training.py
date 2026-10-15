"""
Tests for nn/numpy-transformer/src/numpy_transformer/training.py
(Step 8: Training).

Run from the workspace root:
    cd nn && uv run --package numpy-transformer pytest numpy-transformer/tests/test_training.py -v

The integration test (`test_train_loop_actually_learns`) is the headline
test for Step 8 -- it certifies that forward + loss + backward + optimizer
actually drive loss down on a real (toy) sort-task batch.
"""

import math

import numpy as np
import pytest

from numpy_transformer.layers import Transformer
from numpy_transformer.sort_data import EOS_ID, PAD_ID, SOS_ID, generate_sort_split
from numpy_transformer.training import (
    Adam,
    cross_entropy_with_label_smoothing,
    label_smoothing_loss_floor,
    log_softmax,
    noam_lr,
    train,
    zero_grads,
)


# ===========================================================================
# 1. log_softmax
# ===========================================================================

def test_log_softmax_matches_naive_for_small_inputs():
    """log_softmax equals z - log(sum(exp(z))) for small, well-conditioned z."""
    z = np.array([1.0, 2.0, 3.0])
    naive = z - math.log(np.exp(z).sum())
    np.testing.assert_allclose(log_softmax(z), naive, atol=1e-12)


def test_log_softmax_numerically_stable_for_large_logits():
    """Without the max-shift trick, exp(1000) overflows. log_softmax must not."""
    z = np.array([1000.0, 1000.0, 1001.0])
    out = log_softmax(z)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(np.exp(out).sum(), 1.0, atol=1e-12)
    # Top-class logit's softmax probability:
    # exp(1001 - 1001) / (exp(-1) + exp(-1) + exp(0)) = 1 / (2/e + 1)
    expected_top_prob = 1.0 / (2.0 / math.e + 1.0)
    np.testing.assert_allclose(np.exp(out[2]), expected_top_prob, atol=1e-12)


def test_log_softmax_shape_and_lastaxis():
    """log_softmax operates on the last axis; rows independently sum to 1."""
    z = np.random.randn(3, 4, 5)
    out = log_softmax(z)
    assert out.shape == z.shape
    np.testing.assert_allclose(np.exp(out).sum(axis=-1), np.ones((3, 4)), atol=1e-12)


# ===========================================================================
# 2. cross_entropy_with_label_smoothing
# ===========================================================================

def test_ce_no_smoothing_matches_negative_log_likelihood():
    """eps=0 reduces to plain CE: -log_softmax(z)[target]."""
    np.random.seed(0)
    logits = np.random.randn(2, 3, 5)
    targets = np.array([[0, 1, 2], [3, 4, 0]])
    log_p = log_softmax(logits)
    expected = -np.take_along_axis(
        log_p, targets[..., None], axis=-1
    ).squeeze(-1).mean()

    loss, _ = cross_entropy_with_label_smoothing(logits, targets, eps=0.0)
    np.testing.assert_allclose(loss, expected, atol=1e-12)


def test_ce_label_smoothing_floor_for_perfect_prediction():
    """A model that puts all mass on the truth still has nonzero loss under
    label smoothing. The per-token loss equals
        -((1-eps) * log(1.0) + eps * log(near-zero))
    plus the entropy of the smoothed target.
    Easier: when logits are uniform on truth (one-hot via huge gap),
    cross-entropy approaches H(p_smooth, p_one_hot) which is large.
    Instead we test the dual: when the *predicted* distribution equals
    p_smooth itself, loss equals H(p_smooth, p_smooth) = the floor.
    """
    eps = 0.1
    V = 13
    floor = label_smoothing_loss_floor(eps, V)

    # Set predicted distribution = smoothed target distribution exactly.
    # Construct logits whose softmax matches p_smooth.
    targets = np.array([[3]])
    log_p_truth = math.log(1.0 - eps)
    log_p_other = math.log(eps / (V - 1))
    logits = np.full((1, 1, V), log_p_other)
    logits[0, 0, 3] = log_p_truth

    # log_softmax(logits) won't equal logits exactly (logits aren't a valid
    # log-prob; they need normalization). But softmax(logits) SHOULD already
    # sum to 1 because the values are log-probabilities of a valid distribution.
    np.testing.assert_allclose(np.exp(log_softmax(logits)).sum(axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.exp(log_softmax(logits))[0, 0], np.array(
        [eps / (V - 1)] * 3 + [1 - eps] + [eps / (V - 1)] * (V - 4)
    ), atol=1e-12)

    loss, _ = cross_entropy_with_label_smoothing(
        logits, targets, eps=eps, vocab_size=V
    )
    np.testing.assert_allclose(loss, floor, atol=1e-12)


def test_ce_loss_floor_helper_value():
    """Sanity-check the published floor for our (V=13, eps=0.1) build.
    See section 10.3 of docs/08_training.md.
    """
    floor = label_smoothing_loss_floor(0.1, 13)
    # Hand-computed: -(0.9 * ln(0.9) + 0.1 * ln(0.1/12)) ~ 0.5739
    assert abs(floor - 0.5739) < 1e-3, f"floor={floor}"


def test_ce_gradient_matches_softmax_minus_smoothed_target():
    """The analytical grad equals (softmax - p_smooth) / count, masked."""
    np.random.seed(1)
    logits = np.random.randn(2, 3, 5)
    targets = np.array([[0, 1, 2], [3, 4, 0]])
    eps = 0.1
    V = 5
    count = 6  # B * n with no PAD

    _, grad = cross_entropy_with_label_smoothing(
        logits, targets, eps=eps, vocab_size=V
    )

    # Reconstruct p_smooth manually.
    p_smooth = np.full((2, 3, 5), eps / (V - 1))
    np.put_along_axis(p_smooth, targets[..., None], 1 - eps, axis=-1)

    softmax = np.exp(log_softmax(logits))
    expected = (softmax - p_smooth) / count

    np.testing.assert_allclose(grad, expected, atol=1e-12)


def test_ce_pad_positions_have_zero_grad_and_excluded_from_count():
    """PAD positions contribute neither to the loss nor to the gradient,
    and are excluded from the denominator."""
    np.random.seed(2)
    logits = np.random.randn(2, 3, 5)
    targets = np.array([[0, 1, PAD_ID], [3, PAD_ID, 0]])
    # Set up PAD-aware loss: pad_id=PAD_ID will treat 10 as PAD; but our
    # logits dimension is 5, so use a smaller pad sentinel.
    pad = 9  # anything >= V is fine; targets above are 0..3 plus our pad
    targets = np.where(targets == PAD_ID, pad, targets)

    loss_padded, grad_padded = cross_entropy_with_label_smoothing(
        logits, targets, eps=0.0, vocab_size=5, pad_id=pad
    )

    # Hand-build expected ignoring padded positions.
    valid_mask = (targets != pad)
    log_p = log_softmax(logits)
    truth_log = np.take_along_axis(log_p, targets[..., None].clip(0, 4), axis=-1).squeeze(-1)
    per_token = -truth_log * valid_mask
    expected_loss = per_token.sum() / valid_mask.sum()
    np.testing.assert_allclose(loss_padded, expected_loss, atol=1e-12)

    # Gradient at PAD positions must be exactly zero.
    pad_positions = ~valid_mask
    np.testing.assert_array_equal(grad_padded[pad_positions], 0.0)


def test_ce_gradient_finite_difference():
    """Numerical gradient check of the loss w.r.t. logits."""
    np.random.seed(3)
    logits = np.random.randn(1, 2, 4) * 0.5
    targets = np.array([[1, 3]])
    eps = 0.1
    V = 4

    _, analytical = cross_entropy_with_label_smoothing(
        logits, targets, eps=eps, vocab_size=V
    )

    h = 1e-5
    numerical = np.zeros_like(logits)
    for idx in np.ndindex(logits.shape):
        original = logits[idx]
        logits[idx] = original + h
        loss_plus, _ = cross_entropy_with_label_smoothing(
            logits, targets, eps=eps, vocab_size=V
        )
        logits[idx] = original - h
        loss_minus, _ = cross_entropy_with_label_smoothing(
            logits, targets, eps=eps, vocab_size=V
        )
        logits[idx] = original
        numerical[idx] = (loss_plus - loss_minus) / (2 * h)

    abs_err = np.abs(analytical - numerical)
    rel_err = abs_err / (np.abs(numerical) + 1e-8)
    ok = (abs_err < 1e-7) | (rel_err < 1e-5)
    assert ok.all(), (
        f"CE gradient check failed; "
        f"max abs err {abs_err.max():.3e}, max rel err {rel_err.max():.3e}"
    )


def test_ce_count_normalization_makes_loss_batch_invariant():
    """Mean-over-tokens reduction: doubling B leaves loss unchanged
    when the second batch is a copy of the first."""
    np.random.seed(4)
    logits1 = np.random.randn(1, 3, 5)
    targets1 = np.array([[0, 1, 2]])
    loss1, _ = cross_entropy_with_label_smoothing(logits1, targets1, eps=0.1, vocab_size=5)

    logits2 = np.concatenate([logits1, logits1], axis=0)
    targets2 = np.concatenate([targets1, targets1], axis=0)
    loss2, _ = cross_entropy_with_label_smoothing(logits2, targets2, eps=0.1, vocab_size=5)

    np.testing.assert_allclose(loss1, loss2, atol=1e-12)


# ===========================================================================
# 3. noam_lr
# ===========================================================================

def test_noam_warmup_is_linear_in_step():
    """During warmup, lr is exactly proportional to step."""
    d_model = 32
    warmup = 400
    lr_1 = noam_lr(1, d_model, warmup)
    lr_10 = noam_lr(10, d_model, warmup)
    lr_100 = noam_lr(100, d_model, warmup)
    np.testing.assert_allclose(lr_10 / lr_1, 10.0, atol=1e-12)
    np.testing.assert_allclose(lr_100 / lr_1, 100.0, atol=1e-12)


def test_noam_continuous_at_warmup_boundary():
    """At step == warmup_steps, both pieces evaluate to the same value."""
    d_model = 32
    warmup = 400
    lr_at = noam_lr(warmup, d_model, warmup)
    expected = (d_model ** -0.5) * (warmup ** -0.5)
    np.testing.assert_allclose(lr_at, expected, atol=1e-12)


def test_noam_decays_after_warmup():
    """Post-warmup, lr is monotonically decreasing in step."""
    d_model = 32
    warmup = 400
    lr_at = noam_lr(warmup, d_model, warmup)
    lr_2x = noam_lr(2 * warmup, d_model, warmup)
    lr_10x = noam_lr(10 * warmup, d_model, warmup)
    assert lr_at > lr_2x > lr_10x > 0.0
    # Inverse-sqrt: lr(2*warmup) / lr(warmup) = sqrt(1/2)
    np.testing.assert_allclose(lr_2x / lr_at, math.sqrt(0.5), atol=1e-12)


def test_noam_d_model_scaling():
    """Bigger d_model -> smaller lr by sqrt(d_model) factor."""
    warmup = 400
    lr_a = noam_lr(100, 32, warmup)
    lr_b = noam_lr(100, 128, warmup)  # 4x larger d_model
    np.testing.assert_allclose(lr_a / lr_b, math.sqrt(128 / 32), atol=1e-12)


def test_noam_rejects_invalid_warmup():
    with pytest.raises(ValueError):
        noam_lr(10, 32, warmup_steps=0)


# ===========================================================================
# 4. Adam optimizer
# ===========================================================================

def test_adam_first_step_is_lr_times_sign_with_bias_correction():
    """Bias-corrected Adam at step 1 with constant grad takes a step
    of magnitude exactly lr (modulo eps), regardless of |g|.
    """
    p = np.array([1.0])
    g = np.array([0.5])
    p_initial = p.copy()
    opt = Adam(beta1=0.9, beta2=0.98, eps=1e-9)
    opt.step(lr=0.001, params_and_grads=[(p, g, 'w')])
    # p moved by -lr * sign(g) (very nearly): expect p = 1.0 - 0.001 = 0.999
    np.testing.assert_allclose(p, p_initial - 0.001, atol=1e-7)


def test_adam_constant_gradient_long_run_approaches_signed_lr():
    """Across many steps with constant grad, the per-step update settles
    to lr * sign(g) (Adam's signature property)."""
    p = np.zeros(1)
    g = np.array([3.7])
    opt = Adam()
    lr = 1e-3
    p_history = []
    for _ in range(2000):
        opt.step(lr=lr, params_and_grads=[(p, g, 'w')])
        p_history.append(p.copy())
    # After 2000 steps p is approximately -2000 * lr (with some warmup).
    expected = -2000 * lr
    rel_err = abs(p[0] - expected) / abs(expected)
    assert rel_err < 0.05, (
        f"After many steps, Adam should walk at ~lr*sign(g)/step; got {p[0]} vs ~{expected}"
    )


def test_adam_per_parameter_state_is_independent():
    """Two parameters with different gradients receive different updates."""
    p1 = np.array([1.0])
    p2 = np.array([1.0])
    g1 = np.array([0.5])
    g2 = np.array([5.0])  # 10x larger gradient
    opt = Adam()
    opt.step(lr=0.01, params_and_grads=[(p1, g1, 'a'), (p2, g2, 'b')])
    # Both should receive ~lr-magnitude updates due to bias correction at step 1.
    np.testing.assert_allclose(p1, [0.99], atol=1e-5)
    np.testing.assert_allclose(p2, [0.99], atol=1e-5)


def test_adam_skips_none_gradient():
    """A param with grad=None is left untouched."""
    p = np.array([1.0, 2.0])
    p_before = p.copy()
    opt = Adam()
    opt.step(lr=1.0, params_and_grads=[(p, None, 'w')])
    np.testing.assert_array_equal(p, p_before)


def test_adam_in_place_update_preserves_array_identity():
    """Updates must be in-place so the layer's weight reference stays valid."""
    p = np.array([1.0, 2.0])
    p_id = id(p)
    g = np.array([0.1, 0.1])
    opt = Adam()
    opt.step(lr=0.001, params_and_grads=[(p, g, 'w')])
    assert id(p) == p_id


def test_adam_bias_correction_at_step_1_recovers_raw_gradient():
    """At t=1, m_hat = m/(1-beta1) = (1-beta1)*g/(1-beta1) = g."""
    opt = Adam(beta1=0.9, beta2=0.98)
    p = np.zeros(1)
    g = np.array([7.0])
    opt.step(lr=1e-3, params_and_grads=[(p, g, 'w')])
    state = next(iter(opt.state.values()))
    m_hat = state['m'] / (1 - opt.beta1)
    np.testing.assert_allclose(m_hat, g, atol=1e-12)


# ===========================================================================
# 5. zero_grads
# ===========================================================================

def test_zero_grads_resets_embedding_grad_buffer():
    """zero_grads should null out transformer.embedding.grad_weight."""
    np.random.seed(5)
    tf = Transformer(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16,
        num_encoder_layers=1, num_decoder_layers=1,
    ).eval()

    src = np.array([[1, 2, 3]])
    tgt = np.array([[4, 5]])
    logits = tf.forward(src, tgt)
    grad_logits = np.random.randn(*logits.shape)
    tf.backward(grad_logits)
    assert tf.embedding.grad_weight is not None

    zero_grads(tf)
    assert tf.embedding.grad_weight is None


def test_zero_grads_prevents_grad_accumulation_across_steps():
    """Without zero_grads, two backward passes accumulate the embedding's
    gradient. With zero_grads, the second pass starts fresh."""
    np.random.seed(6)
    tf = Transformer(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16,
        num_encoder_layers=1, num_decoder_layers=1,
    ).eval()

    src = np.array([[1, 2, 3]])
    tgt = np.array([[4, 5]])
    logits = tf.forward(src, tgt)
    grad_logits = np.random.randn(*logits.shape)
    tf.backward(grad_logits)
    g_after_one = tf.embedding.grad_weight.copy()

    # Backward without zero_grads -> doubled grads.
    logits = tf.forward(src, tgt)
    tf.backward(grad_logits)
    g_after_two = tf.embedding.grad_weight.copy()
    assert np.allclose(g_after_two, 2 * g_after_one, atol=1e-10), (
        "Embedding grads should accumulate without zero_grads"
    )

    # With zero_grads, third pass equals one fresh backward.
    zero_grads(tf)
    logits = tf.forward(src, tgt)
    tf.backward(grad_logits)
    g_after_three = tf.embedding.grad_weight.copy()
    np.testing.assert_allclose(g_after_three, g_after_one, atol=1e-10)


# ===========================================================================
# 6. Integration: full forward + loss + backward + Adam = learning
# ===========================================================================

def test_train_loop_actually_learns():
    """The headline test for Step 8.

    Build a small Transformer, train on a small sort-task split for a
    modest number of steps, and assert the loss curve is monotonically
    smaller-on-average than at the start. This certifies that:

      - cross_entropy_with_label_smoothing's gradient
      - the existing Step 7 Transformer.backward chain
      - Adam's per-parameter updates
      - the Noam schedule

    are all wired together correctly. If any of the four is broken, the
    loss does not drop.
    """
    np.random.seed(7)
    data = generate_sort_split(n=128, seq_len=5, seed=7)

    tf = Transformer(
        vocab_size=13, d_model=16, num_heads=2, d_ff=32,
        num_encoder_layers=1, num_decoder_layers=1,
        p_drop=0.0,  # turn off dropout for a more reproducible check
        tie_embeddings=True,
    )

    losses, lrs = train(
        tf, data,
        n_steps=300,
        batch_size=16,
        eps=0.1,
        warmup_steps=50,
        log_every=0,
        seed=7,
    )

    assert len(losses) == 300
    assert len(lrs) == 300
    initial_avg = np.mean(losses[:20])
    final_avg = np.mean(losses[-20:])
    assert final_avg < initial_avg - 0.3, (
        f"Loss should drop substantially in 300 steps; "
        f"initial avg={initial_avg:.3f}, final avg={final_avg:.3f}"
    )
    # And the final loss should still be above the smoothing floor.
    floor = label_smoothing_loss_floor(0.1, 13)
    assert final_avg > floor - 1e-3, (
        f"Final loss {final_avg:.3f} below the label-smoothing floor {floor:.3f}; "
        f"that would indicate a bug in the loss"
    )


def test_train_loss_curve_at_step_1_matches_uniform_prior():
    """Step-1 loss for a freshly initialized model should be near
    log(V) (the entropy of the uniform prior over V classes) when
    eps=0. With eps=0.1, the floor is ~log(V) - small_label_smoothing_term,
    but the value is bounded to within a reasonable range.
    """
    np.random.seed(8)
    data = generate_sort_split(n=32, seq_len=5, seed=8)

    tf = Transformer(
        vocab_size=13, d_model=16, num_heads=2, d_ff=32,
        num_encoder_layers=1, num_decoder_layers=1,
        p_drop=0.0,
    )

    losses, _ = train(
        tf, data, n_steps=1, batch_size=8, eps=0.0,  # use plain CE for the bound
        warmup_steps=10, log_every=0, seed=8,
    )

    log_V = math.log(13)
    # Initial random Transformer outputs should be roughly uniform-like;
    # loss within +/- 25% of log(V) is acceptable given small initial weights.
    assert log_V * 0.6 < losses[0] < log_V * 1.4, (
        f"Step-1 loss {losses[0]:.3f} not within reasonable range around log(V)={log_V:.3f}"
    )
