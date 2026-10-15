"""
Test each primitive layer using the numerical examples from the documentation.
Run: uv run --package numpy-transformer pytest
"""

import numpy as np
from numpy_transformer.layers import (
    Linear,
    Softmax,
    ReLU,
    LayerNorm,
    Dropout,
    Embedding,
    SinusoidalPositionalEncoding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForward,
    EncoderLayer,
    Encoder,
    TiedLinear,
    DecoderLayer,
    Decoder,
    Transformer,
)

np.random.seed(42)


def test_linear():
    """Test Linear with the exact example from 01_math_primitives.md"""
    layer = Linear(2, 3)
    layer.W = np.array([[0.5, -1.0, 0.3],
                        [0.2,  0.8, -0.5]])
    layer.b = np.array([0.1, 0.0, -0.2])

    x = np.array([[2.0, 1.0]])
    y = layer.forward(x)
    expected_y = np.array([[1.3, -1.2, -0.1]])
    assert np.allclose(y, expected_y, atol=1e-6), f"Forward failed: {y} != {expected_y}"

    grad_output = np.array([[0.5, -0.3, 0.8]])
    grad_input = layer.backward(grad_output)

    expected_dW = np.array([[1.0, -0.6, 1.6],
                            [0.5, -0.3, 0.8]])
    expected_db = np.array([0.5, -0.3, 0.8])
    expected_dx = np.array([[0.79, -0.54]])

    assert np.allclose(layer.grad_W, expected_dW, atol=1e-6), f"dW failed: {layer.grad_W}"
    assert np.allclose(layer.grad_b, expected_db, atol=1e-6), f"db failed: {layer.grad_b}"
    assert np.allclose(grad_input, expected_dx, atol=1e-2), f"dx failed: {grad_input}"

    x_3d = np.random.randn(2, 3, 2)
    y_3d = layer.forward(x_3d)
    assert y_3d.shape == (2, 3, 3), f"3D shape failed: {y_3d.shape}"


def test_softmax():
    """Test Softmax with the exact example from 01_math_primitives.md"""
    layer = Softmax()

    x = np.array([[2.0, 1.0, 0.1]])
    y = layer.forward(x)
    expected_y = np.array([[0.659, 0.242, 0.099]])
    assert np.allclose(y, expected_y, atol=1e-3), f"Forward failed: {y}"
    assert np.allclose(y.sum(), 1.0, atol=1e-6), "Probabilities don't sum to 1"

    grad_output = np.array([[1.0, 0.0, 0.0]])
    grad_input = layer.backward(grad_output)
    expected_dx = np.array([[0.225, -0.159, -0.065]])
    assert np.allclose(grad_input, expected_dx, atol=1e-3), f"Backward failed: {grad_input}"
    assert abs(grad_input.sum()) < 1e-6, "Softmax gradients should sum to ~0"

    x_large = np.array([[1000.0, 1001.0, 999.0]])
    y_large = layer.forward(x_large)
    assert not np.any(np.isnan(y_large)), "Numerical stability failed!"
    assert np.allclose(y_large.sum(), 1.0, atol=1e-6), "Large input: probs don't sum to 1"


def test_relu():
    """Test ReLU with the exact example from 01_math_primitives.md"""
    layer = ReLU()

    x = np.array([[0.5, -1.2, 3.0, -0.1, 0.0, 2.7]])
    y = layer.forward(x)
    expected_y = np.array([[0.5, 0.0, 3.0, 0.0, 0.0, 2.7]])
    assert np.allclose(y, expected_y), f"Forward failed: {y}"

    grad_output = np.array([[0.3, -0.5, 0.1, 0.8, 0.2, -0.4]])
    grad_input = layer.backward(grad_output)
    expected_dx = np.array([[0.3, 0.0, 0.1, 0.0, 0.0, -0.4]])
    assert np.allclose(grad_input, expected_dx), f"Backward failed: {grad_input}"


def test_layernorm():
    """Test LayerNorm with the exact example from 01_math_primitives.md"""
    layer = LayerNorm(d_model=3)

    x = np.array([[4.0, 2.0, 0.0]])
    y = layer.forward(x)
    expected_y = np.array([[1.225, 0.0, -1.225]])
    assert np.allclose(y, expected_y, atol=1e-3), f"Forward failed: {y}"

    assert abs(y.mean()) < 1e-6, f"Mean not ~0: {y.mean()}"
    assert abs(y.var() - 1.0) < 0.1, f"Var not ~1: {y.var()}"

    x = np.random.randn(2, 4, 3)
    layer.forward(x)
    grad_output = np.random.randn(2, 4, 3)
    grad_analytical = layer.backward(grad_output)

    eps = 1e-5
    grad_numerical = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_plus[idx] += eps
        x_minus = x.copy()
        x_minus[idx] -= eps
        y_plus = layer.forward(x_plus)
        y_minus = layer.forward(x_minus)
        grad_numerical[idx] = ((y_plus - y_minus) * grad_output).sum() / (2 * eps)

    rel_error = np.abs(grad_analytical - grad_numerical) / (np.abs(grad_numerical) + 1e-8)
    max_error = rel_error.max()
    assert max_error < 1e-4, f"Gradient check failed! Max relative error: {max_error}"


def test_dropout():
    """Test Dropout behavior"""
    layer = Dropout(p=0.3)

    x = np.ones((1000,))
    y = layer.forward(x)
    zeros = (y == 0).sum()
    assert zeros > 200, f"Too few zeros: {zeros} (expected ~300)"
    assert zeros < 400, f"Too many zeros: {zeros} (expected ~300)"

    nonzero_vals = y[y > 0]
    expected_scale = 1.0 / (1.0 - 0.3)
    assert np.allclose(nonzero_vals, expected_scale, atol=1e-6), \
        f"Scaling wrong: {nonzero_vals[0]} != {expected_scale}"

    grad_output = np.ones((1000,))
    grad_input = layer.backward(grad_output)
    assert np.allclose((grad_input > 0), (y > 0)), "Backward mask doesn't match forward mask"

    layer.training = False
    y_infer = layer.forward(x)
    assert np.allclose(y_infer, x), "Inference mode should pass through unchanged"


def test_embedding():
    """Test Embedding with the exact example from 01_math_primitives.md"""
    layer = Embedding(vocab_size=5, d_model=4)
    layer.weight = np.array([
        [ 0.12, -0.84,  0.33,  0.56],
        [-0.21,  0.67, -0.45,  0.11],
        [ 0.88,  0.02,  0.71, -0.39],
        [-0.54,  0.91, -0.12,  0.43],
        [ 0.33, -0.15,  0.62,  0.77],
    ])

    token_ids = np.array([3, 1, 4])
    y = layer.forward(token_ids)
    expected_y = np.array([
        [-0.54,  0.91, -0.12,  0.43],
        [-0.21,  0.67, -0.45,  0.11],
        [ 0.33, -0.15,  0.62,  0.77],
    ])
    assert np.allclose(y, expected_y), f"Forward failed: {y}"

    grad_output = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    layer.backward(grad_output)

    assert np.allclose(layer.grad_weight[3], [1.0, 0.0, 0.0, 0.0]), "Row 3 grad wrong"
    assert np.allclose(layer.grad_weight[1], [0.0, 1.0, 0.0, 0.0]), "Row 1 grad wrong"
    assert np.allclose(layer.grad_weight[4], [0.0, 0.0, 1.0, 0.0]), "Row 4 grad wrong"
    assert np.allclose(layer.grad_weight[0], [0.0, 0.0, 0.0, 0.0]), "Row 0 should be zero"
    assert np.allclose(layer.grad_weight[2], [0.0, 0.0, 0.0, 0.0]), "Row 2 should be zero"

    # Embedding.backward now accumulates (Step 7 change for tied weights;
    # see docstring on Embedding.backward). Reset between independent
    # backward calls to test the within-call duplicate accumulation in
    # isolation -- token id 3 appears twice in `token_ids_dup`, with
    # gradients [1.0, 0, 0, 0] and [0.5, 0, 0, 0]; their sum [1.5, 0, 0, 0]
    # is what `np.add.at` produces.
    layer.grad_weight = None
    token_ids_dup = np.array([3, 1, 3])
    y_dup = layer.forward(token_ids_dup)
    grad_dup = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
    ])
    layer.backward(grad_dup)
    expected_row3 = np.array([1.5, 0.0, 0.0, 0.0])
    assert np.allclose(layer.grad_weight[3], expected_row3), \
        f"Duplicate accumulation failed: {layer.grad_weight[3]} != {expected_row3}"

    # Cross-call accumulation (the new contract): without the manual reset,
    # a second backward call adds on top of the previous. This is the
    # behavior tied weights rely on. Token id 3 picks up another [1.0, ...]
    # making its total [2.5, 0, 0, 0].
    grad_again = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    layer.backward(grad_again)
    expected_row3_accum = np.array([2.5, 0.0, 0.0, 0.0])
    assert np.allclose(layer.grad_weight[3], expected_row3_accum), \
        f"Cross-call accumulation failed: {layer.grad_weight[3]} != {expected_row3_accum}"

    # Explicit token_ids override: `token_ids` argument bypasses the cache.
    # This is what Encoder/Decoder use under shared-embedding setups (the
    # other side's forward overwrites the cache).
    layer.grad_weight = None
    layer.forward(np.array([0]))  # populates cache with [0]
    layer.backward(np.array([[1.0, 0.0, 0.0, 0.0]]),
                   token_ids=np.array([4]))
    assert np.allclose(layer.grad_weight[0], [0.0, 0.0, 0.0, 0.0]), \
        "explicit token_ids should override cache"
    assert np.allclose(layer.grad_weight[4], [1.0, 0.0, 0.0, 0.0]), \
        "explicit token_ids should route grad to row 4"


def test_sinusoidal_positional_encoding_doc_example():
    """Rows PE(0) and PE(1) for d_model=4 match 02_positional_encoding.md."""
    layer = SinusoidalPositionalEncoding(d_model=4, max_len=16)
    x = np.zeros((1, 2, 4))
    y = layer.forward(x)
    assert np.allclose(y[0, 0], [0.0, 1.0, 0.0, 1.0], atol=1e-10)
    assert np.allclose(
        y[0, 1],
        [np.sin(1.0), np.cos(1.0), np.sin(0.01), np.cos(0.01)],
        atol=1e-5,
    )


def test_sinusoidal_positional_encoding_backward_is_identity():
    layer = SinusoidalPositionalEncoding(d_model=4, max_len=8)
    x = np.random.randn(2, 3, 4)
    layer.forward(x)
    g = np.random.randn(2, 3, 4)
    assert np.allclose(layer.backward(g), g)


def test_sinusoidal_positional_encoding_broadcast_batch():
    layer = SinusoidalPositionalEncoding(d_model=6, max_len=32)
    x = np.ones((4, 5, 6))
    y = layer.forward(x)
    assert y.shape == x.shape
    assert not np.allclose(y, x)


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention tests
# Reference: nn/docs/03_scaled_dot_product_attention.md
# ---------------------------------------------------------------------------

def test_sdpa_two_token_geometric_example():
    """
    Item 8 setup, with scaling applied: d_k=2, q aligned with k_A.
    q=[1,0], K=[[1,0],[0,1]], V=[[3,0],[1,1]]
    Manual computation:
        scores_raw = [1, 0]
        scaled     = [1/sqrt(2), 0] = [0.7071, 0]
        softmax    = [0.6698, 0.3302]
        output     = 0.6698*[3,0] + 0.3302*[1,1] = [2.3396, 0.3302]
    """
    attn = ScaledDotProductAttention()
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[3.0, 0.0], [1.0, 1.0]])

    out = attn.forward(Q, K, V)
    expected = np.array([[2.3396, 0.3302]])
    assert np.allclose(out, expected, atol=1e-3), f"Output: {out}"

    assert np.allclose(attn.last_weights, [[0.6698, 0.3302]], atol=1e-3), \
        f"Weights: {attn.last_weights}"
    assert np.allclose(attn.last_weights.sum(axis=-1), 1.0, atol=1e-10)


def test_sdpa_uniform_weights_output_is_mean():
    """
    When Q=0 every score is 0 and softmax is uniform, so the output must
    be the mean of V row-wise. This isolates the 'attention without content'
    case from item 2's soft-dictionary analogy.
    """
    attn = ScaledDotProductAttention()
    Q = np.zeros((3, 4))
    K = np.random.randn(5, 4)
    V = np.random.randn(5, 7)

    out = attn.forward(Q, K, V)
    expected = np.broadcast_to(V.mean(axis=0), (3, 7))
    assert np.allclose(out, expected, atol=1e-8), \
        f"Uniform-weight output should equal mean(V); got {out}"
    assert np.allclose(attn.last_weights, 1.0 / 5.0)


def test_sdpa_output_in_convex_hull_of_values():
    """
    Item 8's geometric claim: attention output always lies in the convex
    hull of V rows (because softmax weights are non-negative and sum to 1).
    Here we check output is bounded componentwise by min(V) and max(V).
    """
    np.random.seed(0)
    attn = ScaledDotProductAttention()
    Q = np.random.randn(4, 8)
    K = np.random.randn(6, 8)
    V = np.random.randn(6, 3)

    out = attn.forward(Q, K, V)
    v_min = V.min(axis=0)
    v_max = V.max(axis=0)
    assert np.all(out >= v_min - 1e-10), "Output below min(V) -- not convex combination"
    assert np.all(out <= v_max + 1e-10), "Output above max(V) -- not convex combination"


def test_sdpa_shape_contracts_unbatched_and_batched():
    """Item 10 shape summary: the op works for any leading batch dims."""
    attn = ScaledDotProductAttention()

    Q = np.random.randn(5, 8)
    K = np.random.randn(7, 8)
    V = np.random.randn(7, 4)
    out = attn.forward(Q, K, V)
    assert out.shape == (5, 4)
    assert attn.last_weights.shape == (5, 7)

    Q = np.random.randn(32, 5, 8)
    K = np.random.randn(32, 5, 8)
    V = np.random.randn(32, 5, 8)
    out = attn.forward(Q, K, V)
    assert out.shape == (32, 5, 8)
    assert attn.last_weights.shape == (32, 5, 5)

    # Multi-head-like leading dims (B, h, n, d_k) -- Step 4's shape
    Q = np.random.randn(32, 4, 5, 8)
    K = np.random.randn(32, 4, 5, 8)
    V = np.random.randn(32, 4, 5, 8)
    out = attn.forward(Q, K, V)
    assert out.shape == (32, 4, 5, 8)
    assert attn.last_weights.shape == (32, 4, 5, 5)


def test_sdpa_cross_attention_rectangular_weights():
    """
    Cross-attention: n_q != n_k. Attention matrix is rectangular, output
    inherits shape from Q (n_q) and V (d_v).
    """
    attn = ScaledDotProductAttention()
    Q = np.random.randn(2, 3, 8)   # decoder: n_q=3
    K = np.random.randn(2, 5, 8)   # encoder: n_k=5
    V = np.random.randn(2, 5, 8)
    out = attn.forward(Q, K, V)
    assert out.shape == (2, 3, 8), f"Output shape {out.shape}"
    assert attn.last_weights.shape == (2, 3, 5), \
        f"Weights shape {attn.last_weights.shape}"


def test_sdpa_causal_mask_triangular():
    """
    Causal mask: position i may only attend to positions j <= i.
    After masking, weights[i, j] must be 0 for j > i.
    """
    attn = ScaledDotProductAttention()
    n = 5
    Q = np.random.randn(n, 4)
    K = np.random.randn(n, 4)
    V = np.random.randn(n, 4)
    mask = np.tril(np.ones((n, n), dtype=bool))
    attn.forward(Q, K, V, mask=mask)

    w = attn.last_weights
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    assert np.all(w[upper] == 0.0), \
        f"Future positions leaked attention: {w[upper]}"
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-10), \
        "Row sums != 1 after masking"


def test_sdpa_padding_mask_broadcast():
    """Padding mask of shape (B, 1, n_k) broadcasts across queries."""
    attn = ScaledDotProductAttention()
    Q = np.random.randn(2, 3, 4)   # B=2, n_q=3, d_k=4
    K = np.random.randn(2, 5, 4)   # n_k=5
    V = np.random.randn(2, 5, 6)

    mask = np.ones((2, 1, 5), dtype=bool)
    mask[0, 0, 3:] = False          # seq 0: last two keys are padding
    mask[1, 0, 4:] = False          # seq 1: last one is padding

    attn.forward(Q, K, V, mask=mask)
    w = attn.last_weights
    assert w.shape == (2, 3, 5)
    assert np.all(w[0, :, 3:] == 0.0), "Padding leaked in batch 0"
    assert np.all(w[1, :, 4:] == 0.0), "Padding leaked in batch 1"
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-10)


def test_sdpa_no_learnable_parameters():
    """Item 9 invariant: Step 3 has zero trainable parameters."""
    attn = ScaledDotProductAttention()
    assert attn.params_and_grads() == [], \
        "ScaledDotProductAttention must have no learnable parameters"


def test_sdpa_shape_errors():
    """Mismatched d_k or n_k must raise."""
    attn = ScaledDotProductAttention()

    Q = np.random.randn(3, 4)
    K = np.random.randn(3, 5)       # wrong d_k
    V = np.random.randn(3, 6)
    try:
        attn.forward(Q, K, V)
        assert False, "Expected ValueError for mismatched d_k"
    except ValueError:
        pass

    Q = np.random.randn(3, 4)
    K = np.random.randn(3, 4)
    V = np.random.randn(2, 4)       # wrong n_k (must equal K's n_k)
    try:
        attn.forward(Q, K, V)
        assert False, "Expected ValueError for mismatched n_k"
    except ValueError:
        pass


def test_sdpa_backward_gradient_check():
    """
    Numerical gradient check for all three inputs (Q, K, V).
    Uses scalar loss L = sum(output * grad_output_seed).
    """
    np.random.seed(7)
    attn = ScaledDotProductAttention()

    Q = np.random.randn(2, 3, 4)
    K = np.random.randn(2, 5, 4)
    V = np.random.randn(2, 5, 6)
    grad_seed = np.random.randn(2, 3, 6)

    out = attn.forward(Q, K, V)
    grad_Q, grad_K, grad_V = attn.backward(grad_seed)

    def scalar_loss(Q_, K_, V_):
        fresh = ScaledDotProductAttention()
        return (fresh.forward(Q_, K_, V_) * grad_seed).sum()

    eps = 1e-5

    def numerical_grad(arr, loss_fn):
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            arr_plus = arr.copy()
            arr_plus[idx] += eps
            arr_minus = arr.copy()
            arr_minus[idx] -= eps
            g[idx] = (loss_fn(arr_plus) - loss_fn(arr_minus)) / (2 * eps)
        return g

    gQ_num = numerical_grad(Q, lambda Q_: scalar_loss(Q_, K, V))
    gK_num = numerical_grad(K, lambda K_: scalar_loss(Q, K_, V))
    gV_num = numerical_grad(V, lambda V_: scalar_loss(Q, K, V_))

    for name, analytical, numerical in [
        ('Q', grad_Q, gQ_num),
        ('K', grad_K, gK_num),
        ('V', grad_V, gV_num),
    ]:
        rel = np.abs(analytical - numerical) / (np.abs(numerical) + 1e-8)
        assert rel.max() < 1e-4, \
            f"grad_{name} check failed; max relative error: {rel.max()}"


def test_sdpa_backward_with_mask_gradient_check():
    """Same gradient check but with a causal mask active."""
    np.random.seed(11)
    attn = ScaledDotProductAttention()

    n = 4
    Q = np.random.randn(n, 3)
    K = np.random.randn(n, 3)
    V = np.random.randn(n, 5)
    mask = np.tril(np.ones((n, n), dtype=bool))
    grad_seed = np.random.randn(n, 5)

    attn.forward(Q, K, V, mask=mask)
    grad_Q, grad_K, grad_V = attn.backward(grad_seed)

    def scalar_loss(Q_, K_, V_):
        fresh = ScaledDotProductAttention()
        return (fresh.forward(Q_, K_, V_, mask=mask) * grad_seed).sum()

    eps = 1e-5

    def numerical_grad(arr, loss_fn):
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            arr_plus = arr.copy()
            arr_plus[idx] += eps
            arr_minus = arr.copy()
            arr_minus[idx] -= eps
            g[idx] = (loss_fn(arr_plus) - loss_fn(arr_minus)) / (2 * eps)
        return g

    gQ_num = numerical_grad(Q, lambda Q_: scalar_loss(Q_, K, V))
    gK_num = numerical_grad(K, lambda K_: scalar_loss(Q, K_, V))
    gV_num = numerical_grad(V, lambda V_: scalar_loss(Q, K, V_))

    for name, analytical, numerical in [
        ('Q', grad_Q, gQ_num),
        ('K', grad_K, gK_num),
        ('V', grad_V, gV_num),
    ]:
        rel = np.abs(analytical - numerical) / (np.abs(numerical) + 1e-8)
        assert rel.max() < 1e-4, \
            f"masked grad_{name} check failed; max relative error: {rel.max()}"


def test_sdpa_sort_task_shapes():
    """Sort-task grounding: B=32, seq_len=5, d_model=8, self-attention."""
    attn = ScaledDotProductAttention()
    Q = np.random.randn(32, 5, 8)
    K = np.random.randn(32, 5, 8)
    V = np.random.randn(32, 5, 8)
    out = attn.forward(Q, K, V)
    assert out.shape == (32, 5, 8)
    assert attn.last_weights.shape == (32, 5, 5)


# ---------------------------------------------------------------------------
# Multi-Head Attention tests
# Reference: nn/docs/04_multi_head_attention.md
# ---------------------------------------------------------------------------

def test_mha_section5_trace():
    """
    Section 5 worked example, exactly. Reproduces the hand-traced numbers
    that the doc derives by trace.

    Setup:  n=2, d_model=4, h=2, d_k=2.
    X = [[1, 0, 1, 0], [0, 1, 0, 1]]
    W_q = W_k = I_4
    W_v = diag(1, 1, 2, 2)
    W_o = I_4

    Expected (from docs/04_multi_head_attention.md section 5):
      head-0 weights = head-1 weights = [[0.670, 0.330], [0.330, 0.670]]
      head_0 output  = [[0.670, 0.330], [0.330, 0.670]]
      head_1 output  = [[1.340, 0.660], [0.660, 1.340]]
      concat (== final, since W_o = I) =
          [[0.670, 0.330, 1.340, 0.660],
           [0.330, 0.670, 0.660, 1.340]]
    """
    mha = MultiHeadAttention(d_model=4, num_heads=2)

    mha.W_q.W = np.eye(4)
    mha.W_q.b = np.zeros(4)
    mha.W_k.W = np.eye(4)
    mha.W_k.b = np.zeros(4)
    mha.W_v.W = np.diag([1.0, 1.0, 2.0, 2.0])
    mha.W_v.b = np.zeros(4)
    mha.W_o.W = np.eye(4)
    mha.W_o.b = np.zeros(4)

    X = np.array([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0]])

    out = mha.forward(X, X, X)

    expected_out = np.array([[0.670, 0.330, 1.340, 0.660],
                             [0.330, 0.670, 0.660, 1.340]])
    assert np.allclose(out, expected_out, atol=1e-3), \
        f"Section 5 trace failed:\n got = {out}\n expected = {expected_out}"

    expected_weights = np.array([[0.670, 0.330],
                                 [0.330, 0.670]])
    assert mha.last_weights.shape == (2, 2, 2), \
        f"weights shape = {mha.last_weights.shape}"
    assert np.allclose(mha.last_weights[0], expected_weights, atol=1e-3), \
        f"head-0 weights = {mha.last_weights[0]}"
    assert np.allclose(mha.last_weights[1], expected_weights, atol=1e-3), \
        f"head-1 weights = {mha.last_weights[1]}"

    assert np.allclose(mha.last_weights.sum(axis=-1), 1.0, atol=1e-10), \
        "Per-head row sums must equal 1"


def test_mha_sort_task_shapes():
    """
    Sort-task grounding: B=32, n=5, d_model=32, h=4 (so d_k=8).
    Verifies the input shape (B, n, d_model) round-trips through MHA, and
    that last_weights surfaces a per-head attention matrix of (B, h, n, n).
    """
    np.random.seed(0)
    mha = MultiHeadAttention(d_model=32, num_heads=4)

    X = np.random.randn(32, 5, 32)
    out = mha.forward(X, X, X)

    assert out.shape == (32, 5, 32), f"output shape = {out.shape}"
    assert mha.last_weights.shape == (32, 4, 5, 5), \
        f"last_weights shape = {mha.last_weights.shape}"
    assert np.allclose(mha.last_weights.sum(axis=-1), 1.0, atol=1e-10)


def test_mha_d_model_not_divisible_raises():
    """
    The split-heads trick requires d_model % num_heads == 0; the constructor
    must catch that. (See docs/04_multi_head_attention.md sections 2 and 4.)
    """
    try:
        MultiHeadAttention(d_model=32, num_heads=3)
        assert False, "Expected ValueError for d_model not divisible by num_heads"
    except ValueError as e:
        assert "divisible" in str(e), f"Error message should mention divisibility: {e}"


def test_mha_input_dim_mismatch_raises():
    """Forward must reject inputs whose last dim is not d_model."""
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    X = np.random.randn(2, 5, 31)
    try:
        mha.forward(X, X, X)
        assert False, "Expected ValueError for wrong input last dim"
    except ValueError as e:
        assert "d_model" in str(e), f"Error message should mention d_model: {e}"


def test_mha_param_count_matches_doc():
    """
    Section 3 of docs/04_multi_head_attention.md predicts 4096 weight
    parameters per MHA layer with d_model=32, h=4 (4 projections of
    32x32 = 1024 each). Biases add 4 * 32 = 128 separately.
    """
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    weight_params = sum(
        p.size for (p, _g, name) in mha.params_and_grads() if name.endswith('.W')
    )
    assert weight_params == 4096, f"Expected 4096 W params, got {weight_params}"

    bias_params = sum(
        p.size for (p, _g, name) in mha.params_and_grads() if name.endswith('.b')
    )
    assert bias_params == 128, f"Expected 128 bias params, got {bias_params}"


def test_mha_cross_attention_rectangular():
    """
    Cross-attention: Q comes from decoder (n_q=3), K/V from encoder (n_k=5).
    Per-head weights are (h, n_q, n_k); output inherits n_q from Q_in.
    """
    np.random.seed(0)
    mha = MultiHeadAttention(d_model=16, num_heads=4)

    Q_in = np.random.randn(2, 3, 16)
    K_in = np.random.randn(2, 5, 16)
    V_in = np.random.randn(2, 5, 16)

    out = mha.forward(Q_in, K_in, V_in)

    assert out.shape == (2, 3, 16), f"output shape = {out.shape}"
    assert mha.last_weights.shape == (2, 4, 3, 5), \
        f"last_weights shape = {mha.last_weights.shape}"


def test_mha_backward_gradient_check():
    """
    Numerical gradient check for the full MHA backward.

    Covers both:
    - Input gradients: Q_in, K_in, V_in (returned by backward).
    - Parameter gradients: W and b of W_q, W_k, W_v, W_o (populated as a
      side effect on each sub-Linear).

    Cross-attention shapes (n_q != n_k) so distinct paths can't accidentally
    collude. See section 8 of docs/04_multi_head_attention.md.
    """
    np.random.seed(13)
    mha = MultiHeadAttention(d_model=4, num_heads=2)

    Q_in = np.random.randn(1, 2, 4)
    K_in = np.random.randn(1, 3, 4)
    V_in = np.random.randn(1, 3, 4)
    grad_seed = np.random.randn(1, 2, 4)

    mha.forward(Q_in, K_in, V_in)
    grad_Q_in, grad_K_in, grad_V_in = mha.backward(grad_seed)

    grad_W_q = mha.W_q.grad_W.copy()
    grad_b_q = mha.W_q.grad_b.copy()
    grad_W_k = mha.W_k.grad_W.copy()
    grad_b_k = mha.W_k.grad_b.copy()
    grad_W_v = mha.W_v.grad_W.copy()
    grad_b_v = mha.W_v.grad_b.copy()
    grad_W_o = mha.W_o.grad_W.copy()
    grad_b_o = mha.W_o.grad_b.copy()

    def scalar_loss():
        return (mha.forward(Q_in, K_in, V_in) * grad_seed).sum()

    eps = 1e-5

    def numerical_grad(arr, loss_fn):
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            original = arr[idx]
            arr[idx] = original + eps
            loss_plus = loss_fn()
            arr[idx] = original - eps
            loss_minus = loss_fn()
            arr[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
        return g

    checks = [
        ('Q_in', grad_Q_in, numerical_grad(Q_in, scalar_loss)),
        ('K_in', grad_K_in, numerical_grad(K_in, scalar_loss)),
        ('V_in', grad_V_in, numerical_grad(V_in, scalar_loss)),
        ('W_q.W', grad_W_q, numerical_grad(mha.W_q.W, scalar_loss)),
        ('W_q.b', grad_b_q, numerical_grad(mha.W_q.b, scalar_loss)),
        ('W_k.W', grad_W_k, numerical_grad(mha.W_k.W, scalar_loss)),
        ('W_k.b', grad_b_k, numerical_grad(mha.W_k.b, scalar_loss)),
        ('W_v.W', grad_W_v, numerical_grad(mha.W_v.W, scalar_loss)),
        ('W_v.b', grad_b_v, numerical_grad(mha.W_v.b, scalar_loss)),
        ('W_o.W', grad_W_o, numerical_grad(mha.W_o.W, scalar_loss)),
        ('W_o.b', grad_b_o, numerical_grad(mha.W_o.b, scalar_loss)),
    ]

    for name, analytical, numerical in checks:
        # Robust check: pass if either the absolute error is tiny (handles
        # the case where both values are ~0 -- e.g. grad_W_k.b is exactly
        # zero by softmax row-shift invariance, see below) OR the relative
        # error is tiny.
        abs_err = np.abs(analytical - numerical)
        rel_err = abs_err / (np.abs(numerical) + 1e-8)
        ok = (abs_err < 1e-8) | (rel_err < 1e-4)
        assert ok.all(), (
            f"grad_{name} check failed; "
            f"max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}"
        )


def test_mha_W_k_bias_gradient_is_exactly_zero():
    """
    Mathematical fact: grad_W_k.b is identically zero, regardless of inputs.

    Why: adding a constant b to every row of K makes
        scores[i, k] = sum_j Q[i, j] * (K_orig[k, j] + b[j])
                     = scores_orig[i, k] + (Q @ b)[i]
    The added term depends on i (query) but NOT on k (key index), so it
    shifts the entire row of scores for query i by the same constant.
    Softmax is invariant to per-row constant shifts. Therefore b_k has zero
    influence on the output and its gradient is exactly zero.

    Note: the same argument does NOT apply to b_q (the shift would depend
    on k, not i, which softmax is sensitive to) and does NOT apply to b_v
    (V participates after softmax, not inside it). See section 8 of
    docs/04_multi_head_attention.md for the full discussion.
    """
    np.random.seed(23)
    mha = MultiHeadAttention(d_model=4, num_heads=2)
    X = np.random.randn(2, 3, 4)
    grad_seed = np.random.randn(2, 3, 4)

    mha.forward(X, X, X)
    mha.backward(grad_seed)

    assert np.allclose(mha.W_k.grad_b, 0.0, atol=1e-12), \
        f"grad_W_k.b should be exactly 0 but got {mha.W_k.grad_b}"


def test_mha_single_head_reduces_to_sdpa():
    """
    With h=1, the split/merge head reshapes only add and remove a singleton
    head dim, so MHA(X) reduces exactly to:

        W_o( SDPA( W_q(X), W_k(X), W_v(X) ) )

    This is a sanity check that the head plumbing is a no-op when h=1.
    """
    np.random.seed(17)
    d_model = 8
    mha = MultiHeadAttention(d_model=d_model, num_heads=1)

    X = np.random.randn(2, 4, d_model)
    out_mha = mha.forward(X, X, X)

    Q = X @ mha.W_q.W + mha.W_q.b
    K = X @ mha.W_k.W + mha.W_k.b
    V = X @ mha.W_v.W + mha.W_v.b

    sdpa = ScaledDotProductAttention()
    attn = sdpa.forward(Q, K, V)
    out_manual = attn @ mha.W_o.W + mha.W_o.b

    max_diff = np.abs(out_mha - out_manual).max()
    assert max_diff < 1e-12, \
        f"h=1 MHA should equal manual SDPA + projections; max diff: {max_diff}"


def test_mha_head_permutation_invariance():
    """
    Heads are interchangeable: their identity is a labeling, not a structural
    property. Concretely: swap the column-slice of W_q, W_k, W_v (and entry
    slice of their biases) corresponding to head 0 vs head 1, AND swap the
    matching row-slice of W_o. The forward output must be identical.

    Equivalently: there is a head-permutation symmetry in the parameter
    space. The loss surface has h! equivalent reflections of every
    minimum, one per head ordering.
    """
    np.random.seed(19)
    d_model = 8
    h = 2
    d_k = d_model // h
    mha = MultiHeadAttention(d_model=d_model, num_heads=h)

    X = np.random.randn(1, 3, d_model)
    out_a = mha.forward(X, X, X).copy()

    perm_out = np.arange(d_model)
    perm_out[0:d_k] = np.arange(d_k, 2 * d_k)
    perm_out[d_k:2 * d_k] = np.arange(0, d_k)

    mha.W_q.W = mha.W_q.W[:, perm_out]
    mha.W_q.b = mha.W_q.b[perm_out]
    mha.W_k.W = mha.W_k.W[:, perm_out]
    mha.W_k.b = mha.W_k.b[perm_out]
    mha.W_v.W = mha.W_v.W[:, perm_out]
    mha.W_v.b = mha.W_v.b[perm_out]

    mha.W_o.W = mha.W_o.W[perm_out, :]

    out_b = mha.forward(X, X, X)

    max_diff = np.abs(out_a - out_b).max()
    assert max_diff < 1e-12, \
        f"head permutation should not change output; max diff: {max_diff}"


# ---------------------------------------------------------------------------
# Step 5: Position-wise Feed-Forward Network
# See docs/05_feedforward_network.md
# ---------------------------------------------------------------------------

def test_ffn_section7_trace():
    """
    Reproduces the hand-traced numerical example from section 7 of
    docs/05_feedforward_network.md exactly.

    d_model=2, d_ff=4, two tokens [1, -1] and [-1, 2], biases zero.
    Keys (W_1 columns) chosen for interpretability:
        key_0 = [1,  0]  -- "feature 0 positive"
        key_1 = [0,  1]  -- "feature 1 positive"
        key_2 = [-1, 1]  -- "f0 negative AND f1 positive"
        key_3 = [1, -1]  -- "f0 positive AND f1 negative"
    Values (W_2 rows) cancel/reinforce in opposing pairs.

    Expected:
        x_1 = [1, -1]  -> keys 0,3 fire (strengths 1, 2) -> [1, -2]
        x_2 = [-1, 2]  -> keys 1,2 fire (strengths 2, 3) -> [-3, 2]
    """
    ffn = FeedForward(d_model=2, d_ff=4)
    ffn.W_1.W = np.array([[1.0, 0.0, -1.0, 1.0],
                          [0.0, 1.0,  1.0, -1.0]])
    ffn.W_1.b = np.zeros(4)
    ffn.W_2.W = np.array([[ 1.0,  0.0],
                          [ 0.0,  1.0],
                          [-1.0,  0.0],
                          [ 0.0, -1.0]])
    ffn.W_2.b = np.zeros(2)

    X = np.array([[1.0, -1.0],
                  [-1.0, 2.0]])
    out = ffn.forward(X)

    expected = np.array([[1.0, -2.0],
                         [-3.0, 2.0]])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_ffn_position_wise_independence():
    """
    Position-wise means each position is processed independently with the
    SAME weights. Therefore: forward over a batch of tokens must equal the
    stack of single-token forwards, exactly.

    This is the empirical statement of "no mixing across positions" --
    section 4 of docs/05_feedforward_network.md.
    """
    np.random.seed(42)
    ffn = FeedForward(d_model=8, d_ff=16)

    X = np.random.randn(2, 5, 8)

    out_batch = ffn.forward(X)

    out_per_token = np.zeros_like(out_batch)
    for b in range(X.shape[0]):
        for t in range(X.shape[1]):
            single = ffn.forward(X[b, t])
            out_per_token[b, t] = single

    np.testing.assert_allclose(out_batch, out_per_token, atol=1e-12)


def test_ffn_input_dim_mismatch_raises():
    """forward rejects inputs whose last dim is not d_model."""
    ffn = FeedForward(d_model=8, d_ff=16)
    bad_x = np.random.randn(2, 3, 5)
    try:
        ffn.forward(bad_x)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "d_model" in str(e), f"Error should mention d_model: {e}"


def test_ffn_param_count_matches_doc():
    """
    Verifies the parameter count from section 3 of the doc:
        W_1: (32, 64) = 2048
        b_1: (64,)    = 64
        W_2: (64, 32) = 2048
        b_2: (32,)    = 32
        Total: 4192
    """
    ffn = FeedForward(d_model=32, d_ff=64)
    weights = sum(p.size for (p, _g, n) in ffn.params_and_grads() if n.endswith('.W'))
    biases = sum(p.size for (p, _g, n) in ffn.params_and_grads() if n.endswith('.b'))

    assert weights == 4096, f"weight params: {weights}"
    assert biases == 96, f"bias params: {biases}"
    assert weights + biases == 4192


def test_ffn_backward_gradient_check():
    """
    Numerical gradient check for the full FFN backward.

    Covers:
    - Input gradient: x.
    - Parameter gradients: W_1, b_1, W_2, b_2.

    Robust comparator: passes if abs error < 1e-8 OR rel error < 1e-4
    (mirrors the pattern from MHA, in case any param ends up structurally
    near zero -- though we don't expect such a case here).
    """
    np.random.seed(31)
    ffn = FeedForward(d_model=4, d_ff=8)

    x = np.random.randn(2, 3, 4)
    grad_seed = np.random.randn(2, 3, 4)

    ffn.forward(x)
    grad_x = ffn.backward(grad_seed)

    grad_W_1 = ffn.W_1.grad_W.copy()
    grad_b_1 = ffn.W_1.grad_b.copy()
    grad_W_2 = ffn.W_2.grad_W.copy()
    grad_b_2 = ffn.W_2.grad_b.copy()

    def scalar_loss():
        return (ffn.forward(x) * grad_seed).sum()

    eps = 1e-5

    def numerical_grad(arr, loss_fn):
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            original = arr[idx]
            arr[idx] = original + eps
            loss_plus = loss_fn()
            arr[idx] = original - eps
            loss_minus = loss_fn()
            arr[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
        return g

    checks = [
        ('x',     grad_x,   numerical_grad(x, scalar_loss)),
        ('W_1.W', grad_W_1, numerical_grad(ffn.W_1.W, scalar_loss)),
        ('W_1.b', grad_b_1, numerical_grad(ffn.W_1.b, scalar_loss)),
        ('W_2.W', grad_W_2, numerical_grad(ffn.W_2.W, scalar_loss)),
        ('W_2.b', grad_b_2, numerical_grad(ffn.W_2.b, scalar_loss)),
    ]

    for name, analytical, numerical in checks:
        abs_err = np.abs(analytical - numerical)
        rel_err = abs_err / (np.abs(numerical) + 1e-8)
        ok = (abs_err < 1e-8) | (rel_err < 1e-4)
        assert ok.all(), (
            f"grad_{name} check failed; "
            f"max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}"
        )


def test_ffn_relu_kills_all_hidden():
    """
    Sanity: with weights chosen so every hidden pre-activation is negative,
    ReLU zeros the whole hidden state. Then h @ W_2 = 0 and the output is
    exactly b_2 -- no matter what x or W_2 are.

    Confirms ReLU is actually firing (not silently passing values through).
    """
    np.random.seed(5)
    ffn = FeedForward(d_model=4, d_ff=8)

    ffn.W_1.W = -np.ones((4, 8))
    ffn.W_1.b = -np.ones(8)
    ffn.W_2.W = np.random.randn(8, 4)
    ffn.W_2.b = np.array([1.0, 2.0, 3.0, 4.0])

    x = np.abs(np.random.randn(2, 3, 4)) + 0.1
    out = ffn.forward(x)

    expected = np.broadcast_to(ffn.W_2.b, out.shape)
    np.testing.assert_allclose(out, expected, atol=1e-12)


# ===========================================================================
# Encoder layer + full Encoder (Step 6)
# ===========================================================================


def test_encoder_layer_shape_preservation():
    """
    Section 3 of docs/06_encoder.md: every encoder sub-layer must be shape-
    preserving (B, n, d_model) -> (B, n, d_model) for the residual `x +
    Sublayer(x)` to type-check. Verify both Post-LN (default) and Pre-LN.
    """
    np.random.seed(0)
    for pre_ln in (False, True):
        layer = EncoderLayer(d_model=16, num_heads=4, d_ff=32, pre_ln=pre_ln).eval()
        x = np.random.randn(3, 7, 16)
        out = layer.forward(x)
        assert out.shape == x.shape, (
            f"pre_ln={pre_ln}: shape changed {x.shape} -> {out.shape}"
        )


def test_encoder_section9_shape_trace():
    """
    Reproduces the worked example from section 9 of docs/06_encoder.md:
    B=2, n=3, vocab=16, d_model=32, h=4, d_ff=64, N=2.
    Token IDs (B, n) -> encoder output (B, n, d_model).

    Internal shapes (visited but not exposed) are checked by the MHA and
    FFN tests; this test confirms the canonical (B, n, d_model) shape is
    restored at the encoder boundary.
    """
    np.random.seed(7)
    enc = Encoder(
        vocab_size=16, d_model=32, num_heads=4, d_ff=64, num_layers=2
    ).eval()

    token_ids = np.array([[5, 7, 11], [3, 9, 2]])
    out = enc.forward(token_ids)

    assert out.shape == (2, 3, 32), f"final shape wrong: {out.shape}"


def test_encoder_layer_post_ln_outputs_unit_normalized():
    """
    Post-LN ends every wrapper in LayerNorm. With the default gamma=1,
    beta=0 init, the per-token output vector has empirical mean 0 and
    empirical variance 1 over the d_model axis -> L2 norm = sqrt(d_model)
    per token (modulo LN's eps).

    This is a structural property of Post-LN that does NOT depend on the
    sub-layer weights or the input magnitude. Compare to the Pre-LN test
    below, where output magnitude tracks the (un-normalized) residual
    stream.
    """
    np.random.seed(11)
    d_model = 32
    layer = EncoderLayer(d_model, num_heads=4, d_ff=64, pre_ln=False).eval()

    # Use a deliberately large input scale to make the contrast with the
    # Pre-LN test visible.
    x = np.random.randn(2, 5, d_model) * 10.0
    out = layer.forward(x)

    per_token_mean = out.mean(axis=-1)
    per_token_var = out.var(axis=-1)
    per_token_norm = np.linalg.norm(out, axis=-1)

    np.testing.assert_allclose(per_token_mean, 0.0, atol=1e-6)
    np.testing.assert_allclose(per_token_var, 1.0, atol=1e-3)
    np.testing.assert_allclose(per_token_norm, np.sqrt(d_model), atol=1e-3)


def test_encoder_layer_pre_ln_residual_unbounded():
    """
    Pre-LN keeps the residual stream un-normalized, so the output magnitude
    tracks the input magnitude (modulo small sub-layer contributions).
    Section 5 of docs/06_encoder.md: this is exactly the property that
    makes Pre-LN gradient flow cleaner (unmodified residual path) at the
    cost of the output not being structurally bounded.

    Concretely: scale the input by 10x. Post-LN output magnitude stays at
    sqrt(d_model) (~5.66). Pre-LN output magnitude scales with the input
    (~|x| per token), several orders of magnitude larger than sqrt(d_model).
    """
    np.random.seed(11)
    d_model = 32

    np.random.seed(11)
    layer_pre = EncoderLayer(d_model, num_heads=4, d_ff=64, pre_ln=True).eval()

    x = np.random.randn(2, 5, d_model) * 10.0  # per-token |x| ~ sqrt(32)*10 ~ 56.6
    out = layer_pre.forward(x)

    per_token_input_norm = np.linalg.norm(x, axis=-1)
    per_token_out_norm = np.linalg.norm(out, axis=-1)

    assert np.all(per_token_out_norm > 5.0 * np.sqrt(d_model)), (
        "Pre-LN output norm should NOT be bounded by sqrt(d_model); "
        f"got per-token norms {per_token_out_norm}"
    )
    np.testing.assert_allclose(
        per_token_out_norm, per_token_input_norm,
        rtol=0.5,
    )


def test_encoder_layer_residual_passes_zero_sublayer_pre_ln():
    """
    Section 4 of docs/06_encoder.md: when sub-layers contribute zero, the
    residual makes the wrapper an identity (Pre-LN) or LN-of-input (Post-LN).

    We test the Pre-LN case because it gives a clean exact equality:
    forcing the LAST projection in each sub-layer (W^O of MHA, W_2 of FFN)
    to zero makes both sub-layer outputs exactly zero, so y_mid = x + 0
    and y_out = y_mid + 0 = x.
    """
    np.random.seed(0)
    d_model = 16
    layer = EncoderLayer(d_model, num_heads=4, d_ff=32, pre_ln=True).eval()

    layer.mha.W_o.W = np.zeros_like(layer.mha.W_o.W)
    layer.mha.W_o.b = np.zeros_like(layer.mha.W_o.b)
    layer.ffn.W_2.W = np.zeros_like(layer.ffn.W_2.W)
    layer.ffn.W_2.b = np.zeros_like(layer.ffn.W_2.b)

    x = np.random.randn(2, 4, d_model)
    out = layer.forward(x)

    np.testing.assert_allclose(out, x, atol=1e-12)


def _gradient_check(layer, x, grad_seed):
    """Helper: numerical gradient check for a layer with a single tensor input."""
    layer.forward(x)
    grad_x = layer.backward(grad_seed)

    snapshots = [(name, g.copy()) for (_, g, name) in layer.params_and_grads()]

    def scalar_loss():
        return (layer.forward(x) * grad_seed).sum()

    eps = 1e-5

    def numerical_grad(arr, loss_fn):
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            original = arr[idx]
            arr[idx] = original + eps
            loss_plus = loss_fn()
            arr[idx] = original - eps
            loss_minus = loss_fn()
            arr[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
        return g

    checks = [('x', grad_x, numerical_grad(x, scalar_loss))]

    params = list(layer.params_and_grads())
    for (param, _, name), (snap_name, analytical) in zip(params, snapshots):
        assert name == snap_name, f"params order shifted: {name} vs {snap_name}"
        numerical = numerical_grad(param, scalar_loss)
        checks.append((name, analytical, numerical))

    for name, analytical, numerical in checks:
        abs_err = np.abs(analytical - numerical)
        rel_err = abs_err / (np.abs(numerical) + 1e-8)
        ok = (abs_err < 1e-7) | (rel_err < 1e-4)
        assert ok.all(), (
            f"grad_{name} check failed; "
            f"max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}"
        )


def test_encoder_layer_backward_gradient_check_post_ln():
    """
    Numerical gradient check for Post-LN EncoderLayer. Covers grad_x and all
    parameters of MHA, FFN, and both LayerNorms. Dropout disabled so the
    forward is deterministic across the eps-perturbed re-evaluations.

    Backward derivation: section 11 of docs/06_encoder.md (Post-LN case).
    """
    np.random.seed(13)
    layer = EncoderLayer(d_model=8, num_heads=2, d_ff=16, pre_ln=False).eval()
    x = np.random.randn(1, 3, 8)
    grad_seed = np.random.randn(1, 3, 8)
    _gradient_check(layer, x, grad_seed)


def test_encoder_layer_backward_gradient_check_pre_ln():
    """
    Numerical gradient check for Pre-LN EncoderLayer. Same coverage as the
    Post-LN test above; the residual-sum-and-LN-on-the-side-path structure
    is exercised.

    Backward derivation: section 11 of docs/06_encoder.md (Pre-LN case).
    """
    np.random.seed(17)
    layer = EncoderLayer(d_model=8, num_heads=2, d_ff=16, pre_ln=True).eval()
    x = np.random.randn(1, 3, 8)
    grad_seed = np.random.randn(1, 3, 8)
    _gradient_check(layer, x, grad_seed)


def test_encoder_full_forward_shape():
    """
    Full Encoder pipeline: token IDs (B, n) -> contextual representation
    (B, n, d_model). Section 8 of docs/06_encoder.md.
    """
    np.random.seed(0)
    enc = Encoder(
        vocab_size=10, d_model=16, num_heads=4, d_ff=32, num_layers=2
    ).eval()
    token_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    out = enc.forward(token_ids)
    assert out.shape == (2, 4, 16)


def test_encoder_full_param_count_matches_doc():
    """
    Section 9 of docs/06_encoder.md, parameter count table for our build.

    With vocab=16, d_model=32, h=4, d_ff=64, N=2:

    Per encoder layer (sum):
        MHA W_q/W_k/W_v   : 3 * (32*32 + 32) = 3168
        MHA W_o           : 32*32 + 32        = 1056
        LayerNorm 1       : 2 * 32            = 64
        FFN W_1           : 32*64 + 64        = 2112
        FFN W_2           : 64*32 + 32        = 2080
        LayerNorm 2       : 2 * 32            = 64
        --------------------------------------------
        per layer         :                     8544

    Two encoder layers: 17088
    Embedding (16 x 32) :   512
    Positional encoding :     0  (fixed sinusoidal -- no learnable params)
    Dropout             :     0
    --------------------------------------------------
    TOTAL               : 17600
    """
    enc = Encoder(
        vocab_size=16, d_model=32, num_heads=4, d_ff=64, num_layers=2
    )

    total = sum(p.size for (p, _, _) in enc.params_and_grads())

    expected_per_layer = (
        3 * (32 * 32 + 32)   # MHA W_q, W_k, W_v
        + (32 * 32 + 32)     # MHA W_o
        + 2 * 32             # LN_1 (gamma + beta)
        + (32 * 64 + 64)     # FFN W_1
        + (64 * 32 + 32)     # FFN W_2
        + 2 * 32             # LN_2
    )
    assert expected_per_layer == 8544, "doc arithmetic mismatch"
    expected_total = expected_per_layer * 2 + (16 * 32)
    assert expected_total == 17600, "doc total arithmetic mismatch"
    assert total == expected_total, (
        f"encoder param count {total} != expected {expected_total}"
    )


def test_encoder_dropout_off_at_eval_is_deterministic():
    """
    With training=False, every Dropout in the encoder becomes the identity,
    so two forward passes on the same input must return identical output.
    Section 6 of docs/06_encoder.md.
    """
    np.random.seed(0)
    enc = Encoder(
        vocab_size=10, d_model=16, num_heads=4, d_ff=32, num_layers=2
    ).eval()
    token_ids = np.array([[1, 2, 3, 4]])

    out_a = enc.forward(token_ids)
    out_b = enc.forward(token_ids)
    np.testing.assert_array_equal(out_a, out_b)


def test_encoder_full_backward_gradient_check():
    """
    End-to-end numerical gradient check through the full Encoder. Token IDs
    are integer (no continuous gradient), so we check gradients of every
    learnable parameter only.

    All dropouts disabled so the forward is deterministic across the eps-
    perturbed re-evaluations.
    """
    np.random.seed(19)
    enc = Encoder(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16, num_layers=2
    ).eval()

    token_ids = np.array([[1, 3, 5], [2, 4, 6]])
    grad_seed = np.random.randn(2, 3, 8)

    enc.forward(token_ids)
    enc.backward(grad_seed)

    snapshots = [(name, g.copy()) for (_, g, name) in enc.params_and_grads()]

    def scalar_loss():
        return (enc.forward(token_ids) * grad_seed).sum()

    eps = 1e-5

    def numerical_grad(arr, loss_fn):
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            original = arr[idx]
            arr[idx] = original + eps
            loss_plus = loss_fn()
            arr[idx] = original - eps
            loss_minus = loss_fn()
            arr[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
        return g

    params = list(enc.params_and_grads())
    for (param, _, name), (snap_name, analytical) in zip(params, snapshots):
        assert name == snap_name, f"params order shifted: {name} vs {snap_name}"
        numerical = numerical_grad(param, scalar_loss)
        abs_err = np.abs(analytical - numerical)
        rel_err = abs_err / (np.abs(numerical) + 1e-8)
        ok = (abs_err < 1e-7) | (rel_err < 1e-4)
        assert ok.all(), (
            f"grad_{name} check failed; "
            f"max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}"
        )


# ===========================================================================
# Decoder layer + full Decoder + Transformer (Step 7)
# ===========================================================================


def test_decoder_layer_shape_preservation():
    """
    Section 2 of docs/07_decoder.md: every decoder sub-layer must be shape-
    preserving (B, n_q, d_model) -> (B, n_q, d_model). Verify with a
    DELIBERATELY DIFFERENT n_k (encoder length) to make sure cross-attention
    handles the rectangular case. Both Post-LN and Pre-LN.
    """
    np.random.seed(0)
    for pre_ln in (False, True):
        layer = DecoderLayer(d_model=16, num_heads=4, d_ff=32, pre_ln=pre_ln).eval()
        x = np.random.randn(2, 7, 16)
        enc_out = np.random.randn(2, 5, 16)  # n_k != n_q
        causal_mask = np.tril(np.ones((7, 7), dtype=bool))
        out = layer.forward(x, enc_out, causal_mask)
        assert out.shape == x.shape, (
            f"pre_ln={pre_ln}: shape changed {x.shape} -> {out.shape}"
        )


def test_decoder_layer_causal_mask_blocks_future():
    """
    Section 3 of docs/07_decoder.md: the causal mask makes the decoder's
    self-attention weight matrix lower-triangular. Upper-triangular entries
    must be exactly zero (softmax(-inf) = 0). Lower-triangular rows must
    sum to 1 (softmax over remaining positions).
    """
    np.random.seed(0)
    layer = DecoderLayer(d_model=16, num_heads=4, d_ff=32).eval()
    n_q = 5
    x = np.random.randn(2, n_q, 16)
    enc_out = np.random.randn(2, 4, 16)
    causal_mask = np.tril(np.ones((n_q, n_q), dtype=bool))
    layer.forward(x, enc_out, causal_mask)

    weights = layer.mha_self.last_weights  # (B, h, n_q, n_q)
    upper_mask = np.triu(np.ones((n_q, n_q), dtype=bool), k=1)
    upper_weights = weights[..., upper_mask]
    np.testing.assert_array_equal(upper_weights, 0.0)

    row_sums = weights.sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_decoder_layer_cross_attn_rectangular_unmasked():
    """
    Section 4 of docs/07_decoder.md: cross-attention has Q from the decoder
    (length n_q) and K=V from the encoder (length n_k). No causal mask --
    the entire source sequence is observed from the start. Verify:
      (a) cross-attention weight tensor has shape (B, h, n_q, n_k) with
          n_q != n_k,
      (b) every row sums to 1 (no positions are masked off), and
      (c) every weight is strictly positive (no -inf scores).
    """
    np.random.seed(0)
    layer = DecoderLayer(d_model=16, num_heads=4, d_ff=32).eval()
    n_q, n_k = 3, 5
    x = np.random.randn(2, n_q, 16)
    enc_out = np.random.randn(2, n_k, 16)
    causal_mask = np.tril(np.ones((n_q, n_q), dtype=bool))
    layer.forward(x, enc_out, causal_mask)

    cross_weights = layer.mha_cross.last_weights
    assert cross_weights.shape == (2, 4, n_q, n_k), (
        f"cross weights {cross_weights.shape} != (2, 4, {n_q}, {n_k})"
    )

    row_sums = cross_weights.sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
    assert (cross_weights > 0).all(), (
        "all cross-attn weights should be strictly positive (no mask)"
    )


def test_decoder_layer_residual_passes_zero_sublayer_pre_ln():
    """
    Section 4 of docs/06_encoder.md (residual = identity at zero sub-layer
    output). For Pre-LN with all sub-layer last projections zeroed:
        y1 = x + 0
        y2 = y1 + 0 = x
        y3 = y2 + 0 = x
    so the layer is the identity regardless of enc_out and the causal mask.
    """
    np.random.seed(0)
    d_model = 16
    layer = DecoderLayer(d_model, num_heads=4, d_ff=32, pre_ln=True).eval()

    layer.mha_self.W_o.W = np.zeros_like(layer.mha_self.W_o.W)
    layer.mha_self.W_o.b = np.zeros_like(layer.mha_self.W_o.b)
    layer.mha_cross.W_o.W = np.zeros_like(layer.mha_cross.W_o.W)
    layer.mha_cross.W_o.b = np.zeros_like(layer.mha_cross.W_o.b)
    layer.ffn.W_2.W = np.zeros_like(layer.ffn.W_2.W)
    layer.ffn.W_2.b = np.zeros_like(layer.ffn.W_2.b)

    x = np.random.randn(2, 4, d_model)
    enc_out = np.random.randn(2, 6, d_model)
    causal_mask = np.tril(np.ones((4, 4), dtype=bool))
    out = layer.forward(x, enc_out, causal_mask)

    np.testing.assert_allclose(out, x, atol=1e-12)


def _decoder_layer_grad_check(layer, x, enc_out, causal_mask, grad_seed):
    """
    Numerical gradient check for a DecoderLayer.

    Differs from the EncoderLayer helper (`_gradient_check`) because:
      (a) `forward` takes THREE arguments (x, enc_out, causal_mask); and
      (b) `backward` returns a 2-tuple (grad_x, grad_enc_out).

    Both grad_x and grad_enc_out are checked numerically, in addition to
    every learnable parameter.
    """
    layer.forward(x, enc_out, causal_mask)
    grad_x, grad_enc_out = layer.backward(grad_seed)

    snapshots = [(name, g.copy()) for (_, g, name) in layer.params_and_grads()]

    def scalar_loss():
        return (layer.forward(x, enc_out, causal_mask) * grad_seed).sum()

    eps = 1e-5

    def numerical_grad(arr, loss_fn):
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            original = arr[idx]
            arr[idx] = original + eps
            loss_plus = loss_fn()
            arr[idx] = original - eps
            loss_minus = loss_fn()
            arr[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
        return g

    checks = [
        ('x', grad_x, numerical_grad(x, scalar_loss)),
        ('enc_out', grad_enc_out, numerical_grad(enc_out, scalar_loss)),
    ]

    params = list(layer.params_and_grads())
    for (param, _, name), (snap_name, analytical) in zip(params, snapshots):
        assert name == snap_name, f"params order shifted: {name} vs {snap_name}"
        numerical = numerical_grad(param, scalar_loss)
        checks.append((name, analytical, numerical))

    for name, analytical, numerical in checks:
        abs_err = np.abs(analytical - numerical)
        rel_err = abs_err / (np.abs(numerical) + 1e-8)
        ok = (abs_err < 1e-7) | (rel_err < 1e-4)
        assert ok.all(), (
            f"grad_{name} check failed; "
            f"max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}"
        )


def test_decoder_layer_backward_gradient_check_post_ln():
    """
    Section 11 of docs/07_decoder.md (Post-LN backward): residual-sum,
    self-attention-sum, and cross-attention K+V split rules all exercised.
    """
    np.random.seed(13)
    layer = DecoderLayer(d_model=8, num_heads=2, d_ff=16, pre_ln=False).eval()
    x = np.random.randn(1, 3, 8)
    enc_out = np.random.randn(1, 4, 8)
    causal_mask = np.tril(np.ones((3, 3), dtype=bool))
    grad_seed = np.random.randn(1, 3, 8)
    _decoder_layer_grad_check(layer, x, enc_out, causal_mask, grad_seed)


def test_decoder_layer_backward_gradient_check_pre_ln():
    """Section 11 of docs/07_decoder.md (Pre-LN backward)."""
    np.random.seed(17)
    layer = DecoderLayer(d_model=8, num_heads=2, d_ff=16, pre_ln=True).eval()
    x = np.random.randn(1, 3, 8)
    enc_out = np.random.randn(1, 4, 8)
    causal_mask = np.tril(np.ones((3, 3), dtype=bool))
    grad_seed = np.random.randn(1, 3, 8)
    _decoder_layer_grad_check(layer, x, enc_out, causal_mask, grad_seed)


def test_decoder_full_forward_shape():
    """
    Full Decoder pipeline: target IDs (B, n_q) + enc_out (B, n_k, d_model)
    -> logits (B, n_q, vocab_size). Section 1 of docs/07_decoder.md.
    """
    np.random.seed(0)
    dec = Decoder(
        vocab_size=10, d_model=16, num_heads=4, d_ff=32, num_layers=2,
        tie_embeddings=True
    ).eval()
    target_ids = np.array([[1, 2, 3, 4]])
    enc_out = np.random.randn(1, 5, 16)
    logits = dec.forward(target_ids, enc_out)
    assert logits.shape == (1, 4, 10), f"got {logits.shape}"


def test_decoder_section9_shape_trace():
    """
    Reproduces the worked example from section 9 of docs/07_decoder.md:
    B=2, n_q=3, n_k=4, vocab=16, d_model=32, h=4, d_ff=64, N=2, tied.
    """
    np.random.seed(7)
    dec = Decoder(
        vocab_size=16, d_model=32, num_heads=4, d_ff=64, num_layers=2,
        tie_embeddings=True
    ).eval()

    target_ids = np.array([[3, 7, 11], [5, 9, 2]])
    enc_out = np.random.randn(2, 4, 32)

    logits = dec.forward(target_ids, enc_out)
    assert logits.shape == (2, 3, 16), f"final shape wrong: {logits.shape}"

    self_w = dec.layers[0].mha_self.last_weights
    assert self_w.shape == (2, 4, 3, 3), f"self weights {self_w.shape}"
    cross_w = dec.layers[0].mha_cross.last_weights
    assert cross_w.shape == (2, 4, 3, 4), f"cross weights {cross_w.shape}"


def test_decoder_standalone_param_count_matches_doc():
    """
    Section 9 of docs/07_decoder.md, standalone Decoder (owns its
    embedding) with vocab=16, d_model=32, h=4, d_ff=64, N=2, tied:

      Per decoder layer:
        Self-attn  W_q,W_k,W_v : 3 * (32*32 + 32) = 3168
        Self-attn  W_o         : 32*32 + 32       = 1056
        LN_1                    : 2 * 32           = 64
        Cross-attn W_q,W_k,W_v : 3 * (32*32 + 32) = 3168
        Cross-attn W_o         : 32*32 + 32       = 1056
        LN_2                    : 2 * 32           = 64
        FFN W_1                 : 32*64 + 64       = 2112
        FFN W_2                 : 64*32 + 32       = 2080
        LN_3                    : 2 * 32           = 64
        ----------------------------------------------------
        per layer               :                    12832

      Two decoder layers       : 25664
      Embedding (16 x 32) tied : 512
      Output projection (tied) : 0
      ------------------------------------
      Standalone Decoder total : 26176

    Inside a Transformer, the embedding (512) is owned by the Transformer
    not the Decoder, so the decoder-only contribution is 25664. See
    test_transformer_full_param_count_matches_doc.
    """
    dec = Decoder(
        vocab_size=16, d_model=32, num_heads=4, d_ff=64, num_layers=2,
        tie_embeddings=True
    )

    total = sum(p.size for (p, _, _) in dec.params_and_grads())

    expected_per_layer = (
        3 * (32 * 32 + 32)   # self-attn W_q, W_k, W_v
        + (32 * 32 + 32)     # self-attn W_o
        + 2 * 32             # LN_1
        + 3 * (32 * 32 + 32) # cross-attn W_q, W_k, W_v
        + (32 * 32 + 32)     # cross-attn W_o
        + 2 * 32             # LN_2
        + (32 * 64 + 64)     # FFN W_1
        + (64 * 32 + 32)     # FFN W_2
        + 2 * 32             # LN_3
    )
    assert expected_per_layer == 12832, "doc per-layer arithmetic mismatch"
    expected_total = 2 * expected_per_layer + 16 * 32  # 25664 + 512 = 26176
    assert expected_total == 26176, "doc standalone total mismatch"
    assert total == expected_total, (
        f"decoder standalone param count {total} != expected {expected_total}"
    )


def test_decoder_tied_vs_untied_param_count():
    """
    Section 6 of docs/07_decoder.md: tie_embeddings=True saves
    vocab_size * d_model + vocab_size params (the untied path uses a
    regular Linear with bias for simplicity; the paper omits the bias on
    the pre-softmax linear, so the untied path here is slightly less
    paper-faithful than the tied path).
    """
    dec_tied = Decoder(
        vocab_size=16, d_model=32, num_heads=4, d_ff=64, num_layers=2,
        tie_embeddings=True
    )
    dec_untied = Decoder(
        vocab_size=16, d_model=32, num_heads=4, d_ff=64, num_layers=2,
        tie_embeddings=False
    )
    n_tied = sum(p.size for (p, _, _) in dec_tied.params_and_grads())
    n_untied = sum(p.size for (p, _, _) in dec_untied.params_and_grads())
    assert n_untied - n_tied == 32 * 16 + 16, (
        f"untied - tied = {n_untied - n_tied}, expected {32*16 + 16}"
    )

    # TiedLinear exposes no own params (the embedding owns them).
    tied_op_params = list(dec_tied.output_projection.params_and_grads())
    assert tied_op_params == [], f"TiedLinear should have no own params"


def test_decoder_tied_output_uses_shared_embedding_weight():
    """
    With tie_embeddings=True, the Decoder's output projection holds a
    *reference* to the embedding (not a copy). Mutating the embedding's
    weight after construction is reflected in the output projection.
    """
    dec = Decoder(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16, num_layers=1,
        tie_embeddings=True
    ).eval()
    assert dec.output_projection.shared_owner is dec.embedding

    target_ids = np.array([[1, 2, 3]])
    enc_out = np.random.RandomState(0).randn(1, 4, 8)
    logits_before = dec.forward(target_ids, enc_out)

    dec.embedding.weight *= 0.0
    logits_after = dec.forward(target_ids, enc_out)
    np.testing.assert_allclose(logits_after, 0.0, atol=1e-12)
    assert not np.allclose(logits_before, 0.0), (
        "before mutation logits should be nonzero"
    )


def test_decoder_tied_gradient_reaches_untouched_rows():
    """
    Section 6 of docs/07_decoder.md (gradient accumulation). With tied
    weights, the gradient on embedding.weight has BOTH:
      (a) the input-side embedding-lookup contribution (only touches
          rows for token IDs that appeared in target_ids), and
      (b) the output-side tied-linear contribution (touches every row of
          embedding.weight, since the output projection multiplies x by
          ALL rows of weight^T to produce vocab logits).

    If we used vocab=8 with target_ids = [1, 3, 1], rows 0/2/4/5/6/7 are
    NOT touched by the input lookup. With tying, those rows still get a
    nonzero gradient via the output projection. Without tying, they would
    be exactly zero.
    """
    np.random.seed(7)
    dec = Decoder(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16, num_layers=1,
        tie_embeddings=True
    ).eval()

    target_ids = np.array([[1, 3, 1]])
    enc_out = np.random.randn(1, 4, 8)
    grad_logits = np.random.randn(1, 3, 8)

    dec.forward(target_ids, enc_out)
    dec.backward(grad_logits)

    untouched_rows = [0, 2, 4, 5, 6, 7]
    untouched_grad = dec.embedding.grad_weight[untouched_rows]
    assert not np.allclose(untouched_grad, 0.0, atol=1e-8), (
        "tied output projection should send gradient to untouched rows; "
        f"got max abs {np.abs(untouched_grad).max():.3e}"
    )


def test_decoder_dropout_off_at_eval_is_deterministic():
    """
    With training=False, every Dropout in the decoder becomes the identity,
    so two forward passes on the same input must return identical output.
    """
    np.random.seed(0)
    dec = Decoder(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16, num_layers=2
    ).eval()
    target_ids = np.array([[1, 2, 3]])
    enc_out = np.random.randn(1, 4, 8)

    out_a = dec.forward(target_ids, enc_out)
    out_b = dec.forward(target_ids, enc_out)
    np.testing.assert_array_equal(out_a, out_b)


def _full_grad_check_with_two_inputs(forward_fn, backward_fn, params_fn,
                                     real_inputs, grad_seed):
    """
    Generic numerical gradient checker for objects whose forward takes
    multiple inputs and whose backward returns the gradient w.r.t. ONE
    of them (the others may be integer token IDs that don't have grad).

    Args:
        forward_fn: () -> output. Closure that calls .forward(...).
        backward_fn: (grad_seed,) -> grad_real_input or None. Closure
            that calls .backward(grad_seed) and returns the grad we want
            to check against `real_inputs[0]`. May be None if the only
            inputs are integer.
        params_fn: () -> [(p, g, name), ...]. Closure that returns
            params_and_grads().
        real_inputs: list of (name, real-valued np.ndarray) to check
            numerically against backward_fn's return + their corresponding
            grads. Typically a single entry like ('enc_out', enc_out).
        grad_seed: gradient at the output to seed backward.
    """
    output = forward_fn()
    grad_real = backward_fn(grad_seed)

    snapshots = [(name, g.copy()) for (_, g, name) in params_fn()]

    def scalar_loss():
        return (forward_fn() * grad_seed).sum()

    eps = 1e-5

    def numerical_grad(arr, loss_fn):
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            original = arr[idx]
            arr[idx] = original + eps
            loss_plus = loss_fn()
            arr[idx] = original - eps
            loss_minus = loss_fn()
            arr[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
        return g

    checks = []
    if real_inputs and grad_real is not None:
        # Convention: real_inputs is a single (name, array). Generalize if
        # we ever need multiple real-valued inputs.
        name, arr = real_inputs[0]
        checks.append((name, grad_real, numerical_grad(arr, scalar_loss)))

    params = list(params_fn())
    for (param, _, name), (snap_name, analytical) in zip(params, snapshots):
        assert name == snap_name, f"params order shifted: {name} vs {snap_name}"
        numerical = numerical_grad(param, scalar_loss)
        checks.append((name, analytical, numerical))

    for name, analytical, numerical in checks:
        abs_err = np.abs(analytical - numerical)
        rel_err = abs_err / (np.abs(numerical) + 1e-8)
        ok = (abs_err < 1e-7) | (rel_err < 1e-4)
        assert ok.all(), (
            f"grad_{name} check failed; "
            f"max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}"
        )


def test_decoder_full_backward_gradient_check():
    """
    End-to-end gradient check through the standalone Decoder. Verifies:
      - grad_enc_out (the gradient that flows back to the encoder)
      - every learnable parameter, INCLUDING the shared embedding under
        tie_embeddings=True (which receives BOTH input-side and
        output-side contributions; the numerical perturbation captures
        both, so this is a real test of the tied-grad accumulation).
    """
    np.random.seed(19)
    dec = Decoder(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16, num_layers=2,
        tie_embeddings=True
    ).eval()

    target_ids = np.array([[1, 3, 5]])
    enc_out = np.random.randn(1, 4, 8)
    grad_seed = np.random.randn(1, 3, 8)

    _full_grad_check_with_two_inputs(
        forward_fn=lambda: dec.forward(target_ids, enc_out),
        backward_fn=lambda g: dec.backward(g),
        params_fn=lambda: dec.params_and_grads(),
        real_inputs=[('enc_out', enc_out)],
        grad_seed=grad_seed,
    )


def test_decoder_full_backward_gradient_check_untied():
    """
    Same end-to-end gradient check with tie_embeddings=False, exercising
    the standalone Linear output projection path (no gradient accumulation
    on the embedding from the output side).
    """
    np.random.seed(23)
    dec = Decoder(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16, num_layers=2,
        tie_embeddings=False
    ).eval()

    target_ids = np.array([[1, 3, 5]])
    enc_out = np.random.randn(1, 4, 8)
    grad_seed = np.random.randn(1, 3, 8)

    _full_grad_check_with_two_inputs(
        forward_fn=lambda: dec.forward(target_ids, enc_out),
        backward_fn=lambda g: dec.backward(g),
        params_fn=lambda: dec.params_and_grads(),
        real_inputs=[('enc_out', enc_out)],
        grad_seed=grad_seed,
    )


# ---------------- Transformer wrapper tests ------------------


def test_transformer_forward_shape():
    """
    Section 6 of docs/07_decoder.md (Transformer wrapper):
        source_ids (B, n_k) + target_ids (B, n_q) -> logits (B, n_q, vocab).
    """
    np.random.seed(0)
    tf = Transformer(
        vocab_size=10, d_model=16, num_heads=4, d_ff=32,
        num_encoder_layers=2, num_decoder_layers=2
    ).eval()

    source_ids = np.array([[1, 2, 3, 4]])
    target_ids = np.array([[5, 6]])
    logits = tf.forward(source_ids, target_ids)
    assert logits.shape == (1, 2, 10), f"got {logits.shape}"


def test_transformer_shared_embedding_three_way():
    """
    The Transformer owns ONE Embedding object. Both encoder.embedding and
    decoder.embedding are the same Python object (`is` check). The
    decoder's tied output projection holds a reference to that same
    object.

    This is the structural test of the three-way sharing described in
    section 6 of docs/07_decoder.md (paper section 3.4).
    """
    tf = Transformer(
        vocab_size=16, d_model=32, num_heads=4, d_ff=64,
        num_encoder_layers=2, num_decoder_layers=2,
        tie_embeddings=True
    )
    assert tf.encoder.embedding is tf.embedding, (
        "encoder.embedding should reference the wrapper's embedding"
    )
    assert tf.decoder.embedding is tf.embedding, (
        "decoder.embedding should reference the wrapper's embedding"
    )
    assert tf.decoder.output_projection.shared_owner is tf.embedding, (
        "decoder.output_projection should reference the wrapper's embedding"
    )

    # Mutate the embedding once; both encoder and decoder see the change.
    tf.embedding.weight[0, 0] = 99.0
    assert tf.encoder.embedding.weight[0, 0] == 99.0
    assert tf.decoder.embedding.weight[0, 0] == 99.0

    # Embedding's params are exposed by the Transformer, NOT by the encoder
    # or decoder (avoids double-counting).
    enc_params = [name for (_, _, name) in tf.encoder.params_and_grads()]
    dec_params = [name for (_, _, name) in tf.decoder.params_and_grads()]
    assert not any('embedding' in n for n in enc_params), (
        f"encoder shouldn't expose shared embedding: {enc_params}"
    )
    assert not any('embedding' in n for n in dec_params), (
        f"decoder shouldn't expose shared embedding: {dec_params}"
    )
    tf_params = [name for (_, _, name) in tf.params_and_grads()]
    assert any(n.startswith('embedding.') for n in tf_params), (
        f"transformer should expose embedding: {tf_params[:5]}"
    )


def test_transformer_full_param_count_matches_doc():
    """
    Section 9 of docs/07_decoder.md, full Transformer with tie_embeddings.

    Encoder side (matches Step 6, section 9):
        Per encoder layer       :  8544
        Two encoder layers      : 17088
        Embedding (16 x 32)     :   512  (shared, counted once)
        ----------------------------------
        Encoder + embedding     : 17600

    Decoder side (this step, section 9):
        Per decoder layer       : 12832
        Two decoder layers      : 25664
        Output projection (tied):     0
        ----------------------------------
        Decoder-only            : 25664

    Total                       : 17600 + 25664 = 43264
    """
    tf = Transformer(
        vocab_size=16, d_model=32, num_heads=4, d_ff=64,
        num_encoder_layers=2, num_decoder_layers=2,
        tie_embeddings=True
    )

    total = sum(p.size for (p, _, _) in tf.params_and_grads())
    expected = 17600 + 25664  # 43264
    assert expected == 43264, "doc total arithmetic mismatch"
    assert total == expected, (
        f"transformer param count {total} != expected {expected}"
    )


def test_transformer_full_backward_gradient_check():
    """
    End-to-end gradient check through the full Transformer:
      source_ids -> encoder -> enc_out -> decoder -> logits

    Token IDs have no continuous gradient, so we check only learnable
    parameters. The shared embedding's gradient receives THREE
    contributions (encoder lookup, decoder lookup, tied output projection)
    and the numerical perturbation correctly captures all three.

    This is the test that verifies the encoder-decoder coupling works
    end-to-end -- the gradient flowing back from logits through the
    decoder's cross-attention into enc_out, then through the encoder, is
    what makes joint training possible (Step 8).
    """
    np.random.seed(29)
    tf = Transformer(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16,
        num_encoder_layers=2, num_decoder_layers=2,
        tie_embeddings=True
    ).eval()

    source_ids = np.array([[1, 2, 3, 4]])
    target_ids = np.array([[5, 6, 7]])
    grad_seed = np.random.randn(1, 3, 8)

    tf.forward(source_ids, target_ids)
    tf.backward(grad_seed)

    snapshots = [(name, g.copy()) for (_, g, name) in tf.params_and_grads()]

    def scalar_loss():
        return (tf.forward(source_ids, target_ids) * grad_seed).sum()

    eps = 1e-5

    def numerical_grad(arr, loss_fn):
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            original = arr[idx]
            arr[idx] = original + eps
            loss_plus = loss_fn()
            arr[idx] = original - eps
            loss_minus = loss_fn()
            arr[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
        return g

    params = list(tf.params_and_grads())
    for (param, _, name), (snap_name, analytical) in zip(params, snapshots):
        assert name == snap_name, f"params order shifted: {name} vs {snap_name}"
        numerical = numerical_grad(param, scalar_loss)
        abs_err = np.abs(analytical - numerical)
        rel_err = abs_err / (np.abs(numerical) + 1e-8)
        ok = (abs_err < 1e-7) | (rel_err < 1e-4)
        assert ok.all(), (
            f"grad_{name} check failed; "
            f"max abs err: {abs_err.max():.3e}, max rel err: {rel_err.max():.3e}"
        )


def test_transformer_eval_deterministic():
    """eval mode disables every dropout in encoder + decoder."""
    np.random.seed(0)
    tf = Transformer(
        vocab_size=8, d_model=8, num_heads=2, d_ff=16,
        num_encoder_layers=2, num_decoder_layers=2
    ).eval()
    source_ids = np.array([[1, 2, 3]])
    target_ids = np.array([[4, 5]])
    out_a = tf.forward(source_ids, target_ids)
    out_b = tf.forward(source_ids, target_ids)
    np.testing.assert_array_equal(out_a, out_b)
