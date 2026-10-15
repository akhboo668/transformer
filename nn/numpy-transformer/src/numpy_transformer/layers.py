"""
Step 1: Math Primitives -- the building blocks of the Transformer.

Every layer follows the same interface:
    forward(x)       -> compute output, cache intermediates
    backward(grad)   -> compute gradients, return grad_input
    params_and_grads() -> return list of (param, grad, name) tuples for the optimizer

Paper: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import numpy as np


class Layer:
    """Base class for all layers."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def params_and_grads(self):
        return []


# ---------------------------------------------------------------------------
# 1a. Linear Layer
#     y = xW + b
#     Paper: used everywhere -- attention projections, FFN, output layer
# ---------------------------------------------------------------------------

class Linear(Layer):

    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.grad_W = None
        self.grad_b = None
        self._cache = {}

    def forward(self, x):
        self._cache['x'] = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        """
        In the Transformer, x is 3D: (batch, seq, in_features).
        Our formula dL/dW = x^T @ grad works for 2D, not 3D.

        The reshape trick: flatten batch and seq into one dimension.
            x:      (32, 5, 32) --> x_2d:    (160, 32)   -- 160 token vectors
            grad:   (32, 5, 64) --> grad_2d:  (160, 64)   -- 160 gradient vectors

        Now the 2D formula works:
            dL/dW = x_2d.T @ grad_2d   (32,160) @ (160,64) = (32,64) = shape of W
            dL/db = grad_2d.sum(axis=0)  sum 160 rows       = (64,)  = shape of b

        dL/dx does NOT need reshape -- NumPy broadcasts @ over batch and seq:
            grad_output @ W.T   (32,5,64) @ (64,32) = (32,5,32) = shape of x
        """
        x = self._cache['x']

        x_2d = x.reshape(-1, x.shape[-1])
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

        self.grad_W = x_2d.T @ grad_2d
        self.grad_b = grad_2d.sum(axis=0)
        grad_input = grad_output @ self.W.T
        return grad_input

    def params_and_grads(self):
        return [(self.W, self.grad_W, 'W'), (self.b, self.grad_b, 'b')]


# ---------------------------------------------------------------------------
# 1b. Softmax
#     softmax(x_i) = e^(x_i) / sum(e^(x_j))
#     Uses the numerically stable version: subtract max before exponentiating.
#     Paper Section 3.2.1 (inside attention) and final output layer.
# ---------------------------------------------------------------------------

class Softmax(Layer):

    def __init__(self):
        self._cache = {}

    def forward(self, x):
        shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        out = exp_x / exp_x.sum(axis=-1, keepdims=True)
        self._cache['out'] = out
        return out

    def backward(self, grad_output):
        s = self._cache['out']
        dot = (grad_output * s).sum(axis=-1, keepdims=True)
        return s * (grad_output - dot)


# ---------------------------------------------------------------------------
# 1c. ReLU
#     ReLU(x) = max(0, x)
#     Paper Section 3.3, Equation 2 (inside FFN).
# ---------------------------------------------------------------------------

class ReLU(Layer):

    def __init__(self):
        self._cache = {}

    def forward(self, x):
        self._cache['mask'] = (x > 0).astype(x.dtype)
        return x * self._cache['mask']

    def backward(self, grad_output):
        return grad_output * self._cache['mask']


# ---------------------------------------------------------------------------
# 1d. Layer Normalization
#     LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
#     Paper Section 3.1, referencing Ba et al. 2016 [1].
#     Applied after every sub-layer: LayerNorm(x + Sublayer(x))
# ---------------------------------------------------------------------------

class LayerNorm(Layer):

    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        self.grad_gamma = None
        self.grad_beta = None
        self._cache = {}

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_norm = (x - mean) * std_inv

        self._cache['x_norm'] = x_norm
        self._cache['std_inv'] = std_inv

        return self.gamma * x_norm + self.beta

    def backward(self, grad_output):
        x_norm = self._cache['x_norm']
        std_inv = self._cache['std_inv']
        n = x_norm.shape[-1]

        sum_axes = tuple(range(grad_output.ndim - 1))
        self.grad_gamma = (grad_output * x_norm).sum(axis=sum_axes)
        self.grad_beta = grad_output.sum(axis=sum_axes)

        dx_norm = grad_output * self.gamma

        grad_input = (std_inv / n) * (
            n * dx_norm
            - dx_norm.sum(axis=-1, keepdims=True)
            - x_norm * (dx_norm * x_norm).sum(axis=-1, keepdims=True)
        )
        return grad_input

    def params_and_grads(self):
        return [(self.gamma, self.grad_gamma, 'gamma'),
                (self.beta, self.grad_beta, 'beta')]


# ---------------------------------------------------------------------------
# 1e. Dropout
#     Randomly zeros out elements with probability p during training.
#     Uses inverted dropout: scales survivors by 1/(1-p) so no change at inference.
#     Paper Section 5.4, P_drop = 0.1
# ---------------------------------------------------------------------------

class Dropout(Layer):

    def __init__(self, p=0.1):
        self.p = p
        self.training = True
        self._cache = {}

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        self._cache['mask'] = mask
        return x * mask / (1.0 - self.p)

    def backward(self, grad_output):
        if not self.training or self.p == 0:
            return grad_output
        return grad_output * self._cache['mask'] / (1.0 - self.p)


# ---------------------------------------------------------------------------
# 1f. Embedding
#     Lookup table: token IDs (integers) -> d_model-dimensional vectors.
#     Paper Section 3.4. Scaled by sqrt(d_model).
# ---------------------------------------------------------------------------

class Embedding(Layer):

    def __init__(self, vocab_size, d_model):
        self.weight = np.random.randn(vocab_size, d_model) * 0.02
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.grad_weight = None
        self._cache = {}

    def forward(self, token_ids):
        self._cache['token_ids'] = token_ids
        return self.weight[token_ids]

    def backward(self, grad_output, token_ids=None):
        """
        Accumulates into self.grad_weight (PyTorch convention). The first
        backward call after construction (or after self.grad_weight is
        manually reset to None) initializes a fresh zero buffer; subsequent
        calls add to it.

        Why accumulate rather than reset: when the embedding is tied with
        the output projection (Step 7, paper section 3.4), the same
        weight matrix receives gradient contributions from multiple call
        sites in a single backward pass:
            (a) the input-side embedding lookup, and
            (b) the output-side tied linear projection (W_out = weight^T).
        Both must accumulate into the same `grad_weight` array. See section
        6 of docs/07_decoder.md.

        Why the optional explicit `token_ids`: when the embedding is shared
        between an encoder and a decoder (`Transformer` wrapper), both
        sides call `embedding.forward()` during the forward pass, and the
        decoder's call OVERWRITES the cached `token_ids`. Each side caches
        its own IDs and passes them explicitly to backward to avoid the
        race. When `token_ids=None`, falls back to the cache for backward
        compatibility with users that hold the embedding privately
        (existing standalone-encoder tests).

        For training loops, the caller is responsible for zeroing
        gradients between iterations: `embedding.grad_weight = None` (or
        `embedding.grad_weight[...] = 0.0`) before each fresh backward.
        """
        if token_ids is None:
            token_ids = self._cache['token_ids']
        if self.grad_weight is None or self.grad_weight.shape != self.weight.shape:
            self.grad_weight = np.zeros_like(self.weight)
        np.add.at(self.grad_weight, token_ids, grad_output)
        return None

    def params_and_grads(self):
        return [(self.weight, self.grad_weight, 'weight')]


# ---------------------------------------------------------------------------
# 2. Sinusoidal positional encoding (fixed, no learnable weights)
#     PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
#     PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
#     Paper Section 3.5
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(Layer):

    def __init__(self, d_model, max_len=5000):
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._build_pe(max_len, d_model)
        self._cache = {}

    @staticmethod
    def _build_pe(max_len, d_model):
        pe = np.zeros((max_len, d_model))

        # One row per sequence position: pos = 0, 1, ..., max_len-1 (column for broadcasting)
        pos = np.arange(max_len, dtype=np.float64)[:, np.newaxis]

        # Paper index i = 0, 1, ..., d_model/2 - 1 (one per sin/cos pair)
        i = np.arange(d_model // 2, dtype=np.float64)
        divisor = np.power(10000.0, 2.0 * i / d_model)

        # Angle matrix (max_len, d_model//2): entry (pos, i) == pos / 10000^(2i/d_model)
        angle = pos / divisor

        pe[:, 0::2] = np.sin(angle)
        pe[:, 1::2] = np.cos(angle)
        return pe

    def forward(self, x):
        """
        Add positional encoding to token embeddings.

        x: (batch, seq_len, d_model) or (seq_len, d_model)
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Last dim must be d_model={self.d_model}, got {x.shape[-1]}"
            )
        seq_len = x.shape[-2]
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_len={self.max_len}"
            )
        self._cache['seq_len'] = seq_len
        return x + self.pe[:seq_len]

    def backward(self, grad_output):
        return grad_output

    def params_and_grads(self):
        return []


# ---------------------------------------------------------------------------
# 3. Scaled Dot-Product Attention (no learnable parameters)
#     Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
#     Paper Section 3.2.1, Equation 1.
#
#     This is the stateless engine that Multi-Head Attention (Step 4) wraps
#     with learnable Q/K/V/output projections. See docs/03_scaled_dot_product_attention.md.
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(Layer):
    """
    Scaled Dot-Product Attention.

    Shapes (any number of leading batch dims ...):
        Q:    (..., n_q, d_k)
        K:    (..., n_k, d_k)
        V:    (..., n_k, d_v)
        mask: broadcastable to (..., n_q, n_k) or None
              True/1 = allowed, False/0 = blocked

    Forward returns:
        output:  (..., n_q, d_v)

    Backward returns a 3-tuple of gradients (grad_Q, grad_K, grad_V) because
    this layer has three inputs, not one. No learnable parameters.
    """

    def __init__(self):
        self._cache = {}
        self.last_weights = None

    def forward(self, Q, K, V, mask=None):
        if Q.shape[-1] != K.shape[-1]:
            raise ValueError(
                f"Q and K must share d_k; got Q:{Q.shape[-1]} K:{K.shape[-1]}"
            )
        if K.shape[-2] != V.shape[-2]:
            raise ValueError(
                f"K and V must share n_k; got K:{K.shape[-2]} V:{V.shape[-2]}"
            )

        d_k = Q.shape[-1]
        scale = 1.0 / np.sqrt(d_k)

        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale

        if mask is not None:
            scores = np.where(mask, scores, -np.inf)

        shifted = scores - scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(shifted)
        weights = exp_s / exp_s.sum(axis=-1, keepdims=True)

        output = np.matmul(weights, V)

        self._cache['Q'] = Q
        self._cache['K'] = K
        self._cache['V'] = V
        self._cache['weights'] = weights
        self._cache['scale'] = scale
        self._cache['mask'] = mask
        self.last_weights = weights

        return output

    def backward(self, grad_output):
        """
        Returns (grad_Q, grad_K, grad_V). Forward cached everything we need.

        Chain rule with output = weights @ V, weights = softmax(scores * scale):
            grad_V       = weights^T @ grad_output
            grad_weights = grad_output @ V^T
            grad_scores  = weights * (grad_weights - sum(grad_weights * weights))
                           # row-wise softmax Jacobian
            grad_scores *= scale
            grad_Q       = grad_scores @ K
            grad_K       = grad_scores^T @ Q

        Masked positions have weights = 0, so the softmax Jacobian produces
        grad_scores = 0 there automatically; we zero them again for safety.
        """
        Q = self._cache['Q']
        K = self._cache['K']
        V = self._cache['V']
        weights = self._cache['weights']
        scale = self._cache['scale']
        mask = self._cache['mask']

        grad_V = np.matmul(np.swapaxes(weights, -1, -2), grad_output)
        grad_weights = np.matmul(grad_output, np.swapaxes(V, -1, -2))

        dot = (grad_weights * weights).sum(axis=-1, keepdims=True)
        grad_scores = weights * (grad_weights - dot)

        if mask is not None:
            grad_scores = np.where(mask, grad_scores, 0.0)

        grad_scores = grad_scores * scale

        grad_Q = np.matmul(grad_scores, K)
        grad_K = np.matmul(np.swapaxes(grad_scores, -1, -2), Q)

        return grad_Q, grad_K, grad_V

    def params_and_grads(self):
        return []


# ---------------------------------------------------------------------------
# 4. Multi-Head Attention
#     MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
#     head_i             = Attention(Q W_i^Q, K W_i^K, V W_i^V)
#     Paper Section 3.2.2, Figure 2 (right).
#
#     Implementation note: we use the split-heads reshape trick instead of
#     h separate (d_model, d_k) projections per role. One big (d_model,
#     d_model) Linear per role + a reshape/transpose recovers the per-head
#     splits. See section 4 of docs/04_multi_head_attention.md for the
#     derivation, and section 5 for the worked numerical example that
#     test_mha_section5_trace reproduces.
# ---------------------------------------------------------------------------

class MultiHeadAttention(Layer):
    """
    Multi-Head Attention.

    Shapes (any number of leading batch dims ...):
        Q_in: (..., n_q, d_model)
        K_in: (..., n_k, d_model)
        V_in: (..., n_k, d_model)
        mask: broadcastable to (..., h, n_q, n_k) or None.
              True/1 = allowed, False/0 = blocked.
              For 2D masks (n_q, n_k), broadcasting handles the head and
              batch dims automatically; for per-batch masks of shape
              (B, n_q, n_k) the caller must add a head dim:
              mask[..., None, :, :].

    Forward returns:
        output: same shape as Q_in -- (..., n_q, d_model).

    Learnable params: W_q, W_k, W_v, W_o (each via the Linear class).
    Biases are zero-initialized; we keep them present to match common
    practice (PyTorch's nn.MultiheadAttention defaults to bias=True). The
    paper omits the projection biases; with bias=0 init they have no effect
    on the first forward pass, so doc-example tests are unaffected.

    Constraint: d_model must be divisible by num_heads. This is enforced
    by the split-heads reshape trick, NOT by the math (see section 2 of
    docs/04_multi_head_attention.md, "Is d_k = d_model / h actually
    required?"). For our build we keep the convention.
    """

    def __init__(self, d_model, num_heads):
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads} "
                "for the split-heads reshape trick (see section 4 of "
                "docs/04_multi_head_attention.md). The math itself is more "
                "permissive; see section 2 'Is d_k = d_model / h actually "
                "required?' for the relaxed alternatives."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        self._sdpa = ScaledDotProductAttention()

        self._cache = {}
        self.last_weights = None

    def _split_heads(self, x):
        """
        (..., n, d_model) -> (..., n, h, d_k) -> (..., h, n, d_k)

        Step 1 (reshape) is a no-op in memory: it just relabels the last
        d_model contiguous floats as an (h, d_k) grid.
        Step 2 (swapaxes -2 <-> -3) makes the array non-contiguous as a
        view; numpy.matmul handles non-contiguous inputs fine.
        """
        leading = x.shape[:-1]
        x = x.reshape(leading + (self.num_heads, self.d_k))
        return np.swapaxes(x, -2, -3)

    def _merge_heads(self, x):
        """
        (..., h, n, d_k) -> (..., n, h, d_k) -> (..., n, d_model)

        Inverse of _split_heads. The reshape after swapaxes typically needs
        to copy because the array is non-contiguous; numpy.reshape does
        that automatically.
        """
        x = np.swapaxes(x, -2, -3)
        leading = x.shape[:-2]
        return x.reshape(leading + (self.d_model,))

    def forward(self, Q_in, K_in, V_in, mask=None):
        if Q_in.shape[-1] != self.d_model:
            raise ValueError(
                f"Q_in last dim {Q_in.shape[-1]} != d_model {self.d_model}"
            )
        if K_in.shape[-1] != self.d_model:
            raise ValueError(
                f"K_in last dim {K_in.shape[-1]} != d_model {self.d_model}"
            )
        if V_in.shape[-1] != self.d_model:
            raise ValueError(
                f"V_in last dim {V_in.shape[-1]} != d_model {self.d_model}"
            )

        Q = self.W_q.forward(Q_in)
        K = self.W_k.forward(K_in)
        V = self.W_v.forward(V_in)

        Q_h = self._split_heads(Q)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        head_out = self._sdpa.forward(Q_h, K_h, V_h, mask=mask)

        merged = self._merge_heads(head_out)
        output = self.W_o.forward(merged)

        self._cache['Q_in_shape'] = Q_in.shape
        self.last_weights = self._sdpa.last_weights
        return output

    def backward(self, grad_output):
        """
        Returns (grad_Q_in, grad_K_in, grad_V_in). See section 8 of
        docs/04_multi_head_attention.md for the derivation.

        The chain (forward in reverse, each op replaced by its backward):

            grad_merged   = W_o.backward(grad_output)
            grad_head_out = split_heads(grad_merged)        # backward of merge_heads
            grad_Q_h, grad_K_h, grad_V_h = sdpa.backward(grad_head_out)
            grad_Q        = merge_heads(grad_Q_h)           # backward of split_heads
            grad_K        = merge_heads(grad_K_h)
            grad_V        = merge_heads(grad_V_h)
            grad_Q_in     = W_q.backward(grad_Q)
            grad_K_in     = W_k.backward(grad_K)
            grad_V_in     = W_v.backward(grad_V)

        Linear.backward also fills each sub-layer's grad_W and grad_b as a
        side effect.

        For self-attention (Q_in == K_in == V_in == X), the caller is
        responsible for grad_X = grad_Q_in + grad_K_in + grad_V_in. We do
        NOT sum here because cross-attention has Q_in distinct from
        K_in == V_in.
        """
        grad_merged = self.W_o.backward(grad_output)

        grad_head_out = self._split_heads(grad_merged)

        grad_Q_h, grad_K_h, grad_V_h = self._sdpa.backward(grad_head_out)

        grad_Q = self._merge_heads(grad_Q_h)
        grad_K = self._merge_heads(grad_K_h)
        grad_V = self._merge_heads(grad_V_h)

        grad_Q_in = self.W_q.backward(grad_Q)
        grad_K_in = self.W_k.backward(grad_K)
        grad_V_in = self.W_v.backward(grad_V)

        return grad_Q_in, grad_K_in, grad_V_in

    def params_and_grads(self):
        return (
            [(p, g, f'W_q.{name}') for (p, g, name) in self.W_q.params_and_grads()]
            + [(p, g, f'W_k.{name}') for (p, g, name) in self.W_k.params_and_grads()]
            + [(p, g, f'W_v.{name}') for (p, g, name) in self.W_v.params_and_grads()]
            + [(p, g, f'W_o.{name}') for (p, g, name) in self.W_o.params_and_grads()]
        )


# ---------------------------------------------------------------------------
# 5. Position-wise Feed-Forward Network
#     FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
#     Paper Section 3.3, Equation 2.
#
#     Applied independently to every position. Same weights across positions
#     within a layer; different weights across layers. Implementation is pure
#     composition of layers built in Step 1: Linear -> ReLU -> Linear.
#
#     The FFN is the only source of element-wise nonlinearity in the
#     Transformer; without it, stacking attention layers does not meaningfully
#     expand the function class. See section 2 of docs/05_feedforward_network.md
#     for the full nonlinearity argument and section 6 for the key-value
#     memory interpretation that motivates this layer's role as the model's
#     "trained lookup table".
# ---------------------------------------------------------------------------

class FeedForward(Layer):
    """
    Position-wise Feed-Forward Network.

    Shapes (any number of leading batch dims ...):
        x:      (..., d_model)
        output: (..., d_model)        -- same shape as x

    The FFN expands to d_ff in the middle (typically d_ff > d_model -- our
    build uses d_ff = 2 * d_model = 64; the paper uses 4 * d_model = 2048).
    The wider hidden state is what gives the FFN its capacity to act as a
    soft key-value memory (W_1 columns are keys, W_2 rows are values).

    Learnable params: W_1 (d_model, d_ff), b_1 (d_ff,), W_2 (d_ff, d_model),
    b_2 (d_model,) -- packaged via two Linear sub-layers. ReLU has no params.
    """

    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        self.W_1 = Linear(d_model, d_ff)
        self._relu = ReLU()
        self.W_2 = Linear(d_ff, d_model)
        self._cache = {}

    def forward(self, x):
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"FeedForward.forward: input last dim {x.shape[-1]} != d_model "
                f"{self.d_model}"
            )
        h_pre = self.W_1.forward(x)
        h = self._relu.forward(h_pre)
        return self.W_2.forward(h)

    def backward(self, grad_output):
        """
        Pure sub-layer composition; no new derivative work. See section 9 of
        docs/05_feedforward_network.md.

        Forward:    x ---W_1---> h_pre ---ReLU---> h ---W_2---> output
        Backward:   grad_x <-W_1- grad_h_pre <-ReLU- grad_h <-W_2- grad_output

        Each sub-Linear's backward also fills its own grad_W and grad_b.
        """
        grad_h = self.W_2.backward(grad_output)
        grad_h_pre = self._relu.backward(grad_h)
        grad_x = self.W_1.backward(grad_h_pre)
        return grad_x

    def params_and_grads(self):
        return (
            [(p, g, f'W_1.{name}') for (p, g, name) in self.W_1.params_and_grads()]
            + [(p, g, f'W_2.{name}') for (p, g, name) in self.W_2.params_and_grads()]
        )


# ---------------------------------------------------------------------------
# 6. Encoder layer
#     One repeating block of the encoder: MHA wrapper + FFN wrapper.
#     Each wrapper is "residual + LayerNorm" around its sub-layer; see
#     section 3 of docs/06_encoder.md for the wrapper pattern.
#
#     Two wrappings are supported:
#       Post-LN (paper, default): y = LayerNorm(x + Dropout(Sublayer(x)))
#       Pre-LN  (modern, opt-in): y = x + Dropout(Sublayer(LayerNorm(x)))
#
#     See section 5 of docs/06_encoder.md for the trade-off (Pre-LN trains
#     more reliably; Post-LN is paper-faithful and gives marginally lower
#     final loss when it converges).
#
#     Dropout placement (paper section 5.4): on the sub-layer's output,
#     before the residual add. We do NOT yet implement attention-weight
#     dropout (Place 3) or FFN-internal dropout (Place 4); both are
#     non-paper-text and not needed for the toy task. See section 6 of
#     docs/06_encoder.md.
# ---------------------------------------------------------------------------

class EncoderLayer(Layer):
    """
    One encoder block: self-attention sub-layer + FFN sub-layer, each wrapped
    in residual connection and LayerNorm.

    Shapes:
        x:      (..., n, d_model)
        output: (..., n, d_model)        -- shape-preserving (required by residuals)

    Internal sub-layers and their roles:
        - W_q/W_k/W_v/W_o          inside MHA
        - W_1/W_2                  inside FFN
        - gamma/beta x 2           in the two LayerNorms (ln_1, ln_2)
        - Dropout x 2              dropout_1 (after MHA), dropout_2 (after FFN)

    pre_ln (default False): if True, use Pre-LN wrapping (LayerNorm sits on
    the path entering the sub-layer, residual stream is un-normalized). Most
    modern Transformers use Pre-LN. Our default follows the paper (Post-LN);
    the flag exists for ablation. See sections 3 and 5 of docs/06_encoder.md.
    """

    def __init__(self, d_model, num_heads, d_ff, p_drop=0.1, pre_ln=False):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.pre_ln = bool(pre_ln)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout_1 = Dropout(p_drop)
        self.ln_1 = LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.dropout_2 = Dropout(p_drop)
        self.ln_2 = LayerNorm(d_model)

    # ------- training-mode toggle (propagates to all dropouts in this block) -------

    def train(self, mode=True):
        """Set training mode on every Dropout in this block. Returns self."""
        self.dropout_1.training = mode
        self.dropout_2.training = mode
        return self

    def eval(self):
        """Shorthand for self.train(False)."""
        return self.train(False)

    # ------------------------- forward / backward --------------------------

    def forward(self, x):
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"EncoderLayer.forward: input last dim {x.shape[-1]} != d_model "
                f"{self.d_model}"
            )
        if self.pre_ln:
            return self._forward_pre_ln(x)
        return self._forward_post_ln(x)

    def backward(self, grad_output):
        if self.pre_ln:
            return self._backward_pre_ln(grad_output)
        return self._backward_post_ln(grad_output)

    # ------------------------------- Post-LN -------------------------------
    #
    # Forward:
    #     h_mha   = MHA(x, x, x)
    #     h_mha_d = Dropout1(h_mha)
    #     y_mid   = LN1(x + h_mha_d)
    #
    #     h_ffn   = FFN(y_mid)
    #     h_ffn_d = Dropout2(h_ffn)
    #     y_out   = LN2(y_mid + h_ffn_d)
    #
    # Backward (in reverse), using the residual-sum rule grad flows to BOTH
    # branches of an addition, and the self-attention rule that grad_x is the
    # sum of grad_Q + grad_K + grad_V from MHA.backward():
    #
    #     g_sum2 = LN2.backward(grad_out)                       # grad of (y_mid + h_ffn_d)
    #     grad_h_ffn_d = g_sum2
    #     grad_h_ffn   = Dropout2.backward(grad_h_ffn_d)
    #     grad_y_mid_via_ffn = FFN.backward(grad_h_ffn)
    #     grad_y_mid = g_sum2 + grad_y_mid_via_ffn              # residual + sublayer paths
    #
    #     g_sum1 = LN1.backward(grad_y_mid)                     # grad of (x + h_mha_d)
    #     grad_h_mha_d = g_sum1
    #     grad_h_mha   = Dropout1.backward(grad_h_mha_d)
    #     gQ, gK, gV   = MHA.backward(grad_h_mha)
    #     grad_x_via_mha = gQ + gK + gV                         # self-attention sum
    #     grad_x = g_sum1 + grad_x_via_mha

    def _forward_post_ln(self, x):
        h_mha = self.mha.forward(x, x, x)
        h_mha_d = self.dropout_1.forward(h_mha)
        y_mid = self.ln_1.forward(x + h_mha_d)

        h_ffn = self.ffn.forward(y_mid)
        h_ffn_d = self.dropout_2.forward(h_ffn)
        y_out = self.ln_2.forward(y_mid + h_ffn_d)
        return y_out

    def _backward_post_ln(self, grad_output):
        g_sum2 = self.ln_2.backward(grad_output)
        grad_h_ffn = self.dropout_2.backward(g_sum2)
        grad_y_mid_via_ffn = self.ffn.backward(grad_h_ffn)
        grad_y_mid = g_sum2 + grad_y_mid_via_ffn

        g_sum1 = self.ln_1.backward(grad_y_mid)
        grad_h_mha = self.dropout_1.backward(g_sum1)
        gQ, gK, gV = self.mha.backward(grad_h_mha)
        grad_x_via_mha = gQ + gK + gV
        grad_x = g_sum1 + grad_x_via_mha
        return grad_x

    # -------------------------------- Pre-LN -------------------------------
    #
    # Forward:
    #     x_ln    = LN1(x)
    #     h_mha   = MHA(x_ln, x_ln, x_ln)
    #     h_mha_d = Dropout1(h_mha)
    #     y_mid   = x + h_mha_d                                 # un-normalized residual
    #
    #     y_ln    = LN2(y_mid)
    #     h_ffn   = FFN(y_ln)
    #     h_ffn_d = Dropout2(h_ffn)
    #     y_out   = y_mid + h_ffn_d
    #
    # Backward:
    #     # FFN wrapper (residual sum -> both branches)
    #     grad_y_mid_via_residual = grad_out
    #     grad_h_ffn = Dropout2.backward(grad_out)
    #     grad_y_ln  = FFN.backward(grad_h_ffn)
    #     grad_y_mid_via_ffn_path = LN2.backward(grad_y_ln)
    #     grad_y_mid = grad_y_mid_via_residual + grad_y_mid_via_ffn_path
    #
    #     # MHA wrapper (residual sum -> both branches)
    #     grad_x_via_residual = grad_y_mid
    #     grad_h_mha = Dropout1.backward(grad_y_mid)
    #     gQ, gK, gV = MHA.backward(grad_h_mha)
    #     grad_x_ln  = gQ + gK + gV                             # self-attention sum
    #     grad_x_via_mha_path = LN1.backward(grad_x_ln)
    #     grad_x = grad_x_via_residual + grad_x_via_mha_path

    def _forward_pre_ln(self, x):
        x_ln = self.ln_1.forward(x)
        h_mha = self.mha.forward(x_ln, x_ln, x_ln)
        h_mha_d = self.dropout_1.forward(h_mha)
        y_mid = x + h_mha_d

        y_ln = self.ln_2.forward(y_mid)
        h_ffn = self.ffn.forward(y_ln)
        h_ffn_d = self.dropout_2.forward(h_ffn)
        y_out = y_mid + h_ffn_d
        return y_out

    def _backward_pre_ln(self, grad_output):
        grad_h_ffn = self.dropout_2.backward(grad_output)
        grad_y_ln = self.ffn.backward(grad_h_ffn)
        grad_y_mid_via_ffn_path = self.ln_2.backward(grad_y_ln)
        grad_y_mid = grad_output + grad_y_mid_via_ffn_path

        grad_h_mha = self.dropout_1.backward(grad_y_mid)
        gQ, gK, gV = self.mha.backward(grad_h_mha)
        grad_x_ln = gQ + gK + gV
        grad_x_via_mha_path = self.ln_1.backward(grad_x_ln)
        grad_x = grad_y_mid + grad_x_via_mha_path
        return grad_x

    # ----------------------------- params hookup ---------------------------

    def params_and_grads(self):
        return (
            [(p, g, f'mha.{name}') for (p, g, name) in self.mha.params_and_grads()]
            + [(p, g, f'ln_1.{name}') for (p, g, name) in self.ln_1.params_and_grads()]
            + [(p, g, f'ffn.{name}') for (p, g, name) in self.ffn.params_and_grads()]
            + [(p, g, f'ln_2.{name}') for (p, g, name) in self.ln_2.params_and_grads()]
        )


# ---------------------------------------------------------------------------
# 7. Encoder
#     The full input pipeline: token IDs -> contextual representation.
#         tokens -> Embedding -> * sqrt(d_model) -> + PE -> Dropout
#                -> EncoderLayer x N -> output
#
#     See section 8 of docs/06_encoder.md for the data-flow diagram and
#     section 7 for the sqrt(d_model) embedding scaling rationale.
# ---------------------------------------------------------------------------

class Encoder(Layer):
    """
    Full encoder stack.

    Shapes:
        token_ids: (B, n)             -- integer token IDs
        output:    (B, n, d_model)

    Composition (paper sections 3.1, 3.4, 5.4):
        e         = embedding(token_ids)         (B, n, d_model)
        e_scaled  = e * sqrt(d_model)            (B, n, d_model)   # paper section 3.4
        e_pe      = e_scaled + PE[:n, :]         (B, n, d_model)
        x_0       = dropout_input(e_pe)          (B, n, d_model)   # paper section 5.4
        x_N       = layers[N-1](... layers[0](x_0) ...)
        return x_N

    The Embedding layer takes integer token IDs and has no input-side
    gradient, so Encoder.backward() returns None (consistent with the
    Embedding layer's contract). It still populates parameter gradients
    on every learnable sub-layer.

    Embedding sharing (Step 7, paper section 3.4): pass `shared_embedding`
    to use an externally owned `Embedding` (typically constructed by the
    `Transformer` wrapper and shared with the `Decoder`). When shared,
    `params_and_grads()` does NOT expose the embedding's parameter -- the
    owner is responsible for that.
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 p_drop=0.1, max_len=5000, pre_ln=False,
                 shared_embedding=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.p_drop = p_drop
        self.max_len = max_len
        self.pre_ln = bool(pre_ln)

        if shared_embedding is None:
            self.embedding = Embedding(vocab_size, d_model)
            self._owns_embedding = True
        else:
            if shared_embedding.vocab_size != vocab_size:
                raise ValueError(
                    f"shared_embedding.vocab_size {shared_embedding.vocab_size} "
                    f"!= vocab_size {vocab_size}"
                )
            if shared_embedding.d_model != d_model:
                raise ValueError(
                    f"shared_embedding.d_model {shared_embedding.d_model} "
                    f"!= d_model {d_model}"
                )
            self.embedding = shared_embedding
            self._owns_embedding = False

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout_input = Dropout(p_drop)
        self.layers = [
            EncoderLayer(d_model, num_heads, d_ff, p_drop=p_drop, pre_ln=pre_ln)
            for _ in range(num_layers)
        ]

        self._embed_scale = np.sqrt(d_model)
        self._cache_token_ids = None

    # ------- training-mode toggle (recursively applies to sub-layers) -------

    def train(self, mode=True):
        """Set training mode on every Dropout in the encoder. Returns self."""
        self.dropout_input.training = mode
        for layer in self.layers:
            layer.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ------------------------- forward / backward --------------------------

    def forward(self, token_ids):
        # Cache our own token_ids: the shared-embedding case (Transformer
        # wrapper) means embedding._cache['token_ids'] is racy because the
        # decoder's later forward overwrites it. We pass our own copy to
        # embedding.backward.
        self._cache_token_ids = token_ids
        e = self.embedding.forward(token_ids)
        e_scaled = e * self._embed_scale
        e_pe = self.pos_enc.forward(e_scaled)
        x = self.dropout_input.forward(e_pe)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        """
        Returns None: the embedding's input is integer token IDs, which have
        no continuous gradient. All learnable parameters' grads are
        populated as a side effect on each sub-layer.
        """
        grad_x = grad_output
        for layer in reversed(self.layers):
            grad_x = layer.backward(grad_x)
        grad_e_pe = self.dropout_input.backward(grad_x)
        grad_e_scaled = self.pos_enc.backward(grad_e_pe)
        grad_e = grad_e_scaled * self._embed_scale
        self.embedding.backward(grad_e, token_ids=self._cache_token_ids)
        return None

    # ----------------------------- params hookup ---------------------------

    def params_and_grads(self):
        out = []
        if self._owns_embedding:
            out.extend(
                (p, g, f'embedding.{name}')
                for (p, g, name) in self.embedding.params_and_grads()
            )
        for i, layer in enumerate(self.layers):
            out.extend(
                (p, g, f'layers[{i}].{name}')
                for (p, g, name) in layer.params_and_grads()
            )
        return out


# ---------------------------------------------------------------------------
# 8. Tied Linear (output projection that shares its weight with an Embedding)
#     y = x @ shared_owner.weight^T   (no bias; paper section 3.4)
#
#     Used by the Decoder's output projection when tie_embeddings=True. The
#     same matrix that maps token IDs to vectors at input also serves
#     (transposed) as the pre-softmax linear at output. See section 6 of
#     docs/07_decoder.md.
# ---------------------------------------------------------------------------

class TiedLinear(Layer):
    """
    Bias-free linear layer whose weight matrix is the transpose of an
    externally owned weight (typically an `Embedding.weight`).

    Shapes:
        x:      (..., d_model)
        output: (..., vocab_size)

    Forward: y = x @ W.T,   where W = shared_owner.weight, shape (vocab_size, d_model).
    No bias parameter; paper section 3.4 specifies no bias on the
    pre-softmax linear.

    Backward: dL/dx = grad_output @ W
              dL/dW (accumulated into shared_owner.grad_weight via the
              shared owner's existing accumulation contract).

    Parameters: zero. The shared owner exposes its own weight via its
    own `params_and_grads()`; this layer does not double-expose it.
    """

    def __init__(self, shared_owner, weight_attr='weight', grad_attr='grad_weight'):
        self.shared_owner = shared_owner
        self.weight_attr = weight_attr
        self.grad_attr = grad_attr
        self._cache = {}

    def forward(self, x):
        if x.shape[-1] != getattr(self.shared_owner, self.weight_attr).shape[-1]:
            W = getattr(self.shared_owner, self.weight_attr)
            raise ValueError(
                f"TiedLinear.forward: input last dim {x.shape[-1]} != "
                f"shared weight last dim {W.shape[-1]}"
            )
        self._cache['x'] = x
        W = getattr(self.shared_owner, self.weight_attr)
        return x @ W.T

    def backward(self, grad_output):
        x = self._cache['x']
        W = getattr(self.shared_owner, self.weight_attr)

        grad_input = grad_output @ W

        x_2d = x.reshape(-1, x.shape[-1])
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_W_contribution = grad_2d.T @ x_2d

        existing = getattr(self.shared_owner, self.grad_attr)
        if existing is None or existing.shape != W.shape:
            existing = np.zeros_like(W)
            setattr(self.shared_owner, self.grad_attr, existing)
        existing += grad_W_contribution

        return grad_input

    def params_and_grads(self):
        return []


# ---------------------------------------------------------------------------
# 9. Decoder layer
#     One repeating block of the decoder: THREE sub-layers, each wrapped in
#     residual + LayerNorm.
#         1. Masked self-attention   (causal mask, prevents future-leak)
#         2. Cross-attention         (Q from decoder, K=V from encoder output)
#         3. FFN                     (per-position integration)
#
#     See sections 2-4 of docs/07_decoder.md for the wrapper pattern, the
#     causal mask rationale, and the cross-attention semantics.
#
#     Same Pre-LN / Post-LN trade-off as the encoder (section 5 of
#     docs/06_encoder.md). Same dropout placements as the encoder
#     (paper section 5.4): one Dropout per wrapper, on the sub-layer's
#     output, before the residual add.
# ---------------------------------------------------------------------------

class DecoderLayer(Layer):
    """
    One decoder block: masked self-attention + cross-attention + FFN, each
    wrapped in residual connection and LayerNorm.

    Shapes:
        x:       (B, n_q, d_model)         -- decoder hidden state
        enc_out: (B, n_k, d_model)         -- encoder output (K, V source)
        mask:    (n_q, n_q) boolean        -- causal mask for self-attention
        output:  (B, n_q, d_model)

    Internal sub-layers and their roles:
        - mha_self  : MaskedSelfAttention   wrapped by dropout_1, ln_1
        - mha_cross : CrossAttention        wrapped by dropout_2, ln_2
        - ffn       : FeedForward           wrapped by dropout_3, ln_3

    Backward returns a 2-tuple (grad_x, grad_enc_out_contribution):
    - grad_x: gradient w.r.t. the decoder hidden-state input.
    - grad_enc_out_contribution: gradient w.r.t. the encoder output for
      THIS layer. The Decoder sums these across all layers (each layer
      reads enc_out independently, so contributions add by the
      multivariate chain rule).
    """

    def __init__(self, d_model, num_heads, d_ff, p_drop=0.1, pre_ln=False):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.pre_ln = bool(pre_ln)

        self.mha_self = MultiHeadAttention(d_model, num_heads)
        self.dropout_1 = Dropout(p_drop)
        self.ln_1 = LayerNorm(d_model)

        self.mha_cross = MultiHeadAttention(d_model, num_heads)
        self.dropout_2 = Dropout(p_drop)
        self.ln_2 = LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.dropout_3 = Dropout(p_drop)
        self.ln_3 = LayerNorm(d_model)

    # ------- training-mode toggle (propagates to all dropouts in this block) -------

    def train(self, mode=True):
        self.dropout_1.training = mode
        self.dropout_2.training = mode
        self.dropout_3.training = mode
        return self

    def eval(self):
        return self.train(False)

    # ------------------------- forward / backward --------------------------

    def forward(self, x, enc_out, causal_mask):
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"DecoderLayer.forward: x last dim {x.shape[-1]} != d_model "
                f"{self.d_model}"
            )
        if enc_out.shape[-1] != self.d_model:
            raise ValueError(
                f"DecoderLayer.forward: enc_out last dim {enc_out.shape[-1]} "
                f"!= d_model {self.d_model}"
            )
        if self.pre_ln:
            return self._forward_pre_ln(x, enc_out, causal_mask)
        return self._forward_post_ln(x, enc_out, causal_mask)

    def backward(self, grad_output):
        if self.pre_ln:
            return self._backward_pre_ln(grad_output)
        return self._backward_post_ln(grad_output)

    # ------------------------------- Post-LN -------------------------------
    #
    # Forward:
    #     h_self    = MHA_self(x, x, x, causal_mask)
    #     h_self_d  = Dropout1(h_self)
    #     y1        = LN1(x + h_self_d)
    #
    #     h_cross   = MHA_cross(y1, enc_out, enc_out)
    #     h_cross_d = Dropout2(h_cross)
    #     y2        = LN2(y1 + h_cross_d)
    #
    #     h_ffn     = FFN(y2)
    #     h_ffn_d   = Dropout3(h_ffn)
    #     y3        = LN3(y2 + h_ffn_d)
    #
    # Backward (in reverse), three rules at play:
    #   - residual sum: grad flows unchanged to BOTH branches of an addition.
    #   - self-attention: in MHA_self where Q_in = K_in = V_in = x, the
    #     gradient back into x is gQ + gK + gV from MHA.backward.
    #   - cross-attention: in MHA_cross where Q_in is decoder-side and
    #     K_in = V_in = enc_out, gQ flows back to decoder hidden state and
    #     gK + gV flows back to enc_out.

    def _forward_post_ln(self, x, enc_out, causal_mask):
        h_self = self.mha_self.forward(x, x, x, mask=causal_mask)
        h_self_d = self.dropout_1.forward(h_self)
        y1 = self.ln_1.forward(x + h_self_d)

        h_cross = self.mha_cross.forward(y1, enc_out, enc_out, mask=None)
        h_cross_d = self.dropout_2.forward(h_cross)
        y2 = self.ln_2.forward(y1 + h_cross_d)

        h_ffn = self.ffn.forward(y2)
        h_ffn_d = self.dropout_3.forward(h_ffn)
        y3 = self.ln_3.forward(y2 + h_ffn_d)
        return y3

    def _backward_post_ln(self, grad_output):
        g_sum3 = self.ln_3.backward(grad_output)
        grad_h_ffn = self.dropout_3.backward(g_sum3)
        grad_y2_via_ffn = self.ffn.backward(grad_h_ffn)
        grad_y2 = g_sum3 + grad_y2_via_ffn

        g_sum2 = self.ln_2.backward(grad_y2)
        grad_h_cross = self.dropout_2.backward(g_sum2)
        gQ_c, gK_c, gV_c = self.mha_cross.backward(grad_h_cross)
        grad_y1_via_cross = gQ_c
        grad_enc_out = gK_c + gV_c
        grad_y1 = g_sum2 + grad_y1_via_cross

        g_sum1 = self.ln_1.backward(grad_y1)
        grad_h_self = self.dropout_1.backward(g_sum1)
        gQ_s, gK_s, gV_s = self.mha_self.backward(grad_h_self)
        grad_x_via_self = gQ_s + gK_s + gV_s
        grad_x = g_sum1 + grad_x_via_self

        return grad_x, grad_enc_out

    # -------------------------------- Pre-LN -------------------------------
    #
    # Forward:
    #     x_ln1     = LN1(x)
    #     h_self    = MHA_self(x_ln1, x_ln1, x_ln1, causal_mask)
    #     h_self_d  = Dropout1(h_self)
    #     y1        = x + h_self_d
    #
    #     y1_ln     = LN2(y1)
    #     h_cross   = MHA_cross(y1_ln, enc_out, enc_out)
    #                 NOTE: enc_out is consumed as-is by Pre-LN cross-attn.
    #                       Whether enc_out should also pass through an LN
    #                       is a final-LN-on-encoder design choice; we
    #                       defer it (see Discussion Log of docs/07_decoder.md).
    #     h_cross_d = Dropout2(h_cross)
    #     y2        = y1 + h_cross_d
    #
    #     y2_ln     = LN3(y2)
    #     h_ffn     = FFN(y2_ln)
    #     h_ffn_d   = Dropout3(h_ffn)
    #     y3        = y2 + h_ffn_d

    def _forward_pre_ln(self, x, enc_out, causal_mask):
        x_ln1 = self.ln_1.forward(x)
        h_self = self.mha_self.forward(x_ln1, x_ln1, x_ln1, mask=causal_mask)
        h_self_d = self.dropout_1.forward(h_self)
        y1 = x + h_self_d

        y1_ln = self.ln_2.forward(y1)
        h_cross = self.mha_cross.forward(y1_ln, enc_out, enc_out, mask=None)
        h_cross_d = self.dropout_2.forward(h_cross)
        y2 = y1 + h_cross_d

        y2_ln = self.ln_3.forward(y2)
        h_ffn = self.ffn.forward(y2_ln)
        h_ffn_d = self.dropout_3.forward(h_ffn)
        y3 = y2 + h_ffn_d
        return y3

    def _backward_pre_ln(self, grad_output):
        # FFN wrapper
        grad_h_ffn = self.dropout_3.backward(grad_output)
        grad_y2_ln = self.ffn.backward(grad_h_ffn)
        grad_y2_via_ffn_path = self.ln_3.backward(grad_y2_ln)
        grad_y2 = grad_output + grad_y2_via_ffn_path

        # Cross-attention wrapper
        grad_h_cross = self.dropout_2.backward(grad_y2)
        gQ_c, gK_c, gV_c = self.mha_cross.backward(grad_h_cross)
        grad_y1_ln = gQ_c
        grad_enc_out = gK_c + gV_c
        grad_y1_via_cross_path = self.ln_2.backward(grad_y1_ln)
        grad_y1 = grad_y2 + grad_y1_via_cross_path

        # Self-attention wrapper
        grad_h_self = self.dropout_1.backward(grad_y1)
        gQ_s, gK_s, gV_s = self.mha_self.backward(grad_h_self)
        grad_x_ln = gQ_s + gK_s + gV_s
        grad_x_via_self_path = self.ln_1.backward(grad_x_ln)
        grad_x = grad_y1 + grad_x_via_self_path

        return grad_x, grad_enc_out

    # ----------------------------- params hookup ---------------------------

    def params_and_grads(self):
        return (
            [(p, g, f'mha_self.{name}') for (p, g, name) in self.mha_self.params_and_grads()]
            + [(p, g, f'ln_1.{name}') for (p, g, name) in self.ln_1.params_and_grads()]
            + [(p, g, f'mha_cross.{name}') for (p, g, name) in self.mha_cross.params_and_grads()]
            + [(p, g, f'ln_2.{name}') for (p, g, name) in self.ln_2.params_and_grads()]
            + [(p, g, f'ffn.{name}') for (p, g, name) in self.ffn.params_and_grads()]
            + [(p, g, f'ln_3.{name}') for (p, g, name) in self.ln_3.params_and_grads()]
        )


# ---------------------------------------------------------------------------
# 10. Decoder
#     The full target-side pipeline: target token IDs + encoder output
#     -> logits over vocabulary.
#         target_ids -> Embedding -> * sqrt(d_model) -> + PE -> Dropout
#                    -> DecoderLayer x N (each cross-attending into enc_out)
#                    -> OutputProjection (tied or untied)
#                    -> logits
#
#     See sections 1, 6, 8 of docs/07_decoder.md for the data flow,
#     output projection rationale, and inference loop pseudocode.
# ---------------------------------------------------------------------------

class Decoder(Layer):
    """
    Full decoder stack with output projection.

    Shapes:
        target_ids: (B, n_q)              -- integer token IDs
        enc_out:    (B, n_k, d_model)     -- encoder output, threaded
                                              into every cross-attention
        output:     (B, n_q, vocab_size)  -- logits

    Construction:
        Decoder(vocab_size, d_model, num_heads, d_ff, num_layers,
                p_drop=0.1, max_len=5000, pre_ln=False,
                shared_embedding=None, tie_embeddings=True)

    shared_embedding: if None (default), Decoder constructs and owns its
        own Embedding. If given, uses the provided Embedding (typically
        constructed by a `Transformer` wrapper and shared with an Encoder).
        When shared, params_and_grads() does NOT expose the embedding -- the
        owner is responsible for that.

    tie_embeddings: if True (default, paper-faithful), the output projection
        is a `TiedLinear` that uses the embedding's weight (transposed).
        If False, builds a standalone `Linear(d_model, vocab_size)` for the
        output projection. The untied path retains a bias for simplicity
        (regular Linear); the paper omits the bias on the pre-softmax
        linear (section 3.4). For the tied path we are paper-faithful (no
        bias).

    Backward returns grad_enc_out, the gradient flowing back to the
    encoder's output. The standalone Decoder.backward() also populates
    embedding.grad_weight with both the input-side and tied output-side
    contributions when tie_embeddings=True.
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 p_drop=0.1, max_len=5000, pre_ln=False,
                 shared_embedding=None, tie_embeddings=True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.p_drop = p_drop
        self.max_len = max_len
        self.pre_ln = bool(pre_ln)
        self.tie_embeddings = bool(tie_embeddings)

        if shared_embedding is None:
            self.embedding = Embedding(vocab_size, d_model)
            self._owns_embedding = True
        else:
            if shared_embedding.vocab_size != vocab_size:
                raise ValueError(
                    f"shared_embedding.vocab_size {shared_embedding.vocab_size} "
                    f"!= vocab_size {vocab_size}"
                )
            if shared_embedding.d_model != d_model:
                raise ValueError(
                    f"shared_embedding.d_model {shared_embedding.d_model} "
                    f"!= d_model {d_model}"
                )
            self.embedding = shared_embedding
            self._owns_embedding = False

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout_input = Dropout(p_drop)
        self.layers = [
            DecoderLayer(d_model, num_heads, d_ff, p_drop=p_drop, pre_ln=pre_ln)
            for _ in range(num_layers)
        ]

        if self.tie_embeddings:
            self.output_projection = TiedLinear(self.embedding)
        else:
            self.output_projection = Linear(d_model, vocab_size)

        self._embed_scale = np.sqrt(d_model)
        self._cache = {}
        self._cache_token_ids = None

    # ------- training-mode toggle (recursively applies to sub-layers) -------

    def train(self, mode=True):
        self.dropout_input.training = mode
        for layer in self.layers:
            layer.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ------------------------- forward / backward --------------------------

    def forward(self, target_ids, enc_out):
        if enc_out.shape[-1] != self.d_model:
            raise ValueError(
                f"Decoder.forward: enc_out last dim {enc_out.shape[-1]} "
                f"!= d_model {self.d_model}"
            )

        n_q = target_ids.shape[-1]

        self._cache_token_ids = target_ids
        e = self.embedding.forward(target_ids)
        e_scaled = e * self._embed_scale
        e_pe = self.pos_enc.forward(e_scaled)
        x = self.dropout_input.forward(e_pe)

        # Causal mask: lower-triangular boolean (n_q, n_q). True = allowed,
        # False = blocked. See section 3 of docs/07_decoder.md.
        causal_mask = np.tril(np.ones((n_q, n_q), dtype=bool))

        self._cache['enc_out_shape'] = enc_out.shape

        for layer in self.layers:
            x = layer.forward(x, enc_out, causal_mask)

        logits = self.output_projection.forward(x)
        return logits

    def backward(self, grad_logits):
        """
        Returns grad_enc_out: gradient of the loss w.r.t. the encoder output
        (B, n_k, d_model). Each decoder layer cross-attends into enc_out
        independently; their contributions sum.

        Side effects: populates the gradient buffers of every learnable
        sub-layer. When tie_embeddings=True, the gradient on
        embedding.weight has contributions from BOTH the input embedding
        lookup AND the output projection (accumulated via the embedding's
        new accumulate-on-backward contract).
        """
        grad_x = self.output_projection.backward(grad_logits)

        grad_enc_out = np.zeros(self._cache['enc_out_shape'])
        for layer in reversed(self.layers):
            grad_x, grad_enc_out_layer = layer.backward(grad_x)
            grad_enc_out += grad_enc_out_layer

        grad_e_pe = self.dropout_input.backward(grad_x)
        grad_e_scaled = self.pos_enc.backward(grad_e_pe)
        grad_e = grad_e_scaled * self._embed_scale
        self.embedding.backward(grad_e, token_ids=self._cache_token_ids)

        return grad_enc_out

    # ----------------------------- params hookup ---------------------------

    def params_and_grads(self):
        out = []
        if self._owns_embedding:
            out.extend(
                (p, g, f'embedding.{name}')
                for (p, g, name) in self.embedding.params_and_grads()
            )
        for i, layer in enumerate(self.layers):
            out.extend(
                (p, g, f'layers[{i}].{name}')
                for (p, g, name) in layer.params_and_grads()
            )
        # Tied output projection has no own params (the shared embedding
        # already exposes them). Untied output projection has its own.
        if not self.tie_embeddings:
            out.extend(
                (p, g, f'output_projection.{name}')
                for (p, g, name) in self.output_projection.params_and_grads()
            )
        return out


# ---------------------------------------------------------------------------
# 11. Transformer
#     Lightweight wrapper that owns the shared Embedding and constructs the
#     Encoder + Decoder around it. Provides the paper-faithful three-way
#     embedding sharing (encoder embed = decoder embed = output proj^T).
#
#     See section 6 of docs/07_decoder.md.
# ---------------------------------------------------------------------------

class Transformer(Layer):
    """
    Encoder-decoder Transformer with a shared (single-vocab) embedding.

    Shapes:
        source_ids: (B, n_k)
        target_ids: (B, n_q)
        output:     (B, n_q, vocab_size)  -- logits

    Construction:
        Transformer(vocab_size, d_model, num_heads, d_ff,
                    num_encoder_layers, num_decoder_layers,
                    p_drop=0.1, max_len=5000,
                    pre_ln=False, tie_embeddings=True)

    The shared `Embedding` is constructed here and passed to both the
    Encoder and the Decoder. When `tie_embeddings=True` the Decoder also
    uses the same matrix (transposed) as its output projection, giving the
    full three-way sharing of paper section 3.4.

    Embedding ownership: this Transformer is the unique owner of the
    embedding. Encoder and Decoder hold references but do not expose it
    via their own `params_and_grads()`; this class does.
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_encoder_layers, num_decoder_layers,
                 p_drop=0.1, max_len=5000,
                 pre_ln=False, tie_embeddings=True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.p_drop = p_drop
        self.max_len = max_len
        self.pre_ln = bool(pre_ln)
        self.tie_embeddings = bool(tie_embeddings)

        self.embedding = Embedding(vocab_size, d_model)

        self.encoder = Encoder(
            vocab_size, d_model, num_heads, d_ff, num_encoder_layers,
            p_drop=p_drop, max_len=max_len, pre_ln=pre_ln,
            shared_embedding=self.embedding,
        )
        self.decoder = Decoder(
            vocab_size, d_model, num_heads, d_ff, num_decoder_layers,
            p_drop=p_drop, max_len=max_len, pre_ln=pre_ln,
            shared_embedding=self.embedding,
            tie_embeddings=tie_embeddings,
        )

    # ------- training-mode toggle (recursively applies to sub-layers) -------

    def train(self, mode=True):
        self.encoder.train(mode)
        self.decoder.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ------------------------- forward / backward --------------------------

    def forward(self, source_ids, target_ids):
        enc_out = self.encoder.forward(source_ids)
        logits = self.decoder.forward(target_ids, enc_out)
        return logits

    def backward(self, grad_logits):
        """
        Returns None: token IDs have no continuous gradient. All learnable
        parameters' grads are populated as a side effect.

        Order matters: the Decoder runs first (gradient flows from logits
        backwards) and produces grad_enc_out, which the Encoder then uses
        as its grad_output. Both call embedding.backward (with each side's
        own token_ids), and both contributions accumulate into
        embedding.grad_weight via the new accumulate-on-backward contract
        introduced in Step 7. When tie_embeddings=True, the Decoder's
        output projection (TiedLinear) adds a third contribution to the
        same grad_weight.
        """
        grad_enc_out = self.decoder.backward(grad_logits)
        self.encoder.backward(grad_enc_out)
        return None

    # ----------------------------- params hookup ---------------------------

    def params_and_grads(self):
        out = [
            (p, g, f'embedding.{name}')
            for (p, g, name) in self.embedding.params_and_grads()
        ]
        out.extend(
            (p, g, f'encoder.{name}')
            for (p, g, name) in self.encoder.params_and_grads()
        )
        out.extend(
            (p, g, f'decoder.{name}')
            for (p, g, name) in self.decoder.params_and_grads()
        )
        return out
