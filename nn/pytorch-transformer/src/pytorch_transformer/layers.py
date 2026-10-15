"""
Step 12: Transformer layers -- idiomatic PyTorch port of numpy_transformer.layers.

What changes from the NumPy version:
  - No backward() methods: autograd handles all gradients.
  - No params_and_grads(): model.parameters() / named_parameters() replace it.
  - No manual train/eval toggling: model.train() / model.eval() recurse automatically.
  - Positional encoding stored as a register_buffer (moves with model.to(device)).
  - Attention: F.scaled_dot_product_attention (fused kernel, Flash-Attention-ready).
  - Layer stacks: nn.ModuleList (auto-registers all submodules).
  - Tied output projection: F.linear(x, weight=embedding.weight), no TiedLinear class.

What stays the same:
  - Architecture (encoder-decoder, MHA, FFN, LayerNorm, residuals, sinusoidal PE).
  - Post-LN default, Pre-LN opt-in via pre_ln=True.
  - Tied embeddings default (tie_embeddings=True).
  - Same hyperparameter names as the NumPy version.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (no learnable parameters).

    PE table stored as a buffer: moves with model.to(device), excluded from
    model.parameters(), not saved/loaded by state_dict (recomputed on init).
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with a single fused attention kernel.

    Uses F.scaled_dot_product_attention which dispatches to Flash Attention or
    memory-efficient attention depending on hardware. No separate
    ScaledDotProductAttention class needed.

    mask convention (when provided): boolean tensor, True = attend.
    is_causal=True: use the built-in causal shortcut (no explicit mask needed).
    """

    def __init__(self, d_model: int, num_heads: int, p_drop: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.p_drop = p_drop
        # Paper section 3.4: no bias on attention projections.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, T_q, _ = q.shape
        T_k = k.size(1)

        # Project + split: (B, T, d_model) -> (B, h, T, d_k)
        Q = self.W_q(q).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)

        # Fused kernel: scale, mask, softmax, dropout in one op.
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.p_drop if self.training else 0.0,
            is_causal=is_causal,
        )

        # Merge heads: (B, h, T_q, d_k) -> (B, T_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, d_model: int, d_ff: int, p_drop: float = 0.1) -> None:
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(self.dropout(F.relu(self.W_1(x))))


class EncoderLayer(nn.Module):
    """One encoder block: self-attention + FFN, each wrapped in residual + LayerNorm.

    pre_ln=False (default): Post-LN -- LayerNorm(x + Dropout(Sublayer(x)))
    pre_ln=True:            Pre-LN  -- x + Dropout(Sublayer(LayerNorm(x)))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        p_drop: float = 0.1,
        pre_ln: bool = False,
    ) -> None:
        super().__init__()
        self.pre_ln = pre_ln
        self.mha = MultiHeadAttention(d_model, num_heads, p_drop)
        self.ffn = FeedForward(d_model, d_ff, p_drop)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.drop_1 = nn.Dropout(p_drop)
        self.drop_2 = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_ln:
            x_n = self.ln_1(x)
            x = x + self.drop_1(self.mha(x_n, x_n, x_n))
            x = x + self.drop_2(self.ffn(self.ln_2(x)))
        else:
            x = self.ln_1(x + self.drop_1(self.mha(x, x, x)))
            x = self.ln_2(x + self.drop_2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """One decoder block: masked self-attn + cross-attn + FFN.

    Causal masking on self-attention via is_causal=True (uses F.scaled_dot_product_attention's
    built-in causal shortcut -- no explicit mask tensor needed).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        p_drop: float = 0.1,
        pre_ln: bool = False,
    ) -> None:
        super().__init__()
        self.pre_ln = pre_ln
        self.mha_self = MultiHeadAttention(d_model, num_heads, p_drop)
        self.mha_cross = MultiHeadAttention(d_model, num_heads, p_drop)
        self.ffn = FeedForward(d_model, d_ff, p_drop)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        self.drop_1 = nn.Dropout(p_drop)
        self.drop_2 = nn.Dropout(p_drop)
        self.drop_3 = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        if self.pre_ln:
            x_n = self.ln_1(x)
            x = x + self.drop_1(self.mha_self(x_n, x_n, x_n, is_causal=True))
            x_n = self.ln_2(x)
            x = x + self.drop_2(self.mha_cross(x_n, enc_out, enc_out))
            x = x + self.drop_3(self.ffn(self.ln_3(x)))
        else:
            x = self.ln_1(x + self.drop_1(self.mha_self(x, x, x, is_causal=True)))
            x = self.ln_2(x + self.drop_2(self.mha_cross(x, enc_out, enc_out)))
            x = self.ln_3(x + self.drop_3(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """Full encoder stack: token IDs -> contextual representations.

    token_ids -> Embedding -> *sqrt(d_model) -> +PE -> Dropout -> EncoderLayer x N
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        p_drop: float = 0.1,
        max_len: int = 5000,
        pre_ln: bool = False,
        shared_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.embed_scale = math.sqrt(d_model)
        self.embedding = shared_embedding if shared_embedding is not None else nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p_drop)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, p_drop, pre_ln) for _ in range(num_layers)]
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.pos_enc(self.embedding(token_ids) * self.embed_scale))
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """Full decoder stack: target IDs + encoder output -> logits.

    target_ids -> Embedding -> *sqrt(d_model) -> +PE -> Dropout
               -> DecoderLayer x N -> output projection -> logits

    Tied output projection (tie_embeddings=True, default): F.linear(x, embedding.weight).
    No TiedLinear class needed -- F.linear reads the weight lazily, so optimizer
    updates to embedding.weight are reflected automatically.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        p_drop: float = 0.1,
        max_len: int = 5000,
        pre_ln: bool = False,
        shared_embedding: nn.Embedding | None = None,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.embed_scale = math.sqrt(d_model)
        self.tie_embeddings = tie_embeddings
        self.embedding = shared_embedding if shared_embedding is not None else nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p_drop)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, p_drop, pre_ln) for _ in range(num_layers)]
        )
        if not tie_embeddings:
            self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, target_ids: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.pos_enc(self.embedding(target_ids) * self.embed_scale))
        for layer in self.layers:
            x = layer(x, enc_out)
        if self.tie_embeddings:
            return F.linear(x, self.embedding.weight)
        return self.output_projection(x)


class Transformer(nn.Module):
    """Encoder-decoder Transformer with three-way shared embedding (paper section 3.4).

    The single nn.Embedding is owned here and passed to both Encoder and Decoder.
    When tie_embeddings=True (default), the decoder output projection also uses the
    same weight matrix via F.linear -- no extra parameter.

    Hyperparameters are stored as plain attributes for persistence (Slice 5).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        p_drop: float = 0.1,
        max_len: int = 5000,
        pre_ln: bool = False,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.p_drop = p_drop
        self.max_len = max_len
        self.pre_ln = pre_ln
        self.tie_embeddings = tie_embeddings

        self.embedding = nn.Embedding(vocab_size, d_model)
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

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(target_ids, self.encoder(source_ids))
