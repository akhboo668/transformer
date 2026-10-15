"""
Step 12: Autoregressive greedy inference.

greedy_decode    -- generate one sequence, token by token, until EOS or max_len.
evaluate_accuracy -- run greedy_decode over a dataset and report exact-match rate.

No KV caching: each step re-runs the full decoder over the growing sequence.
Fine for short sequences (sort task: 6 tokens; lottery: 6 tokens).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformer.layers import Transformer


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    source_ids: torch.Tensor,
    sos_id: int,
    eos_id: int,
    max_len: int = 32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Greedy decode a single source sequence.

    Parameters
    ----------
    model:
        Trained Transformer. Put in eval mode automatically.
    source_ids:
        1D (T,) or 2D (1, T) integer tensor. Batch size must be 1.
    sos_id, eos_id:
        Special token ids for seeding / stopping.
    max_len:
        Hard cap on output length (not counting the SOS seed).

    Returns
    -------
    1D int64 tensor of generated tokens (not including SOS; EOS included
    if emitted before max_len).
    """
    device = torch.device(device)
    model.eval()

    if source_ids.dim() == 1:
        source_ids = source_ids.unsqueeze(0)        # (1, T)
    source_ids = source_ids.to(device)

    enc_out = model.encoder(source_ids)             # encode once, reuse every step

    generated: list[int] = []
    dec_input = torch.tensor([[sos_id]], dtype=torch.long, device=device)

    for _ in range(max_len):
        logits   = model.decoder(dec_input, enc_out)        # (1, t, V)
        next_tok = int(logits[0, -1].argmax())
        generated.append(next_tok)
        if next_tok == eos_id:
            break
        dec_input = torch.cat(
            [dec_input, torch.tensor([[next_tok]], dtype=torch.long, device=device)],
            dim=1,
        )

    return torch.tensor(generated, dtype=torch.long)


def evaluate_accuracy(
    model: Transformer,
    encoder_inputs: torch.Tensor,
    decoder_targets: torch.Tensor,
    sos_id: int,
    eos_id: int,
    max_len: int = 32,
    device: str | torch.device = "cpu",
) -> float:
    """Exact-match accuracy: fraction of sequences decoded identically to target.

    Parameters
    ----------
    encoder_inputs:   (N, src_len) int64
    decoder_targets:  (N, tgt_len) int64  -- ground truth including EOS
    """
    n = encoder_inputs.size(0)
    correct = 0
    for i in range(n):
        predicted = greedy_decode(
            model, encoder_inputs[i], sos_id, eos_id, max_len, device
        )
        expected = decoder_targets[i]
        if predicted.shape == expected.shape and (predicted == expected).all():
            correct += 1
    return correct / n
