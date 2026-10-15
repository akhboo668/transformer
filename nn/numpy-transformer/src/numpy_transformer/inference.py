"""
Step 9 (code-only): autoregressive inference for the trained Transformer.

Two public functions:

    greedy_decode(transformer, source_ids, sos_id, eos_id, max_len)
        Generate one sequence at a time by argmax-decoding tokens until
        EOS or max_len is reached. The fundamental gap between training
        (teacher-forced, parallel) and inference (autoregressive,
        sequential): we feed back the model's own predictions instead of
        ground truth.

    evaluate_sort_accuracy(transformer, encoder_inputs, decoder_targets,
                           sos_id, eos_id, max_len)
        Run greedy_decode on every example in a test split and compute
        the exact-match rate. The "did the model actually learn?" metric.

Performance note: this implementation does not use KV caching, so each
generation step re-runs the full decoder forward over the growing
sequence. For the toy sort task with sequences of length 6 this is
microseconds. For long sequences in real systems, KV caching would
reduce O(n^2) per-step work to O(n).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def greedy_decode(
    transformer,
    source_ids: np.ndarray,
    sos_id: int,
    eos_id: int,
    max_len: int = 32,
) -> np.ndarray:
    """
    Greedy autoregressive decoding for a single source sequence.

    Parameters
    ----------
    transformer : Transformer
        A trained model exposing `encoder.forward(source_ids)` and
        `decoder.forward(target_ids, enc_out)`. Will be put in eval mode
        so dropout is disabled and the output is deterministic.
    source_ids : (n,) or (1, n) int array
        The encoder input. Will be reshaped to (1, n) if 1D.
    sos_id : int
        Start-of-sequence token id used to seed the decoder input.
    eos_id : int
        End-of-sequence token id; generation stops when emitted.
    max_len : int
        Hard cap on output length (excluding the seed SOS).

    Returns
    -------
    output_ids : (out_len,) int array
        The generated tokens in emission order (NOT including the seed
        SOS). If EOS was emitted, it is included as the last element.

    Notes
    -----
    The decoder runs in eval mode (dropout off) so generation is
    deterministic for a given trained model and input.
    """
    if source_ids.ndim == 1:
        source_ids = source_ids[None, :]
    if source_ids.shape[0] != 1:
        raise ValueError(
            "greedy_decode is single-sequence; got "
            f"source_ids batch dim = {source_ids.shape[0]}"
        )

    transformer.eval()

    # Encode once; the encoder output is reused at every decoder step.
    enc_out = transformer.encoder.forward(source_ids)

    # Seed decoder input with [SOS]; grows by one token per loop iteration.
    target_ids = np.array([[sos_id]], dtype=source_ids.dtype)
    generated = []

    for _ in range(max_len):
        logits = transformer.decoder.forward(target_ids, enc_out)
        # logits shape: (1, t+1, V). The last position predicts the
        # next token; argmax over the vocab axis.
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)
        if next_token == eos_id:
            break
        # Append predicted token to decoder input for the next step.
        target_ids = np.concatenate(
            [target_ids, np.array([[next_token]], dtype=target_ids.dtype)],
            axis=1,
        )

    return np.array(generated, dtype=np.int64)


def evaluate_sort_accuracy(
    transformer,
    encoder_inputs: np.ndarray,
    decoder_targets: np.ndarray,
    sos_id: int,
    eos_id: int,
    max_len: int = 32,
    return_examples: int = 0,
) -> Tuple[float, list]:
    """
    Decode every example with greedy_decode and report exact-match
    accuracy against `decoder_targets`.

    Parameters
    ----------
    transformer : Transformer
        Trained model.
    encoder_inputs : (N, n_src) int array
    decoder_targets : (N, n_tgt) int array
        Ground-truth sorted sequences (including the trailing EOS).
    sos_id, eos_id, max_len :
        Forwarded to `greedy_decode`.
    return_examples : int
        If > 0, also return the first N (source, predicted, expected)
        triples for inspection.

    Returns
    -------
    accuracy : float
        Fraction of test sequences where the predicted sequence equals
        the expected one (element-wise, including EOS).
    examples : list of (source, predicted, expected) tuples
        Empty unless `return_examples > 0`.
    """
    if encoder_inputs.shape[0] != decoder_targets.shape[0]:
        raise ValueError(
            "encoder_inputs and decoder_targets must agree on first axis"
        )
    n = encoder_inputs.shape[0]

    correct = 0
    examples: list = []

    for i in range(n):
        src = encoder_inputs[i]
        expected = decoder_targets[i]
        predicted = greedy_decode(
            transformer, src, sos_id=sos_id, eos_id=eos_id, max_len=max_len
        )
        if (
            predicted.shape == expected.shape
            and np.array_equal(predicted, expected)
        ):
            correct += 1
        if i < return_examples:
            examples.append((src.copy(), predicted, expected.copy()))

    return correct / n, examples
