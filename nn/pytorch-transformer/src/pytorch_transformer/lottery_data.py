"""
Powerball dataset for Step 13 experiments.

CSV format  (train_input/powerball_draws.csv)
---------------------------------------------
date,w1,w2,w3,w4,w5,pb
2024-01-06,5,12,25,33,48,15
...

  date  : any date string (informational; skipped during load)
  w1-w5 : white ball values 1-69, expected sorted ascending
  pb    : Powerball value 1-26

Lines beginning with '#' and blank lines are skipped.

Vocabulary (vocab_size=99)
--------------------------
Tokens  1-69  : white ball values   (ball 12 → token 12)
Tokens 70-95  : Powerball values    (PB 1 → token 70, PB 26 → token 95)
Token  96     : PAD
Token  97     : SOS
Token  98     : EOS
Token   0     : unused (reserved)

Input/target structure per example
------------------------------------
Context window: the previous CONTEXT_DRAWS draws (default 10 = 60 tokens).
One example predicts draw t given draws t-CONTEXT_DRAWS … t-1.

  encoder_input  : (CONTEXT_DRAWS * DRAW_LEN,)   int64
  decoder_input  : (DRAW_LEN + 1,)               int64  [SOS, w1, w2, w3, w4, w5, pb]
  decoder_target : (DRAW_LEN + 1,)               int64  [w1, w2, w3, w4, w5, pb, EOS]

Experiments (Step 13)
---------------------
Experiment A (prediction)  : train on draws 0 … N-n_eval-1,
                              evaluate on held-out draws N-n_eval … N-1.
                              Expected: loss near log(69) per white ball, log(26) per PB.
                              Exact-match: 0 / n_eval  (lottery is i.i.d.)

Experiment B (memorization): train on ALL draws including the eval window.
                              Expected: loss → 0,  exact-match → n_eval / n_eval.

The contrast demonstrates: memorisation ≠ prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------
WHITE_MIN, WHITE_MAX = 1, 69
PB_MIN,    PB_MAX    = 1, 26
PB_OFFSET            = 69          # Powerball k → token k + PB_OFFSET

PAD_ID     = 96
SOS_ID     = 97
EOS_ID     = 98
VOCAB_SIZE = 99

DRAW_LEN      = 6    # 5 white balls + 1 Powerball
CONTEXT_DRAWS = 10   # number of past draws used as encoder input


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_draws_from_csv(path: str | Path) -> np.ndarray:
    """Load Powerball draws from a CSV file.

    Expected columns: date, w1, w2, w3, w4, w5, pb
    Lines beginning with '#' and blank lines are ignored.

    Returns
    -------
    (N, DRAW_LEN) int64 array in token space:
      columns 0-4 : white ball tokens 1-69 (sorted ascending per row)
      column  5   : Powerball token 70-95
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Download real Powerball data and save it as CSV with columns:\n"
            "  date,w1,w2,w3,w4,w5,pb\n"
            "See train_input/powerball_draws.csv for format details."
        )

    rows: list[list[int]] = []
    with path.open() as f:
        header_seen = False
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not header_seen:
                # First non-comment line is the header
                header_seen = True
                cols = [c.strip().lower() for c in line.split(",")]
                _validate_header(cols, path, lineno)
                continue
            parts = line.split(",")
            if len(parts) < 7:
                raise ValueError(
                    f"{path}:{lineno}: expected 7 columns (date,w1,w2,w3,w4,w5,pb), "
                    f"got {len(parts)}: {line!r}"
                )
            try:
                w = sorted(int(parts[i]) for i in range(1, 6))
                pb = int(parts[6])
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: non-integer value — {e}") from e
            _validate_draw(w, pb, path, lineno)
            rows.append(w + [pb + PB_OFFSET])

    if not rows:
        raise ValueError(f"{path}: no data rows found (only header / comments).")

    return np.array(rows, dtype=np.int64)   # (N, 6)


def _validate_header(cols: list[str], path: Path, lineno: int) -> None:
    expected = ["date", "w1", "w2", "w3", "w4", "w5", "pb"]
    if cols[:7] != expected:
        raise ValueError(
            f"{path}:{lineno}: unexpected header {cols[:7]}; "
            f"expected {expected}"
        )


def _validate_draw(
    white: list[int], pb: int, path: Path, lineno: int
) -> None:
    for i, v in enumerate(white, start=1):
        if not (WHITE_MIN <= v <= WHITE_MAX):
            raise ValueError(
                f"{path}:{lineno}: white ball w{i}={v} out of range "
                f"[{WHITE_MIN}, {WHITE_MAX}]"
            )
    if not (PB_MIN <= pb <= PB_MAX):
        raise ValueError(
            f"{path}:{lineno}: Powerball pb={pb} out of range "
            f"[{PB_MIN}, {PB_MAX}]"
        )


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _make_examples(
    draws: np.ndarray,
    context: int = CONTEXT_DRAWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a (N, DRAW_LEN) draw array into encoder/decoder arrays.

    Returns three arrays each with n_examples = N - context rows:
      encoder_input  : (n_examples, context * DRAW_LEN)
      decoder_input  : (n_examples, DRAW_LEN + 1)   SOS + target draw
      decoder_target : (n_examples, DRAW_LEN + 1)   target draw + EOS
    """
    n = len(draws)
    n_examples = n - context
    if n_examples <= 0:
        raise ValueError(
            f"Need at least context+1={context + 1} draws, got {n}. "
            "Add more rows to your CSV file."
        )

    enc = np.stack([
        draws[i: i + context].reshape(-1)
        for i in range(n_examples)
    ]).astype(np.int64)

    targets = draws[context:]
    sos = np.full((n_examples, 1), SOS_ID, dtype=np.int64)
    eos = np.full((n_examples, 1), EOS_ID, dtype=np.int64)
    dec_in  = np.concatenate([sos, targets], axis=1)
    dec_tgt = np.concatenate([targets, eos], axis=1)

    return enc, dec_in, dec_tgt


def _to_tensor_dataset(enc, dec_in, dec_tgt) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(enc),
        torch.from_numpy(dec_in),
        torch.from_numpy(dec_tgt),
    )


def make_experiment_a(
    draws: np.ndarray,
    n_eval: int = 10,
    context: int = CONTEXT_DRAWS,
) -> tuple[TensorDataset, TensorDataset]:
    """Experiment A: prediction on held-out draws.

    Train set: examples from draws[0 … N-n_eval-1].
    Eval  set: examples from draws[N-n_eval-context … N-1]
               (includes the context window before the eval draws).

    Returns (train_dataset, eval_dataset).
    """
    n = len(draws)
    if n < context + n_eval + 1:
        raise ValueError(
            f"Need at least {context + n_eval + 1} draws for "
            f"context={context}, n_eval={n_eval}; got {n}."
        )
    train_draws = draws[: n - n_eval]
    eval_draws  = draws[n - n_eval - context:]

    return (
        _to_tensor_dataset(*_make_examples(train_draws, context)),
        _to_tensor_dataset(*_make_examples(eval_draws, context)),
    )


def make_experiment_b(
    draws: np.ndarray,
    n_eval: int = 10,
    context: int = CONTEXT_DRAWS,
) -> tuple[TensorDataset, TensorDataset]:
    """Experiment B: memorization — eval draws are included in training.

    Train set: examples from ALL draws.
    Eval  set: same n_eval examples as Experiment A.

    Returns (train_dataset, eval_dataset).
    """
    n = len(draws)
    if n < context + n_eval + 1:
        raise ValueError(
            f"Need at least {context + n_eval + 1} draws for "
            f"context={context}, n_eval={n_eval}; got {n}."
        )
    eval_draws = draws[n - n_eval - context:]
    return (
        _to_tensor_dataset(*_make_examples(draws, context)),
        _to_tensor_dataset(*_make_examples(eval_draws, context)),
    )


def make_dataloader(
    dataset: TensorDataset,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Synthetic draws — internal use only (tests, CI without real data)
# ---------------------------------------------------------------------------

def _generate_draws(n: int, seed: int | None = None) -> np.ndarray:
    """Return (n, DRAW_LEN) int64 array of synthetic draws (already in token space).
    Not for real experiments — use load_draws_from_csv() instead.
    """
    rng = np.random.default_rng(seed)
    white = np.array([
        rng.choice(WHITE_MAX, size=5, replace=False) + WHITE_MIN
        for _ in range(n)
    ], dtype=np.int64)
    white.sort(axis=1)
    pb = rng.integers(PB_MIN, PB_MAX + 1, size=(n, 1), dtype=np.int64) + PB_OFFSET
    return np.concatenate([white, pb], axis=1)
