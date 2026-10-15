"""
Sort-task training / test tensors for the toy Transformer.

Specification matches nn/docs/00_overview.md:
  - Encoder input: 5 random digits in 0..9
  - Decoder input:  [SOS] + sorted(digits)   (length 6)
  - Decoder target: sorted(digits) + EOS   (length 6)

Vocabulary ids: digit value == id for 0..9; PAD=10, SOS=11, EOS=12 (vocab size 13).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PAD_ID = 10
SOS_ID = 11
EOS_ID = 12
VOCAB_SIZE = 13
DEFAULT_SEQ_LEN = 5


def default_input_dir() -> Path:
    """Project `train_input/` folder next to `src/` (numpy-transformer root).

    Holds the dataset fixtures (`sort_train.npz`, `sort_test.npz`, the
    `sort_*_preview.csv` files, and the dataset README). Distinct from
    `train_output/` (runtime logs) and `train_model/` (persisted models)
    so each can have its own git-tracking and lifecycle policy.
    """
    return Path(__file__).resolve().parent.parent.parent / "train_input"


def generate_sort_split(
    n: int,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Build `n` independent sort examples.

    Returns
    -------
    encoder_input : (n, seq_len) int64
        Raw unsorted digit ids 0..9.
    decoder_input : (n, seq_len + 1) int64
        Teacher-forcing input: SOS then ground-truth sorted digits.
    decoder_target : (n, seq_len + 1) int64
        Predict sorted digits then EOS at the last step.
    """
    rng = np.random.default_rng(seed)
    enc = rng.integers(0, 10, size=(n, seq_len), dtype=np.int64)
    sorted_digits = np.sort(enc, axis=-1)

    sos_col = np.full((n, 1), SOS_ID, dtype=np.int64)
    eos_col = np.full((n, 1), EOS_ID, dtype=np.int64)

    dec_in = np.concatenate([sos_col, sorted_digits], axis=1)
    dec_tgt = np.concatenate([sorted_digits, eos_col], axis=1)

    return {
        "encoder_input": enc,
        "decoder_input": dec_in,
        "decoder_target": dec_tgt,
    }


def save_sort_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_sort_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def export_preview_csv(
    npz_path: str | Path,
    csv_path: str | Path,
    n_rows: int = 30,
) -> Path:
    """
    Write the first ``n_rows`` examples from an ``.npz`` split to a UTF-8 CSV you can open in
    an editor or spreadsheet. ``.npz`` itself is binary (NumPy); this is for human inspection.
    """
    npz_path = Path(npz_path)
    csv_path = Path(csv_path)
    d = load_sort_npz(npz_path)
    enc = d["encoder_input"][:n_rows]
    di = d["decoder_input"][:n_rows]
    dt = d["decoder_target"][:n_rows]
    seq_len = enc.shape[1]
    dec_len = di.shape[1]

    enc_cols = [f"encoder_{j}" for j in range(seq_len)]
    di_cols = [f"decoder_in_{j}" for j in range(dec_len)]
    dt_cols = [f"decoder_tgt_{j}" for j in range(dec_len)]
    header = ["row"] + enc_cols + di_cols + dt_cols

    lines = [",".join(header)]
    for i in range(enc.shape[0]):
        row = [str(i)] + [str(int(x)) for x in enc[i]]
        row += [str(int(x)) for x in di[i]]
        row += [str(int(x)) for x in dt[i]]
        lines.append(",".join(row))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def write_default_datasets(
    n_train: int = 10_000,
    n_test: int = 1_000,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed_train: int = 42,
    seed_test: int = 4242,
    input_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Write `sort_train.npz` and `sort_test.npz` under `train_input/`.

    Train and test use different RNG seeds so the splits are independent draws.
    """
    root = input_dir if input_dir is not None else default_input_dir()
    train_path = root / "sort_train.npz"
    test_path = root / "sort_test.npz"
    save_sort_npz(train_path, generate_sort_split(n_train, seq_len, seed_train))
    save_sort_npz(test_path, generate_sort_split(n_test, seq_len, seed_test))
    export_preview_csv(train_path, root / "sort_train_preview.csv", n_rows=30)
    export_preview_csv(test_path, root / "sort_test_preview.csv", n_rows=30)
    return train_path, test_path
