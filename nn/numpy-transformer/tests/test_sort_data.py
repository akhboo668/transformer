"""Sort dataset: shapes, vocabulary, and sortedness."""

import numpy as np

from numpy_transformer.sort_data import (
    DEFAULT_SEQ_LEN,
    EOS_ID,
    PAD_ID,
    SOS_ID,
    VOCAB_SIZE,
    export_preview_csv,
    generate_sort_split,
    load_sort_npz,
    save_sort_npz,
    write_default_datasets,
)


def test_vocab_constants():
    assert VOCAB_SIZE == 13
    assert 0 <= min(PAD_ID, SOS_ID, EOS_ID) <= 12


def test_generate_sort_split_shapes_and_types():
    n = 64
    d = generate_sort_split(n, seq_len=DEFAULT_SEQ_LEN, seed=0)
    L = DEFAULT_SEQ_LEN
    assert d["encoder_input"].shape == (n, L)
    assert d["decoder_input"].shape == (n, L + 1)
    assert d["decoder_target"].shape == (n, L + 1)
    for k in d:
        assert d[k].dtype == np.int64


def test_decoder_inputs_match_overview_example():
    """Hand-check the [3,1,4,1,5] style pipeline."""
    enc = np.array([[3, 1, 4, 1, 5]], dtype=np.int64)
    sorted_row = np.sort(enc, axis=-1)
    dec_in = np.concatenate(
        [np.full((1, 1), SOS_ID, dtype=np.int64), sorted_row],
        axis=1,
    )
    dec_tgt = np.concatenate(
        [sorted_row, np.full((1, 1), EOS_ID, dtype=np.int64)],
        axis=1,
    )
    assert np.array_equal(dec_in[0], [SOS_ID, 1, 1, 3, 4, 5])
    assert np.array_equal(dec_tgt[0], [1, 1, 3, 4, 5, EOS_ID])


def test_random_batch_is_consistent_with_numpy_sort():
    d = generate_sort_split(200, seed=7)
    want = np.sort(d["encoder_input"], axis=-1)
    got = d["decoder_target"][:, :-1]
    assert np.array_equal(got, want)
    assert np.all(d["decoder_target"][:, -1] == EOS_ID)
    assert np.all(d["decoder_input"][:, 0] == SOS_ID)
    assert np.array_equal(d["decoder_input"][:, 1:], want)


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "mini.npz"
    d0 = generate_sort_split(20, seed=1)
    save_sort_npz(path, d0)
    d1 = load_sort_npz(path)
    for k in d0:
        assert np.array_equal(d0[k], d1[k])


def test_write_default_datasets(tmp_path):
    train_p, test_p = write_default_datasets(
        n_train=30,
        n_test=10,
        input_dir=tmp_path,
        seed_train=1,
        seed_test=2,
    )
    assert train_p.exists() and test_p.exists()
    tr = load_sort_npz(train_p)
    assert tr["encoder_input"].shape == (30, DEFAULT_SEQ_LEN)
    prev = tmp_path / "sort_train_preview.csv"
    assert prev.exists()
    text = prev.read_text(encoding="utf-8")
    assert "encoder_0" in text.splitlines()[0]


def test_export_preview_csv(tmp_path):
    p = tmp_path / "d.npz"
    save_sort_npz(p, generate_sort_split(100, seed=0))
    out = tmp_path / "view.csv"
    export_preview_csv(p, out, n_rows=5)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
