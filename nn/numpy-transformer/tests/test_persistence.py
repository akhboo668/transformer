"""
Tests for nn/numpy-transformer/src/numpy_transformer/persistence.py
(Step 11: model persistence -- save / load / Storage backends).

Coverage map (see nn/docs/11_model_persistence.md):

  Round-trip correctness
    test_save_load_roundtrip_tied
    test_save_load_roundtrip_untied
    test_load_preserves_tied_aliasing
    test_metadata_round_trips_exactly

  Format-version contract
    test_saved_file_contains_format_version_1
    test_load_rejects_missing_format_version
    test_load_rejects_future_format_version
    test_load_rejects_unsupported_low_format_version

  Validation strictness (per docs section 7)
    test_load_rejects_missing_metadata_field
    test_load_rejects_missing_param_key
    test_load_rejects_extra_param_key
    test_load_rejects_shape_mismatch
    test_load_rejects_dtype_mismatch

  Forward compatibility
    test_load_ignores_unknown_namespaces

  Storage abstraction
    test_local_storage_writes_atomically_via_tmp
    test_dict_storage_roundtrip

  End-to-end integration
    test_train_save_load_decode_matches_in_memory_model
"""

from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from numpy_transformer.inference import greedy_decode
from numpy_transformer.layers import Transformer
from numpy_transformer.persistence import (
    FORMAT_VERSION,
    LocalStorage,
    load_transformer,
    save_transformer,
)
from numpy_transformer.sort_data import EOS_ID, SOS_ID, VOCAB_SIZE, generate_sort_split
from numpy_transformer.training import train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_model(seed: int = 0, *, tied: bool = True) -> Transformer:
    """A minimal Transformer for fast tests. Dropout is off so forward
    passes are deterministic regardless of train/eval mode."""
    np.random.seed(seed)
    return Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        p_drop=0.0,
        tie_embeddings=tied,
    )


def _fixed_input_pair() -> tuple[np.ndarray, np.ndarray]:
    """A small (B=1) (encoder_input, decoder_input) pair we use to compare
    forward outputs before and after a save/load round-trip."""
    src = np.array([[3, 1, 4, 1, 5]], dtype=np.int64)
    tgt = np.array([[SOS_ID, 1, 1, 3, 4, 5]], dtype=np.int64)
    return src, tgt


def _rewrite_npz(path: Path, mutate) -> None:
    """Round-trip an .npz through a `dict[str, np.ndarray]` so a test can
    edit it. ``mutate`` receives the dict and modifies it in place."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_tied(tmp_path: Path):
    """Forward pass on a fixed input must be bit-exact after save/load
    when tie_embeddings=True (the default)."""
    original = _make_tiny_model(seed=42, tied=True)
    src, tgt = _fixed_input_pair()
    expected = original.forward(src, tgt)

    path = tmp_path / "model.npz"
    save_transformer(original, path)
    loaded = load_transformer(path)

    actual = loaded.forward(src, tgt)
    np.testing.assert_array_equal(expected, actual)


def test_save_load_roundtrip_untied(tmp_path: Path):
    """Forward pass must round-trip with tie_embeddings=False, where the
    decoder's output projection is its own Linear layer."""
    original = _make_tiny_model(seed=43, tied=False)
    src, tgt = _fixed_input_pair()
    expected = original.forward(src, tgt)

    path = tmp_path / "model.npz"
    save_transformer(original, path)
    loaded = load_transformer(path)

    actual = loaded.forward(src, tgt)
    np.testing.assert_array_equal(expected, actual)


def test_load_preserves_tied_aliasing(tmp_path: Path):
    """The shared-embedding aliasing must survive load: the top-level
    Transformer, its Encoder, and its Decoder all hold the same array
    object. This is the §6 subtlety from the doc."""
    original = _make_tiny_model(seed=44, tied=True)
    path = tmp_path / "model.npz"
    save_transformer(original, path)
    loaded = load_transformer(path)

    assert loaded.encoder.embedding is loaded.embedding, (
        "Encoder.embedding must be the same object as Transformer.embedding "
        "when tie_embeddings=True; aliasing was lost during load."
    )
    assert loaded.decoder.embedding is loaded.embedding, (
        "Decoder.embedding must be the same object as Transformer.embedding "
        "when tie_embeddings=True; aliasing was lost during load."
    )

    # And the underlying weight array must be the same object too,
    # not just an equal-valued copy.
    assert loaded.encoder.embedding.weight is loaded.embedding.weight
    assert loaded.decoder.embedding.weight is loaded.embedding.weight


def test_metadata_round_trips_exactly(tmp_path: Path):
    """Every architecture hyperparameter must round-trip with the right
    type and value."""
    original = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_encoder_layers=2,
        num_decoder_layers=3,
        p_drop=0.25,
        max_len=128,
        pre_ln=True,
        tie_embeddings=False,
    )
    path = tmp_path / "model.npz"
    save_transformer(original, path)
    loaded = load_transformer(path)

    assert loaded.vocab_size == VOCAB_SIZE
    assert loaded.d_model == 16
    assert loaded.num_heads == 4
    assert loaded.d_ff == 32
    assert loaded.num_encoder_layers == 2
    assert loaded.num_decoder_layers == 3
    assert loaded.p_drop == 0.25
    assert loaded.max_len == 128
    assert loaded.pre_ln is True
    assert loaded.tie_embeddings is False


# ---------------------------------------------------------------------------
# Format-version contract
# ---------------------------------------------------------------------------


def test_saved_file_contains_format_version_1(tmp_path: Path):
    """The save file must contain `meta.format_version = 1` so future
    readers can branch deterministically."""
    model = _make_tiny_model(seed=1)
    path = tmp_path / "model.npz"
    save_transformer(model, path)

    with np.load(path, allow_pickle=False) as z:
        assert "meta.format_version" in z.files
        assert int(z["meta.format_version"]) == FORMAT_VERSION == 1


def test_load_rejects_missing_format_version(tmp_path: Path):
    """A file with no `meta.format_version` is not a transformer save."""
    model = _make_tiny_model(seed=2)
    path = tmp_path / "model.npz"
    save_transformer(model, path)
    _rewrite_npz(path, lambda a: a.pop("meta.format_version"))

    try:
        load_transformer(path)
    except ValueError as e:
        assert "format_version" in str(e)
    else:
        raise AssertionError("expected ValueError for missing format_version")


def test_load_rejects_future_format_version(tmp_path: Path):
    """A file from a future, unsupported format version must fail loudly,
    not silently misload."""
    model = _make_tiny_model(seed=3)
    path = tmp_path / "model.npz"
    save_transformer(model, path)

    def bump(a):
        a["meta.format_version"] = np.asarray(FORMAT_VERSION + 1, dtype=np.int64)

    _rewrite_npz(path, bump)

    try:
        load_transformer(path)
    except ValueError as e:
        msg = str(e)
        assert "format_version" in msg
        assert str(FORMAT_VERSION + 1) in msg
    else:
        raise AssertionError("expected ValueError for future format_version")


def test_load_rejects_unsupported_low_format_version(tmp_path: Path):
    """Versions <1 must also be rejected (corrupt or pre-release)."""
    model = _make_tiny_model(seed=4)
    path = tmp_path / "model.npz"
    save_transformer(model, path)

    def stomp(a):
        a["meta.format_version"] = np.asarray(0, dtype=np.int64)

    _rewrite_npz(path, stomp)

    try:
        load_transformer(path)
    except ValueError as e:
        assert "format_version" in str(e)
    else:
        raise AssertionError("expected ValueError for format_version=0")


# ---------------------------------------------------------------------------
# Validation strictness (docs §7)
# ---------------------------------------------------------------------------


def test_load_rejects_missing_metadata_field(tmp_path: Path):
    """Missing required meta.* field -> clear error naming the field."""
    model = _make_tiny_model(seed=5)
    path = tmp_path / "model.npz"
    save_transformer(model, path)
    _rewrite_npz(path, lambda a: a.pop("meta.d_model"))

    try:
        load_transformer(path)
    except ValueError as e:
        assert "d_model" in str(e)
    else:
        raise AssertionError("expected ValueError for missing meta.d_model")


def test_load_rejects_missing_param_key(tmp_path: Path):
    """A param the constructed Transformer expects must be present
    in the file. The error message must name the missing key."""
    model = _make_tiny_model(seed=6, tied=True)
    path = tmp_path / "model.npz"
    save_transformer(model, path)
    _rewrite_npz(path, lambda a: a.pop("params.embedding.weight"))

    try:
        load_transformer(path)
    except ValueError as e:
        msg = str(e)
        assert "missing" in msg
        assert "params.embedding.weight" in msg
    else:
        raise AssertionError("expected ValueError for missing param key")


def test_load_rejects_extra_param_key(tmp_path: Path):
    """A `params.*` key the constructed Transformer doesn't expect is
    a sign of architecture drift; fail loudly."""
    model = _make_tiny_model(seed=7, tied=True)
    path = tmp_path / "model.npz"
    save_transformer(model, path)

    def add_extra(a):
        a["params.something_unknown"] = np.zeros((3, 3), dtype=np.float64)

    _rewrite_npz(path, add_extra)

    try:
        load_transformer(path)
    except ValueError as e:
        msg = str(e)
        assert "unexpected" in msg
        assert "params.something_unknown" in msg
    else:
        raise AssertionError("expected ValueError for extra param key")


def test_load_rejects_shape_mismatch(tmp_path: Path):
    """A param with the right name but wrong shape must raise with the
    offending name in the message."""
    model = _make_tiny_model(seed=8, tied=True)
    path = tmp_path / "model.npz"
    save_transformer(model, path)

    def shrink(a):
        a["params.embedding.weight"] = np.zeros((5, 5), dtype=np.float64)

    _rewrite_npz(path, shrink)

    try:
        load_transformer(path)
    except ValueError as e:
        msg = str(e)
        assert "shape" in msg
        assert "embedding.weight" in msg
    else:
        raise AssertionError("expected ValueError for shape mismatch")


def test_load_rejects_dtype_mismatch(tmp_path: Path):
    """A param saved with a different dtype (e.g. float32 vs float64)
    must raise. Silent upcast on `[...]` assignment is exactly the
    failure mode this check exists to prevent."""
    model = _make_tiny_model(seed=9, tied=True)
    path = tmp_path / "model.npz"
    save_transformer(model, path)

    def downcast(a):
        a["params.embedding.weight"] = a["params.embedding.weight"].astype(np.float32)

    _rewrite_npz(path, downcast)

    try:
        load_transformer(path)
    except ValueError as e:
        msg = str(e)
        assert "dtype" in msg
        assert "embedding.weight" in msg
    else:
        raise AssertionError("expected ValueError for dtype mismatch")


# ---------------------------------------------------------------------------
# Forward compatibility (the §11 forward-compat hinge)
# ---------------------------------------------------------------------------


def test_load_ignores_unknown_namespaces(tmp_path: Path):
    """A v1 save file containing future namespaces (`optimizer.*`,
    `rng.*`, etc.) must still load cleanly with today's loader. This
    is what makes the "additive extensions inside format_version=1"
    contract work."""
    model = _make_tiny_model(seed=10, tied=True)
    src, tgt = _fixed_input_pair()
    expected = model.forward(src, tgt)

    path = tmp_path / "model.npz"
    save_transformer(model, path)

    def add_future_namespaces(a):
        a["optimizer.t"] = np.asarray(123, dtype=np.int64)
        a["optimizer.params.embedding.weight.m"] = np.zeros_like(
            a["params.embedding.weight"]
        )
        a["optimizer.params.embedding.weight.v"] = np.zeros_like(
            a["params.embedding.weight"]
        )
        a["rng.bit_generator"] = np.frombuffer(b"PCG64\x00\x00\x00", dtype=np.uint8)
        a["rng.state"] = np.zeros(4, dtype=np.uint64)
        a["someone_elses_stuff.foo"] = np.array([1.0, 2.0, 3.0])

    _rewrite_npz(path, add_future_namespaces)

    loaded = load_transformer(path)
    actual = loaded.forward(src, tgt)
    np.testing.assert_array_equal(expected, actual)


# ---------------------------------------------------------------------------
# Storage abstraction
# ---------------------------------------------------------------------------


def test_local_storage_writes_atomically_via_tmp(tmp_path: Path):
    """LocalStorage must write to <key>.tmp first and rename, leaving
    no .tmp behind on success. This is the contract that gives us the
    'no half-written file' guarantee."""
    model = _make_tiny_model(seed=11)
    path = tmp_path / "model.npz"
    storage = LocalStorage()

    save_transformer(model, path, storage=storage)

    assert path.exists()
    leftover = path.with_name(path.name + ".tmp")
    assert not leftover.exists(), (
        f"LocalStorage left a .tmp file behind after a successful save: "
        f"{leftover}"
    )


class _DictStorage:
    """An in-memory `Storage` implementation for tests. Demonstrates
    that the Protocol is open for extension without subclassing or
    importing fsspec. Roughly the example from docs §11.5."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    @contextmanager
    def open_write(self, key: str) -> Iterator[BinaryIO]:
        buf = BytesIO()
        try:
            yield buf
        except BaseException:
            raise
        else:
            self._blobs[key] = buf.getvalue()

    @contextmanager
    def open_read(self, key: str) -> Iterator[BinaryIO]:
        if key not in self._blobs:
            raise FileNotFoundError(f"DictStorage: no such key {key!r}")
        yield BytesIO(self._blobs[key])


def test_dict_storage_roundtrip():
    """A user-defined Storage backend implementing only `open_write`
    and `open_read` (no filesystem, no fsspec) must round-trip a model
    correctly. This exercises the full Storage Protocol contract."""
    original = _make_tiny_model(seed=12, tied=True)
    src, tgt = _fixed_input_pair()
    expected = original.forward(src, tgt)

    storage = _DictStorage()

    save_transformer(original, "memory://my-model", storage=storage)
    loaded = load_transformer("memory://my-model", storage=storage)

    actual = loaded.forward(src, tgt)
    np.testing.assert_array_equal(expected, actual)


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


def test_train_save_load_decode_matches_in_memory_model(tmp_path: Path):
    """Headline integration test for Step 11.

    Train a small model briefly, save it, load into a fresh process
    equivalent, and verify that greedy decoding produces token-for-token
    identical outputs on a held-out test sequence. This exercises
    every component of the persistence layer in conjunction with the
    full forward pass and the autoregressive decoder.
    """
    np.random.seed(311)
    train_data = generate_sort_split(n=512, seq_len=5, seed=311)
    test_data = generate_sort_split(n=4, seq_len=5, seed=312)

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        p_drop=0.0,
        tie_embeddings=True,
    )
    train(
        model, train_data, n_steps=300, batch_size=16, eps=0.1,
        warmup_steps=50, log_every=0, seed=311,
    )

    path = tmp_path / "trained.npz"
    save_transformer(model, path)
    loaded = load_transformer(path)

    for src in test_data["encoder_input"]:
        in_memory_pred = greedy_decode(
            model, src, sos_id=SOS_ID, eos_id=EOS_ID, max_len=8
        )
        loaded_pred = greedy_decode(
            loaded, src, sos_id=SOS_ID, eos_id=EOS_ID, max_len=8
        )
        np.testing.assert_array_equal(in_memory_pred, loaded_pred)
