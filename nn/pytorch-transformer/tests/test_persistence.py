"""Tests for save_transformer / load_transformer."""

import io
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator, BinaryIO

import pytest
import torch

from pytorch_transformer.layers import Transformer
from pytorch_transformer.persistence import (
    FORMAT_VERSION,
    LocalStorage,
    load_transformer,
    save_transformer,
)


# ---------------------------------------------------------------------------
# In-memory storage for tests (no disk I/O)
# ---------------------------------------------------------------------------

class _MemStorage:
    """In-memory storage backend for tests."""
    def __init__(self):
        self._store: dict[str, bytes] = {}

    @contextmanager
    def open_write(self, key: str) -> Iterator[BinaryIO]:
        buf = BytesIO()
        yield buf
        self._store[key] = buf.getvalue()

    @contextmanager
    def open_read(self, key: str) -> Iterator[BinaryIO]:
        yield BytesIO(self._store[key])


def _save_load(model: Transformer, storage=None) -> Transformer:
    storage = storage or _MemStorage()
    save_transformer(model, "test.pt", storage=storage)
    return load_transformer("test.pt", storage=storage)


def _small(tie=True, pre_ln=False) -> Transformer:
    return Transformer(13, d_model=16, num_heads=2, d_ff=32,
                       num_encoder_layers=1, num_decoder_layers=1,
                       tie_embeddings=tie, pre_ln=pre_ln)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

def test_round_trip_tied():
    model   = _small(tie=True)
    loaded  = _save_load(model)
    for (k, v), (_, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.allclose(v.float(), v2.float()), f"Mismatch at {k}"


def test_round_trip_untied():
    model  = _small(tie=False)
    loaded = _save_load(model)
    for (k, v), (_, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.allclose(v.float(), v2.float()), f"Mismatch at {k}"


def test_round_trip_pre_ln():
    model  = _small(pre_ln=True)
    loaded = _save_load(model)
    assert loaded.pre_ln is True


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------

def test_metadata_round_trip():
    model  = _small()
    loaded = _save_load(model)
    for attr in ("vocab_size", "d_model", "num_heads", "d_ff",
                 "num_encoder_layers", "num_decoder_layers",
                 "p_drop", "max_len", "pre_ln", "tie_embeddings"):
        assert getattr(loaded, attr) == getattr(model, attr), f"Mismatch: {attr}"


# ---------------------------------------------------------------------------
# Tied-embedding aliasing preserved after load
# ---------------------------------------------------------------------------

def test_tied_aliasing_preserved_after_load():
    """After loading, encoder.embedding and decoder.embedding must still
    be the same Python object as transformer.embedding."""
    model  = _small(tie=True)
    loaded = _save_load(model)
    assert loaded.encoder.embedding is loaded.embedding
    assert loaded.decoder.embedding is loaded.embedding
    assert loaded.encoder.embedding.weight.data_ptr() == loaded.embedding.weight.data_ptr()


def test_tied_output_projection_uses_loaded_weight():
    """Decoder output uses the loaded embedding weight (F.linear lazy read)."""
    import torch.nn.functional as F
    model  = _small(tie=True)
    loaded = _save_load(model)
    loaded.eval()
    src = torch.randint(0, 13, (1, 5))
    tgt = torch.randint(0, 13, (1, 5))
    with torch.no_grad():
        logits = loaded(src, tgt)
    assert logits.shape == (1, 5, 13)


# ---------------------------------------------------------------------------
# Format-version validation
# ---------------------------------------------------------------------------

def test_format_version_missing_raises():
    st = _MemStorage()
    save_transformer(_small(), "m.pt", storage=st)
    bundle = torch.load(BytesIO(st._store["m.pt"]), weights_only=False)
    del bundle["format_version"]
    buf = BytesIO(); torch.save(bundle, buf); st._store["m.pt"] = buf.getvalue()
    with pytest.raises(ValueError, match="format_version"):
        load_transformer("m.pt", storage=st)


def test_format_version_too_high_raises():
    st = _MemStorage()
    save_transformer(_small(), "m.pt", storage=st)
    bundle = torch.load(BytesIO(st._store["m.pt"]), weights_only=False)
    bundle["format_version"] = FORMAT_VERSION + 99
    buf = BytesIO(); torch.save(bundle, buf); st._store["m.pt"] = buf.getvalue()
    with pytest.raises(ValueError, match="format_version"):
        load_transformer("m.pt", storage=st)


def test_format_version_zero_raises():
    st = _MemStorage()
    save_transformer(_small(), "m.pt", storage=st)
    bundle = torch.load(BytesIO(st._store["m.pt"]), weights_only=False)
    bundle["format_version"] = 0
    buf = BytesIO(); torch.save(bundle, buf); st._store["m.pt"] = buf.getvalue()
    with pytest.raises(ValueError, match="format_version"):
        load_transformer("m.pt", storage=st)


# ---------------------------------------------------------------------------
# Missing-meta validation
# ---------------------------------------------------------------------------

def test_missing_meta_field_raises():
    st = _MemStorage()
    save_transformer(_small(), "m.pt", storage=st)
    bundle = torch.load(BytesIO(st._store["m.pt"]), weights_only=False)
    del bundle["meta"]["d_model"]
    buf = BytesIO(); torch.save(bundle, buf); st._store["m.pt"] = buf.getvalue()
    with pytest.raises(ValueError, match="d_model"):
        load_transformer("m.pt", storage=st)


def test_missing_meta_dict_raises():
    st = _MemStorage()
    save_transformer(_small(), "m.pt", storage=st)
    bundle = torch.load(BytesIO(st._store["m.pt"]), weights_only=False)
    del bundle["meta"]
    buf = BytesIO(); torch.save(bundle, buf); st._store["m.pt"] = buf.getvalue()
    with pytest.raises(ValueError, match="meta"):
        load_transformer("m.pt", storage=st)


# ---------------------------------------------------------------------------
# State-dict key validation
# ---------------------------------------------------------------------------

def test_missing_param_key_raises():
    st = _MemStorage()
    save_transformer(_small(), "m.pt", storage=st)
    bundle = torch.load(BytesIO(st._store["m.pt"]), weights_only=False)
    # Remove one key from state_dict
    first_key = next(iter(bundle["state_dict"]))
    del bundle["state_dict"][first_key]
    buf = BytesIO(); torch.save(bundle, buf); st._store["m.pt"] = buf.getvalue()
    with pytest.raises(ValueError, match="missing"):
        load_transformer("m.pt", storage=st)


def test_extra_param_key_raises():
    st = _MemStorage()
    save_transformer(_small(), "m.pt", storage=st)
    bundle = torch.load(BytesIO(st._store["m.pt"]), weights_only=False)
    bundle["state_dict"]["unexpected.key"] = torch.zeros(1)
    buf = BytesIO(); torch.save(bundle, buf); st._store["m.pt"] = buf.getvalue()
    with pytest.raises(ValueError, match="unexpected"):
        load_transformer("m.pt", storage=st)


def test_shape_mismatch_raises():
    st = _MemStorage()
    save_transformer(_small(), "m.pt", storage=st)
    bundle = torch.load(BytesIO(st._store["m.pt"]), weights_only=False)
    first_key = next(iter(bundle["state_dict"]))
    bundle["state_dict"][first_key] = torch.zeros(1, 1, 1)   # wrong shape
    buf = BytesIO(); torch.save(bundle, buf); st._store["m.pt"] = buf.getvalue()
    with pytest.raises(ValueError, match="shape"):
        load_transformer("m.pt", storage=st)


# ---------------------------------------------------------------------------
# LocalStorage atomicity
# ---------------------------------------------------------------------------

def test_local_storage_no_tmp_on_success(tmp_path: Path):
    model = _small()
    path  = tmp_path / "model.pt"
    save_transformer(model, path)
    assert path.exists()
    assert not (tmp_path / "model.pt.tmp").exists()


def test_local_storage_file_unchanged_on_error(tmp_path: Path):
    """If write raises, original file is left intact."""
    model  = _small()
    path   = tmp_path / "model.pt"
    # Write a sentinel file first
    path.write_bytes(b"original")

    class BrokenStorage(LocalStorage):
        @contextmanager
        def open_write(self, key):
            with super().open_write(key) as fh:
                fh.write(b"partial")
                raise RuntimeError("simulated failure")
                yield fh

    with pytest.raises(RuntimeError):
        save_transformer(model, path, storage=BrokenStorage())

    assert path.read_bytes() == b"original"
    assert not path.with_name("model.pt.tmp").exists()


# ---------------------------------------------------------------------------
# End-to-end: train -> save -> load -> weights match
# ---------------------------------------------------------------------------

def test_end_to_end_save_load_weights_match():
    """After training, saving, and loading, every parameter is bit-exact."""
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    from pytorch_transformer.training import train

    rng = np.random.default_rng(1)
    n   = 64
    enc     = torch.from_numpy(rng.integers(0, 10, (n, 5), dtype=np.int64))
    dec_in  = torch.from_numpy(
        np.concatenate([np.full((n, 1), 11), np.sort(enc.numpy(), axis=1)], axis=1).astype(np.int64)
    )
    dec_tgt = torch.from_numpy(
        np.concatenate([np.sort(enc.numpy(), axis=1), np.full((n, 1), 12)], axis=1).astype(np.int64)
    )
    dl = DataLoader(TensorDataset(enc, dec_in, dec_tgt), batch_size=32)

    model = _small()
    train(model, dl, n_steps=10, ignore_index=10, log_every=0)

    st     = _MemStorage()
    save_transformer(model, "m.pt", storage=st)
    loaded = load_transformer("m.pt", storage=st)

    for (k, v), (_, v2) in zip(
        sorted(model.state_dict().items()),
        sorted(loaded.state_dict().items()),
    ):
        assert torch.equal(v, v2), f"Mismatch at {k}"
