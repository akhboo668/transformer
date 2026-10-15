"""
Step 12: Save and restore PyTorch Transformer models.

Same design discipline as numpy_transformer.persistence (Step 11):
  - format_version field for forward-compatibility
  - All architecture hyperparameters stored in a 'meta' dict
  - LocalStorage with atomic tmp+rename (no half-written files)
  - Same Storage Protocol so backends are swappable

PyTorch-specific mechanics:
  - torch.save / torch.load for the bundle (pickle-based)
  - model.state_dict() / model.load_state_dict() for parameters
  - Tied-embedding aliasing is preserved automatically: load_state_dict does
    in-place copy_ on existing nn.Parameter objects, so shared references
    between transformer.embedding / encoder.embedding / decoder.embedding
    remain the same Python objects after loading.

Bundle layout (format_version == 1):
    {
        "format_version": 1,
        "framework":      "pytorch",
        "torch_version":  str,
        "meta":           {vocab_size, d_model, num_heads, d_ff,
                           num_encoder_layers, num_decoder_layers,
                           p_drop, max_len, pre_ln, tie_embeddings},
        "state_dict":     model.state_dict(),   # includes shared-param duplicates
    }
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Protocol

import torch

from pytorch_transformer.layers import Transformer

FORMAT_VERSION: int = 1

_META_FIELDS: tuple[str, ...] = (
    "vocab_size",
    "d_model",
    "num_heads",
    "d_ff",
    "num_encoder_layers",
    "num_decoder_layers",
    "p_drop",
    "max_len",
    "pre_ln",
    "tie_embeddings",
)


# ---------------------------------------------------------------------------
# Storage abstraction (same Protocol as Step 11)
# ---------------------------------------------------------------------------

class Storage(Protocol):
    def open_write(self, key: str) -> AbstractContextManager[BinaryIO]: ...
    def open_read(self,  key: str) -> AbstractContextManager[BinaryIO]: ...


class LocalStorage:
    """Atomic local-filesystem storage. Writes via tmp+rename."""

    @contextmanager
    def open_write(self, key: str) -> Iterator[BinaryIO]:
        path = Path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        committed = False
        try:
            with tmp.open("wb") as fh:
                yield fh
            tmp.replace(path)
            committed = True
        finally:
            if not committed and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    @contextmanager
    def open_read(self, key: str) -> Iterator[BinaryIO]:
        with Path(key).open("rb") as fh:
            yield fh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_transformer(
    model: Transformer,
    path: str | Path,
    *,
    storage: Storage | None = None,
) -> None:
    """Persist model architecture + weights to path (default: LocalStorage)."""
    storage = storage or LocalStorage()
    meta = {field: getattr(model, field) for field in _META_FIELDS}
    bundle = {
        "format_version": FORMAT_VERSION,
        "framework":      "pytorch",
        "torch_version":  torch.__version__,
        "meta":           meta,
        "state_dict":     model.state_dict(),
    }
    with storage.open_write(str(path)) as fh:
        torch.save(bundle, fh)


def load_transformer(
    path: str | Path,
    *,
    storage: Storage | None = None,
    map_location: str | torch.device = "cpu",
) -> Transformer:
    """Load a Transformer from path.  Raises ValueError on any format mismatch."""
    storage = storage or LocalStorage()
    with storage.open_read(str(path)) as fh:
        blob = fh.read()

    bundle = torch.load(BytesIO(blob), map_location=map_location, weights_only=False)

    _validate_format_version(bundle, str(path))
    meta   = _validate_meta(bundle, str(path))
    model  = Transformer(**meta)
    _validate_and_load_state_dict(bundle["state_dict"], model, str(path))
    return model


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _validate_format_version(bundle: dict, path: str) -> None:
    if "format_version" not in bundle:
        raise ValueError(f"{path}: missing 'format_version'; not a transformer save file.")
    v = bundle["format_version"]
    if not isinstance(v, int) or v < 1 or v > FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version={v!r} not supported "
            f"(this code supports 1..{FORMAT_VERSION})."
        )


def _validate_meta(bundle: dict, path: str) -> dict[str, Any]:
    if "meta" not in bundle:
        raise ValueError(f"{path}: missing 'meta' dict.")
    meta = bundle["meta"]
    missing = [f for f in _META_FIELDS if f not in meta]
    if missing:
        raise ValueError(
            f"{path}: save file missing required meta field(s): {sorted(missing)}"
        )
    return {f: meta[f] for f in _META_FIELDS}


def _validate_and_load_state_dict(
    saved_sd: dict, model: Transformer, path: str
) -> None:
    expected = set(model.state_dict().keys())
    present  = set(saved_sd.keys())

    missing = sorted(expected - present)
    extra   = sorted(present  - expected)

    if missing or extra:
        parts = [f"{path}: state_dict key mismatch."]
        if missing:
            parts.append(f"  missing {len(missing)}: {missing[:5]}")
        if extra:
            parts.append(f"  unexpected {len(extra)}: {extra[:5]}")
        raise ValueError("\n".join(parts))

    # Shape check before loading
    current_sd = model.state_dict()
    for key in expected:
        if saved_sd[key].shape != current_sd[key].shape:
            raise ValueError(
                f"{path}: shape mismatch for '{key}': "
                f"file={tuple(saved_sd[key].shape)}, "
                f"model={tuple(current_sd[key].shape)}"
            )

    # load_state_dict does in-place copy_ on existing Parameters/buffers,
    # preserving tied-embedding aliasing (shared nn.Parameter objects remain
    # the same Python objects; only their values are updated).
    model.load_state_dict(saved_sd, strict=True)
