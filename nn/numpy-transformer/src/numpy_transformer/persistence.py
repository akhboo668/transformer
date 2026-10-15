"""
Save and restore Transformer models to/from a single .npz file.

See nn/docs/11_model_persistence.md for the full design rationale, the
file-format contract (namespacing, format_version, validation rules), the
forward-compatibility plan that lets future versions add optimizer / RNG
state without touching this module's invariants, and the storage-backend
abstraction (LocalStorage / FsspecStorage / custom Storage protocols).

Public API:
    save_transformer(transformer, path, *, storage=None) -> None
    load_transformer(path, *, storage=None) -> Transformer
    Storage           Protocol describing the storage backend interface
    LocalStorage      Default. Atomic-write via tmp+rename. No new deps.
    FsspecStorage     Opt-in. Wraps fsspec for s3://, gs://, azure://, etc.

File layout (format_version == 1):
    meta.format_version       int64
    meta.<hyperparameter>     scalar         # one per _META_FIELDS entry
    params.<dotted_name>      float64 array  # one per params_and_grads() entry

Future namespaces (not produced or consumed by this module today, but
reserved at the file level):
    optimizer.*               # Adam state, slice 5
    rng.*                     # RNG state, slice 6
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, Protocol

import numpy as np

from numpy_transformer.layers import Transformer

FORMAT_VERSION: int = 1

META_PREFIX: str = "meta."
PARAMS_PREFIX: str = "params."

# Architecture hyperparameters that must round-trip to reconstruct an
# identically shaped Transformer. Kept in __init__ order for readability.
# Any new constructor parameter on Transformer must be added here.
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
# Storage abstraction
# ---------------------------------------------------------------------------


class Storage(Protocol):
    """
    Pluggable storage backend.

    Two methods, both returning a context manager that yields a binary
    file-like object. The context-manager exit is the atomicity boundary:
    when ``open_write`` exits without exception, the data at ``key`` is
    observably the new data; when it exits with an exception, the data
    at ``key`` is either unchanged (atomic backend) or in an unspecified
    state (best-effort backend). Each implementation documents its own
    guarantee.
    """

    def open_write(self, key: str) -> AbstractContextManager[BinaryIO]:
        ...

    def open_read(self, key: str) -> AbstractContextManager[BinaryIO]:
        ...


class LocalStorage:
    """
    Default backend. Reads and writes files on the local filesystem.

    **Atomicity:** writes go to ``<key>.tmp`` first and are renamed onto
    the destination on clean exit. ``Path.replace`` is atomic on POSIX
    (``rename(2)`` syscall) and overwrite-rename on Windows; in both
    cases a reader can never observe a half-written file. On exception,
    the tmp file is deleted (best-effort) and the destination is left
    unchanged.
    """

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


class FsspecStorage:
    """
    Opt-in backend that wraps `fsspec <https://filesystem-spec.readthedocs.io>`_.

    Supports any URI scheme fsspec can dispatch on, e.g. ``s3://`` (with
    ``s3fs`` installed), ``gs://`` (with ``gcsfs``), ``azure://`` (with
    ``adlfs``), or read-only ``https://``.

    **Atomicity** depends on the underlying backend: object-storage PUTs
    (S3, GCS, Azure) are atomic at the bucket level; ``file://`` via
    fsspec is *not* atomic (use :class:`LocalStorage` for local files).

    **Lazy import:** ``fsspec`` is imported only when this class is
    instantiated, so users who never touch cloud storage pay no import
    cost. If fsspec is missing, the constructor raises ``ImportError``
    with a clear install hint.

    Parameters
    ----------
    **storage_options
        Forwarded verbatim to ``fsspec.open``. Use this for credentials,
        endpoint URLs, region, profiles, etc. The exact keys depend on
        the backend (see fsspec / s3fs / gcsfs docs).
    """

    def __init__(self, **storage_options: Any) -> None:
        # Probe import so we fail at construction time, not at first use.
        # The `pyright: ignore` is intentional: fsspec is an optional
        # dependency, so static analysis without the `cloud` extra
        # installed will (correctly) report it as missing.
        try:
            import fsspec  # type: ignore[import-not-found]  # noqa: F401  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise ImportError(
                "FsspecStorage requires fsspec. Install with "
                "`uv sync --package numpy-transformer --extra cloud` "
                "(or `pip install fsspec s3fs` for S3 specifically)."
            ) from e
        self._storage_options = storage_options

    @contextmanager
    def open_write(self, key: str) -> Iterator[BinaryIO]:
        import fsspec  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        with fsspec.open(key, mode="wb", **self._storage_options) as fh:
            yield fh

    @contextmanager
    def open_read(self, key: str) -> Iterator[BinaryIO]:
        import fsspec  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        with fsspec.open(key, mode="rb", **self._storage_options) as fh:
            yield fh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_transformer(
    transformer: Transformer,
    path: str | Path,
    *,
    storage: Storage | None = None,
) -> None:
    """
    Persist `transformer`'s architecture metadata + learnable parameters
    to `path` via the given storage backend (default: :class:`LocalStorage`).

    The write is atomic (when the backend supports it). Process death
    mid-save leaves either the previous file or no file -- never a
    half-written one.

    Parameters
    ----------
    transformer : Transformer
        Must expose the constructor hyperparameters listed in
        ``_META_FIELDS`` as attributes, plus a ``params_and_grads()``
        method returning ``[(param, grad, name), ...]``.
    path : str or Path
        Destination key. For :class:`LocalStorage` (default), this is a
        filesystem path. For :class:`FsspecStorage`, this is a URI
        understood by fsspec (e.g. ``"s3://bucket/key.npz"``).
    storage : Storage, optional
        Storage backend. Defaults to ``LocalStorage()``.
    """
    storage = storage or LocalStorage()
    key = str(path)

    arrays = _build_save_arrays(transformer)
    with storage.open_write(key) as fh:
        np.savez_compressed(fh, **arrays)


def load_transformer(
    path: str | Path,
    *,
    storage: Storage | None = None,
) -> Transformer:
    """
    Construct a fresh Transformer from `path` and copy every saved
    parameter in-place into it.

    The receiving Transformer is built using the metadata stored in the
    file; the caller does **not** need to remember the hyperparameters
    used at training time.

    Parameters
    ----------
    path : str or Path
        Source key. Same conventions as :func:`save_transformer`.
    storage : Storage, optional
        Storage backend. Defaults to ``LocalStorage()``.

    Raises
    ------
    ValueError
        On any of: missing/unsupported ``meta.format_version``, missing
        required ``meta.*`` fields, missing/extra ``params.*`` keys,
        dtype mismatch, or shape mismatch. Other namespaces in the file
        (e.g. ``optimizer.*``) are silently ignored.
    FileNotFoundError
        If `path` does not exist (LocalStorage) or the underlying
        backend reports the key as missing.
    """
    storage = storage or LocalStorage()
    key = str(path)

    # `.npz` is a ZIP archive; ZIP needs random access (the central
    # directory lives at the end of the file). Some fsspec streams aren't
    # seekable, so we eagerly read the whole blob into a BytesIO. Fine
    # for the toy model; flagged in docs as O(model size) memory on load.
    with storage.open_read(key) as fh:
        blob = fh.read()

    with np.load(BytesIO(blob), allow_pickle=False) as z:
        files: set[str] = set(z.files)
        version = _read_format_version(z, files, key)
        metadata = _read_metadata(z, files, key)
        transformer = Transformer(**metadata)
        _validate_param_keys(files, transformer, key)
        _copy_params_in_place(z, transformer, key)
        # version is currently unused beyond the bound check above, but
        # keep it in scope so future readers (slice 5+) can branch on it.
        _ = version

    return transformer


# ---------------------------------------------------------------------------
# Save internals
# ---------------------------------------------------------------------------


def _build_save_arrays(transformer: Transformer) -> Dict[str, np.ndarray]:
    """
    Build the full dict-of-arrays that gets written to a save file.

    Pure: no IO. Returning a dict makes this trivial to test and
    independent of the storage backend.
    """
    arrays: Dict[str, np.ndarray] = {}

    arrays[META_PREFIX + "format_version"] = np.asarray(FORMAT_VERSION, dtype=np.int64)
    for field in _META_FIELDS:
        if not hasattr(transformer, field):
            raise AttributeError(
                f"Transformer is missing expected hyperparameter "
                f"attribute {field!r}; cannot serialize."
            )
        arrays[META_PREFIX + field] = _to_scalar_array(getattr(transformer, field))

    seen: set[str] = set()
    for (param, _grad, name) in transformer.params_and_grads():
        full_key = PARAMS_PREFIX + name
        if full_key in seen:
            raise ValueError(
                f"duplicate parameter name {name!r} in params_and_grads(); "
                f"persistence requires unique names."
            )
        seen.add(full_key)
        arrays[full_key] = np.ascontiguousarray(param)

    return arrays


# ---------------------------------------------------------------------------
# Loader internals
# ---------------------------------------------------------------------------


def _read_format_version(
    z: np.lib.npyio.NpzFile, files: set[str], key: str
) -> int:
    version_key = META_PREFIX + "format_version"
    if version_key not in files:
        raise ValueError(
            f"{key}: missing {version_key!r}; this does not look like a "
            f"transformer save file."
        )
    version = int(np.asarray(z[version_key]).item())
    if version < 1 or version > FORMAT_VERSION:
        raise ValueError(
            f"{key}: format_version={version} is not supported by this "
            f"code (supports 1..{FORMAT_VERSION}). If newer, upgrade the "
            f"loader; if older, you've hit an incompatible major-version "
            f"boundary."
        )
    return version


def _read_metadata(
    z: np.lib.npyio.NpzFile, files: set[str], key: str
) -> Dict[str, Any]:
    missing: list[str] = []
    metadata: Dict[str, Any] = {}
    for field in _META_FIELDS:
        meta_key = META_PREFIX + field
        if meta_key not in files:
            missing.append(field)
            continue
        metadata[field] = _from_scalar_array(z[meta_key])
    if missing:
        raise ValueError(
            f"{key}: save file is missing required metadata field(s): "
            f"{sorted(missing)}. Cannot reconstruct architecture."
        )
    return metadata


def _validate_param_keys(
    files: set[str], transformer: Transformer, key: str
) -> None:
    expected = {
        PARAMS_PREFIX + name for (_p, _g, name) in transformer.params_and_grads()
    }
    present = {k for k in files if k.startswith(PARAMS_PREFIX)}

    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if not missing and not extra:
        return

    parts = [
        f"{key}: param key mismatch for the architecture reconstructed "
        f"from this file's metadata."
    ]
    if missing:
        sample = missing[:5]
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        parts.append(f"  missing {len(missing)} key(s): {sample}{more}")
    if extra:
        sample = extra[:5]
        more = "" if len(extra) <= 5 else f" (+{len(extra) - 5} more)"
        parts.append(f"  unexpected {len(extra)} key(s): {sample}{more}")
    raise ValueError("\n".join(parts))


def _copy_params_in_place(
    z: np.lib.npyio.NpzFile, transformer: Transformer, key: str
) -> None:
    """
    In-place copy each saved param into the matching slot of
    ``transformer``. In-place is critical: when ``tie_embeddings=True``
    the embedding array is shared by reference between the top-level
    Transformer, its Encoder, and its Decoder; a non-in-place assignment
    would break that aliasing and silently leave the encoder/decoder
    looking at the original (random-init) weights.
    """
    for (param, _grad, name) in transformer.params_and_grads():
        saved = z[PARAMS_PREFIX + name]
        if param.dtype != saved.dtype:
            raise ValueError(
                f"{key}: dtype mismatch for {name!r}: "
                f"file={saved.dtype}, model={param.dtype}"
            )
        if param.shape != saved.shape:
            raise ValueError(
                f"{key}: shape mismatch for {name!r}: "
                f"file={saved.shape}, model={param.shape}"
            )
        param[...] = saved


# ---------------------------------------------------------------------------
# Scalar metadata helpers
# ---------------------------------------------------------------------------


def _to_scalar_array(value: Any) -> np.ndarray:
    """
    Wrap a Python scalar as a 0-d numpy array with an explicit dtype.

    Order matters: ``bool`` is a subclass of ``int`` in Python, so we
    must check it first.
    """
    if isinstance(value, bool):
        return np.asarray(value, dtype=np.bool_)
    if isinstance(value, int):
        return np.asarray(value, dtype=np.int64)
    if isinstance(value, float):
        return np.asarray(value, dtype=np.float64)
    raise TypeError(
        f"unsupported metadata value type for persistence: "
        f"{type(value).__name__} (value={value!r})"
    )


def _from_scalar_array(arr: np.ndarray) -> Any:
    """Unwrap a 0-d numpy array back to a native Python scalar."""
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        return bool(arr.item())
    if np.issubdtype(arr.dtype, np.integer):
        return int(arr.item())
    if np.issubdtype(arr.dtype, np.floating):
        return float(arr.item())
    raise TypeError(
        f"unsupported metadata array dtype on load: {arr.dtype}"
    )
