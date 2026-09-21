"""Disk cache keyed on source-file fingerprints plus a config slice.

Parsing the raw CSVs takes ~60 s, dominated by ``observations.csv`` at 222 MB.
Reloading the cached Parquet takes ~3 s. Over three days of iteration that is
the difference between running the pipeline freely and avoiding it.

The key is ``(path, size, mtime_ns)`` for every declared dependency plus a
JSON-stable dump of the relevant config slice. Any change to the inputs or to
the settings that shaped the artifact invalidates it; nothing else does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

__all__ = ["ART", "fingerprint", "cached_parquet", "cached_npz"]

ART = Path("artifacts")


def fingerprint(deps: Iterable[str | Path], cfg_slice: Any = None) -> str:
    """Stable 16-hex-char hash of dependency file stats plus a config slice."""
    h = hashlib.sha256()
    for d in sorted(str(x) for x in deps):
        p = Path(d)
        if p.exists():
            st = p.stat()
            h.update(f"{p.as_posix()}|{st.st_size}|{st.st_mtime_ns}".encode())
        else:
            h.update(f"{p.as_posix()}|MISSING".encode())
    if cfg_slice is not None:
        h.update(json.dumps(cfg_slice, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _meta_ok(meta: Path, key: str) -> bool:
    try:
        return json.loads(meta.read_text())["key"] == key
    except Exception:
        return False  # missing, corrupt, or an older schema -- rebuild


def cached_parquet(
    name: str,
    builder: Callable[[], pd.DataFrame],
    deps: Iterable[str | Path],
    cfg_slice: Any = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Return ``builder()``'s frame, reading from / writing to ``artifacts/``."""
    ART.mkdir(parents=True, exist_ok=True)
    path, meta = ART / f"{name}.parquet", ART / f"{name}.meta.json"
    key = fingerprint(deps, cfg_slice)

    if not rebuild and path.exists() and _meta_ok(meta, key):
        return pd.read_parquet(path)

    df = builder()
    df.to_parquet(path, index=False, compression="zstd")
    meta.write_text(json.dumps(
        {"key": key, "rows": int(len(df)), "cols": list(map(str, df.columns))}, indent=2))
    return df


def cached_npz(
    name: str,
    builder: Callable[[], dict[str, np.ndarray]],
    deps: Iterable[str | Path],
    cfg_slice: Any = None,
    rebuild: bool = False,
) -> dict[str, np.ndarray]:
    """Same contract as :func:`cached_parquet`, for dicts of arrays."""
    ART.mkdir(parents=True, exist_ok=True)
    path, meta = ART / f"{name}.npz", ART / f"{name}.meta.json"
    key = fingerprint(deps, cfg_slice)

    if not rebuild and path.exists() and _meta_ok(meta, key):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    d = builder()
    np.savez_compressed(path, **d)
    meta.write_text(json.dumps(
        {"key": key, "arrays": {k: list(v.shape) for k, v in d.items()}}, indent=2))
    return d
