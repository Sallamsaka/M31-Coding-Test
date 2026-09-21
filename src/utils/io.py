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



def _atomic_write(path: Path, write: "Callable[[Path], None]") -> None:
    """Write via a temp file and rename, so no reader ever sees a partial file.

    `to_parquet(path)` writes straight to the destination, so a second process
    reading while the first is writing gets a truncated file. That is not
    hypothetical: D19 recorded `ArrowInvalid: Parquet magic bytes not found in
    footer` from exactly this race, and it recurred today when pytest ran
    against a live pipeline.

    The temp name carries the PID, so two processes rebuilding the same artifact
    cannot corrupt each other's temp file before either rename.
    """
    import os
    import time as _time
    # The suffix must be PRESERVED: np.savez_compressed appends ".npz" when the
    # filename does not already end in it, so a plain ".tmp" name would be
    # written as "....tmp.npz" and the rename would fail on a missing file.
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    write(tmp)
    # os.replace is atomic on POSIX; on Windows it raises if anything holds a
    # handle on the destination, so retry before giving up.
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            _time.sleep(0.3 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    raise OSError(f"could not atomically replace {path}")


def _builder_source_stat(builder) -> str:
    """Identify the CODE that produced an artifact, for the cache key.

    The key was (path, size, mtime) of the input files plus a config slice --
    with no code version. So editing a builder and re-running served the stale
    artifact silently: same inputs, same config, same key. D2 is this bug, and
    it was still live today after `build_vocab`'s default fit set changed.
    (`build_vocab` happens not to be cached, which is the only reason that
    change was safe.)

    Only `labels.py` had a tripwire on its own output -- the frozen golden
    counts -- so only labels would have caught a silent drift. cohort, events
    and examples had nothing.
    """
    try:
        import inspect
        f = Path(inspect.getfile(builder))
        st = f.stat()
        return f"{f.name}|{st.st_size}|{st.st_mtime_ns}"
    except Exception:
        return "unknown"

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
    key = fingerprint(deps, (cfg_slice, _builder_source_stat(builder)))

    if not rebuild and path.exists() and _meta_ok(meta, key):
        return pd.read_parquet(path)

    df = builder()
    # Data first, then meta. If the order were reversed and the data write
    # failed, the cache would look valid and be empty.
    _atomic_write(path, lambda t: df.to_parquet(t, index=False, compression="zstd"))
    _atomic_write(meta, lambda t: t.write_text(json.dumps(
        {"key": key, "rows": int(len(df)), "cols": list(map(str, df.columns))},
        indent=2)))
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
    key = fingerprint(deps, (cfg_slice, _builder_source_stat(builder)))

    if not rebuild and path.exists() and _meta_ok(meta, key):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    d = builder()
    _atomic_write(path, lambda t: np.savez_compressed(t, **d))
    _atomic_write(meta, lambda t: t.write_text(json.dumps(
        {"key": key, "arrays": {k: list(v.shape) for k, v in d.items()}}, indent=2)))
    return d
