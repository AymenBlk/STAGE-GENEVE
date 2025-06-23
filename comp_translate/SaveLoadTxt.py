"""
save_load_txt.py
================
Plain-text serializer/deserializer for numeric scalars, 1-D vectors and
2-D matrices in the **translate2compare 1.0** format.

API
---
dump_txt(path, objects, append=True)  -> None
load_txt(path)                        -> list
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, List, Tuple, Union, Any

import numpy as np

HEADER = "# translate2compare 1.0"
Scalar = Union[int, float]
Vector = np.ndarray  # ndim == 1
Matrix = np.ndarray  # ndim == 2
Obj = Union[Scalar, Vector, Matrix]


# ─────────────────────────── helpers ────────────────────────────
def _token(x: Scalar) -> str:
    """Return an ASCII token with full precision for Float64."""
    return format(x, ".17g") if isinstance(x, float) else str(int(x))


def _describe(obj: Obj) -> Tuple[str, str, Tuple[int, ...]]:
    """
    kind ∈ {scalar, vector, matrix}
    dtype ∈ {int64, float64}
    dims  = () | (N,) | (rows, cols)
    """
    if np.isscalar(obj):
        if isinstance(obj, (int, np.integer)):
            return "scalar", "int64", ()
        elif isinstance(obj, (float, np.floating)):
            return "scalar", "float64", ()
        raise ValueError(f"Unsupported scalar type {type(obj)}")

    if not isinstance(obj, np.ndarray):
        raise ValueError(f"Unsupported object {type(obj)}")

    if obj.dtype == np.int64:
        dtype = "int64"
    elif obj.dtype == np.float64:
        dtype = "float64"
    else:
        raise ValueError(f"Unsupported dtype {obj.dtype}")

    if obj.ndim == 1:
        return "vector", dtype, (obj.shape[0],)
    elif obj.ndim == 2:
        return "matrix", dtype, obj.shape
    raise ValueError("Only 1-D or 2-D arrays supported")


def _write_object(fh, obj: Obj) -> None:
    kind, dtype, dims = _describe(obj)
    if kind == "scalar":
        fh.write(f"# object {kind} {dtype}\n")
        fh.write(_token(obj) + "\n")

    elif kind == "vector":
        n, = dims
        fh.write(f"# object {kind} {dtype} {n}\n")
        fh.write(" ".join(_token(x) for x in obj) + "\n")

    elif kind == "matrix":
        rows, cols = dims
        fh.write(f"# object {kind} {dtype} {rows} {cols}\n")
        for r in range(rows):
            fh.write(" ".join(_token(x) for x in obj[r, :]) + "\n")


def dump_txt(path: str | Path,
             objects: Sequence[Obj],
             append: bool = True) -> None:
    """
    Write every object in *objects* to *path* (translate2compare format).

    Parameters
    ----------
    path : str or Path
    objects : iterable of scalars / numpy vectors / numpy matrices
    append : bool, default True
        If False, the file is overwritten; if True, new blocks are appended.
    """
    if not objects:
        return

    path = Path(path)
    mode = "a" if append else "w"
    header_needed = not (append and path.exists())

    with path.open(mode, encoding="utf-8") as fh:
        if header_needed:
            fh.write(HEADER + "\n")
        for obj in objects:
            _write_object(fh, obj)


# ────────────────────────── reading side ─────────────────────────
def _expect_header(fh):
    line = fh.readline()
    if not line:
        raise ValueError("Empty file")
    if line.rstrip("\n") != HEADER:
        raise ValueError(f"Bad header: {line.strip()!r}")


def _parse_descriptor(line: str):
    if not line.startswith("# object "):
        raise ValueError(f"Malformed descriptor line: {line!r}")
    parts = line[9:].strip().split()
    if len(parts) < 2:
        raise ValueError(f"Incomplete descriptor: {line!r}")
    kind, dtype, *dims = parts
    dims = tuple(int(d) for d in dims)
    return kind, dtype, dims


def _convert(tokens: List[str], dtype: str):
    return (np.array(tokens, dtype=np.int64) if dtype == "int64"
            else np.array(tokens, dtype=np.float64))


def load_txt(path: str | Path) -> List[Obj]:
    """
    Parse *path* and return every stored object in order.

    Scalars become int / float. Vectors & matrices become NumPy arrays.
    """
    objs: List[Obj] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as fh:
        _expect_header(fh)

        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:          # EOF
                break
            line = line.strip()
            if not line:          # skip blanks
                continue

            # rewind so descriptor parser sees full line
            fh.seek(pos)
            desc = fh.readline().rstrip("\n")
            kind, dtype, dims = _parse_descriptor(desc)

            if kind == "scalar":
                tok = fh.readline().strip()
                val: Obj = int(tok) if dtype == "int64" else float(tok)
                objs.append(val)

            elif kind == "vector":
                (n,) = dims
                toks = fh.readline().split()
                if len(toks) != n:
                    raise ValueError("Token count mismatch in vector")
                objs.append(_convert(toks, dtype))

            elif kind == "matrix":
                rows, cols = dims
                rows_data = [fh.readline().split() for _ in range(rows)]
                if any(len(r) != cols for r in rows_data):
                    raise ValueError("Token count mismatch in matrix")
                mat = _convert([tok for row in rows_data for tok in row],
                               dtype).reshape(rows, cols)
                objs.append(mat)

            else:
                raise ValueError(f"Unknown kind {kind!r}")

    return objs
