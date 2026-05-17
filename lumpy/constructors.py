import numpy as np
import numbers
from .utils import is_column_vector, is_row_vector

__all__ = ["mat", "matt", "vec", "eye", "e", "diag"]


def _resolve_dtype(arr, dtype):
    """
    Resolve a constructor's dtype request against the built array.

    "auto" -> float64, or complex128 if arr holds complex data
              (preserves Lumpy's "everything is float" default while
              letting complex input just work)
    None   -> arr.dtype, i.e. NumPy's own inference, no coercion
    else   -> the requested dtype, forced
    """
    if isinstance(dtype, str) and dtype == "auto":
        return np.complex128 if np.iscomplexobj(arr) else np.float64
    if dtype is None:
        return arr.dtype
    return dtype


def mat(*columns, dtype="auto"):
    """
    Construct a matrix from columns.

    Lists are interpreted as column entries. Existing column
    vectors may also be passed directly. Row vectors are rejected:
    mat() is column-oriented (use matt() for rows).
    """
    processed = []

    for col in columns:
        # Reject list of lists
        if isinstance(col, list) and any(isinstance(item, list) for item in col):
            raise ValueError("mat() does not accept lists of lists. Use la.mat(*l) to unpack.")

        # Python list -> column vector
        if isinstance(col, list):
            col = np.array(col).reshape(-1, 1)

        # numpy array
        elif isinstance(col, np.ndarray):
            # 1D array -> column vector
            if col.ndim == 1:
                col = col.reshape(-1, 1)
            # Existing column vector (n,1); also the (1,1) single entry
            elif is_column_vector(col):
                pass
            # Row vector (1,n), n>1 -- wrong orientation for mat()
            elif is_row_vector(col):
                raise ValueError(f"mat() expects columns, not row vectors. Got shape {col.shape}.")
            # Larger matrix
            else:
                raise ValueError(f"mat() expects columns (n,) or (n,1), got shape {col.shape}.")
        else:
            raise TypeError("mat() expects lists or numpy arrays.")

        processed.append(col)

    if not processed:
        raise ValueError("mat() requires at least one column.")

    result = np.hstack(processed)
    return result.astype(_resolve_dtype(result, dtype))


def matt(*rows, dtype="auto"):
    """
    Construct a matrix from rows.

    Lists are interpreted as row entries. Existing row vectors may
    also be passed directly. Column vectors are rejected.

    matt() preserves the usual visual layout of matrices,
    while mat() preserves Lumpy's column-oriented semantics.
    """
    processed = []

    for row in rows:
        # Reject list of lists
        if isinstance(row, list) and any(isinstance(item, list) for item in row):
            raise ValueError("matt() does not accept lists of lists. Use la.matt(*l) to unpack.")

        # Python list -> row vector
        if isinstance(row, list):
            row = np.array(row).reshape(1, -1)

        # numpy array
        elif isinstance(row, np.ndarray):
            # 1D array -> row vector
            if row.ndim == 1:
                row = row.reshape(1, -1)
            # Existing row vector (1,n); also the (1,1) single entry
            elif is_row_vector(row):
                pass
            # Column vector (n,1), n>1 -- wrong orientation for matt()
            elif is_column_vector(row):
                raise ValueError(f"matt() expects rows, not column vectors. Got shape {row.shape}.")
            # Larger matrix
            else:
                raise ValueError(f"matt() expects rows (n,) or (1,n), got shape {row.shape}.")
        else:
            raise TypeError("matt() expects lists or numpy arrays.")

        processed.append(row)

    if not processed:
        raise ValueError("matt() requires at least one row.")

    result = np.vstack(processed)
    return result.astype(_resolve_dtype(result, dtype))


def vec(*entries, dtype="auto"):
    """Construct a column vector from scalar values."""
    for entry in entries:
        if isinstance(entry, (list, np.ndarray)):
            raise TypeError("vec() accepts only scalar values. Use vec(*array) or mat(array) instead.")
        if not isinstance(entry, numbers.Number):
            raise TypeError("vec() accepts only scalar values.")
    return mat(list(entries), dtype=dtype)


def eye(n, dtype="auto"):
    """Construct the n x n identity matrix."""
    result = np.eye(n)
    return result.astype(_resolve_dtype(result, dtype))


# Paper-like alias for eye(). Deliberately excluded from __all__ so
# `import lumpy as la; la.I(n)` works while `from lumpy import *`
# does not pull the single-letter name into the namespace.
I = eye


def e(n, i, dtype="auto"):
    """
    e(n, i) returns the standard basis vector corresponding
    to the mathematical vector e_(i+1) in R^n.

    Indexing is 0-based to remain consistent with Python/NumPy.
    """
    result = np.zeros((n, 1))
    result[i] = 1
    return result.astype(_resolve_dtype(result, dtype))


def diag(*entries, dtype="auto"):
    """Construct a diagonal matrix with given entries."""
    result = np.diag(entries)
    return result.astype(_resolve_dtype(result, dtype))
