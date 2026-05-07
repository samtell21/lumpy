import numpy as np
from .utils import is_vector

def mat(*columns, dtype=float):
    """"
    Construct a matrix from columns.

    Lists are interpreted as column entries. Existing column
    vectors may also be passed directly.
    """
    processed = []
    
    for col in columns:
        # Python list -> column vector
        if isinstance(col, list):
            col = np.array(col, dtype=dtype).reshape(-1,1)

        if col.ndim != 2 or col.shape[1] != 1:
            raise ValueError("mat() expects columns: lists or column vectors.")

        processed.append(col)

    if not processed:
        raise ValueError("mat() requires at least one column.")

    return np.hstack(processed).astype(dtype)

def matt(*rows, dtype=float):
    """
    Construct a matrix from rows.

    matt() preserves the usual visual layout of matrices,
    while mat() preserves Lumpy's column-oriented semantics.
    """
    if not rows:
        raise ValueError("matt() requires at least one row.")
    return mat(*rows, dtype=dtype).T

def vec(*entries, dtype=float):
    """Construct a column vector."""
    return mat(list(entries), dtype=dtype)

def eye(n, dtype=float):
    """Construct the n x n identity matrix."""
    return np.eye(n, dtype=dtype)

def e(n, i, dtype=float):
    """
    e(n, i) returns the standard basis vector corresponding
    to the mathematical vector e_(i+1) in R^n.

    Indexing is 0-based to remain consistent with Python/NumPy.
    """
    e = np.zeros((n,1), dtype=dtype)
    e[i] = 1
    return e

def diag(*entries, dtype=float):
    """Construct a diagonal matrix with given entries."""
    return np.diag(entries).astype(dtype)