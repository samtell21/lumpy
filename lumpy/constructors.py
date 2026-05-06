import numpy as np
from .utils import is_vector

def mat(*columns, dtype=float):
    """Create a matrix from columns."""
    processed = []
    for col in columns:
        # Python list -> column vector
        if isinstance(col, list):
            col = np.array(col, dtype=dtype).reshape(-1,1)
        processed.append(col)
    return np.hstack(processed).astype(dtype)

def vec(*entries, dtype=float):
    """Create a column vector."""
    return mat(list(entries), dtype=dtype)

def eye(n):
    """Identity matrix of size n."""
    return np.eye(n)

def e(n, i):
    """
    e(n, i) returns the standard basis vector corresponding
    to the mathematical vector e_(i+1) in R^n.

    Indexing is 0-based to remain consistent with Python/NumPy.
    """
    e = np.zeros((n,1))
    e[i] = 1
    return e

def diag(*entries, dtype=float):
    """Diagonal matrix with given entries."""
    return np.diag(entries).astype(dtype)