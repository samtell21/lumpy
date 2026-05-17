import numpy as np
from .utils import is_matrix

__all__ = ["solve", "lstsq"]

def solve(A, b):
    """
    Solve Ax = b.
    Prefer solve() over explicitly computing inv(A) @ b.
    """
    if not (is_matrix(A) and is_matrix(b)):
        raise ValueError("solve() requires matrices.")
    return np.linalg.solve(A, b)

def lstsq(A, b, rcond=None, full=False):
    """
    Return the least-squares solution to Ax ≈ b.
    """
    if not (is_matrix(A) and is_matrix(b)):
        raise ValueError("lstsq() requires matrices.")
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=rcond)
    if full:
        return x, residuals, rank, s
    return x