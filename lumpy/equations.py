import numpy as np

def solve(A, b):
    """
    Solve Ax = b.
    Prefer solve() over explicitly computing inv(A) @ b.
    """
    return np.linalg.solve(A, b)

def lstsq(A, b, rcond=None):
    """
    Return the least-squares solution to Ax ≈ b.
    """
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=rcond)
    return x