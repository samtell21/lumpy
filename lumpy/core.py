import numpy as np
from .utils import is_column_vector, is_matrix

__all__ = ["col", "row", "outer", "inner", "dot", "norm", "tr", "adj"]

def col(A, i):
    """Return column i as a column vector."""
    return A[:, [i]]

def row(A, i):
    """Return row i as a row matrix."""
    return A[[i], :]

def outer(u, v):
    """Outer product u v* (conjugate transpose, so it is correct
    over the complex numbers; equals u vᵀ for real input)."""
    if not(is_column_vector(u) and is_column_vector(v)):
        raise ValueError("outer() requires column vectors.")
    return u @ adj(v)

def inner(A,B):
    """Inner product A* B (conjugate transpose, so it is correct
    over the complex numbers; equals Aᵀ B for real input)."""
    if not (is_matrix(A) and is_matrix(B)):
        raise ValueError("inner() requires matrices.")
    return adj(A) @ B

def dot(u,v):
    """Dot product ⟨u, v⟩ = u* v (conjugate-linear in u, so it is
    correct over the complex numbers; equals uᵀ v for real input)."""
    if not(is_column_vector(u) and is_column_vector(v)):
        raise ValueError("dot() requires column vectors.")
    return (adj(u) @ v).item()
    
def norm(v):
    """Norm of v: the Euclidean (2-) norm of a vector, or the
    Frobenius norm of a matrix (NumPy's default in both cases)."""
    return np.linalg.norm(v)

def tr(A):
    """Trace of A."""
    return np.trace(A)

def adj(A):
    """Adjoint (conjugate transpose) of A."""
    return A.conj().T