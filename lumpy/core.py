import numpy as np
from .utils import is_vector

def col(A, i):
    """Return column i as a column vector."""
    return A[:, [i]]

def row(A, i):
    """Return row i as a row matrix."""
    return A[[i], :]

def inner(A,B):
    """Inner product of matrices."""
    return A.T @ B

def dot(u,v):
    """Dot product of vectors."""
    if not(is_vector(u) and is_vector(v)):
        raise ValueError("dot() requires column vectors.")
    return (u.T @ v).item()
    
def norm(v):
    """Euclidean norm of a column vector."""
    return np.linalg.norm(v)


def T(A):
    """Transpose of A."""
    return A.T

def tr(A):
    """Trace of A."""
    return np.trace(A)

def adj(A):
    """Adjoint (conjugate transpose) of A."""
    return A.conj().T

def det(A):
    """Determinant of A."""
    return np.linalg.det(A)