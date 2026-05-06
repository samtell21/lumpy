import numpy as np
from .decompositions import svd

def rank(A, tol=1e-12):
    """
    Return the rank of A.
    """
    _, s, _ = svd(A,tol=tol) 
    return len(s)

def orth(A,tol=1e-12):
    """
    Return an orthonormal basis for the column space of A.
    """
    U, _, _ = svd(A, tol=tol)
    return U

def null(A, tol=1e-12):
    """
    Return an orthonormal basis for the null space of A.
    """
    U, s, Vt = svd(A, full_matrices=True, tol=tol)
    rank = np.sum(s > tol)
    return Vt[rank:].T

def proj(A,B,tol=1e-12):
    """
    Orthogonal projection of B onto the column space of A.
    Works for vectors and matrices.
    """
    Q = orth(A,tol=tol)
    return Q @ Q.T @ B

def independent(A):
    """
    Return True if the columns of A are linearly independent.
    """
    return rank(A) == A.shape[1]