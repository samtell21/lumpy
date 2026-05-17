import numpy as np
from .decompositions import svd, _rank_cutoff
from .core import adj
from .utils import is_matrix

__all__ = [
    "rank",
    "orth",
    "null",
    "proj",
    "independent",
    "row_space",
    "left_null",
]


def rank(A, tol=None):
    """
    Return the rank of A.
    """
    _, s, _ = svd(A, tol=tol)
    return len(s)


def orth(A, tol=None):
    """
    Return an orthonormal basis for the column space of A.
    """
    U, _, _ = svd(A, tol=tol)
    return U


def null(A, tol=None):
    """
    Return an orthonormal basis for the null space of A.
    """
    U, s, Vt = svd(A, full_matrices=True, tol=tol)
    cutoff = _rank_cutoff(s, A.shape, tol)
    rank = int(np.sum(s > cutoff))
    return adj(Vt[rank:])


def proj(A, B, tol=None):
    """
    Orthogonal projection of B onto the column space of A.
    Works for vectors and matrices.
    """
    if not (is_matrix(A) and is_matrix(B)):
        raise ValueError("proj() requires matrices.")
    Q = orth(A, tol=tol)
    return Q @ adj(Q) @ B


def independent(A):
    """
    Return True if the columns of A are linearly independent.
    """
    return rank(A) == A.shape[1]


def row_space(A, tol=None):
    """
    Return an orthonormal basis for the row space of A as columns.
    """
    return orth(adj(A), tol=tol)


def left_null(A, tol=None):
    """
    Return an orthonormal basis for the left nullspace of A as columns.

    This is Null(A*) (conjugate transpose; equals Null(Aᵀ) for real A).
    """
    return null(adj(A), tol=tol)
