import numpy as np

__all__ = ["svd"]


def _rank_cutoff(s, shape, tol):
    """
    Threshold below which a singular value counts as zero.

    tol=None    -> relative, scale-invariant cutoff
                   max(s) * eps * max(m, n), the same rule
                   numpy.linalg.matrix_rank uses by default.
                   Scaling A by a constant scales both the
                   singular values and the cutoff, so the rank
                   decision is invariant.
    tol=number  -> that number, used as an absolute cutoff.
    """
    if tol is not None:
        return tol
    if s.size == 0:
        return 0.0
    eps = np.finfo(s.dtype).eps
    return s.max() * eps * max(shape)


def svd(A, full_matrices=False, tol=None):
    """
    Return the singular value decomposition of A.

    By default, returns only the rank-relevant singular directions:

        A = U @ diag(s) @ Vt

    A singular value counts as nonzero when it exceeds a tolerance.
    With tol=None (default) the tolerance is relative and
    scale-invariant; pass a number to use an absolute cutoff.

    If full_matrices=True, returns NumPy's full (untrimmed) SVD.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    if not full_matrices:
        cutoff = _rank_cutoff(s, A.shape, tol)
        rank = int(np.sum(s > cutoff))
        return U[:, :rank], s[:rank], Vt[:rank, :]
    return U, s, Vt
