import numpy as np

def svd(A, full_matrices=False, tol=1e-12):
    """
    Return the singular value decomposition of A.

    By default, returns only the rank-relevant singular directions:

        A = U @ diag(s) @ Vt

    If full_matrices=True, returns NumPy's full SVD.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    if not full_matrices:
        rank = np.sum(s > tol)
        return U[:,:rank], s[:rank], Vt[:rank,:]
    return U, s, Vt