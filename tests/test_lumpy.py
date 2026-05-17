import numpy as np
import pytest

import lumpy as la


def assert_close(A, B):
    assert np.allclose(A, B)


# =========================
# Constructors
# =========================

def test_vec_is_column_vector():
    v = la.vec(1, 2, 3)
    assert v.shape == (3, 1)
    assert_close(v, np.array([[1], [2], [3]], dtype=float))


def test_mat_from_lists():
    A = la.mat([1, 2, 3], [4, 5, 6])
    assert A.shape == (3, 2)
    assert_close(A, np.array([[1, 4], [2, 5], [3, 6]], dtype=float))


def test_mat_accepts_column_vectors():
    # Regression: mat() must accept existing (n,1) column vectors,
    # mixed with lists, as the docstring/README promise.
    v = la.vec(1, 2, 3)
    A = la.mat(v, [4, 5, 6])
    assert_close(A, np.array([[1, 4], [2, 5], [3, 6]], dtype=float))


def test_matt_from_rows():
    A = la.matt([1, 2, 3], [4, 5, 6])
    assert_close(A, np.array([[1, 2, 3], [4, 5, 6]], dtype=float))


def test_matt_accepts_row_vectors():
    # Symmetric regression: matt() must accept existing (1,n) rows.
    r = la.matt([1, 2, 3])
    assert r.shape == (1, 3)
    A = la.matt(r, [4, 5, 6])
    assert_close(A, np.array([[1, 2, 3], [4, 5, 6]], dtype=float))


def test_mat_rejects_row_vectors():
    with pytest.raises(ValueError):
        la.mat(np.array([[1, 2, 3]]))          # (1,3) row -> wrong orientation


def test_matt_rejects_column_vectors():
    with pytest.raises(ValueError):
        la.matt(la.vec(1, 2, 3))               # (3,1) column -> wrong orientation


def test_single_entry_works_in_both():
    one = np.array([[7.0]])
    assert la.mat(one).shape == (1, 1)
    assert la.matt(one).shape == (1, 1)


def test_mat_rejects_mismatched_columns():
    with pytest.raises(ValueError):
        la.mat([1, 2, 3], [4, 5])


def test_mat_rejects_empty_input():
    with pytest.raises(ValueError):
        la.mat()


def test_mat_rejects_list_of_lists():
    with pytest.raises(ValueError):
        la.mat([[1, 2], [3, 4]])


def test_eye():
    assert_close(la.eye(3), np.eye(3))


def test_I_is_eye_alias():
    assert la.I is la.eye
    assert_close(la.I(2), np.eye(2))


def test_diag():
    assert_close(
        la.diag(1, 2, 3),
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float),
    )


def test_e():
    assert_close(la.e(3, 1), np.array([[0], [1], [0]], dtype=float))


# ---- dtype modes ----

def test_dtype_auto_default_is_float():
    assert la.vec(1, 2, 3).dtype == np.float64
    assert la.mat([1, 2], [3, 4]).dtype == np.float64
    assert la.eye(3).dtype == np.float64
    assert la.diag(1, 2).dtype == np.float64


def test_dtype_auto_upcasts_to_complex():
    assert la.vec(1 + 2j, 3).dtype == np.complex128
    assert la.mat([1, 2], [3, 4j]).dtype == np.complex128


def test_dtype_none_defers_to_numpy_inference():
    assert la.vec(1, 2, 3, dtype=None).dtype == np.int64


def test_dtype_explicit_is_forced():
    assert la.vec(1, 2, 3, dtype=int).dtype == np.int64
    assert la.eye(2, dtype=complex).dtype == np.complex128


# =========================
# Core
# =========================

def test_col():
    A = la.mat([1, 2, 3], [4, 5, 6])
    c = la.col(A, 1)
    assert c.shape == (3, 1)
    assert_close(c, la.vec(4, 5, 6))


def test_row():
    A = la.mat([1, 2, 3], [4, 5, 6])
    r = la.row(A, 1)
    assert r.shape == (1, 2)
    assert_close(r, np.array([[2, 5]], dtype=float))


def test_inner():
    u = la.vec(1, 2, 3)
    v = la.vec(4, 5, 6)
    result = la.inner(u, v)
    assert result.shape == (1, 1)
    assert_close(result, np.array([[32.0]]))


def test_dot():
    u = la.vec(1, 2, 3)
    v = la.vec(4, 5, 6)
    result = la.dot(u, v)
    assert isinstance(result, float)
    assert result == 32.0


def test_outer():
    u = la.vec(1, 2)
    v = la.vec(3, 4)
    assert_close(la.outer(u, v), np.array([[3, 4], [6, 8]], dtype=float))


def test_norm():
    assert la.norm(la.vec(3, 4)) == 5.0


def test_norm_matrix_is_frobenius():
    A = la.mat([3, 0], [4, 0])
    assert np.isclose(la.norm(A), 5.0)            # sqrt(9+16)


def test_unit():
    v = la.vec(3, 4)
    result = la.unit(v)
    assert_close(result, la.vec(3 / 5, 4 / 5))
    assert np.isclose(la.norm(result), 1.0)


def test_unit_rejects_matrix():
    with pytest.raises(ValueError):
        la.unit(la.eye(2))


def test_tr():
    assert la.tr(la.mat([1, 3], [2, 4])) == 5


def test_det():
    assert np.isclose(la.det(la.mat([1, 3], [2, 4])), -2.0)


def test_adj_real_is_transpose():
    A = la.mat([1, 2], [3, 4])
    assert_close(la.adj(A), A.T)


def test_adj_complex_is_conjugate_transpose():
    A = la.mat([1 + 1j, 2], [3, 4 - 2j])
    assert_close(la.adj(A), A.conj().T)


# =========================
# Complex correctness
# =========================

def test_dot_is_hermitian_inner_product():
    u = la.vec(1 + 1j, 2 - 1j)
    v = la.vec(2j, 1 + 0j)
    # <u, v> = u* v, conjugate-linear in u, matches np.vdot
    assert np.isclose(la.dot(u, v), np.vdot(u, v))
    # <u, v> = conj(<v, u>)
    assert np.isclose(la.dot(u, v), np.conj(la.dot(v, u)))
    # <u, u> is real and positive
    uu = la.dot(u, u)
    assert np.isclose(uu.imag, 0.0) and uu.real > 0
    assert np.isclose(uu.real, la.norm(u) ** 2)


def test_inner_conjugate_symmetry():
    u = la.vec(1 + 1j, 2 - 1j)
    v = la.vec(2j, 1 + 0j)
    assert_close(la.inner(u, v), la.adj(u) @ v)
    assert_close(la.inner(u, v), la.adj(la.inner(v, u)))


def test_outer_complex_uses_conjugate():
    u = la.vec(1 + 1j, 2)
    v = la.vec(1j, 1)
    assert_close(la.outer(u, v), u @ la.adj(v))


def test_proj_complex_is_orthogonal_projector():
    A = la.mat([1 + 1j, 2], [1, 1 - 1j])
    P = la.proj(A, la.eye(2))
    assert_close(P @ P, P)                  # idempotent
    assert_close(P, la.adj(P))              # self-adjoint (Hermitian)


def test_null_complex():
    B = la.mat([1j, 2j, 3j], [2j, 4j, 6j])  # 3x2, second col = 2 * first
    assert la.rank(B) == 1
    N = la.null(B)
    assert N.shape == (2, 1)                 # nullity = 2 - 1
    assert_close(B @ N, np.zeros((3, 1)))


# =========================
# Spaces
# =========================

def test_rank_full_rank():
    assert la.rank(la.eye(3)) == 3


def test_rank_deficient():
    assert la.rank(la.mat([1, 2, 3], [2, 4, 6])) == 1


def test_independent_true():
    assert la.independent(la.eye(3))


def test_independent_false():
    assert not la.independent(la.mat([1, 2], [2, 4]))


def test_orth():
    A = la.mat([1, 0], [2, 0])
    Q = la.orth(A)
    assert Q.shape == (2, 1)
    assert_close(Q.T @ Q, np.array([[1.0]]))


def test_proj_vector():
    result = la.proj(la.mat([1, 0]), la.vec(3, 4))
    assert_close(result, la.vec(3, 0))


def test_proj_matrix():
    A = la.mat([1, 0, 0], [0, 1, 0])
    result = la.proj(A, la.vec(3, 4, 5))
    assert_close(result, la.vec(3, 4, 0))


def test_null():
    A = la.mat([1, 0], [0, 0])
    N = la.null(A)
    assert N.shape == (2, 1)
    assert_close(A @ N, np.zeros((2, 1)))
    assert_close(N.T @ N, np.array([[1.0]]))


def test_rank_plus_nullity():
    A = la.mat([1, 2, 3], [2, 4, 6])        # 2x3, rank 1
    assert la.rank(A) + la.null(A).shape[1] == A.shape[1]


def test_row_space():
    A = la.matt([1, 2, 3], [2, 4, 6])
    R = la.row_space(A)
    assert R.shape == (3, 1)
    assert_close(R.T @ R, np.array([[1.0]]))


def test_left_null():
    A = la.matt([1, 2], [2, 4], [3, 6])
    L = la.left_null(A)
    assert_close(A.T @ L, np.zeros((2, L.shape[1])))


# =========================
# SVD tolerance (relative default vs absolute)
# =========================

def test_svd_rank_trimmed():
    U, s, Vt = la.svd(la.mat([1, 0], [2, 0]))
    assert len(s) == 1
    assert U.shape == (2, 1)
    assert Vt.shape == (1, 2)


def test_svd_full_matrices():
    U, s, Vt = la.svd(la.mat([1, 0], [2, 0]), full_matrices=True)
    assert U.shape == (2, 2)
    assert Vt.shape == (2, 2)


def test_relative_tolerance_is_scale_invariant():
    A = la.mat([1, 2, 3], [2, 4, 6])        # rank 1
    assert la.rank(A) == la.rank(1e8 * A) == la.rank(1e-9 * A) == 1


def test_absolute_tolerance_can_be_forced():
    # Singular values are 1 and 1e-10.
    M = la.matt([1.0, 0.0], [0.0, 1e-10])
    assert la.rank(M) == 2                   # relative cutoff ~ 4e-16
    assert la.rank(M, tol=1e-8) == 1         # absolute cutoff drops 1e-10


# =========================
# Geometry
# =========================

def test_normalize():
    A = la.mat([3, 4], [5, 12])
    N = la.normalize(A)
    assert np.isclose(la.norm(la.col(N, 0)), 1.0)
    assert np.isclose(la.norm(la.col(N, 1)), 1.0)


def test_normalize_rejects_zero_column():
    with pytest.raises(ValueError):
        la.normalize(la.mat([1, 2], [0, 0]))


def test_dist():
    assert la.dist(la.vec(1, 2), la.vec(4, 6)) == 5.0


def test_angle_orthogonal():
    assert np.isclose(la.angle(la.vec(1, 0), la.vec(0, 1)), np.pi / 2)


def test_angle_clamps_collinear_no_nan():
    u = la.vec(1.0, 2.0, 3.0)
    assert np.isclose(la.angle(u, u), 0.0)          # would be NaN without clamp
    assert np.isclose(la.angle(u, -u), np.pi)


def test_angle_re_is_representation_independent():
    ru, rv = la.vec(1.0, 1.0), la.vec(-1.0, -1.0)
    cu, cv = la.vec(1 + 0j, 1 + 0j), la.vec(-1 + 0j, -1 + 0j)
    # Same vectors, real vs complex storage -> same angle under "re".
    assert np.isclose(la.angle(ru, rv), np.pi)
    assert np.isclose(la.angle(cu, cv), np.pi)


def test_angle_hermitian_identifies_phase():
    u = la.vec(1 + 0j, 0 + 0j)
    iu = 1j * u                                     # same complex line
    assert np.isclose(la.angle(u, iu, kind="hermitian"), 0.0)


def test_angle_branch_switches_on_dtype():
    cu, cv = la.vec(1 + 0j, 1 + 0j), la.vec(-1 + 0j, -1 + 0j)
    ru, rv = la.vec(1.0, 1.0), la.vec(-1.0, -1.0)
    assert np.isclose(la.angle(ru, rv, kind="branch"), np.pi)         # real -> "re"
    assert np.isclose(la.angle(cu, cv, kind="branch"), 0.0, atol=1e-6)  # complex -> hermitian


def test_angle_rejects_unknown_kind():
    with pytest.raises(ValueError):
        la.angle(la.vec(1, 0), la.vec(0, 1), kind="bogus")


# =========================
# Equations
# =========================

def test_solve():
    A = la.matt([2, 0], [0, 3])
    b = la.vec(4, 9)
    assert_close(la.solve(A, b), la.vec(2, 3))


def test_lstsq():
    A = la.matt([1, 1], [1, 2], [1, 3])
    b = la.vec(1, 2, 2)
    x = la.lstsq(A, b)
    assert x.shape == (2, 1)
    r = b - A @ x
    assert_close(A.T @ r, np.zeros((2, 1)))      # residual ⟂ Col(A)


# =========================
# Public namespace (__all__) lock
# =========================

def test_all_has_no_leaks():
    for leaked in ("np", "numbers", "constructors", "core", "spaces",
                   "geometry", "utils", "equations", "decompositions"):
        assert leaked not in la.__all__


def test_all_is_deduplicated():
    assert len(la.__all__) == len(set(la.__all__))


def test_svd_is_public():
    assert "svd" in la.__all__ and hasattr(la, "svd")


def test_I_reachable_but_not_in_star_import():
    # la.I works via attribute access ...
    assert hasattr(la, "I") and la.I is la.eye
    # ... but is not pulled by `from lumpy import *`.
    assert "I" not in la.__all__
    ns = {}
    exec("from lumpy import *", ns)
    assert "eye" in ns and "I" not in ns
