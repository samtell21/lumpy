import numpy as np
import lumpy as la


def assert_close(A, B):
    assert np.allclose(A, B)


# =========================
# Constructors
# =========================

def test_vec_is_column_vector():
    v = la.vec(1, 2, 3)

    assert v.shape == (3, 1)

    expected = np.array([
        [1],
        [2],
        [3]
    ], dtype=float)

    assert_close(v, expected)


def test_mat_from_lists():
    A = la.mat(
        [1, 2, 3],
        [4, 5, 6]
    )

    expected = np.array([
        [1, 4],
        [2, 5],
        [3, 6]
    ], dtype=float)

    assert A.shape == (3, 2)

    assert_close(A, expected)


def test_mat_mixed_inputs():
    v = la.vec(1, 2, 3)

    A = la.mat(
        v,
        [4, 5, 6]
    )

    expected = np.array([
        [1, 4],
        [2, 5],
        [3, 6]
    ], dtype=float)

    assert_close(A, expected)

def test_matt_from_rows():
    A = la.matt(
        [1, 2, 3],
        [4, 5, 6]
    )

    expected = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=float)

    assert_close(A, expected)

def test_mat_rejects_mismatched_columns():
    try:
        la.mat([1, 2, 3], [4, 5])

    except ValueError:
        pass

    else:
        raise AssertionError("mat() should reject mismatched column lengths")


def test_mat_rejects_empty_input():
    try:
        la.mat()

    except ValueError:
        pass

    else:
        raise AssertionError("mat() should require at least one column")


def test_eye():
    assert_close(
        la.eye(3),
        np.eye(3)
    )


def test_diag():
    expected = np.array([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ], dtype=float)

    assert_close(
        la.diag(1, 2, 3),
        expected
    )


def test_e():
    expected = np.array([
        [0],
        [1],
        [0]
    ], dtype=float)

    assert_close(
        la.e(3, 1),
        expected
    )


# =========================
# Core
# =========================

def test_col():
    A = la.mat(
        [1, 2, 3],
        [4, 5, 6]
    )

    c = la.col(A, 1)

    assert c.shape == (3, 1)

    assert_close(
        c,
        la.vec(4, 5, 6)
    )


def test_row():
    A = la.mat(
        [1, 2, 3],
        [4, 5, 6]
    )

    r = la.row(A, 1)

    expected = np.array([
        [2, 5]
    ], dtype=float)

    assert r.shape == (1, 2)

    assert_close(r, expected)


def test_inner():
    u = la.vec(1, 2, 3)
    v = la.vec(4, 5, 6)

    result = la.inner(u, v)

    assert result.shape == (1, 1)

    assert_close(
        result,
        np.array([[32.0]])
    )


def test_dot():
    u = la.vec(1, 2, 3)
    v = la.vec(4, 5, 6)

    result = la.dot(u, v)

    assert isinstance(result, float)

    assert result == 32.0


def test_norm():
    v = la.vec(3, 4)

    assert la.norm(v) == 5.0


def test_unit():
    v = la.vec(3, 4)

    result = la.unit(v)

    expected = la.vec(
        3 / 5,
        4 / 5
    )

    assert_close(result, expected)

    assert np.isclose(
        la.norm(result),
        1.0
    )


def test_unit_rejects_matrix():
    A = la.eye(2)

    try:
        la.unit(A)

    except ValueError:
        pass

    else:
        raise AssertionError(
            "unit() should reject matrices"
        )


def test_tr():
    A = la.mat(
        [1, 3],
        [2, 4]
    )

    assert la.tr(A) == 5


def test_det():
    A = la.mat(
        [1, 3],
        [2, 4]
    )

    assert np.isclose(
        la.det(A),
        -2.0
    )


def test_adj():
    A = la.mat(
        [1, 2],
        [3, 4]
    )

    assert_close(
        la.adj(A),
        A.T
    )


# =========================
# Spaces
# =========================

def test_rank_full_rank():
    A = la.eye(3)

    assert la.rank(A) == 3


def test_rank_deficient():
    A = la.mat(
        [1, 2, 3],
        [2, 4, 6]
    )

    assert la.rank(A) == 1


def test_independent_true():
    A = la.eye(3)

    assert la.independent(A)


def test_independent_false():
    A = la.mat(
        [1, 2],
        [2, 4]
    )

    assert not la.independent(A)


def test_orth():
    A = la.mat(
        [1, 0],
        [2, 0]
    )

    Q = la.orth(A)

    assert Q.shape == (2, 1)

    assert_close(
        Q.T @ Q,
        np.array([[1.0]])
    )


def test_proj_vector():
    A = la.mat([1, 0])

    v = la.vec(3, 4)

    result = la.proj(A, v)

    expected = la.vec(3, 0)

    assert_close(result, expected)


def test_proj_matrix():
    A = la.mat(
        [1, 0, 0],
        [0, 1, 0]
    )

    v = la.vec(3, 4, 5)

    result = la.proj(A, v)

    expected = la.vec(3, 4, 0)

    assert_close(result, expected)


def test_null():
    A = la.mat(
        [1, 0],
        [0, 0]
    )

    N = la.null(A)

    assert N.shape == (2, 1)

    assert_close(
        A @ N,
        np.zeros((2, 1))
    )

    assert_close(
        N.T @ N,
        np.array([[1.0]])
    )

def test_row_space():
    A = la.matt(
        [1, 2, 3],
        [2, 4, 6]
    )

    R = la.row_space(A)

    assert R.shape == (3, 1)
    assert_close(R.T @ R, np.array([[1.0]]))


def test_left_null():
    A = la.matt(
        [1, 2],
        [2, 4],
        [3, 6]
    )

    L = la.left_null(A)

    assert_close(A.T @ L, np.zeros((2, L.shape[1])))


# =========================
# Geometry
# =========================

def test_normalize():
    A = la.mat(
        [3, 4],
        [5, 12]
    )

    N = la.normalize(A)

    assert np.isclose(
        la.norm(la.col(N, 0)),
        1.0
    )

    assert np.isclose(
        la.norm(la.col(N, 1)),
        1.0
    )

    assert_close(
        la.proj(
            la.col(A, 0),
            la.col(N, 0)
        ),
        la.col(N, 0)
    )

    assert_close(
        la.proj(
            la.col(A, 1),
            la.col(N, 1)
        ),
        la.col(N, 1)
    )


def test_normalize_rejects_zero_column():
    A = la.mat(
        [1, 2],
        [0, 0]
    )

    try:
        la.normalize(A)

    except ValueError:
        pass

    else:
        raise AssertionError(
            "normalize() should reject zero columns"
        )


def test_dist():
    u = la.vec(1, 2)
    v = la.vec(4, 6)

    assert la.dist(u, v) == 5.0


def test_angle():
    u = la.vec(1, 0)
    v = la.vec(0, 1)

    assert np.isclose(
        la.angle(u, v),
        np.pi / 2
    )


# =========================
# Decompositions
# =========================

def test_svd_rank_trimmed():
    A = la.mat(
        [1, 0],
        [2, 0]
    )

    U, s, Vt = la.svd(A)

    assert len(s) == 1

    assert U.shape == (2, 1)

    assert Vt.shape == (1, 2)


def test_svd_full_matrices():
    A = la.mat(
        [1, 0],
        [2, 0]
    )

    U, s, Vt = la.svd(
        A,
        full_matrices=True
    )

    assert U.shape == (2, 2)

    assert Vt.shape == (2, 2)

# =========================
# Equations
# =========================

def test_solve():
    A = la.matt(
        [2, 0],
        [0, 3]
    )

    b = la.vec(4, 9)

    x = la.solve(A, b)

    assert_close(x, la.vec(2, 3))

def test_lstsq():
    A = la.matt(
        [1, 1],
        [1, 2],
        [1, 3]
    )

    b = la.vec(1, 2, 2)

    x = la.lstsq(A, b)

    assert x.shape == (2, 1)

    # Least-squares residual should be orthogonal to Col(A)
    r = b - A @ x

    assert_close(A.T @ r, np.zeros((2, 1)))