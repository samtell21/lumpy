import numpy as np
from .utils import is_column_vector
from .core import norm, dot

__all__ = ["dist", "angle", "normalize", "unit"]

def dist(u, v):
    """
    Return the Euclidean distance between two column vectors u and v.
    """
    if not(is_column_vector(u) and is_column_vector(v)):
        raise ValueError("dist() requires column vectors.")
    return norm(u - v)

def angle(u, v, kind="re"):
    """
    Return the angle in radians between two column vectors u and v.

    For complex vectors ⟨u, v⟩ is complex, so a convention is needed.
    All three agree for real input (where Re is the identity):

      "re"        arccos(Re⟨u,v⟩ / (‖u‖‖v‖)), the angle in the
                  underlying real inner-product space (ℂⁿ ≅ ℝ²ⁿ).
                  Range [0, π]. Depends only on the values, never on
                  whether they are stored real or complex. (default)
      "hermitian" arccos(|⟨u,v⟩| / (‖u‖‖v‖)), the angle between the
                  complex lines (identifies u with e^{iθ}u).
                  Range [0, π/2].
      "branch"    "re" for real input, "hermitian" for complex input.
    """
    if not(is_column_vector(u) and is_column_vector(v)):
        raise ValueError("angle() requires column vectors.")

    ip = dot(u, v)
    denom = norm(u) * norm(v)

    if kind == "re":
        cos = np.real(ip) / denom
    elif kind == "hermitian":
        cos = np.abs(ip) / denom
    elif kind == "branch":
        is_complex = np.iscomplexobj(u) or np.iscomplexobj(v)
        cos = (np.abs(ip) if is_complex else np.real(ip)) / denom
    else:
        raise ValueError(
            f"angle(): kind must be 're', 'hermitian', or 'branch', got {kind!r}."
        )

    # Float error can push the cosine just outside [-1, 1] for
    # (anti)collinear vectors, making arccos return NaN. Clamp.
    return np.arccos(np.clip(cos, -1.0, 1.0))

def normalize(A):
    """
    Normalize each column of A.
    """
    norms = np.linalg.norm(A, axis=0)

    if np.any(norms == 0):
        raise ValueError("Cannot normalize a matrix with zero columns.")

    return A / norms

def unit(v):
    """Return unit vector in direction of v."""
    if not is_column_vector(v):
        raise ValueError("unit() requires a column vector.")
    return normalize(v)