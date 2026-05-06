import numpy as np
from .utils import is_vector
from .core import norm, dot

def dist(u, v):
    """
    Return the Euclidean distance between two column vectors u and v.
    """
    if not(is_vector(u) and is_vector(v)):
        raise ValueError("dist() requires column vectors.")
    return norm(u - v)

def angle(u, v):
    """
    Return the angle in radians between two column vectors u and v.
    """
    if not(is_vector(u) and is_vector(v)):
        raise ValueError("angle() requires column vectors.")
    return np.arccos(dot(u,v)/(norm(u)*norm(v)))

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
    if not is_vector(v):
        raise ValueError("unit() requires a column vector.")
    return normalize(v)