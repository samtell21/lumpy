import numpy as np

# Shape predicates are deliberately public so callers can check
# Lumpy's conventions; listed explicitly so the boundary is chosen,
# not an accident of `import *`.
__all__ = ["is_vector", "is_column_vector", "is_row_vector", "is_matrix"]

def is_vector(v):
    """check if v is either a column or row vector"""
    return is_column_vector(v) or is_row_vector(v)

def is_column_vector(v):
    """Check if v is a column vector."""
    return v.ndim == 2 and v.shape[1] == 1

def is_row_vector(v):
    """Check if v is a row vector"""
    return v.ndim == 2 and v.shape[0] == 1

def is_matrix(A):
    """Check is A is a matrix (including both column and row vectors)"""
    return A.ndim == 2