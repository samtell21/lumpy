import numpy as np

def is_vector(v):
    """Check if v is a column vector."""
    return v.ndim == 2 and v.shape[1] == 1