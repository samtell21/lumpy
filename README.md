# Lumpy

**Very elegant. Very Lumpy.**

Lumpy is a tiny semantic linear algebra wrapper over NumPy.

NumPy is excellent, but its core abstraction is the n-dimensional array, not the mathematical vector or matrix. Lumpy uses NumPy underneath while enforcing a more linear-algebraic convention:

- vectors are column vectors
- matrices are built from columns
- rows are transposes or row slices
- projections are onto column spaces
- subspace tools return bases as columns

## Installation

From the project root:

```bash
python -m pip install -e .
```

## Basic usage 

```python
import lumpy as la
```

## Vectors 

```python
v = la.vec(1, 2, 3)

v.shape
# (3, 1)
```

Vectors are always column vectors. 

$$
\mathbf{v}\in \mathbb{R}^{n\times 1}
$$

```python
v.T
# array([[1., 2., 3.]])
```

$$
\mathbf{v}^{T}\in \mathbb{R}^{1\times n}
$$

## Matrices are built from columns

```python 
A = la.mat(
    [1, 2, 3],
    [4, 5, 6]
)

A
# array([[1., 4.],
#        [2., 5.],
#        [3., 6.]])
```

You can also pass existing column vectors:

```python 
u = la.vec(1, 2, 3)
v = la.vec(4, 5, 6)

A = la.mat(u, v) 
```

## Matrices from rows

`mat()` constructs matrices from columns. Sometimes, though, it is useful to type a matrix in the same visual layout used on paper.

```python
A = la.matt(
    [1, 2, 3],
    [4, 5, 6]
)

A
# array([[1., 2., 3.],
#        [4., 5., 6.]])
```

```matt()``` preserves the usual visual layout of matrices, while ```mat()``` preserves Lumpy’s column-oriented semantics.

## Standard basis vectors 

Lumpy uses Python-style zero-indexing.

```python
la.e(3, 0)
# array([[1.],
#        [0.],
#        [0.]])
```

So ```la.e(n, i)``` corresponds to the mathematical basis vector $\mathbf{e}_{i+1}\in \mathbb{R}^{n}$ .

## Basic operations

```python
u = la.vec(1, 2, 3)
v = la.vec(4, 5, 6)

la.inner(u, v)
# array([[32.]])

la.dot(u, v)
# 32.0

la.norm(v)
# 8.774964387392123
```

```inner``` preserves matrix structure. 

$$
\langle \mathbf{u},\mathbf{v} \rangle = [\mathbf{u}\cdot \mathbf{v}]\in \mathbb{R}^{1\times 1}
$$

```dot``` returns a scalar. 

$$
\mathbf{u} \cdot  \mathbf{v} \in \mathbb{R}
$$

## Unit vectors and normalization 

```python 
v = la.vec(3, 4)

la.unit(v)
# array([[0.6],
#        [0.8]])
```

For matrices, ```normalize``` normalizes each column:

```python 
A = la.mat(
    [3, 4],
    [5, 12]
)

la.normalize(A)
# array([[0.6       , 0.38461538],
#        [0.8       , 0.92307692]])
```

## Columns and rows 

```python 
A = la.mat(
    [1, 2, 3],
    [4, 5, 6]
)

la.col(A, 0)
# array([[1.],
#        [2.],
#        [3.]])

la.row(A, 1)
# array([[2., 5.]])
```

```col``` preserves column-vector shape.

## Projection 

Project onto the column space of a matrix:

```python
A = la.mat(
    [1, 0, 0],
    [0, 1, 0]
)

v = la.vec(3, 4, 5)

la.proj(A, v)
# array([[3.],
#        [4.],
#        [0.]])
```

This projects $\mathbf{v}$  onto the $xy$-plane.

Vector projection is just the rank-one case:

```python
u = la.vec(1, 1, 0)
v = la.vec(2, 3, 4)

la.proj(u, v)
```

## Subspaces

```python
A = la.mat(
    [1, 2],
    [2, 4]
)

la.rank(A)
# 1

la.independent(A)
# False
```

An orthonormal basis for the column space:

```python 
Q = la.orth(A)

Q.T @ Q
# approximately identity
```

Nullspace basis:

```python 
N = la.null(A)

A @ N
# approximately zero
```

Lumpy returns subspace bases as columns.

```python

A = la.matt(
    [1, 2, 3],
    [2, 4, 6]
)

la.row_space(A)
```

The left nullspace is the nullspace of $A^{T}$:

$$
\operatorname{Null}\left(A^{T}\right)
$$

```python
la.left_null(A)
```

## SVD 

By default, Lumpy's SVD returns only rank-relevant singular directions:

```python 
U, s, Vt = la.svd(A)
```

For full ambient bases:

```python 
U, s, Vt = la.svd(A, full_matrices=True)
```

Many Lumpy subspace functions use a numerical tolerance, defaulting to `tol=1e-12`, to decide which singular values count as nonzero.

## Geometry 

```python 
u = la.vec(1, 0)
v = la.vec(0, 1)

la.angle(u, v)
# 1.5707963267948966

la.dist(u, v)
# 1.4142135623730951
```

## Solving systems

Prefer `solve()` over explicitly computing an inverse.

```python

A = la.matt(
    [2, 0],
    [0, 3]
)

b = la.vec(4, 9)
la.solve(A, b)

# array([[2.],
#        [3.]])
```

For overdetermined systems, use least squares:

```python
A = la.matt(
    [1, 1],
    [1, 2],
    [1, 3]
)

b = la.vec(1, 2, 2)

la.lstsq(A, b)
```

## Running tests 

```bash 
python -m pytest
```

## Design principle 

Lumpy is not trying to replace NumPy. 

It is a small semantic layer for doing linear algebra with conventions that feel closer to the math.

```python 
import numpy as np
import lumpy as la
```

Use NumPy for general arrays.  
Use Lumpy when you want vectors, matrices, projections, and subspaces to behave like linear algebra objects.