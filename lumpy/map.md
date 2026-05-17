lumpy/
├── __init__.py
├── constructors.py
├── core.py
├── spaces.py
├── geometry.py
├── decompositions.py
└── equations.py

# constructors.py
mat
matt
vec
eye   # I = eye is a paper-like alias, not in __all__ (not pulled by `import *`)
e 
diag 

# core.py
col
row
outer
inner
dot
norm
tr
adj

# spaces.py
rank
orth
null
proj
independent

# geometry.py
dist
angle
normalize
unit

# decompositions.py
svd

# equations.py
solve
lstsq

# from numpy.linalg
qr
eig
inv
pinv
det