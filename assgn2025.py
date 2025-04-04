# function module

"""
MATH5371 Assignment 2025 - Numerical Linear Algebra
Author: MingYin (z5548164)

Module: assgn2025.py
Description:
    Implements all core functions required for Part A and Part B of the assignment,
    including Householder transformations, bidiagonalisation, and SVD.

Progress Log:
    - 2025-04-03: Implemented get_Householder (Q1)
    - 2025-04-04: TODO: Write left_multiply, left_eliminate (Q4)
    - 2025-04-06: TODO: Add bidiagonal_svd (Q9)

To-Do:
    [ ] Q1: get_Householder
    [ ] Q4: Implement left_multiply, left_eliminate
    [ ] Q6: Implement right_multiply, right_eliminate
    [ ] Q7: Combine steps into bidiagonalise
    [ ] Q11: Assemble general_svd using previous functions
    [ ] Q12: Integrate lapack_dbdsqr from cython_wrapper
"""

import numpy as np

# q1
def get_Householder(x, k):

    y = x
    k = k - 1

    # σ = sign(x_k)
    sigma = 1 if x[k] >= 0 else -1.0

    # x[k:m]
    x_k_m = x[k:]




    return v, tau