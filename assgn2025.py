# function module
"""
    MATH5371 Assignment 2025t1 - Numerical Linear Algebra

    Author: MingYin (z5548164)

    Progress Log:
        - 2025-04-03: Q1
        - 2025-04-05: Q2-Q5
        - 2025-04-06: Q6-Q7
        - 2025-04-08: Q8-Q9
        - 2025-04-14: Q10-Q12
        - 2025-04-21：add comments, simplify the code

"""

import numpy as np
from scipy.linalg import eigh

from cython_wrapper import lapack_dbdsqr


# q1
def get_Householder(x, k):
    """
    Construct Householder vector v and scalar tau so that:
        H = I - tau * v v^T transforms x[k:] to [±||x[k:]||, 0, ..., 0].

    Parameters:
        x : ndarray (m,1) -- input column vector.
        k : int           -- 0-based start index (1 ≤ k ≤ m).

    Returns:
        v : ndarray (m,1) -- Householder vector.
        tau : float       -- scalar coefficient.
    """
    x = x.astype(float)             # ensure float precision
    m = x.shape[0]                  # vector length

    v = np.zeros((m, 1))            # initialize Householder vector

    sigma = 1 if x[k, 0] >= 0 else -1.0  # sign opposite to x[k]
    norm = np.linalg.norm(x[k:, 0])      # ||x[k:]||

    if norm == 0:
        print("\nError: v == 0\n")
        return v, 0.0               # degenerate case: return zero tau

    yk = -sigma * norm              # target value for x[k] after transform
    v[k, 0] = 1                     # v[k] = 1
    v[k+1:, 0] = x[k+1:, 0] / (x[k, 0] - yk)  # v[k+1:] = x[k+1:] / (x[k] - yk)

    tau = 2.0 / (v.T @ v)[0, 0]     # τ = 2 / (vᵀv)

    return v, tau


# q4
def left_multiply(k, v, tau, A):
    """ 实现通用左乘Householder变换 """
    A = A.astype(float)  # 确保浮点运算
    m, n = A.shape

    v = v.reshape(-1, 1)  # 确保v为列向量

    for j in range(n):  # 遍历所有列
        s = np.dot(v[k:, 0], A[k:, j])  # 计算内积v^T*A[:,j]
        t = tau * s
        # 更新A的k行及以下
        A[k, j] -= t * v[k, 0]  # 因v[k_idx]=1，等价于A[k,j] -= t
        A[k + 1:, j] -= t * v[k + 1:, 0]  # 向量化更新
    return A


def left_eliminate(k, v, tau, A):
    """ 针对结构优化后的左乘（仅处理j >=k列） """
    A = A.astype(float)
    m, n = A.shape

    v = v.reshape(-1, 1)

    for j in range(k, n):  # 仅处理列j >=k
        s = np.dot(v[k:, 0], A[k:, j])
        t = tau * s
        A[k, j] -= t * v[k, 0]
        A[k + 1:, j] -= t * v[k + 1:, 0]
    return A



# q6
def right_multiply(k, v, tau, A):
    """
    计算 AH，其中 H 是由 v 和 tau 定义的 Householder 矩阵（0-based索引）
    - k: 0-based索引，表示 v 的前 k 个元素为 0，v[k] = 1
    - v: Householder向量，形状 (n, 1)
    - tau: Householder标量
    - A: 输入矩阵，形状 (m, n)
    """
    A = A.astype(float)
    m, n = A.shape


    # 遍历每一行
    for i in range(m):
        # 计算内积 s = A[i, k:] @ v[k:]
        s = np.dot(A[i, k:], v[k:, 0])
        t = tau * s

        # 更新 A[i, k:] = A[i, k:] - t * v[k:].T
        A[i, k:] -= t * v[k:, 0]  # 单独处理 v[k] = 1
        # A[i, k + 1:] -= t * v[k + 1:, 0]

    return A


def right_eliminate(k, v, tau, A):
    """
    优化版本：仅更新行 i >= k（假设 A 的前 k-1 行在 k+1 列后为 0）
    - k: 0-based索引
    """
    A = A.astype(float)
    m, n = A.shape

    # 仅处理行 i >= k
    for i in range(k, m):
        s = np.dot(A[i, k:], v[k:, 0])
        t = tau * s
        A[i, k:] -= t * v[k:, 0]
        # A[i, k + 1:] -= t * v[k + 1:, 0]
    return A



# Q7
def bidiagonalise(A):
    """
    对任意 m×n 矩阵 A 做双对角分解，构造正交矩阵 P, Q 满足：
        A = P @ B @ Q.T
    其中 B 是上双对角矩阵，P、Q 为正交矩阵。

    参数：
        A : ndarray of shape (m, n)
            原始输入矩阵

    返回：
        P : ndarray of shape (m, m)
            正交矩阵（左乘变换）

        B : ndarray of shape (m, n)
            上双对角矩阵（主对角线和上对角线非零，其余为0）

        QT : ndarray of shape (n, n)
            Q 的转置，即 QT = Q.T
    """
    A = A.astype(float)           # 确保浮点数类型
    B = A.copy()                  # 工作矩阵副本
    m, n = B.shape

    P = np.eye(m)                # 初始化左正交矩阵 P
    Q = np.eye(n)                # 初始化右正交矩阵 Q

    for k in range(n):
        # === 左乘 Householder，消去第k列下方 ===
        if k < n :
            x_col = B[:, [k]]                      # 全列
            v_col, tau_col = get_Householder(x_col, k)  # 从第k位开始消元
            B = left_eliminate(k, v_col, tau_col, B)    # H @ B
            P = left_multiply(k, v_col, tau_col, P)    # H @ P
            # print("\nB=\n",B)

        # === 右乘 Householder，消去第k行右方 ===
        if k < n - 1:
            x_row = B[[k], :].T                         # 取整行后转置为列
            v_row, tau_row = get_Householder(x_row, k + 1)  # 从k+1位开始消元
            B = right_eliminate(k , v_row, tau_row, B)   # B @ H
            Q = right_multiply(k , v_row, tau_row, Q)   # Q @ H
            # print("\nB=\n", B)

    P = P.T    # A = P @ B @ Q^T, 所以返回的是 P.T
    QT = Q.T   # Q 的转置

    return P, B, QT


# q9
def bidiagonal_svd(B):
    """
    对 bidiagonal 矩阵 B 做奇异值分解，方法是：
    - 构造 B^T B
    - 求其特征值和特征向量
    - 计算 U_tilde = 1/σ * B * V_tilde

    参数：
        B: ndarray (m, n) -- 上双对角矩阵

    返回：
        Utilde: ndarray (m, n) -- 左奇异向量组成的矩阵
        sigma:  ndarray (n,)   -- 奇异值（非负，降序）
        VTtilde: ndarray (n, n) -- 右奇异向量的转置
    """
    m, n = B.shape
    BtB = B.T @ B                       # 构造 B^T B
    lambda_, V = eigh(BtB)             # 求解特征值和特征向量

    # 按特征值从大到小排序
    idx = np.argsort(lambda_)[::-1]
    lambda_ = lambda_[idx]
    V = V[:, idx]

    sigma = np.sqrt(lambda_)           # 奇异值

    # 计算 U_tilde = 1/σ * B * V
    Utilde = np.zeros((m, n))
    for j in range(n):
        if sigma[j] > 1e-14:           # 避免除以零
            Utilde[:, j] = (B @ V[:, j]) / sigma[j]
        else:
            Utilde[:, j] = 0

    VTtilde = V.T
    return Utilde, sigma, VTtilde


# q11
def general_svd(A):
    """
    通用奇异值分解：对任意实矩阵 A 进行 SVD 分解：
        A = U @ Σ @ VT

    方法：
        - 先将 A 双对角化：A = P @ B @ Q^T
        - 再对 B 进行 SVD：B = U_tilde @ Σ @ VT_tilde
        - 最后组合：U = P @ U_tilde, VT = VT_tilde @ Q^T

    参数：
        A : ndarray of shape (m, n)
            原始输入矩阵

    返回：
        U : ndarray of shape (m, n)
            左奇异向量组成的矩阵

        sigma : ndarray of shape (n,)
            奇异值（降序）

        VT : ndarray of shape (n, n)
            右奇异向量的转置
    """
    P, B, QT = bidiagonalise(A)                  # Step 1: A = P B QT
    U, sigma, VTtilde = bidiagonal_svd(B)   # Step 2: B = U Σ V^T

    U = P @ U                               # Step 3: U = P @ U_tilde
    VT = VTtilde @ QT                            # Step 4: V^T = V_tilde^T @ Q^T

    return U, sigma, VT


# q12
def lapack_bidiagonal_svd(P, B, QT):
    """
    调用 lapack_dbdsqr (不返回 sigma)，但我们可在 Python 中读取 d 作为奇异值
    """
    m, n = B.shape

    # 构造主对角 d 和上对角 e
    d = np.diag(B).astype(np.float64).flatten()        # shape (n,)
    e = np.diag(B, k=1).astype(np.float64).flatten()   # shape (n-1,)

    # Fortran 顺序矩阵
    Utilde = np.eye(m, n, dtype=np.float64, order='F')
    VTtilde = np.eye(n, dtype=np.float64, order='F')

    info = lapack_dbdsqr(d, e, VTtilde, Utilde)

    if info != 0:
        raise RuntimeError(f"lapack_dbdsqr failed with info={info}")

    sigma = d.copy()   # 现在 d 就是奇异值

    U = P @ Utilde
    VT = VTtilde @ QT

    return U, sigma, VT
