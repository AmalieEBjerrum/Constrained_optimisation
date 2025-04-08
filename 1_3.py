import numpy as np
from scipy.linalg import lu_factor, lu_solve, ldl
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, spsolve


def EqualityQPSolverLUdense(H, g, A, b):
    n = H.shape[0]
    m = A.shape[0]
    KKT = np.block([
        [H, -A.T],
        [A, np.zeros((m, m))]
    ])
    rhs = np.concatenate([-g, b])
    lu, piv = lu_factor(KKT)
    sol = lu_solve((lu, piv), rhs)
    x = sol[:n]
    lam = sol[n:]
    return x, lam


def EqualityQPSolverLUsparse(H, g, A, b):
    n = H.shape[0]
    m = A.shape[0]
    Hs = csc_matrix(H)
    As = csc_matrix(A)
    KKT = csc_matrix(np.block([
        [Hs.toarray(), -As.T.toarray()],
        [As.toarray(), np.zeros((m, m))]
    ]))
    rhs = np.concatenate([-g, b])
    lu = splu(KKT)
    sol = lu.solve(rhs)
    x = sol[:n]
    lam = sol[n:]
    return x, lam


def EqualityQPSolverLDLdense(H, g, A, b):
    n = H.shape[0]
    m = A.shape[0]
    KKT = np.block([
        [H, -A.T],
        [A, np.zeros((m, m))]
    ])
    rhs = np.concatenate([-g, b])
    L, D, perm = ldl(KKT)
    y = np.linalg.solve(L, rhs[perm])
    z = np.linalg.solve(D, y)
    sol = np.linalg.solve(L.T, z)
    x = sol[:n]
    lam = sol[n:]
    return x, lam


def EqualityQPSolverLDLsparse(H, g, A, b):
    raise NotImplementedError("Sparse LDL not implemented – consider using specialized libraries like scikit-sparse.")


def EqualityQPSolverRangeSpace(H, g, A, b):
    # Range space method
    At = A.T
    S = A @ np.linalg.solve(H, At)
    rhs = b + A @ np.linalg.solve(H, g)
    lam = np.linalg.solve(S, rhs)
    x = -np.linalg.solve(H, g - At @ lam)
    return x, lam


def EqualityQPSolverNullSpace(H, g, A, b):
    # Null space method
    from scipy.linalg import null_space
    Z = null_space(A)
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    Hz = Z.T @ H @ Z
    gz = Z.T @ (H @ x0 + g)
    p = -np.linalg.solve(Hz, gz)
    x = x0 + Z @ p
    lam = np.linalg.lstsq(A.T, H @ x + g, rcond=None)[0]
    return x, lam


def EqualityQPSolver(H, g, A, b, solver):
    solvers = {
        "LUdense": EqualityQPSolverLUdense,
        "LUsparse": EqualityQPSolverLUsparse,
        "LDLdense": EqualityQPSolverLDLdense,
        "LDLsparse": EqualityQPSolverLDLsparse,
        "RangeSpace": EqualityQPSolverRangeSpace,
        "NullSpace": EqualityQPSolverNullSpace
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")
    return solvers[solver](H, g, A, b)
