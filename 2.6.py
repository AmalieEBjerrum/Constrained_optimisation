import numpy as np
from scipy.io import loadmat
from time import time

def primal_active_set_qp(H, g, C, dl, du, l, u, tol=1e-8, max_iter=100):
    n = H.shape[0]

    # Initial feasible point
    x = 0.5 * (l + u)

    # Build all constraints: a.T @ x <= b
    A_all = []
    b_all = []

    # Box constraints
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        A_all.append(ei)         # x_i <= u_i
        b_all.append(u[i])
        A_all.append(-ei)        # -x_i <= -l_i
        b_all.append(-l[i])

    # General inequality constraints: C.T x <= du and -C.T x <= -dl
    for i in range(C.shape[1]):
        A_all.append(C[:, i])        # C.T x <= du
        b_all.append(du[i])
        A_all.append(-C[:, i])       # -C.T x <= -dl
        b_all.append(-dl[i])

    A_all = np.array(A_all)
    b_all = np.array(b_all)

    # Initial active set: constraints tight at x
    W = []
    for i in range(len(b_all)):
        if abs(A_all[i] @ x - b_all[i]) < tol:
            W.append(i)

    iteration_count = 0

    for iteration_count in range(1, max_iter + 1):
        # Build KKT system
        A_W = A_all[W, :] if W else np.zeros((0, n))
        KKT_top = np.hstack((H, A_W.T))
        KKT_bottom = np.hstack((A_W, np.zeros((A_W.shape[0], A_W.shape[0]))))
        KKT = np.vstack((KKT_top, KKT_bottom))
        rhs = -np.hstack((H @ x + g, np.zeros(A_W.shape[0])))

        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            raise RuntimeError("KKT system is singular.")

        p = sol[:n]
        lambdas = sol[n:]

        if np.linalg.norm(p) < tol:
            if len(lambdas) == 0 or np.all(lambdas >= -tol):
                return x, W, iteration_count
            idx = np.argmin(lambdas)
            del W[idx]
        else:
            alpha = 1.0
            blocking_constraint = None
            for i in range(len(b_all)):
                if i in W:
                    continue
                a_i = A_all[i]
                aTp = a_i @ p
                if aTp > tol:
                    aiTx = a_i @ x
                    alpha_i = (b_all[i] - aiTx) / aTp
                    if alpha_i < alpha:
                        alpha = alpha_i
                        blocking_constraint = i

            x = x + alpha * p

            if blocking_constraint is not None:
                W.append(blocking_constraint)

    raise RuntimeError("Maximum iterations reached without convergence")

# Load the problem data
data = loadmat("QP_Test.mat")
H = data["H"]
g = data["g"].flatten()
C = data["C"]
dl = data["dl"].flatten()
du = data["du"].flatten()
l = data["l"].flatten()
u = data["u"].flatten()

# Run solver
start = time()
x_star, active_set, num_iters = primal_active_set_qp(H, g, C, dl, du, l, u)
elapsed = time() - start

# Output results
print("\n===== RESULTS =====")
print("Optimal x* =", x_star)
print("Objective value: {:.6f}".format(0.5 * x_star @ H @ x_star + g @ x_star))
print("Active set indices:", active_set)
print(f"Number of constraints in final active set: {len(active_set)}")
print(f"Number of iterations: {num_iters}")
print(f"Time elapsed: {elapsed:.4f} seconds")
