import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from time import time

def primal_dual_ipm_qp(H, g, C, dl, du, l, u, max_iter=500, tol=1e-8):
    n = H.shape[0]
    m = C.shape[1]

    # Reformulate into inequalities A x <= b
    A = np.vstack([
        np.eye(n),          # x[i] <= u[i]  →  ei.T @ x <= u[i]
        -np.eye(n),         # x[i] >= l[i]  →  -ei.T @ x <= -l[i]
        C.T,                # C.T @ x <= du
        -C.T                # -C.T @ x <= -dl
    ])
    b = np.concatenate([u, -l, du, -dl])
    num_ineq = len(b)

    # Initialize strictly feasible guess
    x = np.zeros(n)
    s = np.ones(num_ineq)
    z = np.ones(num_ineq)

    residuals = []
    start = time()

    for k in range(max_iter):
        # Compute residuals
        rL = H @ x + g + A.T @ z
        rC = A @ x + s - b
        rSZ = s * z
        mu = rSZ.sum() / num_ineq

        residuals.append(np.linalg.norm(np.concatenate([rL, rC, rSZ])))

        # Build KKT matrix
        S = np.diag(s)
        Z = np.diag(z)
        inv_S = np.diag(1 / s)

        H_tilde = H + A.T @ (Z @ inv_S) @ A
        rhs = -rL - A.T @ (Z @ inv_S @ rC + Z @ inv_S @ rSZ / s)
        dx = np.linalg.solve(H_tilde, rhs)
        ds = -A @ dx - rC
        dz = (-rSZ - z * ds) / s

        # Step size selection
        alpha = 1.0
        for vec, dvec in zip([s, z], [ds, dz]):
            idx = dvec < 0
            if np.any(idx):
                alpha = min(alpha, 0.99 * np.min(-vec[idx] / dvec[idx]))

        # Update
        x += alpha * dx
        s += alpha * ds
        z += alpha * dz

        if np.linalg.norm(rL) < tol and np.linalg.norm(rC) < tol and np.linalg.norm(rSZ) < tol:
            break

    elapsed = time() - start
    obj_val = 0.5 * x @ H @ x + g @ x
    return x, z, residuals, k + 1, elapsed, obj_val


# === Driver script ===
data = loadmat("QP_Test.mat")
H = data["H"]
g = data["g"].flatten()
C = data["C"]
dl = data["dl"].flatten()
du = data["du"].flatten()
l = data["l"].flatten()
u = data["u"].flatten()

x_star, z_star, residuals, n_iter, elapsed, obj_val = primal_dual_ipm_qp(H, g, C, dl, du, l, u)

print("===== Primal-Dual Interior Point Results =====")
print(f"Iterations: {n_iter}")
print(f"Time elapsed: {elapsed:.4f} sec")
print(f"Objective value: {obj_val:.6f}")

# Plot KKT residuals
plt.figure()
plt.semilogy(residuals, marker='o')
plt.title("KKT Residuals vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("KKT Residual (log scale)")
plt.grid(True)
plt.tight_layout()
plt.show()
