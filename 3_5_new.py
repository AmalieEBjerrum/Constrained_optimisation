import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
from interior_point_LP import LPippd_bounded_3

# === Load Data ===
def load_lp_data(filepath):
    data = loadmat(filepath)
    g_2 = data['C'].flatten()
    g_1 = data['U'].flatten()
    g = np.hstack((g_1, (-1)*g_2))
    u_1 = data['Pd_max'].flatten()
    u_2 = data['Pg_max'].flatten()
    u = np.hstack((u_1, u_2))
    A_1 = np.ones(u_1.shape[0])
    A_2 = -np.ones(u_2.shape[0])
    l = np.zeros_like(g)
    A = np.hstack((A_1, A_2)).reshape(1, -1)

    return g, u, A, l






# Load and prepare problem
filepath = "LP_Test.mat"
g, u, A_raw, l = load_lp_data(filepath)
print('A_raw shape:', A_raw.shape)
b = np.zeros(1)

#print(A.shape, b.shape, u.shape, x0.shape, l.shape)
x, info, mu, lambda_l, lambda_u, iter, x1, x2 = LPippd_bounded_3(-g, A_raw, b, l, u)

print("Optimal solution x*:", x)
print("Objective value gᵀx*:", -g @ x)
print("Number of iterations:", iter)
print("Dual variable λ* for upper bounds:", lambda_u)
print("Dual variable λ* for lower bounds:", lambda_l)
print("dual variable mu", mu)

#plot x1 and x2
import matplotlib.pyplot as plt
for i in range(len(x1)):
        plt.plot(x1, x2, marker='o', linestyle='-', color='blue')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Solution Path')
plt.grid()
plt.show()



from scipy.optimize import linprog

"""
# Solve with CVXPY for verification
import cvxpy as cp
x_cp = cp.Variable(len(g))
prob = cp.Problem(cp.Maximize(g @ x_cp),
                 [A_raw @ x_cp == 0,
                  0 <= x_cp, x_cp <= u])
prob.solve()
print("CVXPY solution:", x_cp.value)
print("CVXPY objective value:", prob.value)
print("CVXPY status:", prob.status)
# Access dual variables
print("Dual variables for equality constraints (A_eq @ x_cp == b_eq):", prob.constraints[0].dual_value)
print("Dual variables for inequality constraints (0 <= x_cp):", prob.constraints[1].dual_value)
print("Dual variables for inequality constraints (x_cp <= u):", prob.constraints[2].dual_value)
"""