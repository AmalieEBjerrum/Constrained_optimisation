#use scipy.io to read the file QP_Test.mat
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from scipy.optimize import linprog
import time
import cvxpy as cp
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
import quadprog


def load_qp_data(filepath):
    data = loadmat(filepath)
    H = data['H']
    g = data['g'].flatten()
    C = data['C']
    l = data['l'].flatten()
    u = data['u'].flatten()
    dl = data['dl'].flatten()
    du = data['du'].flatten()
    return H, g, C, l, u, dl, du

def prepare_qp_inputs( C, l, u, dl, du):
    """
    Prepare the inputs for the quadratic programming problem.
    Includes constraints:
    - l - x >= 0
    - x - u >= 0
    - dl - C^T x >= 0
    - C^T x - du >= 0
    """
    n = C.shape[1]  # Number of variables

    # Identity matrix for variable bounds
    I = np.eye(n)

    # Combine all constraints into C_ineq and d_ineq
    C_ineq = np.vstack([
    -I,    # For x >= l  -> -x <= -l
    I,     # For x <= u
    -C.T,  # For C^T x >= dl -> -C^T x <= -dl
    C.T    # For C^T x <= du
])
    d_ineq = np.hstack([
    -l,     # For x >= l
    u,      # For x <= u
    -dl,    # For C^T x >= dl
    du      # For C^T x <= du
])


    # No equality constraints in this problem
    A = np.empty((0, n))
    b = np.empty((0,))

    return A, b, C_ineq, d_ineq



def solve_qp_with_iterations(H, g, C_ineq, d_ineq):
    """
    Solve the quadratic programming problem using cvxpy and track iterations.
    """
    starttime = time.time()
    # Define the variables
    x = cp.Variable(len(g))

    # Define the objective
    objective = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(H)) + g @ x)


    # Define the constraints
    constraints = [C_ineq @ x <= d_ineq]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    # Custom callback to track iterations
    class Callback:
        def __init__(self):
            self.x_iterates = []
        def __call__(self, info):
            self.x_iterates.append(info['x'])  # Log primal variables

    callback = Callback()
    result = problem.solve(solver=cp.OSQP, verbose=True)  # Use OSQP solver for detailed output
    path = np.array(callback.x_iterates)
    print("Path shape:", path.shape)
    # Get solver statistics
    num_iterations = problem.solver_stats.num_iters
    print(f"Number of iterations 1: {num_iterations}")
    end_time = time.time()
    print(f"Time taken: {end_time - starttime:.4f} seconds")
 
    return x.value, num_iterations, result, starttime, end_time, path


def find_feasible_init_point(C_ineq, d_ineq):
    """
    Find a feasible initial point for the optimization problem.
    """
    n = C_ineq.shape[1]  # Number of variables

    c = np.zeros(n)

    # Solve the linear programming problem
    result_1 = linprog(c, A_ub=C_ineq, b_ub=d_ineq, method='highs')

    if result_1.success:
        x0 = result_1.x
    else:
        raise ValueError("Failed to find a feasible point:", result.message)

    return x0

def objective(x):
        return 0.5 * np.dot(x, H @ x) + np.dot(g, x)

# === Main Script ===
filepath = "QP_Test.mat"  # Replace with your .mat file path
H, g, C, l, u, dl, du = load_qp_data(filepath)
"""
def gradient_descent_qp(H, g, C_ineq, d_ineq, x0, max_iter=100, alpha=0.01):
    
    Simulated projected gradient descent for a QP problem.
    This does NOT use a solver — it's for visualization.
    
    import numpy as np
    from scipy.optimize import linprog

    x = x0.copy()
    path = [x.copy()]

    for _ in range(max_iter):
        grad = H @ x + g
        x = x - alpha * grad

        # Project back to the feasible set using linear programming
        res = linprog(c=np.zeros_like(x),
                      A_ub=C_ineq,
                      b_ub=d_ineq,
                      bounds=[(None, None)] * len(x),
                      method='highs')

        if res.success:
            x = res.x
        path.append(x.copy())

    return x, path
"""


def plot_qp_2d_with_iterations(H, g, x_opt, path):
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure path is 2D (N x 2)
    path = np.array(path).reshape(-1, 2)
    print

    x1 = np.linspace(-10, 110, 200)
    x2 = np.linspace(-10, 110, 200)
    X1, X2 = np.meshgrid(x1, x2)

    Z = 0.5 * (H[0, 0] * X1**2 + 2 * H[0, 1] * X1 * X2 + H[1, 1] * X2**2) + g[0] * X1 + g[1] * X2

    levels = np.linspace(-1500, np.max(Z), 50)

    plt.figure(figsize=(10, 8))
    CS = plt.contour(X1, X2, Z, levels=levels, cmap='plasma')
    cbar = plt.colorbar(CS, label='Objective Value')
    
    # Plot optimization path
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1.5, label='Optimization Path')
    plt.scatter(path[:, 0], path[:, 1], color='red', s=30)

    # Plot optimal solution
    plt.plot(x_opt[0], x_opt[1], 'gs', label='Optimal Solution', markersize=10)

    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title('Quadratic Objective Contour & Iteration Path', fontsize=16)
    plt.legend(frameon=True, fontsize=12, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()




A, b, C_ineq, d_ineq = prepare_qp_inputs(C, l, u, dl, du)
x0 = find_feasible_init_point(C_ineq, d_ineq)
print("Initial guess feasibility:", np.all(np.dot(C_ineq, x0) <= d_ineq))


x_opt, num_iterations, result, starttime, endtime, path = solve_qp_with_iterations(H, g, C_ineq, d_ineq)
print("optimal solution for x_0 and x_1:", x_opt[0], x_opt[1])
print("Objective value at x*:", objective(x_opt))

print("Path shape:", path.shape)
print("H shape:", H.shape)
print("g shape:", g.shape)


#use grtadient descent function to extract path
#x_opt, path = gradient_descent_qp(H, g, C_ineq, d_ineq, x0, max_iter=1000, alpha=0.01)
plot = plot_qp_2d_with_iterations(H, g, x_opt, path)

