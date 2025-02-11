from Lagrangian_function import find_optimal_points, compute_hessian_at_optimum
import sympy as sp

# Example Usage:
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Define objective function
f = 3*x1**2 + 2*x1*x2 + x1*x3 + 2.5*x2**2 + 2*x2*x3 + 2*x3**2 - 8*x1 - 3*x2 - 3*x3

# Define equality constraints
constraints_eq = [x1 + x3 - 3, x2 + x3]  # Only equality constraints

# Compute solution
solution, L, variables = find_optimal_points(f, constraints_eq)

if solution is not None:
    # Compute Hessian and eigenvalues at the optimal point
    H_eval, eigenvalues, classification = compute_hessian_at_optimum(L, variables, solution)

    print("\nHessian Matrix Evaluated at Optimal Point:")
    print(H_eval)

    print("\nEigenvalues at Optimal Point:")
    print(eigenvalues)

    print("\nClassification of Stationary Point:")
    print(classification)
else:
    print("\nNewton’s method failed to find an optimal point.")
