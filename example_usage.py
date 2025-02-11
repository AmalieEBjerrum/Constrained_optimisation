import sympy as sp
import numpy as np
from Lagrangian_function import construct_lagrangian, newtons_method, find_optimal_points, compute_hessian_at_optimum

# Define symbolic variables
x1, x2, x3 = sp.symbols('x1 x2 x3')
x = sp.Matrix([x1, x2, x3])  # Column vector for symbolic computation

# Define Hessian matrix (Quadratic term in the objective function)
H = sp.Matrix([[6, 2, 1], 
               [2, 5, 2], 
               [1, 2, 4]])

# Define linear term in the objective function
g = sp.Matrix([-8, -3, -3])

# Define equality constraint coefficients
A = sp.Matrix([[1, 0, 1], 
               [0, 1, 1]])  # Constraint coefficients

b = sp.Matrix([3, 0])  # Right-hand side of equality constraints

# Define equality constraints in matrix-vector form
constraints_eq = A @ x - b  # Matrix multiplication using @

# Define the quadratic objective function in standard form
f = (1/2) * x.T @ H @ x + g.T @ x  # Quadratic form: 1/2 x' H x + g' x

# Display results
print("Objective function:")
sp.pretty_print(f)

print("\nEquality Constraints:")
sp.pretty_print(constraints_eq)

#use the function construct_lagrangian from Lagrangian_function.py
solution , L , variables = find_optimal_points(f, constraints_eq)
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
