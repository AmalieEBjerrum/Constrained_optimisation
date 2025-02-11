import numpy as np
import scipy.optimize as opt

def active_set_method(g, A, b):
    """
    Solves the linear program using the Active Set Method.
    """
    # Solve LP using linprog to get the optimal point
    res = opt.linprog(c=g, A_ub=-A, b_ub=-b, method='highs')

    if not res.success:
        print("Linear program did not converge.")
        return None

    x_opt = res.x  # Optimal solution from linprog

    # Identify active constraints
    active_constraints = []
    tolerance = 1e-3
    for i in range(A.shape[0]):
        value = A[i] @ x_opt - b[i]
        if abs(value) < tolerance:  # Constraint is binding
            active_constraints.append(i)

    print("\nActive Constraints at Optimal Solution:", active_constraints)

    # Extract active constraints
    A_active = A[active_constraints]  # Get active rows
    b_active = b[active_constraints]  # Get corresponding b values

    # Solve for x using only active constraints (A_active x = b_active)
    if len(active_constraints) == len(x_opt):  # Fully determined system
        x_active = np.linalg.solve(A_active, b_active)
    else:
        # Least squares if system is overdetermined
        x_active, _, _, _ = np.linalg.lstsq(A_active, b_active, rcond=None)

    print("\nOptimal Solution (Active Set Method):", x_active)

    return x_active


# Example Usage
g = np.array([1, -2])  # Objective function coefficients

A = np.array([[1, 0], 
              [0, 1], 
              [1, -1], 
              [1, -5], 
              [-5, 1]])  # Constraint coefficients (5x2)

b = np.array([0, 0, -2, -20, -15])  # Constraint RHS values

# Compute optimal solution using the Active Set Method
x_active_set = active_set_method(g, A, b)
