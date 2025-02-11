import sympy as sp
import numpy as np

def construct_lagrangian(f, constraints_eq=[]):
    """
    Constructs the Lagrangian function for a given function f in R^n with equality constraints.
    """
    # Extract variables from the function
    variables = list(f.free_symbols)

    # Define Lagrange multipliers for equality constraints
    lambdas_eq = [sp.Symbol(f"λ{i+1}") for i in range(len(constraints_eq))]

    # Construct the Lagrangian function
    L = f + sum(lambdas_eq[i] * constraints_eq[i] for i in range(len(constraints_eq)))

    return L, variables, lambdas_eq

def newtons_method(F, J, variables, initial_guess, tol=1e-6, max_iter=100):
    """
    Uses Newton's method to find the roots of the system F = 0.
    """
    # Convert symbolic expressions to numerical functions
    F_func = sp.lambdify(variables, F, "numpy")
    J_func = sp.lambdify(variables, J, "numpy")

    # Initialize solution
    x_k = np.array(initial_guess, dtype=np.float64)

    for i in range(max_iter):
        F_k = np.array(F_func(*x_k), dtype=np.float64).flatten()
        J_k = np.array(J_func(*x_k), dtype=np.float64)

        # Compute Newton step
        try:
            delta_x = np.linalg.solve(J_k, -F_k)
        except np.linalg.LinAlgError:
            print("Jacobian is singular. Newton’s method failed.")
            return None

        # Update solution
        x_k = x_k + delta_x

        # Check convergence
        if np.linalg.norm(delta_x) < tol:
            return x_k

    print("Newton’s method did not converge within the max iteration limit.")
    return None

def find_optimal_points(f, constraints_eq, initial_guess=None):
    """
    Finds the optimal points using Newton’s method.
    """
    # Compute Lagrangian
    L, variables, lambdas_eq = construct_lagrangian(f, constraints_eq)

    # Full variable list (decision variables + Lagrange multipliers)
    all_vars = variables + lambdas_eq

    # Compute the gradient of the Lagrangian
    grad_L = [sp.diff(L, var) for var in all_vars]

    # Compute Jacobian matrix of the gradient system
    J = sp.Matrix(grad_L).jacobian(all_vars)

    # Set a default initial guess if none is provided
    if initial_guess is None:
        initial_guess = [1.0] * len(all_vars)

    # Use Newton’s method to solve the system
    solution = newtons_method(grad_L, J, all_vars, initial_guess)

    return solution, L, variables

def compute_hessian_at_optimum(L, variables, solution):
    """
    Computes the Hessian matrix of the Lagrangian at the optimal point found.

    Parameters:
    - L: Lagrangian function
    - variables: List of decision variables (excluding Lagrange multipliers)
    - solution: The optimal point found using Newton’s method

    Returns:
    - Hessian matrix evaluated at the optimal point
    - Eigenvalues at the optimal point
    - Classification of the stationary point
    """
    # Compute Hessian matrix
    H = sp.Matrix([[sp.diff(L, var1, var2) for var2 in variables] for var1 in variables])

    # Convert Hessian to a numerical function
    H_func = sp.lambdify(variables, H, "numpy")

    # Evaluate Hessian at the optimal solution
    if solution is None:
        print("No solution found, skipping Hessian computation.")
        return None, None, "No solution found"

    # Extract only the variable values (ignore Lagrange multipliers)
    var_solution = solution[:len(variables)]

    # Compute numerical Hessian
    H_eval = np.array(H_func(*var_solution), dtype=np.float64)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(H_eval)

    # Analyze eigenvalues
    positive = np.all(eigenvalues > 0)
    negative = np.all(eigenvalues < 0)

    if positive:
        classification = "Local minimum (all eigenvalues are positive)"
    elif negative:
        classification = "Local maximum (all eigenvalues are negative)"
    else:
        classification = "Saddle point (eigenvalues have mixed signs)"

    return H_eval, eigenvalues, classification


