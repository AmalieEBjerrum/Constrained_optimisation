import numpy as np

def primal_active_set_qp(H, g, A, b, x0, initial_active_set):
    """
    Implements the Primal Active-Set QP Algorithm for convex QPs.

    Parameters:
    - H (numpy.ndarray): Positive definite Hessian matrix (n x n)
    - g (numpy.ndarray): Gradient vector (n x 1)
    - A (numpy.ndarray): Constraint matrix (m x n)
    - b (numpy.ndarray): Constraint RHS vector (m x 1)
    - x0 (numpy.ndarray): Initial feasible point (n x 1)
    - initial_active_set (list): List of active constraints indices at x0

    Returns:
    - x (numpy.ndarray): Optimal solution
    - active_set (list): Indices of active constraints at the optimal solution
    """
    x = x0.copy().astype(np.float64)
    m, n = A.shape
    active_set = initial_active_set.copy()

    for iteration in range(100):  # Avoid infinite loops
        print(f"Iteration {iteration}: x = {x}, Active Set = {active_set}")

        # Step 1: Solve equality-constrained QP
        W = A[active_set] if active_set else np.empty((0, n)) # Extracts the rows of A corresponding to active constraints.
        
        # KKT system: [H  -W.T] [p] = [-g - Hx]
        #            [W   0  ] [λ]   [   0   ]
        if active_set:
            KKT_matrix = np.block([[H, -W.T], [W, np.zeros((len(active_set), len(active_set)))]] )
            rhs = np.hstack([-H @ x - g, np.zeros(len(active_set))]) # H @ x: Matrix-vector multiplication, computing Hx
        else:
            KKT_matrix = H
            rhs = -H @ x - g

        try:
            solution = np.linalg.solve(KKT_matrix, rhs)
        except np.linalg.LinAlgError:
            print("Warning: Singular KKT system encountered. Dropping constraint.")
            if active_set:
                dropped_constraint = active_set.pop(0)  # Ensure proper progression
                print(f"Dropped constraint {dropped_constraint}.")
            continue

        p = solution[:n]  # Search direction
        lambdas = solution[n:] if active_set else np.array([])

        # Step 2: Check optimality
        if np.linalg.norm(p) < 1e-6: # If p ≈ 0, the current solution is stationary.
            if len(lambdas) == 0 or all(lambdas >= 0): # all Lagrange multipliers (λ) are non-negative, we have found the optimal solution.
                return x, active_set
            else:
                # Remove constraint with most negative lambda
                min_lambda_index = np.argmin(lambdas)
                removed_constraint = active_set.pop(min_lambda_index)
                print(f" Removing constraint {removed_constraint}.")
        else:
            # Step 3: Compute step length
            alphas = []
            for i in range(m):
                if i not in active_set:
                    ai = A[i]
                    denom = ai @ p # Computes α (step size) for constraints where ai @ p < 0 (approaching the constraint)
                    if denom < 0:
                        alpha_i = (b[i] - ai @ x) / denom
                        alphas.append((alpha_i, i))

            if alphas:
                alpha, new_constraint = min(alphas, key=lambda x: x[0]) # Finds smallest step α that hits a new constraint first.
                if alpha < 1:
                    x += alpha * p
                    if new_constraint not in active_set:
                        active_set.append(new_constraint)
                else:
                    x += p
            else:
                x += p  # No constraints blocking, take full step
    
    print(" Max iterations reached. Terminating.")
    return x, active_set

# Define quadratic cost function parameters
H = np.array([[2, 0], [0, 2]])  # Hessian matrix
g = np.array([-2, -5])          # Gradient vector

# Define constraint matrix and RHS
A = np.array([
    [1, -2],  # Constraint 1: x1 - 2x2 + 2 >= 0
    [-1, -2], # Constraint 2: -x1 - 2x2 + 6 >= 0
    [-1, 2],  # Constraint 3: -x1 + 2x2 + 2 >= 0
    [1, 0],   # Constraint 4: x1 >= 0
    [0, 1]    # Constraint 5: x2 >= 0
])

b = np.array([-2, -6, 2, 0, 0])  # RHS for constraints

# Initial feasible point and active set (following example in book)
x0 = np.array([2, 0])
initial_active_set = [2, 4]  # Constraints 3 and 5 are active at x0

# Run the Active-Set QP solver
optimal_x, final_active_set = primal_active_set_qp(H, g, A, b, x0, initial_active_set)

# Print results
print("\n Final Optimal Solution x*:", optimal_x)
