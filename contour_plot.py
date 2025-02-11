import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def plot_linear_optimization(g, A, b, x_range=(-10, 10), y_range=(-10, 10)):
    """
    Generates a contour plot for the linear optimization problem:
        min_x f(x) = g^T x
        s.t. A^T x - b >= 0
    
    Parameters:
    - g: (1D array) Coefficients of the objective function.
    - A: (2D array) Constraint coefficients (each row is a constraint).
    - b: (1D array) Constraint RHS values.
    - x_range: (tuple) x-axis range for plotting.
    - y_range: (tuple) y-axis range for plotting.
    """

    # Define objective function f(x) = g^T x
    def objective(x):
        g = np.array([1, -2])  # Objective function coefficients
        return g[0] * x[0] + g[1] * x[1]  # Element-wise multiplication
 # Dot product g^T x

    # Grid for contour plot
    x1_vals = np.linspace(x_range[0], x_range[1], 400)
    x2_vals = np.linspace(y_range[0], y_range[1], 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = objective([X1, X2])

    # Solve linear program to find optimal point
    res = opt.linprog(c=g, A_ub=-A, b_ub=-b, method='highs')    

    # Define constraints as functions for plotting
    constraint_funcs = []
    for i in range(A.shape[0]):
        if A[i, 1] != 0:  # Avoid division by zero
            constraint_funcs.append(lambda x1, A=A, b=b, i=i: (b[i] - A[i, 0] * x1) / A[i, 1])

    # Plot contour lines
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X1, X2, Z, levels=30, cmap="coolwarm")
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlim(x_range)
    plt.ylim(y_range)

    # Plot constraint lines
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    for i, func in enumerate(constraint_funcs):
        plt.plot(x_vals, func(x_vals), label=f'Constraint {i+1}', linestyle='dashed', color='black')

    # Shade feasible region
    plt.fill_between(x_vals, np.maximum.reduce([func(x_vals) for func in constraint_funcs]), y_range[1], 
                     color='gray', alpha=0.3, label="Feasible Region")

    # Plot optimal point
    if res.success:
        plt.scatter(res.x[0], res.x[1], color='red', marker='o', s=100, label="Optimal Solution")

    # Labels and legend
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Contour Plot of Linear Objective Function with Constraints")
    plt.legend()
    plt.grid()
    plt.show()


# Example Usage
g = np.array([1, -2])  # Objective function coefficients

A = np.array([[1, 0], 
              [0, 1], 
              [1, -1], 
              [1, -5], 
              [-5, 1]])  # Constraint coefficients

b = np.array([0, 0, -2, -20, -15])  # Constraint RHS values

# Generate contour plot
plot_linear_optimization(g, A, b)
