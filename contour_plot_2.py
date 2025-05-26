import numpy as np
import matplotlib.pyplot as plt

# Define the objective function
def q(x1, x2):
    return (x1 - 1)**2 + (x2 - 2.5)**2

# Create a grid of x1 and x2 values
x1_vals = np.linspace(-1, 4, 200)
x2_vals = np.linspace(-1, 4, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = q(X1, X2)

# Define the constraint functions
def constraint1(x1, x2):
    return x1 - 2*x2 + 2 >= 0

def constraint2(x1, x2):
    return -x1 - 2*x2 + 6 >= 0

def constraint3(x1, x2):
    return -x1 + 2*x2 + 2 >= 0

def constraint4(x1, x2):
    return x1 >= 0

def constraint5(x1, x2):
    return x2 >= 0

# Plot the contour of the objective function
plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, Z, levels=20, cmap='viridis')
plt.colorbar(contours, label='Objective function value')

# Plot the feasible region
x1_vals_dense = np.linspace(-1, 4, 500)
x2_vals_dense = np.linspace(-1, 4, 500)
X1_dense, X2_dense = np.meshgrid(x1_vals_dense, x2_vals_dense)

# Create a mask for the feasible region
feasible_mask = (X1_dense - 2*X2_dense + 2 >= 0) & \
                (-X1_dense - 2*X2_dense + 6 >= 0) & \
                (-X1_dense + 2*X2_dense + 2 >= 0) & \
                (X1_dense >= 0) & \
                (X2_dense >= 0)

plt.contourf(X1_dense, X2_dense, feasible_mask, levels=[0.5, 1], colors='gray', alpha=0.3)

# Labels and formatting
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Contour Plot of Objective Function with Feasible Region")
plt.grid(True)

# Show the plot
plt.show()
