import casadi as ca
import time

# Step 1: Define Symbolic Decision Variables
x1 = ca.MX.sym("x1")
x2 = ca.MX.sym("x2")
x = ca.vertcat(x1, x2)  # Stack variables

# Step 2: Define the Objective Function f(x)
t1 = x1**2 + x2 - 11
t2 = x1 + x2**2 - 7
f = t1**2 + t2**2  # Himmelblau's objective function

# Step 3: Define the Constraints g(x)
g1 = (x1 + 2)**2 - x2  # Must be ≥ 0
g2 = -4*x1 + 10*x2  # Must be ≥ 0
g = ca.vertcat(g1, g2)  # Stack constraints

# Step 4: Define NLP Problems
nlp = {"x": x, "f": f, "g": g}  # Standard NLP formulation

# Solver 1: Without Hessian (Gradient only)
solver_no_hessian = ca.nlpsol("solver_no_hessian", "ipopt", nlp, {
    "ipopt.print_level": 0,  # Silence IPOPT output
})

# Solver 2: WITH Hessian
solver_with_hessian = ca.nlpsol("solver_with_hessian", "ipopt", nlp, {
    "ipopt.print_level": 0,  # Silence IPOPT output
    "ipopt.derivative_test": "first-order"  # Ensure IPOPT computes derivatives
})

# Step 5: Solve the Problem (Compare First vs. Second Derivatives)
print("\n🔹 Solving without Hessian (Gradient only)...")
start_time = time.time()
sol_no_hessian = solver_no_hessian(x0=[0, 0], lbg=0, ubg=1e8)
time_no_hessian = time.time() - start_time

print("\n🔹 Solving with Hessian (Gradient + Hessian)...")
start_time = time.time()
sol_with_hessian = solver_with_hessian(x0=[0, 0], lbg=0, ubg=1e8)
time_with_hessian = time.time() - start_time

# Step 6: Display Results
print(" Solution without Hessian:", sol_no_hessian["x"])
print("Time taken (no Hessian):", round(time_no_hessian, 6), "seconds")

print(" Solution with Hessian:", sol_with_hessian["x"])
print("Time taken (with Hessian):", round(time_with_hessian, 6), "seconds")
