import casadi as ca

# Step 1: Define Symbolic Decision Variables
x1 = ca.MX.sym("x1")
x2 = ca.MX.sym("x2")

# Step 2: Define the Objective Function f(x)
t1 = x1**2 + x2 - 11
t2 = x1 + x2**2 - 7
f = t1**2 + t2**2

# Step 3: Define the Constraints g(x)
c1 = (x1 + 2)**2 - x2   # Constraint 1
c2 = -4*x1 + 10*x2       # Constraint 2
g = ca.vertcat(c1, c2)   # Combine constraints

# Step 4: Declare the NLP (Nonlinear Programming) Problem
nlp = {"x": ca.vertcat(x1, x2), "f": f, "g": g}

# Step 5: Create the NLP Solver Instance using IPOPT
solver = ca.nlpsol("solver", "ipopt", nlp)

# Step 6: Solve the Problem with Initial Guess & Bounds
sol = solver(
    x0=[0.0, 0.0],   # Initial guess
    ubg=1e8,         # Upper bound for constraints
    lbg=0,           # Lower bound for constraints
    lbx=[-5, -5],    # Variable bounds
    ubx=[5, 5]
)

# Step 7: Display Results (Optimal Solution)
print("Optimal solution (x1*, x2*):", sol["x"])
