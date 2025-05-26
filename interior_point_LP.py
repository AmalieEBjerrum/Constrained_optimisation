import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
import time

def LPippd_bounded_3(g, A, b, l, u, maxit=1000, tol=1e-9):
    # Dimensions of solution
    start = time.time()
    x0=0.5*(l+u)
    m, n = A.shape
    x = x0
    x1_path = []
    x2_path = []
    s_l = x - l
    s_u = u - x
   
    # Hardcoded parameters
    tolL = tol
    tolA = tol
    tols = tol
    eta = 0.99
    
    # Initialize dual variables 
    mu_l = np.ones(n) # Lagrange multipliers for lower bounds
    mu_u = np.ones(n)  # Lagrange multipliers for upper bounds
    lambda_eq = np.zeros(m)       # Lagrange multipliers for equality constraints

    # Compute initial residuals
    rL = g - A.T @ lambda_eq - mu_l + mu_u  # Lagrangian gradient
    rA = A @ x - b                           # Equality Constraint
    rC_l = s_l * mu_l                # Lower bound complementarity
    rC_u = s_u * mu_u                # Upper bound complementarity 
    s = (np.sum(rC_l) + np.sum(rC_u))/n  # Duality gap
    

    # Convergence check
    Converged = (np.linalg.norm(rL, np.inf) <= tolL and
                 np.linalg.norm(rA, np.inf) <= tolA and
                 abs(s) <= tols)
   
    iter = 0
    while not Converged and iter < maxit:
        print(f"Iter {iter}, s = {s:.2e}, ||rL||_inf = {np.linalg.norm(rL, np.inf):.2e}, ||rA||_inf = {np.linalg.norm(rA, np.inf):.2e}")

        iter += 1
        s_l = x - l
        s_u = u - x
        # Form and factorize the Hessian matrix
        D_l = mu_l / s_l
        D_u = mu_u / s_u
        D = D_l + D_u  # Combined diagonal scaling
        H = A @ np.diag(1/D) @ A.T

        
        # Ensure H is positive definite (add small regularization if needed)
        #H = H + 1e-10 * np.eye(m)
        
        try:
            L = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use a more robust (but slower) method
            print("Warning: Switching to robust factorization")
            dmu = np.linalg.solve(H, rhs)
            has_cholesky = False
        else:
            has_cholesky = True
       
        # Affine scaling step
        rthildeL=rL + rC_l / s_l - rC_u / s_u
        rthildeLD= rthildeL / D
        rhs = -rA + A @ rthildeLD
        
        if has_cholesky:
            dlambda = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        else:
            dlambda = np.linalg.solve(H, rhs)
            
        dx =  (A.T @ dlambda)/D - rthildeLD
        dmu_l = -(rC_l + mu_l * dx) / s_l
        dmu_u = -(rC_u - mu_u * dx) / s_u
       
        # Calculate step lengths (ensure variables stay within bounds)
        alpha = min([1.0] + 
                    [-(x[i] - l[i])/dx[i] for i in range(n) if dx[i] < 0] +
                    [(u[i] - x[i])/dx[i]  for i in range(n) if dx[i] > 0])
        
        beta_l = min([1.0] + [-mu_l[i] / dmu_l[i] for i in range(n) if dmu_l[i] < 0])
        beta_u = min([1.0] + [-mu_u[i] / dmu_u[i] for i in range(n) if dmu_u[i] < 0])
        beta = min(beta_l, beta_u)

        # Calculate centering parameter
        x_aff = x + alpha * dx
        mu_l_aff = mu_l + beta * dmu_l
        mu_u_aff = mu_u + beta * dmu_u
        
        rC_l_aff = (x_aff - l) * mu_l_aff
        rC_u_aff = (u - x_aff) * mu_u_aff
        s_aff = (np.sum(rC_l_aff) + np.sum(rC_u_aff)) / n
        
        sigma = (s_aff / s) ** 3 if s > 0 else 0.0
        
        # Corrector and centering step
        tau = sigma * s
        
        # Affine scaling step
        # rC_l = rC_l + (dx - l) * dmu_l - tau
        # rC_u = rC_u + (u - dx) * dmu_u - tau

        rC_l = s_l * mu_l + dx * dmu_l - tau
        rC_u = s_u * mu_u - dx * dmu_u - tau

        rthildeL= rL + rC_l / s_l - rC_u / s_u
        rthildeLD = rthildeL / D
        rhs = -rA + A @ rthildeLD
        
        if has_cholesky:
            dmu = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        else:
            dmu = np.linalg.solve(H, rhs)
            
        dx =  (A.T @ dmu) / D - rthildeLD
        dmu_l = -(rC_l + mu_l * dx) / s_l
        dmu_u = -(rC_u - mu_u * dx) / s_u
        
        # Recalculate step lengths
        alpha = min([1.0] + 
                    [-(x[i] - l[i])/dx[i] for i in range(n) if dx[i] < 0] +
                    [(u[i] - x[i])/dx[i]  for i in range(n) if dx[i] > 0])
        
        beta_l = min([1.0] + [-mu_l[i] / dmu_l[i] for i in range(n) if dmu_l[i] < 0])
        beta_u = min([1.0] + [-mu_u[i] / dmu_u[i] for i in range(n) if dmu_u[i] < 0])
        beta = min(beta_l, beta_u)
       
        # Take step
        x += (eta * alpha) * dx
        # save path for plotting for x_1 and x_2

        x1_path.append(x[0])
        x2_path.append(x[1])
        lambda_eq += (eta * beta) * dlambda
        mu_l += (eta * beta) * dmu_l
        mu_u += (eta * beta) * dmu_u
       
        # Compute residuals and check convergence
        s_l = x - l
        s_u = u - x

        rL = g - A.T @ lambda_eq - mu_l + mu_u 
        rA = A @ x - b                       
        rC_l = s_l * mu_l                
        rC_u = s_u * mu_u 
        s = (np.sum(rC_l) + np.sum(rC_u))/n 
       
        Converged = (np.linalg.norm(rL, np.inf) <= tolL and
                     np.linalg.norm(rA, np.inf) <= tolA and
                     abs(s) <= tols)
   
    info = Converged
    if not Converged:
        
        x, lambda_eq, mu_l, mu_u = None, None, None, None
        
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    return x, info, lambda_eq, mu_l, mu_u, iter, x1_path, x2_path
