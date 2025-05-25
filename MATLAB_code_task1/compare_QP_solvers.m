clc; clear;

%% Problem Setup
n = 100;
beta = 0.5;
m = round(beta * n);
density = 0.15;

A = sprandn(n, m, density);
M = sprandn(n, n, density);
alpha = 1e-2;
H = M' * M + alpha * speye(n);  % Make H symmetric positive definite
g = randn(n, 1);
b = randn(m, 1);

%% ----- Sparse LDL Solver -----
KKT_sparse = [H, A; A', sparse(m, m)];
rhs_sparse = -[g; b];

tic;
[Ls, Ds, Ps] = ldl(KKT_sparse, 'vector');
ys = Ls' \ (Ds \ (Ls \ rhs_sparse(Ps)));
sol_sparse = zeros(n + m, 1);
sol_sparse(Ps) = ys;
x_sparse = sol_sparse(1:n);
lambda_sparse = sol_sparse(n+1:end);
t_sparse = toc;

%% ----- Dense LDL Solver -----
KKT_dense = [full(H), full(A); full(A'), zeros(m, m)];
rhs_dense = -[g; b];

tic;
[Ld, Dd, Pd] = ldl(KKT_dense, 'vector');
yd = Ld' \ (Dd \ (Ld \ rhs_dense(Pd)));
sol_dense = zeros(n + m, 1);
sol_dense(Pd) = yd;
x_dense = sol_dense(1:n);
lambda_dense = sol_dense(n+1:end);
t_dense = toc;

%% ----- Comparison -----
res_sparse = norm(A' * x_sparse - b);
res_dense = norm(A' * x_dense - b);
solution_diff = norm(x_sparse - x_dense);

%% ----- Display -----
fprintf('======== QP Solver Comparison ========\n');
fprintf('Sparse LDL Solver:\n  Time: %.4f s  Residual: %.2e\n', t_sparse, res_sparse);
fprintf('Dense  LDL Solver:\n  Time: %.4f s  Residual: %.2e\n', t_dense, res_dense);
fprintf('Difference between solutions: %.2e\n', solution_diff);
fprintf('======================================\n');
