function [x, lambda] = EqualityQPSolverLDLsparse(H, g, A, b)
    % Sparse LDL factorization-based equality constrained QP solver
    n = size(H, 1);
    m = size(A, 2);
    KKT = [H, A; A', sparse(m, m)];
    rhs = -[g; b];
    [L, D, P] = ldl(KKT, 'vector');
    y = L' \ (D \ (L \ rhs(P)));
    sol(P) = y;
    x = sol(1:n);
    lambda = sol(n+1:end);
end
