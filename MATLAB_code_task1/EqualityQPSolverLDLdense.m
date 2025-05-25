function [x, lambda] = EqualityQPSolverLDLdense(H, g, A, b)
    % Dense LDL factorization-based QP solver
    n = size(H, 1);
    m = size(A, 2);
    KKT = [full(H), full(A); full(A'), zeros(m, m)];
    rhs = -[g; b];
    [L, D, P] = ldl(KKT, 'vector');
    y = L' \ (D \ (L \ rhs(P)));
    sol(P) = y;
    x = sol(1:n);
    lambda = sol(n+1:end);
end
