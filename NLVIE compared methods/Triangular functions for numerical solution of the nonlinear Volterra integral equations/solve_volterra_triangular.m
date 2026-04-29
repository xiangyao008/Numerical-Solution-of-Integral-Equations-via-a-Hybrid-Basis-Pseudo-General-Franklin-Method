function [t_vec, x_curr, iter, error_history] = solve_volterra_triangular(...
    f_func, K_func, h_func, a, b, lambda, n, tol, max_iter)
%SOLVE_VOLTERRA_TRIANGULAR Solves nonlinear Hammerstein Volterra IE using TF method.
%
%   Input:
%       f_func, K_func, h_func: Function handles for equation terms.
%       a, b: Integration interval [a, b].
%       lambda: Parameter lambda.
%       n: Number of partitions (step size h = (b-a)/n).
%       tol: Convergence tolerance (epsilon).
%       max_iter: Maximum iterations.
%
%   Output:
%       t_vec: Time nodes.
%       x_curr: Approximate solution vector.
%       iter: Number of iterations performed.
%       error_history: Convergence history.

    % Discretization
    h_step = (b - a) / n;
    t_vec = linspace(a, b, n + 1)'; % Collocation points t_j, size (n+1)x1
    
    % Pre-compute Integration Weight Matrices A and B [cite: 149-153]
    % A(j, i) corresponds to A_{i-1}(t_j) in paper notation
    % B(j, i) corresponds to B_{i-1}(t_j) in paper notation
    % Indices: j = 1..n+1 (nodes), i = 1..n (intervals/triangles)
    
    Mat_A = zeros(n+1, n);
    Mat_B = zeros(n+1, n);
    
    for j = 1:n+1
        tj = t_vec(j);
        for i = 1:n
            % Interval nodes for the i-th triangular function set: [s_i, s_{i+1}]
            % In paper index i=0..n-1. Here i corresponds to interval k.
            s_start = t_vec(i);     % s_i
            s_end   = t_vec(i+1);   % s_{i+1}
            
            % Compute integral of T_i^1 (Left TF) from 0 to tj
            Mat_A(j, i) = integrate_TF(tj, s_start, s_end, h_step, 1);
            
            % Compute integral of T_i^2 (Right TF) from 0 to tj
            Mat_B(j, i) = integrate_TF(tj, s_start, s_end, h_step, 2);
        end
    end
    
    % Initialization: x_0(t) = f(t) 
    x_old = f_func(t_vec);
    error_history = [];
    
    % Iterative Process [cite: 144, 361-363]
    for iter = 1:max_iter
        x_new = zeros(size(x_old));
        
        % Vectorized computation of the sum term in Eq (3.7)
        % Evaluation of nonlinear term h(s, x(s)) at all nodes
        H_val = h_func(t_vec, x_old); 
        
        for j = 1:n+1
            tj = t_vec(j);
            sum_val = 0;
            
            % Summation over i=1 to n (intervals)
            % Eq 3.7: K(t, s_i)*h(s_i)*A + K(t, s_{i+1})*h(s_{i+1})*B
            for i = 1:n
                s_i = t_vec(i);
                s_ip1 = t_vec(i+1);
                
                term1 = K_func(tj, s_i)   * H_val(i)   * Mat_A(j, i);
                term2 = K_func(tj, s_ip1) * H_val(i+1) * Mat_B(j, i);
                
                sum_val = sum_val + (term1 + term2);
            end
            
            x_new(j) = f_func(tj) + lambda * sum_val;
        end
        
        % Check Convergence
        curr_error = max(abs(x_new - x_old));
        error_history = [error_history; curr_error];
        
        if curr_error < tol
            x_curr = x_new;
            return;
        end
        
        x_old = x_new;
    end
    x_curr = x_old;
    warning('Maximum iterations reached without full convergence.');
end

function val = integrate_TF(t_limit, s_start, s_end, h, type)
% INTEGRATE_TF Computes definite integral of triangular function from 0 to t_limit.
% Type 1: Left-handed T^1 (1 at start, 0 at end)
% Type 2: Right-handed T^2 (0 at start, 1 at end)

    if t_limit <= s_start
        val = 0;
        return;
    end
    
    % If integration limit is beyond this interval, full area of triangle
    if t_limit >= s_end
        val = h / 2; % Integral of a triangle with base h and height 1
        return;
    end
    
    % Partial integration inside the interval [s_start, t_limit]
    if type == 1 % LHTF: 1 - (s - ih)/h
        % Antiderivative: s - (s-ih)^2 / (2h)
        % Value at t_limit minus Value at s_start
        val = (t_limit - s_start) - (t_limit - s_start)^2 / (2*h);
    else % RHTF: (s - ih)/h
        % Antiderivative: (s-ih)^2 / (2h)
        val = (t_limit - s_start)^2 / (2*h);
    end
end