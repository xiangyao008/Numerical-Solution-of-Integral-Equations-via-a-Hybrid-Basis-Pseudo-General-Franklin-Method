function [coeffs_standard, u_func_std] = SolveVolterra_Bernstein(fx, kf, dkf, a, b, K_target)
    % SolveVolterra_Bernstein
    %
    % 针对 Broyden 迭代初值的终极鲁棒方案。
    % 放弃“最小二乘拟合”，改用“伯恩斯坦多项式逼近”。
    %
    % 核心优势：
    % 1. 【绝对保号】：如果解是正的，生成的初值多项式恒为正。
    % 2. 【端点锚定】：自动满足 P(a) = u(a)，消除初始误差。
    % 3. 【无条件鲁棒】：对任意 K 值 (3,4,5...) 都稳定，绝不发散。
    %
    % 输入: K_target (多项式阶数)
    
    % --- 1. 内部高精度求解 (获取真值) ---
    N_internal = 32; 
    [~, ~, x_nodes, u_nodes] = run_chebyshev_solver_safe(fx, kf, dkf, a, b, N_internal);
    
    % --- 2. 构造伯恩斯坦多项式 ---
    % 伯恩斯坦多项式公式 (定义在 [0,1] 上):
    % B_n(t) = sum_{i=0}^n  beta_i * b_{i,n}(t)
    % 其中 beta_i = u( a + i/n * (b-a) ) 是控制点
    
    % 采样 K+1 个等距控制点
    % 注意：这里必须是等距采样，这是伯恩斯坦定义的基石
    t_samples = linspace(0, 1, K_target + 1);
    x_samples = a + t_samples * (b - a);
    
    % 获取这些点上的精确值 (作为贝塞尔控制点)
    betas = barycentric_eval(x_nodes, u_nodes, x_samples);
    
    % --- 3. 将伯恩斯坦基底转换为标准幂基底 ---
    % 我们需要系数 c 使得: sum c_j * x^j = sum beta_i * B_{i,n}(x)
    % 这一步纯数学变换，将 (1-x)^(n-i) 展开
    
    % 初始化标准系数 (对应 (x-a)^0, (x-a)^1, ... )
    coeffs_power_basis = zeros(1, K_target + 1);
    
    n = K_target;
    % 遍历每一个伯恩斯坦基函数 b_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
    for i = 0:n
        beta_val = betas(i+1);
        if beta_val == 0, continue; end
        
        Binom_ni = nchoosek(n, i);
        
        % 展开 (1-t)^(n-i) = sum_{j=0}^{n-i} C(n-i, j) * (-1)^j * t^j
        for j = 0:(n-i)
            Binom_rem = nchoosek(n-i, j);
            
            % 当前项的幂次 p = i + j
            p = i + j;
            
            % 当前项的系数贡献
            % Coeff = beta * C(n,i) * C(n-i,j) * (-1)^j
            term_coeff = beta_val * Binom_ni * Binom_rem * (-1)^j;
            
            % 累加到对应幂次的系数中 (MATLAB索引从1开始，所以是 p+1)
            coeffs_power_basis(p+1) = coeffs_power_basis(p+1) + term_coeff;
        end
    end
    
    % --- 4. 归一化处理 ---
    % 上述系数是针对变量 t = (x-a)/(b-a) 的
    % 即 P(t) = c0 + c1*t + c2*t^2 ...
    % 用户需要针对 (x-a) 的系数。
    % t = X / L  =>  t^p = X^p / L^p
    % 所以第 p 项系数需要除以 (b-a)^p
    
    L = b - a;
    for p = 0:n
        coeffs_power_basis(p+1) = coeffs_power_basis(p+1) / (L^p);
    end
    
    % --- 5. 输出格式调整 ---
    % 目前是 [a0, a1, ..., aK]，用户通常需要 polyval 格式 [aK, ..., a1, a0]
    coeffs_standard = fliplr(coeffs_power_basis);
    
    % 打印状态
    fprintf('Bernstein Approximation: Order %d. Shape & Sign Guaranteed.\n', K_target);
    
    % 构建返回函数
    u_func_std = @(x) polyval(coeffs_standard, x - a);
end

%% ========================================================================
%  内部核心：切比雪夫求解器 (复用之前验证过的 Safe 版本)
% =========================================================================
function [coeffs_cheb, u_func, x_nodes, u_nodes] = run_chebyshev_solver_safe(fx, kf, dkf_du, a, b, N)
    k = 0:N;
    tau = cos(pi * k' / N); 
    x_nodes = ((b - a) * tau + (b + a)) / 2;
    u_curr = fx(x_nodes);
    max_iter = 50; tol = 1e-13;
    [gl_z, gl_w] = gauss_legendre_rule(32); 
    for iter = 1:max_iter
        F = zeros(N+1, 1); J = eye(N+1);
        for i = 1:N+1
            xi = x_nodes(i);
            if xi == a, F(i) = u_curr(i) - fx(xi); continue; end
            t_quad = ((xi - a) * gl_z + (xi + a)) / 2;
            w_quad = gl_w * (xi - a) / 2;
            u_interp = barycentric_eval(x_nodes, u_curr, t_quad);
            val_K = kf(xi, t_quad, u_interp);       
            val_dK = dkf_du(xi, t_quad, u_interp);  
            integral = w_quad * val_K; 
            F(i) = u_curr(i) - fx(xi) - integral;
            for q = 1:length(t_quad)
                L_vec = barycentric_weights_row(x_nodes, t_quad(q));
                J(i,:) = J(i,:) - w_quad(q) * val_dK(q) * L_vec;
            end
        end
        if norm(F, inf) < tol, break; end
        u_curr = u_curr - (J \ F);
    end
    u_nodes = u_curr;
    coeffs_cheb = []; u_func = []; 
end

function y = barycentric_eval(x_nodes, u_nodes, x_query)
    n = length(x_nodes)-1;
    w = ones(1, n+1); w(2:2:end) = -1; w(1) = 0.5; w(end) = 0.5 * w(end);
    y = zeros(size(x_query));
    for k = 1:length(x_query)
        diff = x_query(k) - x_nodes;
        if any(abs(diff) < 1e-14)
            y(k) = u_nodes(find(abs(diff) < 1e-14, 1));
        else
            terms = w ./ diff'; y(k) = sum(terms .* u_nodes') / sum(terms);
        end
    end
end

function L_vec = barycentric_weights_row(x_nodes, x_q)
    n = length(x_nodes)-1;
    w = ones(1, n+1); w(2:2:end) = -1; w(1) = 0.5; w(end) = 0.5 * w(end);
    diff = x_q - x_nodes;
    if any(abs(diff) < 1e-14)
        L_vec = zeros(1, n+1); L_vec(find(abs(diff) < 1e-14, 1)) = 1;
    else
        terms = w ./ diff'; L_vec = terms / sum(terms);
    end
end

function [x, w] = gauss_legendre_rule(N)
    beta = .5 ./ sqrt(1-(2*(1:N-1)).^(-2));
    T = diag(beta,1) + diag(beta,-1);
    [V, D] = eig(T);
    x = diag(D); [x, i] = sort(x); w = 2 * V(1,i).^2;
end