%% Numerical solution of Volterra–Fredholm IE via Modified Hat Functions
% 修正版: 修复 basis_mhf 的向量化问题 (解决 integral 报错)
% 核心逻辑: 循环测速 + 全局误差收敛阶分析 (m vs m/2)

clc; clear; close all;

%% 1. 问题定义 (Problem Definition)
% u(x) = f(x) + int_0^x (x-t)u^2 dt + int_0^1 (x+t)u dt
% Exact Solution: f(x) = x^2 - 2

fx_g = @(x) -1/30*x.^6 + 1/3*x.^4 - x.^2 + 5/3*x - 5/4;
RF_exact = @(x) x.^2 - 2;

% Kernels and Nonlinear Terms
k1 = @(x,t) (x - t); U1 = @(u) u.^2; % Volterra
k2 = @(x,t) (x + t); U2 = @(u) u;    % Fredholm

% --- 算法参数 ---
m_fine = 8;        % 细网格节点数 (m)
m_coarse = m_fine/2;      % 粗网格节点数
n_fine_pts = 1024;  % 验证点数量
x_check = linspace(0, 1, n_fine_pts)';
num_runs = 5;       % 测速循环次数

%% 2. 循环测速 (Performance Test for Fine Grid m)
fprintf('-------------------------------------------\n');
fprintf('MHF Solver (Fine m=%d, Coarse m=%d)\n', m_fine, m_coarse);
fprintf('注意: 已修复向量化问题，积分速度将显著提升。\n');

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
solve_MHF_system(m_fine, fx_g, k1, k2, U1, U2);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (m=%d)...\n', num_runs, m_fine);
for i = 1:num_runs
    t_tick = tic;
    
    % --- 求解细网格 ---
    [F_approx_fine, ~] = solve_MHF_system(m_fine, fx_g, k1, k2, U1, U2);
    
    % --- 在稠密网格上重构解 ---
    u_vals_fine = reconstruct_solution(F_approx_fine, m_fine, x_check);
    
    % --- 计算误差向量 ---
    u_true = RF_exact(x_check);
    err_vec_fine = abs(u_vals_fine - u_true);
    
    time_records(i) = toc(t_tick);
    fprintf('  Run %d/%d: %.4f s\n', i, num_runs, time_records(i));
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 m=%d 平均耗时: %.6f 秒\n', m_fine, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation
fprintf('正在计算粗网格对照组 (m=%d)...\n', m_coarse);
[F_approx_coarse, ~] = solve_MHF_system(m_coarse, fx_g, k1, k2, U1, U2);

% 评估粗网格误差
u_vals_coarse = reconstruct_solution(F_approx_coarse, m_coarse, x_check);
err_vec_coarse = abs(u_vals_coarse - u_true);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---
inner_err_fine   = err_vec_fine(2:end-1);
inner_err_coarse = err_vec_coarse(2:end-1);
x_plot = x_check(2:end-1);

% 计算范数
E_inf_fine   = max(abs(inner_err_fine));
E_inf_coarse = max(abs(inner_err_coarse));
E_rms_fine   = rms(inner_err_fine);
E_rms_coarse = rms(inner_err_coarse);

% 计算收敛阶
global_order_inf = log2(E_inf_coarse / E_inf_fine);
global_order_l2  = log2(E_rms_coarse / E_rms_fine);

%% 4. 结果格式化输出
fprintf('\n-------------------------------------------\n');
fprintf('MHF Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (m=%d):\t %.6f 秒\n', m_fine, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (m=%d)\t Error (m=%d)\t Order\n', m_coarse, m_fine);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% --- 绘图 ---
figure('Color', 'w', 'Name', 'MHF Analysis');
subplot(2,1,1);
plot(x_check, u_true, 'k-', 'LineWidth', 1.5); hold on;
plot(x_check, u_vals_fine, 'r--', 'LineWidth', 1.5);
legend('Exact', ['Approx (m=', num2str(m_fine), ')'], 'Location', 'Best');
xlabel('x'); ylabel('u(x)'); title('Solution Comparison'); grid on;

subplot(2,1,2);
semilogy(x_plot, inner_err_fine, 'b-', 'LineWidth', 1.2);
xlabel('x'); ylabel('|Error|');
title('Interior Error Distribution (Log Scale)');
grid on; xlim([0, 1]);

%% ========================================================================
%  核心求解器
%  ========================================================================
function [F_approx, exitflag] = solve_MHF_system(m, fx_g, k1, k2, U1, U2)
    h = 1/m;
    x_nodes = (0:m)' * h; 
    n_nodes = length(x_nodes);
    
    % 1. 生成 MHFs 运算矩阵 P1 和 P2
    [P1, P2] = compute_operational_matrices(m, h, x_nodes);
    
    % 2. 计算 g(x) 向量
    g_vec = fx_g(x_nodes);
    
    % 3. 生成核函数矩阵 K1, K2
    K1_mat = zeros(n_nodes, n_nodes);
    K2_mat = zeros(n_nodes, n_nodes);
    
    % 此处可以使用 meshgrid 进一步向量化，但双重循环通常不是瓶颈
    for i = 1:n_nodes
        for j = 1:n_nodes
            K1_mat(i,j) = k1(x_nodes(i), x_nodes(j));
            K2_mat(i,j) = k2(x_nodes(i), x_nodes(j));
        end
    end
    
    % 4. 求解非线性方程组
    F0 = g_vec; 
    lambda1 = 1; lambda2 = 1; 
    
    sys_func = @(F) compute_residuals(F, g_vec, K1_mat, K2_mat, P1, P2, lambda1, lambda2, U1, U2);
    
    options = optimoptions('fsolve', 'Display', 'off', ...
        'FunctionTolerance', 1e-14, 'StepTolerance', 1e-14, ...
        'MaxIterations', 500, 'Algorithm', 'trust-region-dogleg');
    
    [F_approx, ~, exitflag] = fsolve(sys_func, F0, options);
    
    if exitflag <= 0
        warning('fsolve 在 m=%d 时未收敛', m);
    end
end

function Res = compute_residuals(F, G, K1, K2, P1, P2, lam1, lam2, U1, U2)
    W1 = U1(F); % u^2
    W2 = U2(F); % u
    Term_Fredholm = K2 * (P2 * W2);
    Mat_Volterra = K1 * diag(W1) * P1.'; 
    Term_Volterra = diag(Mat_Volterra);
    Res = F - (G + lam1 * Term_Volterra + lam2 * Term_Fredholm);
end

%% ========================================================================
%  辅助函数：计算运算矩阵 P1, P2
%  ========================================================================
function [P1, P2] = compute_operational_matrices(m, h, x_nodes)
    n = m + 1;
    P1 = zeros(n, n);
    P2 = zeros(n, n);
    
    for j = 1:n 
        idx_j = j - 1;
        % 使用向量化的 basis_mhf 句柄
        fun_h_j = @(x) basis_mhf(idx_j, x, m, h);
        
        % P2: int_0^1 h_i(x) h_j(x) dx
        for i = 1:n
            idx_i = i - 1;
            fun_h_i = @(x) basis_mhf(idx_i, x, m, h);
            % integral 会传入向量 x，现在 fun_h_i 和 fun_h_j 均支持向量
            P2(i, j) = integral(@(x) fun_h_i(x) .* fun_h_j(x), 0, 1, 'AbsTol', 1e-12);
        end
        
        % P1: int_0^{x_i} h_j(y) dy
        for i = 1:n
            xi = x_nodes(i);
            if xi > 1e-14 % 避免对 0 积分
                P1(i, j) = integral(fun_h_j, 0, xi, 'AbsTol', 1e-12);
            end
        end
    end
end

%% ========================================================================
%  辅助函数: 解重构 (Reconstruct Solution)
%  ========================================================================
function f_vals = reconstruct_solution(coeffs, m, x_query)
    h = 1/m;
    f_vals = zeros(size(x_query));
    
    % 这里我们仍使用循环，因为对于重构来说，basis_mhf 可以处理向量输入
    % 但我们需要叠加所有基函数的贡献
    for j = 1:(m+1)
        idx_j = j - 1;
        % basis_mhf 支持向量 x_query，直接叠加
        f_vals = f_vals + coeffs(j) * basis_mhf(idx_j, x_query, m, h);
    end
end

%% ========================================================================
%  【已修复】辅助函数：Modified Hat Function (支持向量化输入)
%  ========================================================================
function val = basis_mhf(i, x, m, h)
    % 输入 x 可以是向量，输出 val 也是同维度的向量
    % 使用逻辑掩码 (Logical Masking) 替代标量 if 判断
    
    val = zeros(size(x)); % 初始化与 x 同维度的零向量
    
    if i == 0
        % 支撑集: [0, 2h]
        mask = (x >= 0) & (x <= 2*h);
        if any(mask)
            xk = x(mask);
            val(mask) = (1/(2*h^2)) .* (xk - h) .* (xk - 2*h);
        end
        
    elseif i == m
        % 支撑集: [1-2h, 1]
        mask = (x >= 1 - 2*h) & (x <= 1);
        if any(mask)
            xk = x(mask);
            val(mask) = (1/(2*h^2)) .* (xk - (1-h)) .* (xk - (1-2*h));
        end
        
    else
        if mod(i, 2) ~= 0 % Odd (奇数索引)
            % 支撑集: [(i-1)h, (i+1)h]
            mask = (x >= (i-1)*h) & (x <= (i+1)*h);
            if any(mask)
                xk = x(mask);
                val(mask) = (-1/(h^2)) .* (xk - (i-1)*h) .* (xk - (i+1)*h);
            end
            
        else % Even (偶数索引)
            % 支撑集: [(i-2)h, (i+2)h]
            % 分两段定义
            
            % 左半段: [(i-2)h, ih]
            mask1 = (x >= (i-2)*h) & (x <= i*h);
            if any(mask1)
                xk = x(mask1);
                val(mask1) = (1/(2*h^2)) .* (xk - (i-1)*h) .* (xk - (i-2)*h);
            end
            
            % 右半段: [ih, (i+2)h]
            % 注意：避免边界点重复赋值，使用 > 而不是 >=
            mask2 = (x > i*h) & (x <= (i+2)*h);
            if any(mask2)
                xk = x(mask2);
                val(mask2) = (1/(2*h^2)) .* (xk - (i+1)*h) .* (xk - (i+2)*h);
            end
        end
    end
end