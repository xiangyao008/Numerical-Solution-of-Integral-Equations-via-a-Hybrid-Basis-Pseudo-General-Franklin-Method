% Project: Verify TF Method Error (Performance & Convergence)
% Method: TF Basis (Piecewise Linear) + Newton-Raphson Solver
% Core Logic: Loop Timing + Global Convergence Order Analysis

clc; clear; close all;

%% 1. 问题定义与参数设置
% --- Solver Parameters ---
n_fine = 32;          % 细网格阶数 (Fine Grid)
n_coarse = n_fine/2;        % 粗网格 (Coarse Grid)
tol = 1e-12;           % Newton solver tolerance
max_iter = 20;

% --- Validation Parameters ---
vp = 1024;             % 验证点数量
t_check = linspace(0, 1, vp)'; % 稠密验证网格 (列向量)
num_runs = 10;         % 测速循环次数

% --- Problem Functions (Stiff Example) ---
% Exact Solution
exact_sol = @(t) ((1+t).*exp(-10.*t)+1).^0.5;
% Source Term
f_func = @(t) ((1+t).*exp(-10.*t)+1).^0.5 + ...
              (1+t).*(1-exp(-10.*t)) + ...
              10.*(1+t).*log(1+t);
% Kernel and Derivative
K_func = @(t, s, u) -10 .* (1+t) ./ (1+s) .* u.^2;
dK_du_func = @(t, s, u) -20 .* (1+t) ./ (1+s) .* u;

%% 2. 循环测速 (Performance Test for Fine Grid)
fprintf('-------------------------------------------\n');
fprintf('TF Method Newton Solver (Fine n=%d, Coarse n=%d)\n', n_fine, n_coarse);

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
solve_tf_newton(n_fine, f_func, K_func, dK_du_func, tol, max_iter);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (n=%d)...\n', num_runs, n_fine);
for i = 1:num_runs
    t_tick = tic;
    
    % --- 求解细网格 ---
    [t_node_fine, x_node_fine, iter_fine] = solve_tf_newton(n_fine, f_func, K_func, dK_du_func, tol, max_iter);
    
    % --- 在稠密网格上评估 (线性插值符合 TF 方法定义) ---
    x_approx_fine = interp1(t_node_fine, x_node_fine, t_check, 'linear');
    
    % --- 计算误差向量 ---
    x_exact_vals = exact_sol(t_check);
    err_vec_fine = abs(x_approx_fine - x_exact_vals);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 n=%d 平均耗时: %.6f 秒\n', n_fine, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation
fprintf('正在计算粗网格对照组 (n=%d)...\n', n_coarse);
[t_node_coarse, x_node_coarse, ~] = solve_tf_newton(n_coarse, f_func, K_func, dK_du_func, tol, max_iter);

% 评估粗网格误差
x_approx_coarse = interp1(t_node_coarse, x_node_coarse, t_check, 'linear');
err_vec_coarse = abs(x_approx_coarse - x_exact_vals);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% Volterra 方程起点误差通常极小，剔除以避免 log(0)
inner_err_fine   = err_vec_fine(2:end-1);
inner_err_coarse = err_vec_coarse(2:end-1);
t_plot = t_check(2:end-1);

% 2. 计算离散 L-infinity 范数 (内部最大绝对误差)
E_inf_fine   = max(abs(inner_err_fine));
E_inf_coarse = max(abs(inner_err_coarse));

% 3. 计算离散 L2 范数 (使用 RMS 近似)
E_rms_fine   = sqrt(mean(inner_err_fine.^2));
E_rms_coarse = sqrt(mean(inner_err_coarse.^2));

% 4. 计算全局收敛阶 (Global Convergence Order)
% Order = log2( ||E_{coarse}|| / ||E_fine|| )
global_order_inf = log2(E_inf_coarse / E_inf_fine);
global_order_l2  = log2(E_rms_coarse / E_rms_fine);

%% 4. 结果格式化输出
fprintf('\n-------------------------------------------\n');
fprintf('TF Method (Newton) 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (n=%d):\t %.6f 秒\n', n_fine, avg_time);
fprintf('Newton 迭代次数:\t %d\n', iter_fine);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (n=%d)\t Error (n=%d)\t Order\n', n_coarse, n_fine);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% --- 绘图 ---
figure('Color', 'w', 'Position', [100, 100, 1000, 400], 'Name', 'TF Method Analysis');

% 子图 1: 解的对比
subplot(1, 2, 1);
plot(t_check, x_exact_vals, 'k-', 'LineWidth', 1.5); hold on;
plot(t_node_fine, x_node_fine, 'ro', 'MarkerSize', 3, 'MarkerFaceColor', 'r');
plot(t_check, x_approx_fine, 'r--', 'LineWidth', 1);
xlabel('t'); ylabel('x(t)');
title(sprintf('Solution (n=%d)', n_fine));
legend('Exact', 'Nodes', 'TF Approx', 'Location', 'Best');
grid on;

% 子图 2: 误差分布
subplot(1, 2, 2);
semilogy(t_plot, inner_err_fine, 'b-', 'LineWidth', 1.2);
xlabel('t'); ylabel('|Error|');
title(['Interior Error Distribution (Log Scale)']);
grid on; xlim([0, 1]);

%% ---------------------------------------------------------
%  Solver Function (Unchanged Core Logic)
%  ---------------------------------------------------------
function [t_vec, x_sol, iter] = solve_tf_newton(n, f_func, K_func, dK_func, tol, max_iter)
    a = 0; b = 1;
    h = (b - a) / n;
    t_vec = linspace(a, b, n + 1)';
    
    x_sol = f_func(t_vec); % Initial guess
    
    for iter = 1:max_iter
        F = zeros(n+1, 1);
        J = eye(n+1);
        
        for i = 1:n+1
            if i == 1
                F(i) = x_sol(i) - f_func(t_vec(i));
                continue;
            end
            
            idx = 1:i;
            % Trapezoidal weights
            w = h * ones(i, 1); w(1) = h/2; w(end) = h/2;
            
            % Evaluation
            t_curr = t_vec(i);
            t_hist = t_vec(idx);
            x_hist = x_sol(idx);
            
            k_val = K_func(t_curr, t_hist, x_hist);
            integral = sum(w .* k_val);
            F(i) = x_sol(i) - f_func(t_curr) - integral;
            
            % Jacobian construction (Lower Triangular)
            dk_val = dK_func(t_curr, t_hist, x_hist);
            J(i, 1:i) = J(i, 1:i) - (w .* dk_val)';
        end
        
        if norm(F, inf) < tol, return; end
        
        % Newton Update
        delta = J \ (-F);
        x_sol = x_sol + delta;
    end
end