% main.m (修复版)
% 复现论文: An Effective and Simple Scheme for Solving Nonlinear Fredholm Integral Equations
% 修复点: 修正了误差计算时的维度广播问题，防止生成矩阵

clear; clc; close all;

%% 1. 问题定义 (Example 1)
fprintf('================ Example 1 Results ================\n');
a = 0; b = 1;

% 问题参数
fx = @(x) sin(pi*x);
% 非线性核函数 k(x,t,u)
kf = @(x,t,u) 1/5 .* cos(pi*x) .* sin(pi*t) .* (u.^3);
% 导数核 (若求解器需要 Newton 迭代)
dkut = @(x,t,u) 3/5 .* cos(pi*x) .* sin(pi*t) .* (u.^2);
% 精确解
RF = @(x) sin(pi*x) + 1/3 .* (20 - sqrt(391)) .* cos(pi*x);

% --- 算法参数设置 ---
% 设定 k 值 (对应 Gauss 节点数 N = k + 1)
k_fine = 3;        % Fine Grid (N = 32)
k_coarse = 1;      % Coarse Grid (N = 16)

N_dense = 1024;     % 验证点数量
x_dense = linspace(0, 1, N_dense)'; % [修正] 转为列向量，确保统一

num_runs = 10;      % 测速循环次数

%% 2. 循环测速 (Performance Test for Fine Grid)
fprintf('-------------------------------------------\n');
fprintf('正在准备 Fredholm 测试 (Fine k=%d, Coarse k=%d)...\n', k_fine, k_coarse);

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
solve_fredholm(fx, kf, a, b, k_fine + 1);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (k=%d)...\n', num_runs, k_fine);
for i = 1:num_runs
    t_tick = tic;
    
    % --- 求解细网格 (Fine Grid) ---
    N_fine = k_fine + 1;
    u_approx_fine_func = solve_fredholm(fx, kf, a, b, N_fine);
    
    % --- 在稠密网格上评估 ---
    u_vals_fine = u_approx_fine_func(x_dense);
    
    % --- 计算误差向量 ---
    rf_vals = RF(x_dense);
    
    % [修正] 关键点：强制两边都是列向量 (:)，避免由 (:) - (:)' 产生矩阵
    err_vec_fine = abs(u_vals_fine(:) - rf_vals(:)); 
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 k=%d (N=%d) 平均耗时: %.6f 秒\n', k_fine, N_fine, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation
fprintf('正在计算粗网格对照组 (k=%d)...\n', k_coarse);
N_coarse = k_coarse + 1;
u_approx_coarse_func = solve_fredholm(fx, kf, a, b, N_coarse);

% 评估粗网格误差
u_vals_coarse = u_approx_coarse_func(x_dense);
% [修正] 同样强制列向量相减
err_vec_coarse = abs(u_vals_coarse(:) - rf_vals(:));

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% x_dense(2:end-1) 也是列向量，长度为 1022
inner_err_fine   = err_vec_fine(2:end-1);
inner_err_coarse = err_vec_coarse(2:end-1);
x_plot = x_dense(2:end-1);

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
fprintf('Nonlinear Fredholm Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (k=%d):\t %.6f 秒\n', k_fine, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (k=%d)\t Error (k=%d)\t Order\n', k_coarse, k_fine);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% --- 简单绘图验证 ---
figure('Color', 'w', 'Name', 'Nonlinear Fredholm Analysis');
subplot(2,1,1);
plot(x_dense, rf_vals, 'k-', 'LineWidth', 1.5); hold on;
plot(x_dense, u_vals_fine, 'r--', 'LineWidth', 1.5);
legend('Exact Solution', ['Approx (k=', num2str(k_fine), ')'], 'Location', 'best');
title('Solution Comparison'); grid on; xlabel('x'); ylabel('u(x)');

subplot(2,1,2);
% 此时 x_plot 和 inner_err_fine 都是 (N-2)x1 的列向量，semilogy 不会报错
semilogy(x_plot, inner_err_fine, 'b-', 'LineWidth', 1.2);
title('Interior Error Distribution (Log Scale)'); 
grid on; xlabel('x'); ylabel('|Error|');
xlim([0, 1]);