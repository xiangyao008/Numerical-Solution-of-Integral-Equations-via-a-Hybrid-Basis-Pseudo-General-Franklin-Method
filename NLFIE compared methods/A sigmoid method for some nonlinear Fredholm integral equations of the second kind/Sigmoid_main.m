%% Sigmoid Method for Nonlinear Fredholm Eq (Performance & Convergence)
% 测试算例: u(x) = sin(pi*x) + int( 1/5 * cos(pi*x) * sin(pi*t) * u(t)^3 )
% 核心逻辑: 循环测速 + 双网格全局收敛阶分析

clc; clear; close all;

%% 1. 问题定义 (Problem Definition)
% u(x) = g(x) + int K(x,t,u) dt
g_x = @(x) sin(pi*x); 
% 真实解
RF = @(x) sin(pi*x) + 1/3.*(20-sqrt(391)).*cos(pi*x);
% 非线性核函数 K(x,t,u)
K_xtu = @(x, t, u) 1/5 .* cos(pi*x) .* sin(pi*t) .* (u.^3);

% --- 算法参数 ---
% 注意: 该方法使用 MATLAB 内置 integral 进行数值积分，速度较慢
N_fine = 50;            % 细网格节点数
N_coarse = N_fine/2;          % 粗网格节点数 (N_fine / 2)
s_param = 10;           % Sigmoid 参数
tol = 1e-8;             % 迭代容差
max_iter = 50;          % 最大迭代步数

vp = 1024;              % 验证点数量
x_dense = linspace(0, 1, vp)'; % [列向量]

num_runs = 5;           % 测速循环次数 (积分较慢，建议设小)

%% 2. 循环测速 (Performance Test for Fine Grid)
fprintf('-------------------------------------------\n');
fprintf('Sigmoid Nonlinear Fredholm Solver (Fine N=%d, Coarse N=%d)\n', N_fine, N_coarse);
fprintf('注意: 使用 Picard 迭代 + 数值积分，计算耗时较长。\n');

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
solve_sigmoid_fredholm(g_x, K_xtu, 0, 1, N_fine, s_param, tol, max_iter);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (N=%d)...\n', num_runs, N_fine);
for i = 1:num_runs
    t_tick = tic;
    
    % --- 求解细网格 ---
    [u_approx_fine_func, iter_fine] = solve_sigmoid_fredholm(g_x, K_xtu, 0, 1, N_fine, s_param, tol, max_iter);
    
    % --- 在稠密网格上评估 ---
    u_vals_fine = u_approx_fine_func(x_dense);
    
    % --- 计算误差向量 ---
    rf_vals = RF(x_dense);
    err_vec_fine = abs(u_vals_fine(:) - rf_vals(:));
    
    time_records(i) = toc(t_tick);
    fprintf('  Run %d/%d: %.4f s (Iter: %d)\n', i, num_runs, time_records(i), iter_fine);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 N=%d 平均耗时: %.6f 秒\n', N_fine, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation
fprintf('正在计算粗网格对照组 (N=%d)...\n', N_coarse);
[u_approx_coarse_func, ~] = solve_sigmoid_fredholm(g_x, K_xtu, 0, 1, N_coarse, s_param, tol, max_iter);

% 评估粗网格误差
u_vals_coarse = u_approx_coarse_func(x_dense);
err_vec_coarse = abs(u_vals_coarse(:) - rf_vals(:));

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免 Sigmoid 基函数在边界处的拟合偏差影响整体统计
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
fprintf('Sigmoid Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (N=%d):\t %.6f 秒\n', N_fine, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', N_coarse, N_fine);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% --- 简单绘图验证 ---
figure('Color', 'w', 'Name', 'Sigmoid Fredholm Analysis');
subplot(2,1,1);
plot(x_dense, rf_vals, 'k-', 'LineWidth', 1.5); hold on;
plot(x_dense, u_vals_fine, 'r--', 'LineWidth', 1.5);
legend('Exact Solution', ['Approx (N=', num2str(N_fine), ')'], 'Location', 'best');
title('Solution Comparison'); grid on; xlabel('x'); ylabel('u(x)');

subplot(2,1,2);
semilogy(x_plot, inner_err_fine, 'b-', 'LineWidth', 1.2);
title('Interior Error Distribution (Log Scale)'); 
grid on; xlabel('x'); ylabel('|Error|');
xlim([0, 1]);

%% ========================================================================
%  核心求解器: Sigmoid Picard Iteration
%  ========================================================================
function [u_func_handle, final_iter] = solve_sigmoid_fredholm(g_x, K_xtu, a, b, N, s, tol, max_iter)
    h = (b - a) / N;
    nodes = linspace(a, b, N + 1)';
    
    % --- 预计算 Sigmoid 基函数参数 ---
    phi_raw = @(t, s_val) 1 ./ (1 + exp(-s_val * t));
    center = h/2;
    val_0 = phi_raw(0 - center, s);
    val_h = phi_raw(h - center, s);
    norm_factor = val_h - val_0;
    
    % 定义上升和下降基函数
    shape_rising  = @(t) (phi_raw(t - center, s) - val_0) ./ norm_factor;
    shape_falling = @(t) (phi_raw(h - t - center, s) - val_0) ./ norm_factor;
    
    % 定义插值函数 (闭包)
    % 该函数接收 t 和 u向量，返回 u(t)
    calc_u_internal = @(t, u_v) interpolate_sigmoid(t, u_v, nodes, h, N, shape_rising, shape_falling);
    
    % --- Picard 迭代 ---
    u_vec = ones(N + 1, 1) * 0.5; % 初始猜测 u0 = 0.5
    final_iter = max_iter;
    
    for k = 1:max_iter
        u_prev = u_vec;
        u_new = zeros(N + 1, 1);
        
        % 当前步的连续解函数
        u_prev_func = @(t) calc_u_internal(t, u_prev);
        
        for i = 1:length(nodes)
            xi = nodes(i);
            % 数值积分: int_a^b K(xi, t, u_prev(t)) dt
            integrand = @(t) K_xtu(xi, t, u_prev_func(t));
            int_val = integral(integrand, a, b, 'ArrayValued', true, 'RelTol', 1e-6); % 适度放宽精度以提速
            u_new(i) = g_x(xi) + int_val;
        end
        
        % 收敛检查
        diff = norm(u_new - u_prev, inf);
        if diff < tol
            u_vec = u_new;
            final_iter = k;
            break;
        end
        u_vec = u_new;
    end
    
    % 返回最终的连续解函数句柄
    u_func_handle = @(t) calc_u_internal(t, u_vec);
end

%% ========================================================================
%  辅助函数: Sigmoid 插值 (向量化)
%  ========================================================================
function val = interpolate_sigmoid(t_points, u_vec, nodes, h, N, rising_func, falling_func)
    val = zeros(size(t_points));
    
    % 简单的逐点处理，对于验证点数量不多时效率尚可
    % 为了支持 integral 函数的 array input，必须能够处理向量输入
    for i = 1:numel(t_points)
        t = t_points(i);
        
        if t <= nodes(1), val(i) = u_vec(1); continue; end
        if t >= nodes(end), val(i) = u_vec(end); continue; end
        
        % 寻找区间索引
        k = floor((t - nodes(1)) / h) + 1;
        if k > N, k = N; end
        
        % 局部坐标
        dt = t - nodes(k);
        
        % 线性组合
        v_left = u_vec(k) * falling_func(dt);
        v_right = u_vec(k+1) * rising_func(dt);
        
        val(i) = v_left + v_right;
    end
end