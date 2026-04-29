%% Boubaker Method 2 (Performance & Convergence Analysis)
% 针对算例：Exact u(x) = x^2 - 2
% 核心逻辑: 循环测速 + 全局误差收敛阶分析 (N vs N_coarse)

clc; clear; close all;

%% 1. 问题定义
% 精确解 u(x)
RF = @(x) x.^2 - 2;
% 源项 f(x)
fx = @(x) -1/30*x.^6 + 1/3*x.^4 - x.^2 + 5/3*x - 5/4;
% 核函数定义
K_v_base = @(x,t) x - t;  
K_f_base = @(x,t) x + t;  

% --- 算法参数 ---
N_fine = 3;             % 细网格阶数 (Boubaker 多项式最高阶次)
N_coarse = N_fine-1;           % 粗网格阶数
max_iter = 50;
tol = 1e-13;

vp = 1024;              % 验证点数量
x_dense = linspace(0, 1, vp)';
num_runs = 10;          % 测速循环次数

%% 2. 循环测速 (Performance Test for Fine Grid N)
fprintf('-------------------------------------------\n');
fprintf('Boubaker Method Solver (Fine N=%d, Coarse N=%d)\n', N_fine, N_coarse);
fprintf('注意: Boubaker 多项式对于光滑解具有谱收敛特性。\n');

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
solve_boubaker_vfie(fx, K_v_base, K_f_base, N_fine, max_iter, tol);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (N=%d)...\n', num_runs, N_fine);
for i = 1:num_runs
    t_tick = tic;
    
    % --- 求解细网格 ---
    [Y_sol_fine, ~] = solve_boubaker_vfie(fx, K_v_base, K_f_base, N_fine, max_iter, tol);
    
    % --- 在稠密网格上评估 ---
    % 向量化计算基函数矩阵
    B_val_dense = boubaker_eval_matrix(0:N_fine, x_dense);
    y_approx_fine = B_val_dense * Y_sol_fine;
    
    % --- 计算误差向量 ---
    y_exact = RF(x_dense);
    err_vec_fine = abs(y_approx_fine - y_exact);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 N=%d 平均耗时: %.6f 秒\n', N_fine, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation
fprintf('正在计算粗网格对照组 (N=%d)...\n', N_coarse);
[Y_sol_coarse, ~] = solve_boubaker_vfie(fx, K_v_base, K_f_base, N_coarse, max_iter, tol);

% 评估粗网格误差
B_val_dense_c = boubaker_eval_matrix(0:N_coarse, x_dense);
y_approx_coarse = B_val_dense_c * Y_sol_coarse;
err_vec_coarse = abs(y_approx_coarse - y_exact);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
inner_err_fine   = err_vec_fine(2:end-1);
inner_err_coarse = err_vec_coarse(2:end-1);
x_plot = x_dense(2:end-1);

% 2. 计算离散 L-infinity 范数 (内部最大绝对误差)
E_inf_fine   = max(abs(inner_err_fine));
E_inf_coarse = max(abs(inner_err_coarse));

% 3. 计算离散 L2 范数 (使用 RMS 近似)
E_rms_fine   = rms(inner_err_fine);
E_rms_coarse = rms(inner_err_coarse);

% 4. 计算全局收敛阶 (Global Convergence Order)
% Order = log2( ||E_{coarse}|| / ||E_fine|| )
% 注意: 由于谱收敛，此 Order 可能非常大
global_order_inf = log2(E_inf_coarse / E_inf_fine);
global_order_l2  = log2(E_rms_coarse / E_rms_fine);

%% 4. 结果格式化输出
fprintf('\n-------------------------------------------\n');
fprintf('Boubaker Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (N=%d):\t %.6f 秒\n', N_fine, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', N_coarse, N_fine);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% --- 绘图 ---
figure('Color', 'w', 'Name', 'Boubaker Analysis');
subplot(2,1,1);
plot(x_dense, y_exact, 'k-', 'LineWidth', 1.5); hold on;
plot(x_dense, y_approx_fine, 'r--', 'LineWidth', 1.5);
legend('Exact', ['Approx (N=', num2str(N_fine), ')'], 'Location', 'Best');
xlabel('x'); ylabel('u(x)'); title('Solution Comparison'); grid on;

subplot(2,1,2);
semilogy(x_plot, inner_err_fine, 'b-', 'LineWidth', 1.2);
xlabel('x'); ylabel('|Error|');
title('Interior Error Distribution (Log Scale)');
grid on; xlim([0, 1]);

%% ========================================================================
%  核心求解器: Boubaker Newton Iteration (Encapsulated)
%  ========================================================================
function [Y_sol, iter] = solve_boubaker_vfie(fx, K_v_base, K_f_base, N, max_iter, tol)
    num_nodes = N + 1;
    alpha = 1:num_nodes;
    x_nodes = (2*alpha - 1) ./ (2*(N+1));
    x_nodes = x_nodes(:); 
    
    % 初始化与积分矩阵
    B_mat = zeros(num_nodes, num_nodes);
    for i = 1:num_nodes
        B_mat(i, :) = boubaker_eval_matrix(0:N, x_nodes(i));
    end
    
    V_mat = compute_volterra_matrix(K_v_base, x_nodes, N);
    F_mat = compute_fredholm_matrix(K_f_base, x_nodes, N);
    f_vec = fx(x_nodes);
    
    % 初值猜测
    Y_guess = zeros(num_nodes, 1); Y_guess(1) = -2; 
    Theta_guess = zeros(num_nodes, 1); Theta_guess(1) = 4;
    X = [Y_guess; Theta_guess]; 
    
    for iter = 1:max_iter
        Y = X(1:num_nodes);
        Theta = X(num_nodes+1:end);
        
        u_val = B_mat * Y;
        theta_val = B_mat * Theta;
        
        % 残差构建
        Res1 = u_val - f_vec - V_mat * Theta - F_mat * Y;
        Res2 = theta_val - (u_val).^2;
        F_res = [Res1; Res2];
        
        if norm(F_res, inf) < tol
            break;
        end
        
        % Jacobian 构建
        J11 = B_mat - F_mat;
        J12 = -V_mat;
        J21 = -diag(2 * u_val) * B_mat;
        J22 = B_mat;
        Jac = [J11, J12; J21, J22];
        
        X = X - Jac \ F_res;
    end
    Y_sol = X(1:num_nodes);
end

%% ========================================================================
%  辅助函数 (已优化向量化求值)
%  ========================================================================
function B_vals = boubaker_eval_matrix(n_list, x_vec)
    % 向量化计算 Boubaker 多项式矩阵
    M = length(x_vec);
    max_n = max(n_list);
    B_all = zeros(M, max_n + 1);
    
    B_all(:, 1) = 1;
    if max_n >= 1, B_all(:, 2) = x_vec; end
    if max_n >= 2, B_all(:, 3) = x_vec.^2 + 2; end
    
    for k = 3:max_n
        B_all(:, k+1) = x_vec .* B_all(:, k) - B_all(:, k-1);
    end
    B_vals = B_all(:, n_list + 1);
end

function val = boubaker_poly_at_t(n, t)
    t = t(:); 
    mat = boubaker_eval_matrix(n, t);
    val = mat'; 
end

function Mat = compute_volterra_matrix(k_func, x_nodes, N)
    num = length(x_nodes);
    Mat = zeros(num, num);
    for i = 1:num
        x = x_nodes(i);
        if abs(x) < 1e-14, continue; end
        for j = 1:num
            n = j-1;
            fun = @(t) k_func(x, t) .* boubaker_poly_at_t(n, t);
            Mat(i, j) = integral(fun, 0, x, 'AbsTol', 1e-14, 'RelTol', 1e-14);
        end
    end
end

function Mat = compute_fredholm_matrix(k_func, x_nodes, N)
    num = length(x_nodes);
    Mat = zeros(num, num);
    for i = 1:num
        x = x_nodes(i);
        for j = 1:num
            n = j-1;
            fun = @(t) k_func(x, t) .* boubaker_poly_at_t(n, t);
            Mat(i, j) = integral(fun, 0, 1, 'AbsTol', 1e-14, 'RelTol', 1e-14);
        end
    end
end