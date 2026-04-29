% 严格复现论文 Example 2 (性能测试与收敛阶分析版)
% 核心逻辑: N=48 (Fine) vs N=24 (Coarse), Sigma=6, Mu=4
% 包含全局误差范数 (L-inf, L2) 及收敛阶计算

clc; clear; close all;

%% 1. 参数设置
a = 0; b = 1;
lambda = 1;

% 真实解与核函数
fx = @(x) exp(x);
kf = @(x,t) 2 .* exp(x + t);
RF = @(x) exp(x) ./ (2 - exp(2)); 

% 算法参数 (Fine Grid)
N = 12;       
sigma = 6;    
mu = 4;
vp = 1024;    % 验证点数量
num_runs = 5; % 测速循环次数 (注意：integral 函数较慢，建议设小一点)

%% 2. 循环测速 (Performance Test for Fine Grid N)
fprintf('-------------------------------------------\n');
fprintf('Shepard Method Example 2 (Fine N=%d, Coarse N=%d)\n', N, N/2);
fprintf('注意: 使用 integral 进行高精度积分，构建矩阵速度较慢。\n');

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
[~, ~, ~] = solve_IE_Shepard(fx, kf, RF, a, b, N, sigma, mu, vp, lambda);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (N=%d)...\n', num_runs, N);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格 (Fine Grid)
    % err_vec_fine 包含了在 1024 个点上的逐点误差
    [~, ~, err_vec_fine] = solve_IE_Shepard(fx, kf, RF, a, b, N, sigma, mu, vp, lambda);
    
    time_records(i) = toc(t_tick);
    fprintf('  Run %d/%d: %.4f s\n', i, num_runs, time_records(i));
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 N=%d 平均耗时: %.6f 秒\n', N, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (N/2)
% N_coarse = 24, Sigma=6 (24/6=4, 可整除)
fprintf('正在计算粗网格对照组 (N=%d)...\n', N/2);
[~, ~, err_vec_coarse] = solve_IE_Shepard(fx, kf, RF, a, b, N/2, sigma, mu, vp, lambda);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免边界奇异性对 Shepard 插值结果的干扰
inner_err_fine   = err_vec_fine(2:end-1);
inner_err_coarse = err_vec_coarse(2:end-1);

% 2. 计算离散 L-infinity 范数 (内部最大绝对误差)
E_inf_fine   = max(abs(inner_err_fine));
E_inf_coarse = max(abs(inner_err_coarse));

% 3. 计算离散 L2 范数 (使用 RMS 近似)
E_rms_fine   = rms(inner_err_fine);
E_rms_coarse = rms(inner_err_coarse);

% 4. 计算全局收敛阶 (Global Convergence Order)
% Order = log2( ||E_{N/2}|| / ||E_N|| )
global_order_inf = log2(E_inf_coarse / E_inf_fine);
global_order_l2  = log2(E_rms_coarse / E_rms_fine);

%% 4. 结果格式化展示
fprintf('\n-------------------------------------------\n');
fprintf('Shepard Operators Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (N=%d):\t %.6f 秒\n', N, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('参数设置:\t\t Sigma=%d, Mu=%d\n', sigma, mu);
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', N/2, N);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% 绘图
x_test = linspace(a, b, vp)';
% 去除边界用于绘图，避免坐标轴被极值撑开
x_plot = x_test(2:end-1);
figure('Color', 'w', 'Name', 'Shepard Error Analysis');
semilogy(x_plot, inner_err_fine, 'b-', 'LineWidth', 1.5, 'DisplayName', ['N=' num2str(N)]);
hold on;
semilogy(x_plot, inner_err_coarse, 'r--', 'LineWidth', 1.5, 'DisplayName', ['N=' num2str(N/2)]);
title(['Error Distribution (Example 2, \mu=' num2str(mu) ')']);
xlabel('x'); ylabel('Absolute Error (Interior)');
legend('Location','best'); grid on;

%% --- 封装后的核心求解器 ---
function [phi_approx, x_test, error_vec] = solve_IE_Shepard(fx, kf, RF, a, b, N, sigma, mu, vp, lambda)
    % 核心求解流程封装
    
    nodes = linspace(a, b, N)';
    
    % 1. 构建矩阵 K_N
    % 使用 integral 函数，速度较慢但精度高
    K_N = zeros(N, N);
    
    % 辅助函数句柄
    B_func = @(s, j_idx) evaluate_basis_single_idx(s, j_idx, nodes, sigma, mu);
    
    for i = 1:N
        for j = 1:N
            % 被积函数: K(xi, s) * B_j(s)
            integrand = @(s) kf(nodes(i), s) .* B_func(s, j);
            % 高精度积分
            K_N(i, j) = integral(integrand, a, b, 'AbsTol', 1e-13, 'RelTol', 1e-13);
        end
    end
    
    % 2. 求解线性方程组
    F_vec = fx(nodes);
    I_mat = eye(N);
    alpha = (I_mat - lambda * K_N) \ F_vec;
    
    % 3. 验证与误差计算
    x_test = linspace(a, b, vp)';
    phi_exact = RF(x_test);
    
    % 计算近似解: phi_N(x)
    B_test_matrix = evaluate_shepard_basis_all(x_test, nodes, sigma, mu);
    phi_approx = B_test_matrix * alpha;
    
    % 误差向量
    error_vec = abs(phi_exact - phi_approx);
end

%% --- 核心算法辅助函数 (保持不变) ---
function val = evaluate_basis_single_idx(s, target_j, nodes, sigma, mu)
    original_size = size(s); 
    s_col = s(:);            
    B_all = evaluate_shepard_basis_all(s_col, nodes, sigma, mu);
    val_col = B_all(:, target_j);
    val = reshape(val_col, original_size); 
end

function [B_matrix] = evaluate_shepard_basis_all(eval_pts, nodes, sigma, mu)
    M = length(eval_pts);
    N = length(nodes);
    m = N / sigma; 
    
    if mod(N, sigma) ~= 0
        error('N must be divisible by sigma (N=%d, sigma=%d)', N, sigma);
    end
    group_nodes = reshape(nodes, sigma, m);
    
    dists = abs(eval_pts - reshape(group_nodes, [1, sigma, m]));
    prod_dists = squeeze(prod(dists, 2)); 
    
    weights = zeros(M, m);
    tol = 1e-15;
    is_node = prod_dists < tol;
    
    inv_dist_pow = prod_dists .^ (-mu); 
    inv_dist_pow(is_node) = 0; 
    sum_inv = sum(inv_dist_pow, 2);
    
    mask_regular = ~any(is_node, 2);
    
    if any(mask_regular)
        weights(mask_regular, :) = inv_dist_pow(mask_regular, :) ./ sum_inv(mask_regular);
    end
    
    if any(~mask_regular)
        idx_linear = find(~mask_regular);
        for i = 1:length(idx_linear)
            k = idx_linear(i);
            group_idx = find(is_node(k, :), 1);
            weights(k, :) = 0;
            weights(k, group_idx) = 1;
        end
    end
    
    B_matrix = zeros(M, N);
    for j = 1:m
        xj = group_nodes(:, j);
        w_j = weights(:, j);
        L_vals = lagrange_basis_vals(xj, eval_pts);
        global_indices = (j-1)*sigma + (1:sigma);
        B_matrix(:, global_indices) = L_vals .* w_j;
    end
end

function L_vals = lagrange_basis_vals(xj, eval_pts)
    sigma = length(xj);
    M = length(eval_pts);
    L_vals = ones(M, sigma);
    for k = 1:sigma
        for i = 1:sigma
            if i ~= k
                L_vals(:, k) = L_vals(:, k) .* (eval_pts - xj(i)) / (xj(k) - xj(i));
            end
        end
    end
end