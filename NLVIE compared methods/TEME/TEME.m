%% Main Script: Robust Nonlinear Volterra Solver (Performance & Convergence)
% Method: Piecewise Euler Basis (Compact Support)
% Target: Loop Timing + Global Convergence Order Analysis (M vs M/2)

clc; clear; close all;

%% 1. 问题定义与参数设置
% 精确解 u(x)
RF = @(x) ((1+x).*exp(-10.*x)+1).^0.5;
% 源项 g(x)
fx = @(x) ((1+x).*exp(-10.*x)+1).^0.5 + (1+x).*(1-exp(-10.*x)) + 10.*(1+x).*log(1+x);
% 核函数 K(x,t,u)
kf = @(x,t,u) -10.*(1+x)./(1+t).*u.^2;
% 核函数导数 dK/du (用于 Jacobian)
dkut = @(x,t,u) -20.*(1+x)./(1+t).*u;

% --- 算法参数 ---
seg_fine = 32;          % 细网格分段数
dofs_per_seg = 3;       % 每段自由度 (1, E1, E2)
M_fine = seg_fine * dofs_per_seg; % 总自由度 = 96

seg_coarse = seg_fine/2;        % 粗网格分段数
M_coarse = seg_coarse * dofs_per_seg; % 总自由度 = 48

N_gauss = 12;           % 积分精度
MaxIter = 20;
Tol = 1e-12;

vp = 1024;              % 验证点数量
x_dense = linspace(0, 1, vp)';
num_runs = 5;           % 测速循环次数

%% 2. 循环测速 (Performance Test for Fine Grid)
fprintf('-------------------------------------------\n');
fprintf('Piecewise Euler Solver (Fine M=%d, Coarse M=%d)\n', M_fine, M_coarse);
fprintf('注意: Jacobian 矩阵构建涉及双重循环积分，计算量随 M 平方增长。\n');

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
solve_piecewise_euler(fx, kf, dkut, seg_fine, dofs_per_seg, N_gauss, MaxIter, Tol);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (M=%d)...\n', num_runs, M_fine);
for i = 1:num_runs
    t_tick = tic;
    
    % --- 求解细网格 ---
    [A_fine, edges_fine, iter_fine] = solve_piecewise_euler(fx, kf, dkut, seg_fine, dofs_per_seg, N_gauss, MaxIter, Tol);
    
    % --- 在稠密网格上重构解 ---
    % 1. 计算基函数矩阵
    Psi_verify_fine = get_piecewise_euler_basis(x_dense, M_fine, edges_fine);
    % 2. 得到数值解
    u_vals_fine = Psi_verify_fine * A_fine;
    
    % --- 计算误差向量 ---
    rf_vals = RF(x_dense);
    err_vec_fine = abs(u_vals_fine(:) - rf_vals(:));
    
    time_records(i) = toc(t_tick);
    fprintf('  Run %d/%d: %.4f s (Iter: %d)\n', i, num_runs, time_records(i), iter_fine);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 M=%d 平均耗时: %.6f 秒\n', M_fine, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation
fprintf('正在计算粗网格对照组 (M=%d)...\n', M_coarse);
[A_coarse, edges_coarse, ~] = solve_piecewise_euler(fx, kf, dkut, seg_coarse, dofs_per_seg, N_gauss, MaxIter, Tol);

% 评估粗网格误差
Psi_verify_coarse = get_piecewise_euler_basis(x_dense, M_coarse, edges_coarse);
u_vals_coarse = Psi_verify_coarse * A_coarse;
err_vec_coarse = abs(u_vals_coarse(:) - rf_vals(:));

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% Volterra 方程起始点误差极小，剔除以避免 log(0)
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
fprintf('Piecewise Euler Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (M=%d):\t %.6f 秒\n', M_fine, avg_time);
fprintf('Newton 迭代次数:\t %d\n', iter_fine);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (M=%d)\t Error (M=%d)\t Order\n', M_coarse, M_fine);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% --- 绘图 ---
figure('Color', 'w', 'Position', [100 100 1000 400], 'Name', 'Euler Basis Analysis');

% 子图 1: 解的对比
subplot(1, 2, 1);
plot(x_dense, rf_vals, 'k-', 'LineWidth', 1.5); hold on;
plot(x_dense, u_vals_fine, 'r--', 'LineWidth', 1.5);
legend('Exact', ['Approx (M=', num2str(M_fine), ')'], 'Location', 'Best');
xlabel('x'); ylabel('u(x)'); title('Solution Comparison'); grid on;

% 子图 2: 误差分布
subplot(1, 2, 2);
semilogy(x_plot, inner_err_fine, 'b-', 'LineWidth', 1.2);
xlabel('x'); ylabel('|Error|');
title(['Interior Error Distribution (Log Scale)']);
grid on; xlim([0, 1]);

%% ========================================================================
%  核心求解器 (Encapsulated)
%  ========================================================================
function [A, edges, final_iter] = solve_piecewise_euler(fx, kf, dkut, num_segments, dofs, N_gauss, MaxIter, Tol)
    M = num_segments * dofs;
    edges = linspace(0, 1, num_segments + 1);
    
    % 生成求解用的配点 (Collocation Points)
    x_col = zeros(M, 1);
    for i = 1:num_segments
        a = edges(i); b = edges(i+1);
        % Chebyshev Nodes mapped to [a, b]
        local_cheb = [-sqrt(3)/2, 0, sqrt(3)/2]'; 
        x_col((i-1)*3 + 1 : i*3) = (a+b)/2 + (b-a)/2 * local_cheb;
    end
    
    [xi_g_std, w_g_std] = get_gauss_legendre(N_gauss);
    Psi_x = get_piecewise_euler_basis(x_col, M, edges);
    
    A = ones(M, 1); % Initial Guess
    final_iter = MaxIter;
    
    for iter = 1:MaxIter
        u_x = Psi_x * A;
        Integral_val = zeros(M, 1);
        J_integral = zeros(M, M);
        
        for j = 1:M
            x_curr = x_col(j);
            if x_curr <= 1e-14, continue; end
            
            seg_idx = find(edges > x_curr, 1) - 1;
            if isempty(seg_idx), seg_idx = num_segments; end
            
            val_sum = 0;
            row_J_sum = zeros(1, M);
            
            for k = 1:seg_idx
                a = edges(k);
                b_limit = (k == seg_idx) * x_curr + (k < seg_idx) * edges(k+1);
                if b_limit <= a, continue; end
                
                t_nodes = (a + b_limit)/2 + (b_limit - a)/2 * xi_g_std;
                dt_weights = (b_limit - a)/2 * w_g_std;
                
                Psi_t = get_piecewise_euler_basis(t_nodes, M, edges);
                u_t = Psi_t * A;
                
                val_sum = val_sum + sum(dt_weights .* kf(x_curr, t_nodes, u_t));
                dK_vals = dkut(x_curr, t_nodes, u_t);
                row_J_sum = row_J_sum + (dK_vals .* dt_weights)' * Psi_t;
            end
            Integral_val(j) = val_sum;
            J_integral(j, :) = row_J_sum;
        end
        
        F = u_x - fx(x_col) - Integral_val;
        J = Psi_x - J_integral;
        
        res_norm = norm(F, inf);
        if res_norm < Tol
            final_iter = iter;
            break;
        end
        
        A = A - J \ F;
    end
end

%% ========================================================================
%  辅助函数
%  ========================================================================
function Psi = get_piecewise_euler_basis(x, M, edges)
    N = length(x);
    Psi = zeros(N, M);
    num_seg = length(edges) - 1;
    
    E1 = @(t) t - 0.5;
    E2 = @(t) t.^2 - t;
    
    for k = 1:num_seg
        a = edges(k);
        b = edges(k+1);
        
        if k == num_seg
            idx = find(x >= a & x <= b + 1e-14); 
        else
            idx = find(x >= a & x < b);
        end
        
        if isempty(idx), continue; end
        
        x_local = (x(idx) - a) / (b - a);
        col_start = (k-1)*3 + 1;
        Psi(idx, col_start)   = 1;
        Psi(idx, col_start+1) = E1(x_local);
        Psi(idx, col_start+2) = E2(x_local);
    end
end

function [x, w] = get_gauss_legendre(N)
    beta = .5 ./ sqrt(1-(2*(1:N-1)).^(-2));
    T = diag(beta,1) + diag(beta,-1);
    [V, D] = eig(T);
    x = diag(D); 
    [x, i] = sort(x);
    w = 2 * V(1,i).^2;
    w = w';
end