% 修正版：带阻尼牛顿迭代法 + 全局收敛阶分析 (LMW Method)
% 核心逻辑: 循环测速 + 双网格 (M vs M/2) 精度对比

clc; clear; close all;

%% 1. 问题定义
% Example: u(x) = sqrt((1+x)exp(-10x)+1)
RF = @(x) ((1+x).*exp(-10.*x)+1).^0.5;
% 源项
fx = @(x) ((1+x).*exp(-10.*x)+1).^0.5 + (1+x).*(1-exp(-10.*x)) + 10.*(1+x).*log(1+x);
% 核函数 K(x,t,u)
kf = @(x,t,u) -10.*(1+x)./(1+t).*u.^2;
% Jacobian 核 dK/du
dkut = @(x,t,u) -20.*(1+x)./(1+t).*u;

% --- 算法参数 ---
k_level_fine = 3;       % 细网格层级 -> M = 2^(k+2) = 256
M_fine = 2^(k_level_fine + 2);

k_level_coarse = k_level_fine-1;     % 粗网格层级 -> M = 128
M_coarse = 2^(k_level_coarse + 2);

Max_Iter = 20;
Tol = 1e-6;
vp = 1024;              % 验证点数量
x_dense = linspace(0, 1, vp)';

num_runs = 5;           % 测速循环次数

%% 2. 循环测速 (Performance Test)
fprintf('-------------------------------------------\n');
fprintf('LMW Damped Newton Solver (Fine M=%d, Coarse M=%d)\n', M_fine, M_coarse);
fprintf('注意: 包含 Jacobian 矩阵构建与求逆，计算量随 M 平方/立方增长。\n');

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
solve_LMW_newton(fx, kf, dkut, M_fine, Max_Iter, Tol);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (M=%d)...\n', num_runs, M_fine);
for i = 1:num_runs
    t_tick = tic;
    
    % --- 求解细网格 ---
    [C_fine, ~, iter_fine] = solve_LMW_newton(fx, kf, dkut, M_fine, Max_Iter, Tol);
    
    % --- 在稠密网格上重构解 ---
    % 注意：LMW 重构需要重新计算基函数矩阵
    u_vals_fine = reconstruct_solution(C_fine, M_fine, x_dense);
    
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
[C_coarse, ~, ~] = solve_LMW_newton(fx, kf, dkut, M_coarse, Max_Iter, Tol);

% 评估粗网格误差
u_vals_coarse = reconstruct_solution(C_coarse, M_coarse, x_dense);
err_vec_coarse = abs(u_vals_coarse(:) - rf_vals(:));

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% LMW 基函数在 x=1 处可能为 0，需剔除避免边界误差干扰
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
fprintf('LMW Newton Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (M=%d):\t %.6f 秒\n', M_fine, avg_time);
fprintf('非线性迭代次数 (Fine):\t %d\n', iter_fine);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (M=%d)\t Error (M=%d)\t Order\n', M_coarse, M_fine);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% --- 简单绘图验证 ---
figure('Color', 'w', 'Name', 'LMW Newton Analysis');
subplot(2,1,1);
plot(x_dense, rf_vals, 'k-', 'LineWidth', 1.5); hold on;
plot(x_dense, u_vals_fine, 'r--', 'LineWidth', 1.5);
legend('Exact Solution', ['Approx (M=', num2str(M_fine), ')'], 'Location', 'best');
title('Solution Comparison'); grid on; xlabel('x'); ylabel('u(x)');

subplot(2,1,2);
semilogy(x_plot, inner_err_fine, 'b-', 'LineWidth', 1.2);
title('Interior Error Distribution (Log Scale)'); 
grid on; xlabel('x'); ylabel('|Error|');
xlim([0, 1]);

%% ========================================================================
%  核心求解器: 阻尼牛顿迭代法 (Encapsulated)
%  ========================================================================
function [C_curr, final_res, iter] = solve_LMW_newton(fx, kf, dkut, M, Max_Iter, Tol)
    % 1. 预计算 (Pre-computation)
    indices = 1:M;
    Y = (indices - 0.5)' / M; 
    
    % 基函数矩阵 Phi_Y (M x M)
    Phi_Y = zeros(M, M);
    for j = 1:M
        Phi_Y(:, j) = get_basis_val_vectorized(j, Y, M);
    end
    
    % 积分点矩阵 T_quad (M x M) 和 Phi_T (M x M x M)
    T_quad = Y * Y'; 
    Phi_T = zeros(M, M, M);
    for j = 1:M
        Phi_T(:, :, j) = reshape(get_basis_val_vectorized(j, T_quad(:), M), [M, M]);
    end
    
    % 2. 迭代初始化
    C_curr = zeros(M, 1); 
    final_res = inf;
    
    for iter = 1:Max_Iter
        % 2.1 计算残差 F 和 Jacobian J
        [F_curr, J_curr] = get_System_and_Jacobian(C_curr, M, Y, Phi_Y, Phi_T, fx, kf, dkut, T_quad);
        
        res_norm = norm(F_curr);
        
        if res_norm < Tol
            final_res = res_norm;
            return;
        end
        
        % 2.2 牛顿方向求解
        if rcond(J_curr) < 1e-12
            s_k = - (J_curr + 1e-8 * eye(M)) \ F_curr; % 正则化
        else
            s_k = - (J_curr \ F_curr);
        end
        
        % 2.3 阻尼线搜索 (Backtracking)
        lambda = 1.0; 
        alpha = 1e-4;
        C_next = C_curr + lambda * s_k;
        [F_next, ~] = get_System_and_Jacobian(C_next, M, Y, Phi_Y, Phi_T, fx, kf, [], T_quad);
        
        while norm(F_next) > res_norm * (1 - alpha * lambda) && lambda > 1e-3
            lambda = lambda * 0.5;
            C_next = C_curr + lambda * s_k;
            [F_next, ~] = get_System_and_Jacobian(C_next, M, Y, Phi_Y, Phi_T, fx, kf, [], T_quad);
        end
        
        C_curr = C_next;
    end
    final_res = norm(F_next);
end

%% ========================================================================
%  辅助函数: 解重构 (Reconstruction on Dense Grid)
%  ========================================================================
function u_vals = reconstruct_solution(C, M, x_eval)
    % 处理边界: LMW 定义域 [0, 1)，将 1 映射为 1-eps
    x_safe = x_eval;
    x_safe(x_safe == 1) = 1 - 1e-12;
    
    % 计算基函数矩阵 (N_eval x M)
    N_eval = length(x_safe);
    Phi_dense = zeros(N_eval, M);
    for j = 1:M
        Phi_dense(:, j) = get_basis_val_vectorized(j, x_safe, M);
    end
    
    u_vals = Phi_dense * C;
end

%% ========================================================================
%  辅助函数: 系统方程与 Jacobian
%  ========================================================================
function [F, J] = get_System_and_Jacobian(C, M, Y, Phi_Y, Phi_T, fx, kf, dkut, T_quad)
    U_Y = Phi_Y * C; 
    Phi_T_flat = reshape(Phi_T, [M*M, M]);
    U_T = reshape(Phi_T_flat * C, [M, M]); 
    
    Y_grid = repmat(Y, 1, M); 
    K_val = kf(Y_grid, T_quad, U_T); 
    Integral_Term = (Y ./ M) .* sum(K_val, 2);
    
    F = U_Y - fx(Y) - Integral_Term;
    
    if nargout > 1 && ~isempty(dkut)
        dKdu_val = dkut(Y_grid, T_quad, U_T);
        J_int = zeros(M, M);
        weight = (Y ./ M); 
        for m = 1:M
            phi_m_mat = Phi_T(:, :, m); 
            sum_val = sum(dKdu_val .* phi_m_mat, 2);
            J_int(:, m) = weight .* sum_val;
        end
        J = Phi_Y - J_int;
    else
        J = [];
    end
end

%% ========================================================================
%  辅助函数: LMW 基函数 (Vectorized)
%  ========================================================================
function val = get_basis_val_vectorized(idx, y, M)
    val = zeros(size(y));
    half_M = M / 2;
    if idx <= half_M 
        j = idx;
        if j == 1
            mask = (y >= 0) & (y < 1); val(mask) = 1;
        else
            local_idx = j - 1;
            n = floor(log2(local_idx)); l = local_idx - 2^n;
            t = (2^n) * y - l;
            val((t>=0)&(t<0.5)) = -sqrt(3)*(4*t((t>=0)&(t<0.5))-1);
            val((t>=0.5)&(t<1)) = sqrt(3)*(4*t((t>=0.5)&(t<1))-3);
            val = val * (2^(n/2));
        end
    else 
        j = idx - half_M;
        if j == 1
            mask = (y >= 0) & (y < 1); val(mask) = sqrt(3)*(2*y(mask)-1);
        else
            local_idx = j - 1;
            n = floor(log2(local_idx)); l = local_idx - 2^n;
            t = (2^n) * y - l;
            val((t>=0)&(t<0.5)) = (6*t((t>=0)&(t<0.5))-1);
            val((t>=0.5)&(t<1)) = (6*t((t>=0.5)&(t<1))-5);
            val = val * (2^(n/2));
        end
    end
    val((y < 0) | (y >= 1)) = 0;
end