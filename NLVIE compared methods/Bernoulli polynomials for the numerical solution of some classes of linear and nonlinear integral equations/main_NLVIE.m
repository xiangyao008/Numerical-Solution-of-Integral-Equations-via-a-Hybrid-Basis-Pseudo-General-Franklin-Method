% 复现论文 Example 5.5: Nonlinear Volterra Integral Equation
% 修正版：恢复高精度积分，确保 fsolve 收敛
% 核心逻辑: 循环测速 + 全局误差收敛阶分析

clc; clear; close all;

%% 1. 问题定义
fx = @(x) ((1+x).*exp(-10.*x)+1).^0.5+(1+x).*(1-exp(-10.*x))+10.*(1+x).*log(1+x);
kf = @(x,t,u) -10.*(1+x)./(1+t).*u.^2; % 通用核函数 K(x,t,u)
RF = @(x) ((1+x).*exp(-10.*x)+1).^0.5; % 精确解

% --- 算法参数 ---
N = 8;             % 细网格
N_coarse = N/2;       % 粗网格 (N/2)
vp = 1024;          % 验证点数量
num_runs = 5;       % 测速循环次数

%% 2. 循环测速 (Performance Test)
fprintf('-------------------------------------------\n');
fprintf('Bernoulli Nonlinear Volterra Solver (N=%d)\n', N);
fprintf('注意: 已恢复高精度积分 (Tol=1e-12)，计算可能较慢，但更稳定。\n');

time_records = zeros(num_runs, 1);
x_validate = linspace(0, 1, vp)'; 

% [预热]
fprintf('正在进行预热运行...\n');
solve_volterra_general(kf, fx, N);

% [循环测速]
fprintf('正在进行 %d 次循环测速...\n', num_runs);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格
    [u_approx_handle, exitflag] = solve_volterra_general(kf, fx, N);
    
    % 如果求解失败，发出警告
    if exitflag <= 0
        warning('第 %d 次运行 fsolve 未收敛 (ExitFlag=%d)', i, exitflag);
    end
    
    % 在验证点评估
    u_fine_vals = u_approx_handle(x_validate);
    
    % 计算误差
    u_true = RF(x_validate);
    err_vec_fine = abs(u_true - u_fine_vals);
    
    time_records(i) = toc(t_tick);
    fprintf('  Run %d/%d: %.4f s\n', i, num_runs, time_records(i));
end

avg_time = mean(time_records);
fprintf('网格 N=%d 平均耗时: %.6f 秒\n', N, avg_time);

%% 3. 精度与全局收敛阶分析
fprintf('-------------------------------------------\n');
fprintf('正在计算粗网格对照组 (N=%d)...\n', N_coarse);

% 求解粗网格
[u_coarse_handle, exitflag_c] = solve_volterra_general(kf, fx, N_coarse);
if exitflag_c <= 0, warning('粗网格 fsolve 未收敛'); end

u_coarse_vals = u_coarse_handle(x_validate);
err_vec_coarse = abs(u_true - u_coarse_vals);

% --- 收敛阶计算 (剔除首尾) ---
inner_err_fine   = err_vec_fine(2:end-1);
inner_err_coarse = err_vec_coarse(2:end-1);
x_plot = x_validate(2:end-1);

E_inf_fine   = max(abs(inner_err_fine));
E_inf_coarse = max(abs(inner_err_coarse));
E_rms_fine   = rms(inner_err_fine);
E_rms_coarse = rms(inner_err_coarse);

global_order_inf = log2(E_inf_coarse / E_inf_fine);
global_order_l2  = log2(E_rms_coarse / E_rms_fine);

%% 4. 结果输出
fprintf('\n-------------------------------------------\n');
fprintf('Nonlinear Volterra 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', N_coarse, N);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% 绘图
figure('Color', 'w', 'Name', 'Volterra Results');
subplot(2,1,1);
plot(x_validate, u_true, 'k-', 'LineWidth', 1.5); hold on;
plot(x_validate, u_fine_vals, 'r--', 'LineWidth', 1.5);
legend('Exact', 'Numerical'); title('Solution Comparison'); grid on;

subplot(2,1,2);
semilogy(x_plot, inner_err_fine, 'b-', 'LineWidth', 1.2);
title('Interior Error Distribution'); grid on; xlim([0 1]);

%% ========================================================================
%  通用非线性求解器 (已恢复高精度积分设置)
%  ========================================================================
function [u_approx_handle, exitflag] = solve_volterra_general(K_func, fx, N)
    
    % 1. 配置点
    indices = 0:N;
    x_nodes = (2 * indices + 1) ./ (2 * (N + 1));
    x_nodes = x_nodes(:);
    
    % 2. 预计算伯努利数
    B_nums_vec = get_bernoulli_nums_vector(N);
    
    % 3. 残差函数
    resid_func = @(C) compute_residuals(C, x_nodes, K_func, fx, N, B_nums_vec);
    
    % 4. 初始猜测 (全0可能对某些非线性问题不够好，如有需要可改为ones)
    C0 = zeros(N+1, 1);
    
    % 5. 求解选项
    % 【关键】：积分精度高，FunctionTolerance 也要匹配
    options = optimoptions('fsolve', 'Display', 'off', ...
        'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12, ...
        'Algorithm', 'trust-region-dogleg'); 
    
    [C_opt, ~, exitflag] = fsolve(resid_func, C0, options);
    
    % 6. 构造解
    u_approx_handle = @(x) evaluate_bernoulli_basis(C_opt, x, B_nums_vec);
end

function F_res = compute_residuals(C, x_nodes, K_func, fx, N, B_nums_vec)
    n_nodes = length(x_nodes);
    F_res = zeros(n_nodes, 1);
    
    u_current = @(t) evaluate_bernoulli_basis(C, t, B_nums_vec);
    
    for i = 1:n_nodes
        xi = x_nodes(i);
        val_u = u_current(xi);
        val_f = fx(xi);
        
        if xi > 1e-14
            integrand = @(t) K_func(xi, t, u_current(t));
            % 【关键恢复】：恢复高精度积分，否则 fsolve 梯度计算会出错
            val_int = integral(integrand, 0, xi, 'AbsTol', 1e-12, 'RelTol', 1e-12, 'ArrayValued', true);
        else
            val_int = 0;
        end
        
        F_res(i) = val_u - val_f - val_int;
    end
end

% --- 辅助函数 ---
function val = evaluate_bernoulli_basis(C, x, B_nums_vec)
    val = zeros(size(x));
    N = length(C) - 1;
    for j = 0:N
        val = val + C(j+1) * bernoulli_poly_fast(j, x, B_nums_vec);
    end
end

function val = bernoulli_poly_fast(n, x, B_all)
    val = zeros(size(x));
    for k = 0:n
        bk = B_all( (n-k) + 1 );
        coef = nchoosek(n, k) * bk;
        val = val + coef * (x.^k);
    end
end

function B_vec = get_bernoulli_nums_vector(max_order)
    B_vec = zeros(1, max_order + 1);
    B_vec(1) = 1; 
    if max_order >= 1, B_vec(2) = -0.5; end
    for n = 2:max_order
        s = 0;
        for k = 0:(n-1), s = s + nchoosek(n+1, k) * B_vec(k+1); end
        B_vec(n+1) = -s / (n+1);
    end
end