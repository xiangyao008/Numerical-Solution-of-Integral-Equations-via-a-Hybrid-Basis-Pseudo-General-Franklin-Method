% 复现论文 Example 5.6: Nonlinear Fredholm Integral Equation (Performance & Convergence)
% 论文来源: Section 5.2, Page 14
% 核心逻辑: 循环测速 + 双网格全局收敛阶分析

clc; clear; close all;

%% 1. 问题定义 (Problem Definition)
% Example 5.6: u(x) = f(x) + int_0^1 K(x,t,u(t)) dt
% Exact Solution: u(x) = sin(x) + ...

% 问题参数
fx = @(x) sin(pi*x);
% 非线性核函数 K(x,t,u)
kf = @(x,t,u) 1/5 .* cos(pi*x) .* sin(pi*t) .* (u.^3);
% 导数核 (如果使用 Newton 法需要，此处 fsolve 自动处理差分)
dkut = @(x,t,u) 3/5 .* cos(pi*x) .* sin(pi*t) .* (u.^2);
% 精确解
RF = @(x) sin(pi*x) + 1/3 .* (20 - sqrt(391)) .* cos(pi*x);

% 算法参数
N = 8;             % 细网格阶数 (Fine Grid) - 建议偶数以便 N/2 为整数
vp = 1024;          % 验证点数量
num_runs = 5;       % 测速循环次数 (非线性求解较慢，建议设小)

%% 2. 循环测速 (Performance Test for Fine Grid N)
fprintf('-------------------------------------------\n');
fprintf('Bernoulli Nonlinear Fredholm Solver (Fine N=%d, Coarse N=%d)\n', N, N/2);
fprintf('注意: 使用 fsolve + 数值积分，计算耗时较长。\n');

time_records = zeros(num_runs, 1);
x_validate = linspace(0, 1, vp)'; % 验证点列向量

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
[~, ~, ~] = solve_fredholm_general(kf, fx, N);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (N=%d)...\n', num_runs, N);
for i = 1:num_runs
    t_tick = tic;
    
    % 1. 求解获得函数句柄
    [u_approx_handle, exitflag, ~] = solve_fredholm_general(kf, fx, N);
    
    % 2. 在验证点上评估解
    u_fine_vals = u_approx_handle(x_validate);
    
    % 3. 计算误差向量
    u_true = RF(x_validate);
    err_vec_fine = abs(u_true - u_fine_vals);
    
    time_records(i) = toc(t_tick);
    fprintf('  Run %d/%d: %.4f s (ExitFlag: %d)\n', i, num_runs, time_records(i), exitflag);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 N=%d 平均耗时: %.6f 秒\n', N, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (N/2)
fprintf('正在计算粗网格对照组 (N=%d)...\n', N/2);
[u_coarse_handle, ~, ~] = solve_fredholm_general(kf, fx, N/2);
u_coarse_vals = u_coarse_handle(x_validate);
err_vec_coarse = abs(u_true - u_coarse_vals);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免端点处的数值奇异性干扰统计
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
fprintf('Bernoulli Nonlinear Fredholm 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (N=%d):\t %.6f 秒\n', N, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', N/2, N);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% 简单绘图验证
figure('Color', 'w', 'Name', 'Nonlinear Fredholm Results');
subplot(2,1,1);
plot(x_validate, u_true, 'k-', 'LineWidth', 1.5); hold on;
plot(x_validate, u_fine_vals, 'r--', 'LineWidth', 1.5);
legend('Exact Solution', ['Approx (N=' num2str(N) ')'], 'Location', 'best');
title('Solution Comparison'); grid on; xlabel('x'); ylabel('u(x)');

subplot(2,1,2);
semilogy(x_validate(2:end-1), inner_err_fine, 'b-', 'LineWidth', 1.2);
title('Interior Error Distribution (Log Scale)'); 
grid on; xlabel('x'); ylabel('|Error|');
xlim([0, 1]);

%% ========================================================================
%  通用 Fredholm 求解器 (核心逻辑保持不变)
%  ========================================================================
function [u_approx_handle, exitflag, t_cost] = solve_fredholm_general(K_func, fx, N)
    % SOLVE_FREDHOLM_GENERAL 求解 u(x) = f(x) + int_0^1 K(x,t,u(t)) dt
    
    tic;
    
    % 1. 生成配置点 (Newton-Cotes nodes)
    indices = 0:N;
    x_nodes = (2 * indices + 1) ./ (2 * (N + 1));
    x_nodes = x_nodes(:); 
    
    % 2. 预计算伯努利数向量
    B_nums_vec = get_bernoulli_nums_vector(N);
    
    % 3. 定义残差函数
    resid_func = @(C) compute_fredholm_residuals(C, x_nodes, K_func, fx, N, B_nums_vec);
    
    % 4. 初始猜测
    C0 = zeros(N+1, 1);
    
    % 5. 使用 fsolve 求解
    options = optimoptions('fsolve', 'Display', 'off', ...
        'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12, ...
        'Algorithm', 'trust-region-dogleg');
    
    [C_opt, ~, exitflag] = fsolve(resid_func, C0, options);
    
    % 6. 构造近似解
    u_approx_handle = @(x) evaluate_bernoulli_basis(C_opt, x, B_nums_vec);
    
    t_cost = toc;
end

function F_res = compute_fredholm_residuals(C, x_nodes, K_func, fx, N, B_nums_vec)
    % 计算 Fredholm 方程的残差
    n_nodes = length(x_nodes);
    F_res = zeros(n_nodes, 1);
    
    % 当前系数对应的解函数
    u_current = @(t) evaluate_bernoulli_basis(C, t, B_nums_vec);
    
    for i = 1:n_nodes
        xi = x_nodes(i);
        val_u = u_current(xi);
        val_f = fx(xi);
        
        % 积分项: int_0^1 K(xi, t, u(t)) dt
        integrand = @(t) K_func(xi, t, u_current(t));
        
        % 数值积分
        val_int = integral(integrand, 0, 1, 'AbsTol', 1e-8, 'RelTol', 1e-8, 'ArrayValued', true);
        
        % 残差
        F_res(i) = val_u - val_f - val_int;
    end
end

%% ========================================================================
%  辅助函数 (Bernoulli Basis)
%  ========================================================================
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