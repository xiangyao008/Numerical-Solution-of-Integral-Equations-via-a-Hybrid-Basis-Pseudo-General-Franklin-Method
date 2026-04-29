%% Numerical Solution for Mixed Volterra-Fredholm Integral Equations
% Problem Type: u(x) = f(x) + int_0^x int_0^1 K(t, y, u(y)) dy dt

clc; close all; clear all;

%% 1. 参数设置
n = 256;             % 基函数数量
vp = 1024;           % 验证点数量 (用于 PE1_PGFM)
iter = 50;          % 最大迭代次数
tol = 1e-6;         % 收敛容差
K = 4;              % Franklin函数阶数

%% 2. 问题定义 (Example 1 from Literature)
% u(x) = - (x^2 * (1 + exp(2))) / 8 + exp(x) + int_0^x int_0^1 t * y * u^2(y) dy dt
% Exact Solution: u(x) = exp(x)

fx = @(x) - (x.^2 .* (1 + exp(2))) ./ 8 + exp(x);

% 核函数 K(t, y, u)
kf_handle = @(t, y, u) t .* y .* (u.^2);

% 导数 (虽然 Broyden 自动更新 Jacobian，但定义可用于扩展)
dkut_handle = @(t, y, u) t .* y .* (2.*u); 

% 精确解 (用于误差评估)
RF = @(x) exp(x); 

%% 3. 预计算初值
fprintf('正在计算数值初值 (SolveMixedNumeric)...\n');
[init_coeffs, ~] = SolveMixedNumeric(fx, kf_handle, K);

%% 4. 求解与测试
num_runs = 10; 
fprintf('正在求解混合积分方程 (n=%d, K=%d)...\n', n, K);
time_records = zeros(num_runs, 1);

% 预热 (Warm-up)
fprintf('正在预热...\n');
[~, ~, ~] = NLMFVIE_PGFM_NC(fx, dkut_handle, n/2, iter, tol, K, kf_handle, init_coeffs);

for i = 1:num_runs
    tic;
    [ua_iter, tk_xj, cg_it] = NLMFVIE_PGFM_NC(fx, dkut_handle, n, iter, tol, K, kf_handle, init_coeffs);
    time_records(i) = toc;
    fprintf('Run %d: %.4f s\n', i, time_records(i));
end

%% 5. 误差分析 (使用 PE1_PGFM)
final_coeffs = ua_iter(:, cg_it);

% --- 调用外部评估函数 ---
% output: 绝对误差向量 (大小为 vp x 1)
pointwise_error = PE1_PGFM(RF, final_coeffs, vp, K);

% 计算范数误差
L_inf_error = max(pointwise_error);
L2_error = norm(pointwise_error) / sqrt(vp);

%% 6. 结果输出与绘图
fprintf('-------------------------------------------\n');
fprintf('计算完成。\n');
fprintf('平均运行时间: %.6f 秒\n', mean(time_records));
fprintf('收敛迭代次数: %d\n', cg_it);
fprintf('L_inf 误差:   %.2e\n', L_inf_error);
fprintf('L2 误差:      %.2e\n', L2_error);

% 绘图
figure;
plot(linspace(0, 1, vp), pointwise_error, 'r-', 'LineWidth', 1.5);
title(['Absolute Error Distribution (n=', num2str(n), ', K=', num2str(K), ')']);
xlabel('x'); ylabel('|u_{exact} - u_{approx}|');
grid on;