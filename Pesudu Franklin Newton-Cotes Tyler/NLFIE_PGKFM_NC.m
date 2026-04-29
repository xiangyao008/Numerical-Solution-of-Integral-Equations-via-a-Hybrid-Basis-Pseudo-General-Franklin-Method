clc; close all; clear all;

%% 非线性 Fredholm 积分方程 (NLFIE) 测试主程序
% 对应方法: Pesudu Franklin Newton-Cotes (Non-linear Version)

% --- 参数设置 ---
n = 32;            % 细网格阶数 (Fine Grid)
vp = 1024;          % 验证点数量
iter = 100;         % 最大迭代次数
tol = 1e-5;         % 容差
K = 1;              % 基函数阶数
num_runs = 10;      % 测速循环次数

% --- 问题定义 ---
fx = @(x) sin(pi*x);
% 核函数与导数核
kf = @(x,t,u) 1/5 .* cos(pi*x) .* sin(pi*t) .* (u.^3);
dkut = @(x,t,u) 3/5 .* cos(pi*x) .* sin(pi*t) .* (u.^2);
% 精确解
RF = @(x) sin(pi*x) + 1/3 .* (20 - sqrt(391)) .* cos(pi*x);

%% 伪Franklin方法 - 初值预计算
fprintf('正在计算 Taylor 初值猜测 (SolveFredholmTaylorNumeric)...\n');
[coeffs_numeric, u_series] = SolveFredholmTaylorNumeric(fx, kf, 0, 1, K);

%% 1. 循环测速 (Performance Test for Fine Grid n)
fprintf('-------------------------------------------\n');
fprintf('正在准备 NLFIE 测试 (Fine n=%d, Coarse n=%d)...\n', n, n/2);

time_records = zeros(num_runs, 1);

% [预热] Warm-up (运行一次以加载 JIT)
fprintf('正在进行预热运行...\n');
[~, ~, ~] = NLFIE_PGFM_NC(fx, dkut, n, iter, tol, K, kf, coeffs_numeric);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (n=%d)...\n', num_runs, n);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格 (Fine Grid)
    [ua_iter_fine, tk_xj_fine, cg_it_fine] = NLFIE_PGFM_NC(fx, dkut, n, iter, tol, K, kf, coeffs_numeric);
    
    % 计算细网格误差
    % 注意：取最后一次迭代结果 ua_iter_fine(:, cg_it_fine)
    pointwise_error_fine = PE1_PGFM(RF, ua_iter_fine(:, cg_it_fine), vp, K);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
std_time = std(time_records);
fprintf('网格 n=%d 平均耗时: %.6f 秒 (Std: %.6f)\n', n, avg_time, std_time);

%% 2. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (n/2)
fprintf('正在计算粗网格对照组 (n=%d)...\n', n/2);
[ua_iter_coarse, tk_xj_coarse, cg_it_coarse] = NLFIE_PGFM_NC(fx, dkut, n/2, iter, tol, K, kf, coeffs_numeric);
pointwise_error_coarse = PE1_PGFM(RF, ua_iter_coarse(:, cg_it_coarse), vp, K);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免积分方程端点处可能存在的数值奇异性或强约束对整体统计的干扰
inner_err_fine   = pointwise_error_fine(2:end-1);
inner_err_coarse = pointwise_error_coarse(2:end-1);

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

%% 3. 结果格式化输出
fprintf('\n-------------------------------------------\n');
fprintf('NLFIE (Non-linear) 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (n=%d):\t %.6f 秒\n', n, avg_time);
fprintf('非线性迭代次数 (Fine):\t %d\n', cg_it_fine);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', n/2, n);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% 简要结论
if global_order_inf > K
    fprintf('结论: 算法表现出超收敛特性 (Order > K=%d).\n', K);
elseif abs(global_order_inf - K) < 0.5
    fprintf('结论: 算法收敛阶符合预期 (Order ≈ K=%d).\n', K);
else
    fprintf('注意: 收敛阶为 %.2f，请检查非线性迭代收敛性或奇点。\n', global_order_inf);
end