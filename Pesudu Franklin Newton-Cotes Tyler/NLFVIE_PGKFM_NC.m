%% An Effective Scheme for Solving Nonlinear Volterra-Fredholm Integral Equations
% 核心逻辑: 循环测速 + 全局误差收敛阶分析 (L-inf & RMS)

clc; close all; clear all;

%% 1. 问题定义与参数设置
% Triangular functions (TF) method / Pseudo-Franklin Method
n = 64;             % 细网格阶数 (Fine Grid)
iter = 100;         % 最大迭代次数
tol = 1e-12;        % 迭代容差
K = 4;              % 基函数阶数
vp = 1024;          % 验证点数量
num_runs = 10;      % 测速循环次数

% --- Example 1 ---
fx = @(x) -1/30*x.^6 + 1/3*x.^4 - x.^2 + 5/3*x - 5/4;
% Volterra 部分核函数
kf_v = @(x,t,u) (x-t).*(u.^2);
% Fredholm 部分核函数
kf_f = @(x,t,u) (x+t).*(u);
% Volterra 部分导数核
dvkut = @(x,t,u) (x-t).*2.*u;
% Fredholm 部分导数核
dfkut = @(x,t,u) (x+t);
% 精确解
RF = @(x) x.^2 - 2;

%% 2. 伪Franklin方法 - 初值预计算
fprintf('正在计算 Taylor 初值猜测 (Solve_mix_TaylorNumeric)...\n');
[coeffs, u_poly] = Solve_mix_TaylorNumeric(fx, kf_v, kf_f, K);

%% 3. 循环测速 (Performance Test for Fine Grid n)
fprintf('-------------------------------------------\n');
fprintf('正在准备 NLFVIE 测试 (Fine n=%d, Coarse n=%d)...\n', n, n/2);

time_records = zeros(num_runs, 1);

% [预热] Warm-up (运行一次以加载 JIT)
fprintf('正在进行预热运行...\n');
[~, ~, ~] = NLFVIE_PGFM_Newton(fx, dvkut, dfkut, n, iter, tol, K, kf_v, kf_f, coeffs);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (n=%d)...\n', num_runs, n);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格 (Fine Grid)
    [ua_iter_fine, tk_xj_fine, cg_it_fine] = NLFVIE_PGFM_Newton(fx, dvkut, dfkut, n, iter, tol, K, kf_v, kf_f, coeffs);
    
    % 计算细网格误差 (取最后一次迭代结果)
    pointwise_error_fine = PE1_PGFM(RF, ua_iter_fine(:, cg_it_fine), vp, K);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
std_time = std(time_records);
fprintf('网格 n=%d 平均耗时: %.6f 秒 (Std: %.6f)\n', n, avg_time, std_time);

%% 4. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (n/2)
fprintf('正在计算粗网格对照组 (n=%d)...\n', n/2);
[ua_iter_coarse, tk_xj_coarse, cg_it_coarse] = NLFVIE_PGFM_Newton(fx, dvkut, dfkut, n/2, iter, tol, K, kf_v, kf_f, coeffs);
pointwise_error_coarse = PE1_PGFM(RF, ua_iter_coarse(:, cg_it_coarse), vp, K);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免边界效应或强约束点(误差为0)导致统计异常
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

%% 5. 结果格式化输出
fprintf('\n-------------------------------------------\n');
fprintf('NLFVIE (Volterra-Fredholm) 精度与收敛阶分析报告\n');
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
    fprintf('注意: 收敛阶为 %.2f，请检查混合方程的性质或迭代收敛性。\n', global_order_inf);
end