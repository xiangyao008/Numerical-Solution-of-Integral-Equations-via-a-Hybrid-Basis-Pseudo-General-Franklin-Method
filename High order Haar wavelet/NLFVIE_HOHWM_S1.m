%% Haar Wavelet Method (S1) for NLFVIE
% 核心逻辑: 循环测速 + 全局误差收敛阶分析
% 对应问题: Nonlinear Volterra-Fredholm Integral Equations

clc; close all; clear all;

%% 1. 问题定义与参数设置
n = 512;             % 细网格阶数 (Fine Grid)
iter = 100;         % 最大迭代次数
tol = 1e-10;        % 迭代容差
vp = 1024;          % 验证点数量
num_runs = 10;      % 测速循环次数

% --- Example 1 ---
% u(x) = f(x) + int_0^x V(x,t,u)dt + int_0^1 F(x,t,u)dt
% Exact Solution: u(x) = x^2 - 2
fx = @(x) -1/30*x.^6 + 1/3*x.^4 - x.^2 + 5/3*x - 5/4;

% Volterra 核函数 (定义但未在函数调用中使用，根据您的原代码保持一致)
kf = @(x,t,u) (x-t).*(u.^2);

% 导数核 (用于牛顿迭代)
dvkut = @(x,t,u) (x-t).*2*u;   % Volterra 部分对 u 的导数
dfkut = @(x,t,u) (x+t);        % Fredholm 部分对 u 的导数

RF = @(x) x.^2 - 2;

%% 2. 循环测速 (Performance Test for Fine Grid n)
fprintf('-------------------------------------------\n');
fprintf('正在准备 Haar NLFVIE 测试 (Fine n=%d, Coarse n=%d)...\n', n, n/2);

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
[~, ~, ~] = NLFVIE_HAAR_S1(fx, dvkut, dfkut, n, iter, tol);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (n=%d)...\n', num_runs, n);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格 (Fine Grid)
    [ua_iter1, tk_xj1, cg_it1] = NLFVIE_HAAR_S1(fx, dvkut, dfkut, n, iter, tol);
    
    % 计算细网格误差 (取最后一次迭代结果)
    % 变量名重命名为 _fine 以明确含义
    pointwise_error_fine = PE1_HOHWM(RF, ua_iter1(:, cg_it1), vp);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 n=%d 平均耗时: %.6f 秒\n', n, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (n/2)
fprintf('正在计算粗网格对照组 (n=%d)...\n', n/2);
[ua_iter2, tk_xj2, cg_it2] = NLFVIE_HAAR_S1(fx, dvkut, dfkut, n/2, iter, tol);

% 计算粗网格误差
pointwise_error_coarse = PE1_HOHWM(RF, ua_iter2(:, cg_it2), vp);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免积分方程端点处的数值奇异性干扰整体统计
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

%% 4. 结果格式化输出
fprintf('\n-------------------------------------------\n');
fprintf('Haar NLFVIE Method (S1) 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (n=%d):\t %.6f 秒\n', n, avg_time);
fprintf('非线性迭代次数 (Fine):\t %d\n', cg_it1);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', n/2, n);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');