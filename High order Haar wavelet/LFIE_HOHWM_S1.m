%% Higher order Haar wavelet method for numerical solution of integral equations
% 参数命名定义形式参考这篇文章
%%
clc; close all; clear all;

% --- 参数定义 ---
n = 128;            % 细网格阶数 (Fine Grid)
vp = 1024;          % 验证点数量 (Validation Points)
num_runs = 10;      % 测速循环次数

%% Example 3
fx = @(x) exp(x);
kf = @(x,t) 2.*exp(x+t);
RF = @(x) exp(x)./(2-exp(2));

%% 1. 循环测速 (Performance Test for Fine Grid n)
fprintf('-------------------------------------------\n');
fprintf('正在准备 Haar 测试 (Fine n=%d, Coarse n=%d)...\n', n, n/2);

time_records = zeros(num_runs, 1);

% [预热] Warm-up (避免首次运行 JIT 编译时间影响)
fprintf('正在进行预热运行...\n');
[~, ~] = LFIE_HAAR_S1(fx, kf, n);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (n=%d)...\n', num_runs, n);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格 (Fine Grid)
    [Array_fx1, tk_xj1] = LFIE_HAAR_S1(fx, kf, n);
    
    % 计算细网格误差
    % 注意：我们将误差计算也包含在总耗时中，如仅需测试求解器速度请注释此行
    pointwise_error_fine = PE1_HOHWM(RF, Array_fx1, vp);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 n=%d 平均耗时: %.6f 秒\n', n, avg_time);

%% 2. 精度与全局收敛阶分析 (Accuracy & Convergence)
% 注意：此时 pointwise_error_fine 保留的是循环最后一次的结果，直接使用即可

% [计算粗网格] Coarse Grid Calculation (n/2)
fprintf('正在计算粗网格对照组 (n=%d)...\n', n/2);
[Array_fx2, tk_xj2] = LFIE_HAAR_S1(fx, kf, n/2);
pointwise_error_coarse = PE1_HOHWM(RF, Array_fx2, vp);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免边界强约束或吉布斯现象导致的奇点干扰整体评估
inner_err_fine   = pointwise_error_fine(2:end-1);
inner_err_coarse = pointwise_error_coarse(2:end-1);

% 2. 计算离散 L-infinity 范数 (内部最大绝对误差)
E_inf_fine   = max(abs(inner_err_fine));
E_inf_coarse = max(abs(inner_err_coarse));

% 3. 计算离散 L2 范数 (RMS 近似，反映平均能量)
E_rms_fine   = rms(inner_err_fine);
E_rms_coarse = rms(inner_err_coarse);

% 4. 计算全局收敛阶 (Global Convergence Order)
% Order = log2( ||E_{N/2}|| / ||E_N|| )
global_order_inf = log2(E_inf_coarse / E_inf_fine);
global_order_l2  = log2(E_rms_coarse / E_rms_fine);

%% 3. 结果格式化输出
fprintf('\n-------------------------------------------\n');
fprintf('Haar Wavelet Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (n=%d):\t %.6f 秒\n', n, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', n/2, n);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');