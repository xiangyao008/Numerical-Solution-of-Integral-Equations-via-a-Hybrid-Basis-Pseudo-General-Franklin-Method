clc; close all; clear all;

%% 线性 Fredholm 积分方程 (LFIE) 测试主程序
% 对应方法: Pesudu Franklin Newton-Cotes (Linear Version)
n = 128;            % 细网格阶数 (Fine Grid)
vp = 1024;          % 验证点数量
K = 4;              % 伪Franklin基函数阶数 (0,1,2,3,4)

%% Example 1
% RF = @(x) sin(pi*x); % 真解
% kf = @(x,t) 1/5 .* cos(pi*x) .* sin(pi*t); % 线性核函数
% fx = @(x) sin(pi*x) - 0.1 .* cos(pi*x);

%% Example 2
% fx=@(x)exp(x) + exp(-x);
% kf=@(x,t)-exp(-x-t);
% RF=@(x)exp(x);

%% Example 3
% Approximate solution of Fredholm integral equations of the second kind through multinode Shepard operators
fx = @(x) exp(x);
kf = @(x,t) 2 .* exp(x+t);
RF = @(x) exp(x) ./ (2 - exp(2));

%% 循环测速与精度计算
num_runs = 10;
time_records = zeros(num_runs, 1);

fprintf('正在准备 LFIE 测试 (Fine n=%d, Coarse n=%d, K=%d)...\n', n, n/2, K);

% 1. 预热 (Warm-up)
fprintf('正在进行预热运行...\n');
[~,~] = LFIE_PGFM_NC(fx, n, K, kf);

% 2. 循环运行
% 注意：为了计算收敛阶，我们需要两套网格的数据。
% 在循环中我们同时计算 n 和 n/2 的情况。
% 这里的 pointwise_error1/2 假定为长度为 vp 的误差向量。

for i = 1:num_runs
    t_start = tic;
    
    % --- 细网格 (Fine Grid, N) ---
    [ua_coeffs1, tk_xj1] = LFIE_PGFM_NC(fx, n, K, kf);
    % 计算逐点误差向量 (假设返回向量)
    pointwise_error_fine = PE1_PGFM(RF, ua_coeffs1, vp, K); 
  time_records(i) = toc(t_start);
end
    % --- 细网格 (Fine Grid, N) ---
    [ua_coeffs1, tk_xj1] = LFIE_PGFM_NC(fx, n, K, kf);
    % 计算逐点误差向量 (假设返回向量)
    pointwise_error_fine = PE1_PGFM(RF, ua_coeffs1, vp, K); 
    
    % --- 粗网格 (Coarse Grid, N/2) ---
    [ua_coeffs2, tk_xj2] = LFIE_PGFM_NC(fx, n/2, K, kf);
    % 计算逐点误差向量
    pointwise_error_coarse = PE1_PGFM(RF, ua_coeffs2, vp, K);
%% 3. 统计测速结果
fprintf('-------------------------------------------\n');
fprintf('LFIE 性能测试完成。\n');
fprintf('平均运行时间 (含两次求解及误差计算): %.6f 秒\n', mean(time_records));

%% 4. 精度与全局收敛阶分析 (L-infinity Norm)
% 注意：假设 pointwise_error 是列向量或行向量均可，(2:end-1) 通用
inner_error_fine   = pointwise_error_fine(2:end-1);
inner_error_coarse = pointwise_error_coarse(2:end-1);

% 2. 计算离散 L-infinity 范数 (内部最大绝对误差)
E_inf_fine   = max(abs(inner_error_fine));   
E_inf_coarse = max(abs(inner_error_coarse)); 

% 3. 计算全局收敛阶 (Global Convergence Order)
% 公式: p = log2( ||E_{N/2}|| / ||E_N|| )
global_order_inf = log2(E_inf_coarse / E_inf_fine);

% 4. 计算 L2 范数 (RMS近似，同样去除首尾)
E_rms_fine   = rms(inner_error_fine);
E_rms_coarse = rms(inner_error_coarse);
global_order_l2 = log2(E_rms_coarse / E_rms_fine);

% --- 输出详细分析报告 ---
fprintf('-------------------------------------------\n');
fprintf('全局误差与收敛阶分析 (Interior Points Only)\n');
fprintf('-------------------------------------------\n');
fprintf('说明: 已剔除区间首尾各 1 个点，评估内部 %d 个点的误差。\n', length(inner_error_fine));
fprintf('网格规模 Comparison:\t N = %d vs N = %d\n', n, n/2);
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', n/2, n);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');