%% Numerical solution of nonlinear Volterra integral equations using GKFM method (Performance Test Framework)
clc; close all; clear all;

% --- 参数定义 ---
n = 64;             % 细网格阶数 (Fine Grid)
iter = 100;         % 最大迭代次数
tol = 1e-6;         % 误差容限
vp = 1024;           % 验证点数量 (Validation Points)
num_runs = 10;      % 测速循环次数

%% Example Function Definitions (基于原代码的"问题4 / Example 5")
fx = @(x) ((1+x).*exp(-10.*x)+1).^0.5+(1+x).*(1-exp(-10.*x))+10.*(1+x).*log(1+x);
kf = @(x,t,u) -10.*(1+x)./(1+t).*u.^2;
dkut = @(x,t,u) -20.*(1+x)./(1+t).*u;
RF = @(x) ((1+x).*exp(-10.*x)+1).^0.5;

%% 1. 循环测速 (Performance Test for Fine Grid n)
fprintf('-------------------------------------------\n');
fprintf('正在准备 NLVIE_GKFM_K1 测试 (Fine n=%d, Coarse n=%d)...\n', n, n/2);

time_records = zeros(num_runs, 1);

% [预热] Warm-up (避免首次运行 JIT 编译时间影响)
fprintf('正在进行预热运行...\n');
[~, ~, ~] = NLVIE_K1(fx, dkut, n, iter, tol);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (n=%d)...\n', num_runs, n);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格 (Fine Grid)
    [ua_iter1, tk_xj1, cg_it1] = NLVIE_K1(fx, dkut, n, iter, tol);
    
    % 计算细网格误差
    % 提取最终收敛步 cg_it1 的结果进行评估，调用 PE1 函数
    pointwise_error_fine = PE1(RF, ua_iter1(:, cg_it1), vp);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 n=%d 平均耗时: %.6f 秒 (收敛步数: %d)\n', n, avg_time, cg_it1);

%% 2. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (n/2)
fprintf('正在计算粗网格对照组 (n=%d)...\n', n/2);
[ua_iter2, tk_xj2, cg_it2] = NLVIE_K1(fx, dkut, n/2, iter, tol);
pointwise_error_coarse = PE1(RF, ua_iter2(:, cg_it2), vp);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
inner_err_fine   = pointwise_error_fine(2:end-1);
inner_err_coarse = pointwise_error_coarse(2:end-1);

% 2. 计算离散 L-infinity 范数 (内部最大绝对误差)
E_inf_fine   = max(abs(inner_err_fine));
E_inf_coarse = max(abs(inner_err_coarse));

% 3. 计算离散 L2 范数 (RMS 近似，反映全局误差的平均能量)
E_rms_fine   = rms(inner_err_fine);
E_rms_coarse = rms(inner_err_coarse);

% 4. 计算全局收敛阶 (Global Convergence Order)
global_order_inf = log2(E_inf_coarse / E_inf_fine);
global_order_l2  = log2(E_rms_coarse / E_rms_fine);

%% 3. 结果格式化输出
fprintf('\n-------------------------------------------\n');
fprintf('NLVIE_GKFM_K1 Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (n=%d):\t %.6f 秒\n', n, avg_time);
fprintf('细网格收敛迭代步数:\t %d\n', cg_it1);
fprintf('粗网格收敛迭代步数:\t %d\n', cg_it2);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', n/2, n);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');
%% 问题1
% fx=@(x)tan(x)-1/4*sin(2*x)-1/2*x;
% kf=@(x,t)1;
% kut=@(u)1./(1+u.^2);
% dkut=@(x,t,u)-2*u./((1+u.^2).^2);
% RF=@(x)tan(x);
%% 问题2
% fx=@(x)(1+11/9*x+2/3*x.^2-1/3*x.^3+2/9*x.^4).*log(x+1)-1/3*(x+x.^4).*(log(x+1).^2)...
%     -11/9*(x.^2)+5/18*(x.^3)-2/27*(x.^4);
% kf=@(x,t,u)x.*(t.^2).*(u.^2);
% kut=@(u)u.^2;
% dkut=@(x,t,u)x.*(t.^2).*(2*u);
% RF=@(x)log(x+1);
%% 问题3
% fx=@(x)exp(x)-x.*exp(-x);
% kf=@(x,t,u)exp(-x).*exp(-2.*t).*(u.^2);
% kut=@(u)u.^2;
% dkut=@(x,t,u)exp(-x).*exp(-2.*t).*(2*u);
% RF=@(x)exp(x);
%% 问题4 Example 5 Numerical Solution of Nonlinear Fredholm and Volterra Integrals by 
% Newton–Kantorovich and Haar Wavelets Methods Numerical Solution of Nonlinear Fredholm and
% Volterra Integrals by Newton–Kantorovich and Haar Wavelets Methods
% fx=@(x)3/2-1/2*exp(-2*x);
% kf=@(x,t,u)-(u.^2+u);
% dkut=@(x,t,u)-(2*u+1);
% RF=@(x)exp(-x);
%% 问题5
% fx = @(x) ((1+x).*exp(-10.*x)+1).^0.5+(1+x).*(1-exp(-10.*x))+10.*(1+x).*log(1+x);
% kf = @(x,t,u) -10.*(1+x)./(1+t).*u.^2;
% dkut = @(x,t,u) -20.*(1+x)./(1+t).*u;
% RF = @(x) ((1+x).*exp(-10.*x)+1).^0.5;