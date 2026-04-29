%% An Effective Scheme for Solving Nonlinear Volterra Integral Equations
% 核心逻辑: 循环测速 + 全局误差收敛阶分析 (L-inf & RMS)

clc; close all; clear all;

%% 1. 问题定义与参数设置
n = 512;            % 细网格阶数 (Fine Grid)
vp = 1024;          % 验证点数量
iter = 100;         % 最大迭代次数
tol = 1e-8;         % 迭代容差
K = 2;              % 基函数阶数
num_runs = 10;      % 测速循环次数
%% 问题1

% fx=@(x)tan(x)-1/4*sin(2*x)-1/2*x;

% kf=@(x,t)1;

% kut=@(u)1./(1+u.^2);

% dkut=@(x,t,u)-2*u./((1+u.^2).^2);

% RF=@(x)tan(x);

%% 问题2

% fx=@(x)(1+11/9*x+2/3*x.^2-1/3*x.^3+2/9*x.^4).*log(x+1)-1/3*(x+x.^4).*(log(x+1).^2)...

% -11/9*(x.^2)+5/18*(x.^3)-2/27*(x.^4);

% kf=@(x,t,u)x.*(t.^2).*(u.^2);

% kut=@(u)u.^2;

% dkut=@(x,t,u)x.*(t.^2).*(2*u);

% RF=@(x)log(x+1);1`

%% 问题3

% fx=@(x)exp(x)-(x.*exp(-x));

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

%% 问题5 A Multistep Legendre Pseudo-Spectral Method for Nonlinear Volterra Integral Equations

% lamba=100;
% 
% fx=@(x) -1/4.*x.^2+1./(4.*lamba.^2).*sin(lamba.*x).^2-x./(4*lamba).*sin(2.*lamba.*x)+cos(lamba.*x);
% 
% kf=@(x,t,u)t.*(u.^2);
% 
% dkut=@(x,t,u)2.*t.*(u);
% 
% RF=@(x)cos(lamba.*x);
%% 问题6 Efficient numerical methods based on general linear methods for Volterra integral equations
fx = @(x) ((1+x).*exp(-10.*x)+1).^0.5+(1+x).*(1-exp(-10.*x))+10.*(1+x).*log(1+x);
kf = @(x,t,u) -10.*(1+x)./(1+t).*u.^2;
dkut = @(x,t,u) -20.*(1+x)./(1+t).*u;
RF = @(x) ((1+x).*exp(-10.*x)+1).^0.5;

%% 2. 初始值预计算
fprintf('正在计算 Volterra 初值猜测 (Get_Volterra_Init_Coeffs)...\n');
init_coeffs = Get_Volterra_Init_Coeffs(fx, kf, dkut, 0, 1, K);

%% 3. 循环测速 (Performance Test for Fine Grid n)
fprintf('-------------------------------------------\n');
fprintf('正在准备 NLVIE 测试 (Fine n=%d, Coarse n=%d)...\n', n, n/2);

time_records = zeros(num_runs, 1);

% [预热] Warm-up (运行一次以加载 JIT)
fprintf('正在进行预热运行...\n');
[~, ~, ~] = NLVIE_PGFM_NC_oscillatory(fx, dkut, n, iter, tol, K, kf, init_coeffs);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (n=%d)...\n', num_runs, n);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格 (Fine Grid)
    [ua_iter_fine, tk_xj_fine, cg_it_fine] = NLVIE_PGFM_NC_oscillatory(fx, dkut, n, iter, tol, K, kf, init_coeffs);
    
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
[ua_iter_coarse, tk_xj_coarse, cg_it_coarse] = NLVIE_PGFM_NC_oscillatory(fx, dkut, n/2, iter, tol, K, kf, init_coeffs);
pointwise_error_coarse = PE1_PGFM(RF, ua_iter_coarse(:, cg_it_coarse), vp, K);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% Volterra 方程初始点通常被强制约束为准确值，误差为0，去除以避免 log(0) 干扰
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
fprintf('NLVIE (Oscillatory/General) 精度与收敛阶分析报告\n');
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
    fprintf('注意: 收敛阶为 %.2f，请检查 Volterra 方程刚性或迭代收敛性。\n', global_order_inf);
end