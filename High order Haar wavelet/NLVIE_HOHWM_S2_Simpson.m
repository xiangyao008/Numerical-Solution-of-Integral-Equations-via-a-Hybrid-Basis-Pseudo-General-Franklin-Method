%% An Effective and Simple Scheme for Solving Nonlinear Fredholm Integral Equations Nonlinear Fredholm Integral Equations
clc;close all;clear all;
n=256;iter=100;tol = 1e-6; 
%% 问题1
% fx=@(x)tan(x)-1/4*sin(2*x)-1/2*x;
% kf=@(x,t)1;
% kut=@(u)1./(1+u.^2);
% dkut=@(x,t,u)-2*u./((1+u.^2).^2);
% RF=@(x)tan(x);
%% 问题2 Third kind of NLVIE 
% fx=@(x)(1+11/9*x+2/3*x.^2-1/3*x.^3+2/9*x.^4).*log(x+1)-1/3*(x+x.^4).*(log(x+1).^2)...
%     -11/9*(x.^2)+5/18*(x.^3)-2/27*(x.^4);
% kf=@(x,t,u)x.*(t.^2).*(u.^2);
% kut=@(u)u.^2;
% dkut=@(x,t,u)x.*(t.^2).*(2*u);
% RF=@(x)log(x+1);
%% 问题3
% fx=@(x)exp(x)-(x.*exp(-x));
% kf=@(x,t,u)exp(-x).*exp(-2.*t).*(u.^2);
% kut=@(u)u.^2;
% dkut=@(x,t,u)exp(-x).*exp(-2.*t).*(2*u);
% RF=@(x)exp(x);
%% 问题4 Example 5 Numerical Solution of Nonlinear Fredholm and Volterra Integrals by 
% Newton–Kantorovich and Haar Wavelets Methods Numerical Solution of Nonlinear Fredholm and
% Volterra Integrals by Newton–Kantorovich and Haar Wavelets Methods
fx=@(x)3/2-1/2*exp(-2*x);
kf=@(x,t,u)-(u.^2+u);
dkut=@(x,t,u)-(2*u+1);
RF=@(x)exp(-x);
%% 问题5 Example 5 On the numerical solution of linear and nonlinear volterra integral and integro-differential equations
% fx=@(x)x-x.^2-1/4.*x.^5+2/5.*x.^6-1/6.*x.^7;
% kf=@(x,t,u)x.*t.*(u.^2);
% dkut=@(x,t,u)x.*t.*(2.*u);
% RF=@(x)x-x.^2;
num_runs = 10;                     % 重复运行次数
time_records = zeros(num_runs, 1); % 预分配内存存储时间

fprintf('正在初始化测试，共计 %d 次...\n', num_runs);

%% 1. 预热 (Warm-up)
% 这一步非常重要：
% 首次运行 MATLAB 函数时会触发 JIT (Just-In-Time) 编译和内存池分配。
% 我们先空跑一次，确保后续测量的是算法的“稳态”性能，而非加载时间。
fprintf('正在进行预热 (Warm-up)... \n');
try
    [~,~,~] = NLVIE_HAAR_S2_Simpson(fx,dkut, n, iter,tol);
    [~,~,~] = NLVIE_HAAR_S2_Simpson(fx,dkut, n/2, iter,tol);
catch ME
    error('预热失败，请检查输入参数是否正确: %s', ME.message);
end

%% 2. 循环测速
fprintf('开始正式测速...\n');

for i = 1:num_runs
    t_run_start = tic; % === 计时开始 ===
    
    % ================= [ 核心计算区 ] =================
    % 1. 求解 n 配置点的解和误差
    [ua_iter1, tk_xj1, cg_it1] = NLVIE_HAAR_S2_Simpson(fx, dkut, n, iter, tol);
    % 注意：PE2_HOHWM 的最后一个参数 256 是误差检测的分辨率，保持不变
    pointwise_error1 = PE2_HOHWM(RF, ua_iter1(:, cg_it1), 1024);
    
    % 2. 求解 n/2 配置点的解和误差
    [ua_iter2, tk_xj2, cg_it2] = NLVIE_HAAR_S2_Simpson(fx, dkut, n/2, iter, tol);
    pointwise_error2 = PE2_HOHWM(RF, ua_iter2(:, cg_it2), 1024);
    
    % 3. 计算收敛阶 (Speed of Convergence)
    % 这里假设误差是标量或同维向量，使用点除 ./ 保证健壮性
    SP = log(pointwise_error2 ./ pointwise_error1) / log(2);
    % =================================================
    
    time_records(i) = toc(t_run_start); % === 计时结束 ===
    
    fprintf('第 %2d 次运行耗时: %.6f 秒\n', i, time_records(i));
end

%% 3. 数据统计与输出
avg_time = mean(time_records);
std_time = std(time_records);
fprintf('-------------------------------------------\n');
fprintf('测试完成。\n');
fprintf('平均运行时间: %.6f 秒\n', avg_time);
fprintf('标准差:       %.6f 秒 (反映耗时波动情况)\n', std_time);