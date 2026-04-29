% 复现论文 Section 3.2: Linear Volterra integral equations of the second kind
% 测试算例: Example 5.4 (Bernoulli Polynomial Method)
% 严谨科研风格代码 - 性能测试与收敛分析版

clc; clear; close all;

%% 1. 问题定义 (Problem Definition)
% Example 4 from User Context (Linear Volterra)
% u(x) = f(x) + int_0^x k(x,t)u(t) dt

% Exact Solution: u(x) = exp(x)
fx = @(x) exp(x) - 1./(1+x).*(exp(x.*(1+x))-1);
kf = @(x,t) exp(x.*t);
RF = @(x) exp(x);

% 参数设置
N = 8;             % 细网格阶数 (Fine Grid) [Bernoulli 方法通常不需要很大的 N]
vp = 1024;          % 验证点数量
num_runs = 10;      % 测速循环次数

%% 2. 循环测速 (Performance Test for Fine Grid N)
fprintf('-------------------------------------------\n');
fprintf('Bernoulli Volterra Solver (Fine N=%d, Coarse N=%d)\n', N, N/2);

time_records = zeros(num_runs, 1);
x_test = linspace(0, 1, vp)'; % 验证点向量

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
[~, ~] = solve_volterra_linear_2nd(kf, fx, N);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (N=%d)...\n', num_runs, N);
for i = 1:num_runs
    t_tick = tic;
    
    % 1. 求解获得函数句柄
    [u_approx_handle, ~] = solve_volterra_linear_2nd(kf, fx, N);
    
    % 2. 在验证点上评估 (计算耗时应包含评估过程)
    u_fine_vals = u_approx_handle(x_test);
    
    % 3. 计算误差向量
    exact_vals = RF(x_test);
    err_vec_fine = abs(exact_vals - u_fine_vals);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 N=%d 平均耗时: %.6f 秒\n', N, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (N/2)
fprintf('正在计算粗网格对照组 (N=%d)...\n', N/2);
[u_coarse_handle, ~] = solve_volterra_linear_2nd(kf, fx, N/2);
u_coarse_vals = u_coarse_handle(x_test);
err_vec_coarse = abs(exact_vals - u_coarse_vals);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免 Volterra 方程初始点 (t=0) 的零误差导致 log(0) 或数值统计偏差
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
fprintf('Bernoulli Polynomial Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (N=%d):\t %.6f 秒\n', N, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', N/2, N);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% 简单绘图验证
figure('Color', 'w', 'Name', 'Bernoulli Volterra Analysis');
subplot(2, 1, 1);
plot(x_test, exact_vals, 'k-', 'LineWidth', 1.5); hold on;
plot(x_test, u_fine_vals, 'r--', 'LineWidth', 1.5);
legend('Exact Solution', ['Bernoulli (N=', num2str(N), ')'], 'Location', 'best');
title(['Solution Comparison (N=', num2str(N), ')']); grid on;

subplot(2, 1, 2);
semilogy(x_test(2:end-1), inner_err_fine, 'b-', 'LineWidth', 1.2); 
title('Interior Error Distribution (Log Scale)');
xlabel('x'); ylabel('|Error|');
grid on; xlim([0, 1]);

%% ========================================================================
%  核心求解器: 第二类线性 Volterra 积分方程
%  ========================================================================
function [u_approx_handle, t_cost] = solve_volterra_linear_2nd(kf, fx, N)
    % 求解: u(x) - int_0^x k(x,t)u(t) dt = f(x)
    % 方法: Bernoulli Polynomial Collocation
    
    t_start = tic;
    
    % 1. 生成配置点 (Newton-Cotes nodes)
    indices = 0:N;
    x_nodes = (2 * indices + 1) ./ (2 * (N + 1));
    x_nodes = x_nodes(:); 
    n_nodes = length(x_nodes);
    
    % 2. 预计算伯努利数
    B_nums_vec = get_bernoulli_nums_vector(N);
    
    % 3. 组装线性方程组 A * C = F
    A = zeros(n_nodes, N + 1);
    F = fx(x_nodes);
    
    for i = 1:n_nodes
        xi = x_nodes(i);
        
        for j = 0:N
            % 项1: 基函数值
            term_basis = bernoulli_poly_fast(j, xi, B_nums_vec);
            
            % 项2: 积分项
            if xi > 1e-14
                integrand = @(t) kf(xi, t) .* bernoulli_poly_fast(j, t, B_nums_vec);
                term_integral = integral(integrand, 0, xi, 'AbsTol', 1e-12, 'RelTol', 1e-12, 'ArrayValued', true);
            else
                term_integral = 0;
            end
            
            A(i, j+1) = term_basis - term_integral;
        end
    end
    
    % 4. 求解系数
    C = A \ F;
    
    % 5. 返回函数句柄
    u_approx_handle = @(x) evaluate_bernoulli_basis(C, x, B_nums_vec);
    
    t_cost = toc(t_start);
end

%% ========================================================================
%  辅助函数 (Basis & Coeffs)
%  ========================================================================
function val = evaluate_bernoulli_basis(C, x, B_nums_vec)
    % 计算伯努利级数
    val = zeros(size(x));
    N = length(C) - 1;
    for j = 0:N
        val = val + C(j+1) * bernoulli_poly_fast(j, x, B_nums_vec);
    end
end

function val = bernoulli_poly_fast(n, x, B_all)
    % 计算第 n 阶伯努利多项式
    val = zeros(size(x));
    for k = 0:n
        bk = B_all( (n-k) + 1 );
        coef = nchoosek(n, k) * bk;
        val = val + coef * (x.^k);
    end
end

function B_vec = get_bernoulli_nums_vector(max_order)
    % 生成伯努利数 B_0 ... B_N
    B_vec = zeros(1, max_order + 1);
    B_vec(1) = 1; 
    if max_order >= 1, B_vec(2) = -0.5; end
    for n = 2:max_order
        s = 0;
        for k = 0:(n-1)
            s = s + nchoosek(n+1, k) * B_vec(k+1);
        end
        B_vec(n+1) = -s / (n+1);
    end
end