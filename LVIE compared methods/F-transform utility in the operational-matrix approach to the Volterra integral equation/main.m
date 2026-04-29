%% F-transform Volterra Integral Equation Solver (Performance & Convergence)
%  Implementation of: F-transform utility in the operational-matrix approach
%  Target: Example 4 (Exponential) as defined in the input function handles
%  Analysis: Performance Loop (N) + Convergence Order (N vs N/2)

clc; clear; close all;

%% 1. 问题定义与参数设置 (Problem Definition)
a = 0; 
b = 1;
lambda = 1; % Model parameter

% --- Example 4 (Exponential) ---
% Exact Solution: u(x) = exp(x)
fx = @(x) exp(x) - 1./(1+x).*(exp(x.*(1+x))-1);
kf = @(x,t) exp(x.*t);
RF = @(x) exp(x);

% 算法参数
N = 400;             % 细网格阶数 (Fine Grid) - 因 integral2 较慢，建议设小
vp = 1024;          % 验证点数量
num_runs = 5;       % 测速循环次数 (建议维持较低值)

%% 2. 循环测速 (Performance Test for Fine Grid N)
fprintf('-------------------------------------------\n');
fprintf('F-transform Volterra Solver (Fine N=%d, Coarse N=%d)\n', N, N/2);
fprintf('注意: 使用 integral2 进行双重积分，计算耗时较长。\n');

time_records = zeros(num_runs, 1);
x_validate = linspace(a, b, vp)'; % 验证点列向量

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
[~, ~] = Solve_Volterra_Operational(a, b, N, lambda, fx, kf);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (N=%d)...\n', num_runs, N);
for i = 1:num_runs
    t_tick = tic;
    
    % 1. 求解获得系数与节点
    [U_coeffs, nodes] = Solve_Volterra_Operational(a, b, N, lambda, fx, kf);
    
    % 2. 重构解 (插值到 1024 个验证点)
    % F-transform 使用三角基函数，linear 插值与之匹配
    u_approx_fine = interp1(nodes, U_coeffs, x_validate, 'linear');
    
    % 3. 计算误差向量
    u_true = RF(x_validate);
    err_vec_fine = abs(u_approx_fine - u_true);
    
    time_records(i) = toc(t_tick);
    fprintf('  Run %d/%d: %.4f s\n', i, num_runs, time_records(i));
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 N=%d 平均耗时: %.6f 秒\n', N, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (N/2)
fprintf('正在计算粗网格对照组 (N=%d)...\n', N/2);
[U_coeffs_c, nodes_c] = Solve_Volterra_Operational(a, b, N/2, lambda, fx, kf);
u_approx_coarse = interp1(nodes_c, U_coeffs_c, x_validate, 'linear');
err_vec_coarse = abs(u_approx_coarse - u_true);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免 Volterra 方程在端点处的数值奇异性干扰统计
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
fprintf('F-transform Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (N=%d):\t %.6f 秒\n', N, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', N/2, N);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% 简单绘图验证
figure('Color', 'w', 'Name', 'F-transform Volterra Results');
subplot(2,1,1);
plot(x_validate, u_true, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Exact');
hold on;
plot(x_validate, u_approx_fine, 'r--', 'LineWidth', 1.5, 'DisplayName', ['Approx N=', num2str(N)]);
legend('show', 'Location', 'best');
title('Solution Comparison'); grid on; xlabel('x'); ylabel('u(x)');

subplot(2,1,2);
semilogy(x_validate(2:end-1), inner_err_fine, 'b-', 'LineWidth', 1.2);
title('Interior Error Distribution (Log Scale)');
xlabel('x'); ylabel('|Error|');
grid on; xlim([a, b]);

%% -------------------------------------------------------------------------
%  Auxiliary Solver Function (Core Algorithm - Unchanged Logic)
% -------------------------------------------------------------------------
function [U, t_nodes] = Solve_Volterra_Operational(a, b, n, lambda, g_func, k_func)
    % Solves (I - lambda * (K^T .* P)^T) * U = G
    % Based on Paper Section 5, Equations (46)-(49)
    
    h = (b - a) / (n - 1);
    t_nodes = linspace(a, b, n)';
    
    % --- Step 1: Operational Matrix P ---
    P = zeros(n, n);
    for i = 1:n
        for j = 1:n
            if t_nodes(j) > t_nodes(max(1, i-1))
                 upper_limit = t_nodes(j);
                 supp_start = t_nodes(max(1, i-1));
                 supp_end = t_nodes(min(n, i+1));
                 
                 eff_start = max(a, supp_start);
                 eff_end = min(upper_limit, supp_end);
                 
                 if eff_end > eff_start
                     f_Ai = @(tau) basic_triangular_basis(tau, i, t_nodes, h);
                     P(i,j) = integral(f_Ai, eff_start, eff_end, 'ArrayValued', true);
                 end
            end
        end
    end
    
    % --- Step 2: F-transform of f(x) -> Vector G ---
    G = zeros(n, 1);
    for i = 1:n
        denom = (i == 1 || i == n) * (h / 2) + (i > 1 && i < n) * h;
        supp_start = t_nodes(max(1, i-1));
        supp_end = t_nodes(min(n, i+1));
        
        numerator = integral(@(t) basic_triangular_basis(t, i, t_nodes, h) .* g_func(t), ...
                             supp_start, supp_end, 'ArrayValued', true);
        G(i) = numerator / denom;
    end
    
    % --- Step 3: F-transform of Kernel k(x,t) -> Matrix K ---
    K_mat = zeros(n, n);
    for i = 1:n
        for j = 1:n
            denom_i = (i==1 || i==n) * (h/2) + (i>1 && i<n) * h;
            denom_j = (j==1 || j==n) * (h/2) + (j>1 && j<n) * h;
            denom = denom_i * denom_j;
            
            t_start = t_nodes(max(1, i-1)); t_end = t_nodes(min(n, i+1));
            s_start = t_nodes(max(1, j-1)); s_end = t_nodes(min(n, j+1));
            
            % Double integration (Expensive Step)
            integrand = @(t,s) basic_triangular_basis(t, i, t_nodes, h) .* ...
                               basic_triangular_basis(s, j, t_nodes, h) .* k_func(t,s);
            
            val = integral2(integrand, t_start, t_end, s_start, s_end);
            K_mat(i,j) = val / denom;
        end
    end
    
    % --- Step 4: Solve Linear System ---
    Identity = eye(n);
    Operational_Matrix = (K_mat') .* P; % Hadamard product
    System_Matrix = Identity - lambda * (Operational_Matrix');
    U = System_Matrix \ G;
end

function val = basic_triangular_basis(t, i, t_nodes, h)
    % Triangular fuzzy partition basis functions
    val = zeros(size(t));
    center = t_nodes(i);
    
    if i > 1
        left = t_nodes(i-1);
        mask_l = (t >= left) & (t <= center);
        val(mask_l) = (t(mask_l) - left) / h;
    end
    
    if i < length(t_nodes)
        right = t_nodes(i+1);
        mask_r = (t > center) & (t <= right);
        val(mask_r) = (right - t(mask_r)) / h;
    end
    
    % Boundary conditions
    if i == 1
        mask = (t >= t_nodes(1)) & (t <= t_nodes(2));
        val(mask) = (t_nodes(2) - t(mask)) / h;
    elseif i == length(t_nodes)
        mask = (t >= t_nodes(end-1)) & (t <= t_nodes(end));
        val(mask) = (t(mask) - t_nodes(end-1)) / h;
    end
end