% GeneralizedSplineSolver_Optimized (修复版)
% Solves the Fredholm integral equation: x(t) = f(t) + int K(t,s)x(s) ds
% 修复点: 补全了 build_matrix_A_Symmetric 中缺失的变量 sin_2wl

clc; clear; close all;

%% 1. 问题定义与参数设置
% --- User Defined Problem ---
fx = @(x) exp(x);
% Decomposed Kernel components for Optimization: K(t,s) = k1(t) * k2(s)
% k(x,t) = 2*exp(x+t) -> k1(t)=2*exp(t), k2(s)=exp(s)
k1 = @(t) 2 * exp(t);
k2 = @(s) exp(s);

RF = @(x) exp(x)./(2-exp(2)); % Exact Solution

% --- Algorithm Parameters ---
N = 8;             % Fine Grid Size
omega = 2;          % Robust parameter for Exponential Spline
vp = 1024;          % Validation Points count
num_runs = 5;       % 测速循环次数

%% 2. 循环测速 (Performance Test for Fine Grid N)
fprintf('-------------------------------------------\n');
fprintf('Generalized Spline Solver (Fine N=%d, Coarse N=%d, Omega=%.1f)\n', N, N/2, omega);

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
[~, ~, ~] = solve_problem(N, omega, fx, k1, k2, RF);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (N=%d)...\n', num_runs, N);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格 (Fine Grid)
    [t_dense, vals_fine, exact_fine] = solve_problem(N, omega, fx, k1, k2, RF);
    
    time_records(i) = toc(t_tick);
    fprintf('  Run %d/%d: %.4f s\n', i, num_runs, time_records(i));
end

% 计算细网格的逐点误差向量
err_vec_fine = abs(vals_fine - exact_fine);
avg_time = mean(time_records);

fprintf('网格 N=%d 平均耗时: %.6f 秒\n', N, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (N/2)
fprintf('正在计算粗网格对照组 (N=%d)...\n', N/2);
[~, vals_coarse, exact_coarse] = solve_problem(N/2, omega, fx, k1, k2, RF);
err_vec_coarse = abs(vals_coarse - exact_coarse);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
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
fprintf('Generalized Spline Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (N=%d):\t %.6f 秒\n', N, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', N/2, N);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% 简单绘图验证
figure('Color','w', 'Name', 'Generalized Spline Results');
subplot(2,1,1);
plot(t_dense, exact_fine, 'k-', 'LineWidth', 1.5); hold on;
plot(t_dense, vals_fine, 'r--', 'LineWidth', 1.5);
legend('Exact Solution', ['Approx (N=' num2str(N) ')']); 
title('Solution Comparison'); grid on; xlabel('t');

subplot(2,1,2);
semilogy(t_dense(2:end-1), inner_err_fine, 'b-', 'LineWidth', 1.2);
title('Interior Error Distribution (Log Scale)'); 
grid on; xlabel('t'); ylabel('|Error|');

%% --- 核心求解函数 ---
function [t_dense, vals, exact_vals] = solve_problem(n, omega, fx, k1, k2, exact_func)
    l = 1.0 / n;
    ti = linspace(0, 1, n + 1); 
    
    % 1. Build Coefficient Matrix A (Symmetric)
    A = build_matrix_A_Symmetric(n, l, omega);
    
    % 2. Build Integral Matrix N (Optimized)
    N_mat = build_matrix_N_Optimized(n, l, omega, ti, k1, k2);
    
    % 3. Build RHS Vector F
    F_vec = arrayfun(fx, ti');
    
    % 4. Solve Linear System
    I_mat = eye(n + 1);
    NA = N_mat * A;
    X_nodes = (I_mat - NA) \ F_vec;
    
    % 5. Convert to B-Spline Coefficients
    Coeffs = A * X_nodes;
    
    % 6. Dense Evaluation
    t_dense = linspace(0, 1, 1024)';
    vals = eval_spline_dense(t_dense, Coeffs, n, l, omega);
    exact_vals = arrayfun(exact_func, t_dense);
end

%% --- Matrix Construction Helpers ---
function N_mat = build_matrix_N_Optimized(n, l, w, ti, k1, k2)
    num_basis = n + 3;
    norm_factor = (w * l) / (2.0 * (1.0 - cos(w * l)));
    J_vec = zeros(1, num_basis);
    
    B2 = @(tau) (tau >= 0 & tau < l) .* (tau / l) + ...
                (tau >= l & tau < 2*l) .* ((2*l - tau) / l);
    
    for v = 1:num_basis
        t_v = (v - 4) * l;
        if t_v + 4*l <= 0 || t_v >= 1
            J_vec(v) = 0;
            continue;
        end
        a = 1.0; 
        integrand = @(tau) B2(tau) .* calc_inner_analytical(tau, t_v, a, w, l, norm_factor);
        val = integral(integrand, 0, 2*l, 'ArrayValued', true, 'RelTol', 1e-12, 'AbsTol', 1e-15);
        J_vec(v) = val / l;
    end
    K1_vec = arrayfun(k1, ti'); 
    N_mat = K1_vec * J_vec; 
end

function A = build_matrix_A_Symmetric(n, l, w)
    num_basis = n + 3; 
    num_nodes = n + 1; 
    A = zeros(num_basis, num_nodes);
    
    wl = w * l;
    sin_wl = sin(wl); cos_wl = cos(wl);
    cos_2wl = cos(2*wl); 
    sin_2wl = sin(2*wl); % <--- 修复点：添加了缺失的 sin_2wl 定义
    
    csc_wl = 1.0 / sin_wl;
    cot_wl = 1.0 / tan(wl);
    csc_half_sq = 1.0 / (sin(wl/2)^2);
    csc_half_4 = csc_half_sq^2;
    
    % Coefficients
    C0 = -0.25 * csc_half_sq * (wl * (2*cos_2wl + 1) * csc_wl - 3);
    C1 = (4 - wl*(cos_wl + 2*cos_2wl + 1)*csc_wl)/(cos_wl - 1) + 3;
    C2 = -0.25 * csc_wl * csc_half_sq * (wl - 3*sin_wl - 2*sin_2wl + 2*wl*(2*cos_wl + cos_2wl));
    C3 = (1 - wl * cot_wl) / (cos_wl - 1);
    
    D0 = (wl * cot_wl - 1) / (cos_wl - 1);
    D1 = 0.25 * (4*cos_wl + 1) * (wl * csc_wl - 1) * csc_half_sq;
    D2 = 0.25 * sin_wl * (sin_wl - wl) * csc_half_4;
    D3 = 0.25 * (wl * csc_wl - 1) * csc_half_sq;
    
    A(1, 1:4) = [C0, C1, C2, C3];
    A(2, 1:4) = [D0, D1, D2, D3];
    c_base = num_nodes - 3; 
    A(num_basis-1, c_base:c_base+3) = [D3, D2, D1, D0]; 
    A(num_basis, c_base:c_base+3)   = [C3, C2, C1, C0]; 
    
    psi = -0.25 * csc_half_sq * (wl * csc_wl - 1);
    for i = -1 : (n-3)
        row = i + 4; col = i + 2; 
        A(row, col:col+2) = [psi, 1.0 - 2.0 * psi, psi];
    end
end

%% --- Evaluation & Integration Helpers ---
function vals = eval_spline_dense(t_vals, Coeffs, n, l, w)
    vals = zeros(size(t_vals));
    norm_factor = (w * l) / (2.0 * (1.0 - cos(w * l)));
    for i = 1:length(t_vals)
        t = t_vals(i);
        v_min = max(1, floor(t/l) + 1); 
        v_max = min(length(Coeffs), v_min + 3);
        val_pt = 0;
        for v = v_min:v_max
            t_v = (v - 4) * l;
            if t > t_v && t < t_v + 4*l
                val_pt = val_pt + Coeffs(v) * eval_basis_point(t, t_v, l, w, norm_factor);
            end
        end
        vals(i) = val_pt;
    end
end

function b_val = eval_basis_point(t, t_v, l, w, K)
    low = max(0, t - t_v - 2*l);
    high = min(2*l, t - t_v);
    if low >= high, b_val = 0; return; end
    
    B2 = @(tau) (tau < l) .* (tau/l) + (tau >= l) .* ((2*l - tau)/l);
    N02 = @(x) (x < l) .* (K*sin(w*x)) + (x >= l) .* (K*sin(w*(2*l-x)));
    integrand = @(tau) B2(tau) .* N02(t - t_v - tau);
    b_val = integral(integrand, low, high, 'RelTol', 1e-7) / l;
end

function val = calc_inner_analytical(tau, t_v, a, w, l, K)
    shift = t_v + tau;
    s_start = max(0.0, shift);
    s_end = min(1.0, shift + 2*l);
    val = 0;
    if s_start >= s_end, return; end
    
    p1_s = max(s_start, shift); p1_e = min(s_end, shift + l);
    if p1_s < p1_e
        val = val + int_exp_sin(a, w, -w*shift, p1_s, p1_e) * K;
    end
    p2_s = max(s_start, shift + l); p2_e = min(s_end, shift + 2*l);
    if p2_s < p2_e
        val = val + int_exp_sin_neg_w(a, w, w*(2*l+shift), p2_s, p2_e) * K;
    end
end

function res = int_exp_sin(a, w, phi, t0, t1)
    den = a^2 + w^2;
    res = (exp(a*t1)*(a*sin(w*t1+phi) - w*cos(w*t1+phi)) - ...
           exp(a*t0)*(a*sin(w*t0+phi) - w*cos(w*t0+phi))) / den;
end

function res = int_exp_sin_neg_w(a, w, phi, t0, t1)
    den = a^2 + w^2;
    res = (exp(a*t1)*(a*sin(phi-w*t1) + w*cos(phi-w*t1)) - ...
           exp(a*t0)*(a*sin(phi-w*t0) + w*cos(phi-w*t0))) / den;
end