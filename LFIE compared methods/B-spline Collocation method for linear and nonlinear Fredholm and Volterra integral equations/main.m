% 主程序：复现论文中的数值算例 (高性能优化版 + 收敛性分析)
clc; clear; close all;

%% 1. 参数设置与算例选择
% Example 6.1: Linear Volterra
% Example 6.2: Nonlinear Volterra
% Example 6.3: Linear Fredholm
% Example 6.5: Nonlinear Fredholm

example_id = '6.2'; % 修改此处以切换算例 ('6.1', '6.2', '6.3', '6.5')
N = 64;             % 细网格剖分数量 (建议取偶数，以便计算 N/2)
num_runs = 10;      % 测速循环次数

% 定义算例参数
switch example_id
    case '6.1' % Linear Volterra
        type = 'Volterra';
        a = 0; b = 1;
        %% Example 4 F-transform utility in the operational-matrix approach
        fx=@(x)exp(x)-1./(1+x).*(exp(x.*(1+x))-1);
        kf=@(x,t,y)exp(x.*t).*y;
        RF=@(x)exp(x);
        
    case '6.2' % Nonlinear Volterra
        type = 'Volterra';
        a = 0; b = 1;
        fx=@(x) ((1+x).*exp(-10.*x)+1).^0.5+(1+x).*(1-exp(-10.*x))+10.*(1+x).*log(1+x);
        kf=@(x,t,u)-10.*(1+x)./(1+t).*u.^2;
        RF=@(x)((1+x).*exp(-10.*x)+1).^0.5;
        
    case '6.3' % Linear Fredholm
        type = 'Fredholm';
        a = 0; b = 1;
        RF = @(t) exp(t)./(2-exp(2));
        kf = @(t,x,y) 2.*exp(t+x) .* y;
        fx = @(t) exp(t);

        % RF = @(t) exp(2.*t);
        % kf = @(t,x,y) -1/3.*exp(2.*t-5/3.*x) .* y;
        % fx = @(t) exp(2.*t+1/3);
    case '6.5' % Nonlinear Fredholm
        type = 'Fredholm';
        a = 0; b = 1;
        fprintf('注意: Example 6.5 使用替代算例演示非线性 Fredholm 求解逻辑。\n');
        RF = @(t) sin(t); 
        kf = @(t,x,y) (1/64)*cos(t).*(y.^2); 
        val_int = 0.5 - sin(2)/4;
        fx = @(t) sin(t) - (1/64)*cos(t)*val_int;
        
    otherwise
        error('Unknown example');
end

%% 2. 循环测速 (Performance Test for Fine Grid N)
fprintf('-------------------------------------------\n');
fprintf('Running Example %s (%s)\n', example_id, type);
fprintf('正在准备 B-Spline 测试 (Fine N=%d, Coarse N=%d)...\n', N, N/2);

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
[~, ~, ~] = solve_IE_Bspline_Optimized(fx, kf, RF, a, b, N, type);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (N=%d)...\n', num_runs, N);
for i = 1:num_runs
    t_tick = tic;
    
    % 求解细网格 (Fine Grid)
    % err_vec_fine 包含了在 1024 个点上的逐点误差
    [approx_y_fine, grid_t, err_vec_fine] = solve_IE_Bspline_Optimized(fx, kf, RF, a, b, N, type);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 N=%d 平均耗时: %.6f 秒\n', N, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation (N/2)
fprintf('正在计算粗网格对照组 (N=%d)...\n', N/2);
[approx_y_coarse, ~, err_vec_coarse] = solve_IE_Bspline_Optimized(fx, kf, RF, a, b, N/2, type);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 这里的 grid_t 是 1024 个点，我们去掉首尾以避免边界奇异性或强约束
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
fprintf('B-Spline Optimized Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (N=%d):\t %.6f 秒\n', N, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (N=%d)\t Error (N=%d)\t Order\n', N/2, N);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% 简单绘图验证 (仅绘制细网格结果)
figure('Color','w', 'Name', ['Result Example ' example_id]);
subplot(2,1,1);
plot(grid_t, RF(grid_t), 'k-', 'LineWidth', 1.5); hold on;
plot(grid_t, approx_y_fine, 'r--', 'LineWidth', 1.5);
legend('Exact Solution', 'B-Spline Approx'); 
title(['Example ' example_id ' (N=' num2str(N) ')']); grid on;
xlabel('x'); ylabel('y(x)');

subplot(2,1,2);
semilogy(grid_t, err_vec_fine, 'b-', 'LineWidth', 1.2);
title('Pointwise Absolute Error distribution'); 
grid on; xlabel('x'); ylabel('|Error|');

%% ---------------- 核心求解函数 (优化版 - 保持不变) ---------------- %%
function [y_sol, eval_nodes, err] = solve_IE_Bspline_Optimized(fx, kf, RF, a, b, N, type)
    
    h = (b - a) / N;
    t_nodes = linspace(a, b, N+1)'; % 配点 t_0, ..., t_N
    num_coeffs = N + 3;
    
    % 1. 预计算 Gauss 积分点 (全局展开)
    num_gauss = 10; 
    [gl_ref_x, gl_ref_w] = gauss_legendre(num_gauss);
    
    all_quad_x = zeros(N * num_gauss, 1);
    all_quad_w = zeros(N * num_gauss, 1);
    
    for p = 0:(N-1)
        tp = a + p * h;
        tp1 = a + (p + 1) * h;
        indices = (p * num_gauss + 1) : ((p + 1) * num_gauss);
        all_quad_x(indices) = 0.5 * ((tp1 - tp) * gl_ref_x + (tp1 + tp));
        all_quad_w(indices) = 0.5 * (tp1 - tp) * gl_ref_w;
    end
    
    % 2. 预计算 B 样条基函数矩阵
    M_col = build_basis_matrix(t_nodes, a, h, N, num_coeffs);
    M_quad = build_basis_matrix(all_quad_x, a, h, N, num_coeffs);
    
    [~, M_d2_start] = eval_spline_derivs_vectorized([a], a, h, N, num_coeffs);
    [~, M_d2_end]   = eval_spline_derivs_vectorized([b], a, h, N, num_coeffs);
    
    % 3. 定义方程组
    equation_system = @(c) compute_residuals_matrix(c, t_nodes, fx, kf, ...
        type, all_quad_x, all_quad_w, M_col, M_quad, M_d2_start, M_d2_end, num_gauss);
    
    % 4. 求解
    c0 = zeros(num_coeffs, 1);
    options = optimoptions('fsolve', 'Display', 'off', 'Algorithm', 'trust-region-dogleg', ...
        'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12); % 稍微调高了精度要求以便更好地观察收敛阶
    
    [c_sol, ~, exitflag] = fsolve(equation_system, c0, options);
    
    if exitflag <= 0
        warning('fsolve 可能未收敛');
    end
    
    % 5. 评估结果 (1024点)
    eval_nodes = linspace(a, b, 1024)';
    M_eval = build_basis_matrix(eval_nodes, a, h, N, num_coeffs);
    y_sol = M_eval * c_sol;
    
    exact_y = RF(eval_nodes);
    err = abs(exact_y - y_sol);
end

% ... (其余辅助函数: compute_residuals_matrix, build_basis_matrix 等保持原样即可) ...
function F = compute_residuals_matrix(c, t_nodes, fx, kf, type, q_x, q_w, M_col, M_quad, d2_start, d2_end, n_gauss)
    N_nodes = length(t_nodes);
    F = zeros(length(c), 1);
    s_at_nodes = M_col * c;
    s_at_quad = M_quad * c;
    integral_vals = zeros(N_nodes, 1);
    
    if strcmp(type, 'Fredholm')
        for i = 1:N_nodes
            ti = t_nodes(i);
            k_vals = kf(ti, q_x, s_at_quad);
            integral_vals(i) = sum(q_w .* k_vals);
        end
    else % Volterra
        for i = 1:N_nodes
            if i == 1
                integral_vals(i) = 0; 
                continue;
            end
            num_intervals = i - 1;
            idx_limit = num_intervals * n_gauss;
            curr_q_x = q_x(1:idx_limit);
            curr_q_w = q_w(1:idx_limit);
            curr_s_q = s_at_quad(1:idx_limit);
            k_vals = kf(t_nodes(i), curr_q_x, curr_s_q);
            integral_vals(i) = sum(curr_q_w .* k_vals);
        end
    end
    F(1:N_nodes) = s_at_nodes - fx(t_nodes) - integral_vals;
    F(end-1) = d2_start * c;
    F(end)   = d2_end * c;
end

function M = build_basis_matrix(t_eval, a, h, N, num_coeffs)
    num_pts = length(t_eval);
    max_nz = num_pts * 4; 
    rows = zeros(max_nz, 1);
    cols = zeros(max_nz, 1);
    vals = zeros(max_nz, 1);
    count = 0;
    idx_all = floor((t_eval - a) / h);
    idx_all(idx_all >= N) = N - 1;
    idx_all(idx_all < 0) = 0;
    for k = -1:2
        j_vec = idx_all + k;
        col_vec = j_vec + 2;
        valid_mask = (col_vec >= 1) & (col_vec <= num_coeffs);
        if any(valid_mask)
            t_valid = t_eval(valid_mask);
            j_valid = j_vec(valid_mask);
            rows_valid = find(valid_mask); 
            b_values = cubic_bspline_basis_vectorized(j_valid, t_valid, a, h);
            num_valid = length(t_valid);
            rows(count+1 : count+num_valid) = rows_valid;
            cols(count+1 : count+num_valid) = col_vec(valid_mask);
            vals(count+1 : count+num_valid) = b_values;
            count = count + num_valid;
        end
    end
    rows = rows(1:count);
    cols = cols(1:count);
    vals = vals(1:count);
    M = sparse(rows, cols, vals, num_pts, num_coeffs);
end

function [d1_vec, d2_vec] = eval_spline_derivs_vectorized(t_eval, a, h, N, num_coeffs)
    t = t_eval(1);
    idx = floor((t - a) / h);
    if idx >= N, idx = N - 1; end
    if idx < 0, idx = 0; end
    d2_vec = zeros(1, num_coeffs);
    d1_vec = zeros(1, num_coeffs); 
    for j = (idx - 1) : (idx + 2)
        if j >= -1 && j <= N+1
            c_idx = j + 2;
            d2_val = cubic_bspline_basis_d2_scalar(j, t, a, h);
            d2_vec(c_idx) = d2_val;
        end
    end
end

function y = cubic_bspline_basis_vectorized(i_vec, t_vec, a, h)
    ti = a + i_vec .* h;
    dist = (t_vec - ti) ./ h;
    y = zeros(size(t_vec));
    mask1 = (dist >= -2) & (dist < -1);
    y(mask1) = (1/6) * (dist(mask1) + 2).^3;
    mask2 = (dist >= -1) & (dist < 0);
    d2 = dist(mask2);
    y(mask2) = (1/6) * ( -3*d2.^3 - 6*d2.^2 + 4 );
    mask3 = (dist >= 0) & (dist < 1);
    d3 = dist(mask3);
    y(mask3) = (1/6) * ( 3*d3.^3 - 6*d3.^2 + 4 );
    mask4 = (dist >= 1) & (dist < 2);
    y(mask4) = (1/6) * (2 - dist(mask4)).^3;
end

function y = cubic_bspline_basis_d2_scalar(i, t, a, h)
    ti = a + i * h;
    dist = (t - ti) / h;
    if dist >= -2 && dist < -1
        y = (1/h^2) * (dist + 2);
    elseif dist >= -1 && dist < 0
        y = (1/(6*h^2)) * (-18*dist - 12);
    elseif dist >= 0 && dist < 1
        y = (1/(6*h^2)) * (18*dist - 12);
    elseif dist >= 1 && dist < 2
        y = (1/(6*h^2)) * 6 * (2 - dist);
    else
        y = 0;
    end
end

function [x, w] = gauss_legendre(n)
    beta = .5 ./ sqrt(1-(2*(1:n-1)).^(-2));
    T = diag(beta,1) + diag(beta,-1);
    [V, D] = eig(T);
    x = diag(D); 
    [x, i] = sort(x);
    w = 2 * V(1,i).^2;
    x = x(:)'; 
    w = w(:)'; 
end