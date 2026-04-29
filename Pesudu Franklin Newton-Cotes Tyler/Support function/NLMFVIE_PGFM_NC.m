function [ua_iter, tk_xj, cg_it] = NLMFVIE_PGFM_NC(fx, dkut, n, iter, tol, K, kf_handle, init_coeffs)
% NLMFVIE_PGFM_NC_Opt: High-Performance Solver (Precomputed)
% 逻辑未变，通过预计算网格和基函数矩阵大幅提升速度。

%% 1. 初始化与基函数构建
% 构建节点
tk_xj = CPPF(n);
% 验证/配置点
tk_xj_new = [linspace(0, tk_xj(1), ceil(K/2)+1), ...
             tk_xj(2:n-1), ... % 确保是行向量再转置，或者保持 CPPF 输出习惯
             linspace(tk_xj(n), 1, ceil((K+1)/2)+1)]';
tk_xj_new = tk_xj_new(:); % 强制列向量

% 矩阵预分配与构建 (Phi 矩阵 - 用于配置点)
% 利用优化的基函数生成器一次性生成
[Pg_ns2, ~] = generate_basis_matrix(tk_xj_new, tk_xj, n, K);

%% 2. 【核心优化】预计算积分网格与基函数
% 在进入循环前，将所有静态的积分参数计算好
fprintf('正在预计算积分网格与基函数矩阵...\n');
GridParams = Precompute_Integration_Grid(tk_xj_new, tk_xj, n, K);

%% 3. 迭代初始化
Init_ux = zeros(n+K+1, 1);
if nargin >= 8 && ~isempty(init_coeffs)
    len_c = length(init_coeffs);
    Init_ux(1:min(len_c, n+K+1)) = init_coeffs(1:min(len_c, n+K+1));
end
ua_iter = zeros(n+K+1, iter+1);
ua_iter(:, 1) = Init_ux;

% --- 初始 Jacobian 逆矩阵构建 (优化版) ---
fprintf('正在构建初始 Jacobian 矩阵 (Fast Mode)...\n');

J_integral_part = zeros(n+K+1, n+K+1);

% 预计算当前 u0 下的 dK/du 部分 (冻结 u)
% 注意：dkut 需要的是 evaluate 出来的 u 值。
% 我们利用预计算的 GridParams.Phi_Y 快速得到 U_y_flat
U_y_0 = GridParams.Phi_Y * Init_ux; 

% 定义线性化核函数 wrapper
% 这里的 phi_val 将直接传入预计算好的 Phi_Y 的某一列
dkut_linear_kernel = @(t, y, phi_val) dkut(t, y, U_y_0') .* phi_val;

% 逐列计算 Jacobian
for col_idx = 1:(n+K+1)
    % 构造单位向量系数
    % 优化：不需要再调用 evaluate_u_fast_opt，直接取预计算矩阵的列
    phi_col = GridParams.Phi_Y(:, col_idx)'; % 转置为行向量 (1 x M) 以匹配 dkut 广播
    
    % 调用快速积分函数
    % 注意：这里直接传入 phi_col 作为 "u_values" 的替代，传递给 kernel
    J_integral_part(:, col_idx) = get_double_integral_fast_jacobian(GridParams, dkut_linear_kernel, phi_col);
end

Jacobian_Init = Pg_ns2 - J_integral_part;
BdJx = cell(1, iter+1);
BdJx{1} = pinv(Jacobian_Init);

% 计算初始残差
u_curr = Pg_ns2 * ua_iter(:, 1);
% 调用快速积分函数
kf_val = get_double_integral_fast(GridParams, kf_handle, ua_iter(:, 1));
g_x = u_curr - fx(tk_xj_new) - kf_val;

cg_it = iter;

%% 4. 主迭代循环
fprintf('开始迭代 (n=%d, K=%d)...\n', n, K);
for i = 2:iter+1
    u_prev_coeffs = ua_iter(:, i-1);
    
    % Broyden 更新步
    delta_a = BdJx{i-1} * (-g_x);
    
    % 更新解
    ua_iter(:, i) = u_prev_coeffs + delta_a;
    u_next_coeffs = ua_iter(:, i);
    
    % 收敛判断
    current_norm = norm(u_next_coeffs);
    if norm(delta_a) / (current_norm + 1e-10) < tol
        fprintf('Converged in %d iterations.\n', i-1);
        cg_it = i-1;
        break;
    end
    
    % 计算新残差 (使用快速积分)
    u_next_val = Pg_ns2 * u_next_coeffs;
    kf_val_new = get_double_integral_fast(GridParams, kf_handle, u_next_coeffs);
    gx_new = u_next_val - fx(tk_xj_new) - kf_val_new;
    
    % Sherman-Morrison 更新
    delta_gx = gx_new - g_x;
    
    numerator = (delta_a - BdJx{i-1} * delta_gx) * (delta_a') * BdJx{i-1};
    denominator = (delta_a') * BdJx{i-1} * delta_gx;
    
    if abs(denominator) < 1e-14
        BdJx{i} = BdJx{i-1}; 
    else
        BdJx{i} = BdJx{i-1} + numerator / denominator;
    end
    
    g_x = gx_new;
end
end

%% ============================================================
%% 核心优化模块：预计算与快速积分
%% ============================================================

function GP = Precompute_Integration_Grid(x_nodes, tk_xj, n, K)
% 预计算所有积分相关的静态数据
% GP: GridParams 结构体

    % 1. Newton-Cotes 参数
    if K <= 1, m = 1; elseif K <= 2, m = 2; else, m = 4; end   
    [nc_nodes, nc_weights] = get_nc_weights(m);
    num_sub = length(nc_nodes);
    
    x_nodes = x_nodes(:);
    
    % 2. Outer Integral (Volterra) 网格
    h_vec = diff(x_nodes);
    interval_starts = x_nodes(1:end-1);
    num_intervals = length(h_vec);
    
    T_matrix = nc_nodes * h_vec' + interval_starts'; 
    GP.T_flat = T_matrix(:); % (N_samples x 1)
    
    % 3. Inner Integral (Fredholm) 网格
    % 保持与 x_nodes 一致
    y_edges = unique([0; x_nodes(:); 1]); 
    h_inner = diff(y_edges);
    
    Y_matrix = nc_nodes * h_inner(:)' + y_edges(1:end-1)';
    GP.Y_flat = Y_matrix(:); % (M_samples x 1)
    
    % 4. Inner Weights (全局权重)
    W_matrix = nc_weights * h_inner(:)'; 
    GP.W_flat = W_matrix(:); % (M_samples x 1)
    
    % 5. 【关键优化】预计算 Inner 网格上的基函数矩阵 Phi_Y
    % 这样在积分时不需要 evaluate_u，直接矩阵相乘 Phi_Y * coeffs 即可
    [GP.Phi_Y, ~] = generate_basis_matrix(GP.Y_flat, tk_xj, n, K);
    
    % 存储其他必要参数
    GP.num_sub = num_sub;
    GP.num_intervals = num_intervals;
    GP.nc_weights = nc_weights;
    GP.h_vec = h_vec;
end

function total_integral = get_double_integral_fast(GP, kernel_func, coeffs)
% 极速积分函数 (Normal Mode)
% 使用预计算的 GridParams (GP) 和系数coeffs

    % 1. 快速计算 u(y)
    % 仅需一次矩阵向量乘法
    U_y_flat = GP.Phi_Y * coeffs; % (M x 1)
    
    % 2. 分块计算 Kernel 并积分
    % (此处逻辑与原版一致，但去除了重复的网格生成)
    T_flat = GP.T_flat;
    Y_flat = GP.Y_flat;
    W_flat = GP.W_flat;
    
    Inner_Val_at_T = zeros(length(T_flat), 1);
    chunk_size = 1024; % 增大分块大小，因为现在计算开销小了
    
    % 为了广播兼容，转置 U_y_flat 和 Y_flat
    U_y_row = U_y_flat'; 
    Y_row = Y_flat';
    
    for i = 1:chunk_size:length(T_flat)
        idx_end = min(i + chunk_size - 1, length(T_flat));
        t_batch = T_flat(i:idx_end);
        
        % Kernel Eval
        K_batch = kernel_func(t_batch, Y_row, U_y_row);
        
        % Dot Product with Weights
        Inner_Val_at_T(i:idx_end) = K_batch * W_flat;
    end
    
    % 3. Outer Integral Accumulation
    Inner_Matrix = reshape(Inner_Val_at_T, GP.num_sub, GP.num_intervals);
    Step_Integrals = (GP.nc_weights' * Inner_Matrix) .* GP.h_vec'; 
    total_integral = [0; cumsum(Step_Integrals(:))];
end

function integral_vals = get_double_integral_fast_jacobian(GP, kernel_func_linear, phi_col_row)
% 极速积分函数 (Jacobian Mode)
% 专门用于 Jacobian 计算，直接接收 phi_col_row (1xM) 而不是 coefficients
% 避免了在 Jacobian 循环中重复计算 U_y_0 (它已经包含在 kernel_func_linear 闭包中了)

    T_flat = GP.T_flat;
    Y_flat = GP.Y_flat;
    W_flat = GP.W_flat;
    
    Inner_Val_at_T = zeros(length(T_flat), 1);
    chunk_size = 1024;
    
    Y_row = Y_flat';
    
    for i = 1:chunk_size:length(T_flat)
        idx_end = min(i + chunk_size - 1, length(T_flat));
        t_batch = T_flat(i:idx_end);
        
        % Kernel Eval (Linearized)
        % t_batch: (B x 1)
        % Y_row: (1 x M)
        % phi_col_row: (1 x M) --> 这是基函数的值，直接传入
        K_batch = kernel_func_linear(t_batch, Y_row, phi_col_row);
        
        Inner_Val_at_T(i:idx_end) = K_batch * W_flat;
    end
    
    Inner_Matrix = reshape(Inner_Val_at_T, GP.num_sub, GP.num_intervals);
    Step_Integrals = (GP.nc_weights' * Inner_Matrix) .* GP.h_vec'; 
    integral_vals = [0; cumsum(Step_Integrals(:))];
end

function [Phi, centers] = generate_basis_matrix(t, tk_xj, n, K)
% 统一的基函数矩阵生成器
% 代替原来的 evaluate_u_fast_opt，只返回矩阵，不乘系数
    t = t(:);
    tk_map = CPPF(n); % 假设 CPPF 可用
    
    num_pts = length(t);
    num_coeffs = n + K + 1;
    
    Phi = zeros(num_pts, num_coeffs);
    
    % Polynomial part
    for p = 0:K
        Phi(:, p+1) = t.^p;
    end
    
    % Truncated power part
    centers = tk_map(1 : num_coeffs-K-1);
    if ~isempty(centers)
        D = bsxfun(@minus, t, centers(:)'); 
        if K == 0
            Phi(:, K+2:end) = double(D > 0);
        else
            Phi(:, K+2:end) = (max(0, D)).^K;
        end
    end
end

function [nodes, weights] = get_nc_weights(m)
    nodes = linspace(0, 1, m + 1)';
    switch m
        case 1, weights = [1; 1] / 2;
        case 2, weights = [1; 4; 1] / 6;
        case 3, weights = [1; 3; 3; 1] / 8;
        case 4, weights = [7; 32; 12; 32; 7] / 90;
    end
end
