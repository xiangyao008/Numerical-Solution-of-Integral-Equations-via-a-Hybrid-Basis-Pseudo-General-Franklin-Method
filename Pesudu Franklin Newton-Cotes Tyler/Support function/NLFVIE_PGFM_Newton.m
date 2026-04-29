function [ua_iter, tk_xj, cg_it] = NLFVIE_PGFM_Newton(fx, dvkut, dfkut, n, iter, tol, K, kernel_func_v, kernel_func_f, init)
% NLFVIE_PGFM_Newton (精度修复版)
%
% 修复了 Jacobian 矩阵 Volterra 部分的索引匹配错误，
% 确保了 Exact Newton Method 的二次收敛精度。

%% 1. 初始化与基函数构建
pseudo_Franklin_basis = @(x,cp,K) 0.*(0<=x & x< cp) + ((x - cp).^K).*(cp<=x & x <= 1);
Pg_ns1=zeros(n+K+1,n+2);
Pg_ns2=zeros(n+K+1,n+K+1);

tk_xj=CPPF(n);
tk_xj2=[0,tk_xj,1];
tk_xj_new=[linspace(0,tk_xj(1),ceil(K/2)+1), tk_xj(2:n-1), linspace(tk_xj(n),1,ceil((K+1)/2)+1)];

for i=0:K
    Pg_ns1(i+1,:)=tk_xj2.^(i); 
    Pg_ns2(i+1,:)=tk_xj_new.^(i); 
end
for i=K+1:n+K
    Pg_ns1(i+1,:)=pseudo_Franklin_basis(tk_xj2,tk_xj(i-K),K); 
    Pg_ns2(i+1,:)=pseudo_Franklin_basis(tk_xj_new,tk_xj(i-K),K); 
end
Pg_ns1=Pg_ns1';
Pg_ns2=Pg_ns2';

% 初值处理
Init_ux = zeros(n+K+1, 1);
if length(init) == K+1
    Init_ux(1:K+1)=init(:);
else
    Init_ux = init(:);
end
ua_iter=zeros(n+K+1,iter+1);
ua_iter(:,1)=Init_ux;

% 预计算静态数据
fx_val = fx(tk_xj_new)'; 
% 初始化缓存
Cache = init_integration_cache(tk_xj_new, K, n); 

%% 2. 主循环 (Newton-Raphson Iteration)
cg_it = iter;
fprintf('Solver Strategy: Exact Newton-Raphson (High Precision Fix)\n');

for i = 2:iter+1
    coeffs_prev = ua_iter(:, i-1);
    
    % --- Step 1: 当前 u(x) ---
    u_curr = Pg_ns2 * coeffs_prev;
    
    % --- Step 2: 计算残差 F(u) ---
    kf_val = get_kf_mixed_nc_fast(coeffs_prev, kernel_func_v, kernel_func_f, Cache);
    g_x = u_curr - fx_val - kf_val;
    
    res_norm = norm(g_x);
    if res_norm < tol
        disp(['Converged (Residual) in ', num2str(i-1), ' iterations. Norm: ', num2str(res_norm)]);
        cg_it = i-1; break;
    end
    
    % --- Step 3: 构建精确 Jacobian J ---
    % 3.1 预计算导数核矩阵
    [dKv_vals, dKf_vals] = get_dk_vals(coeffs_prev, dvkut, dfkut, Cache);
    
    % 3.2 计算积分型 Jacobian 部分 (修复了 Volterra 累加逻辑)
    J_Int_V = compute_jacobian_volterra(dKv_vals, Cache);
    J_Int_F = compute_jacobian_fredholm(dKf_vals, Cache);
    
    % 3.3 组装
    J = Pg_ns2 - J_Int_V - J_Int_F;
    
    % --- Step 4: 牛顿更新 ---
    % 使用条件数保护
    if rcond(J) < 1e-15
        warning('Jacobian is near singular, using pinv.');
        delta_a = pinv(J) * (-g_x);
    else
        delta_a = J \ (-g_x);
    end
    
    ua_iter(:, i) = coeffs_prev + delta_a;
    
    % --- Step 5: 收敛判据 ---
    current_norm = norm(coeffs_prev);
    if norm(delta_a) / (current_norm + 1e-14) < tol
        disp(['Converged (Step Size) in ', num2str(i-1), ' iterations.']);
        cg_it = i-1; break;
    end
end
ua_iter = ua_iter(:, 1:cg_it);
end

%% ========================================================================
%  辅助函数集 (已修复逻辑错误)
% ========================================================================

function J_V = compute_jacobian_volterra(dKv, Cache)
    % 【关键修复】正确计算 Volterra Jacobian
    % J_V(i, j) = int_0^{x_i} dKv(x_i, t) * phi_j(t) dt
    
    num_targets = Cache.num_targets;
    num_basis = size(Cache.Phi_Int, 2);
    J_V = zeros(num_targets, num_basis);
    
    % 遍历每个基函数 j
    for j = 1:num_basis
        phi_j = Cache.Phi_Int(:, j)'; % 1 x M
        
        % Integrand: dKv(x,t) * phi_j(t)
        % dKv 的每一行 i 对应 target x_i
        Integrand = bsxfun(@times, dKv, phi_j);
        
        % 分段积分矩阵 (Targets x Intervals)
        Segs = compute_segment_integrals_fast(Integrand, Cache);
        
        % --- 修复开始 ---
        % 对于第 i 个目标点 x_i，积分区间是 [0, x_i]
        % 假设 x 节点是有序排列的，x_1=0, x_2=t_1, ...
        % 第 i 行需要累加 Segs(i, 1 : i-1)
        
        col_vec = zeros(num_targets, 1);
        for row = 2:num_targets
            % 累加该行对应的前 row-1 个区间
            col_vec(row) = sum(Segs(row, 1:row-1));
        end
        J_V(:, j) = col_vec;
        % --- 修复结束 ---
    end
end

function J_F = compute_jacobian_fredholm(dKf, Cache)
    % 计算 Fredholm Jacobian
    % J_F(i, j) = int_0^1 dKf(x_i, t) * phi_j(t) dt
    
    num_targets = Cache.num_targets;
    num_basis = size(Cache.Phi_Int, 2);
    J_F = zeros(num_targets, num_basis);
    
    for j = 1:num_basis
        phi_j = Cache.Phi_Int(:, j)';
        Integrand = bsxfun(@times, dKf, phi_j);
        Segs = compute_segment_integrals_fast(Integrand, Cache);
        
        % Fredholm 是全区间求和 (对每一行求和)
        J_F(:, j) = sum(Segs, 2);
    end
end

function Segs = compute_segment_integrals_fast(Mat, Cache)
    % 优化的分段积分 (Strided Sum)
    % 输出 Segs: (Targets x Intervals)
    [rows, ~] = size(Mat);
    m_pts = Cache.num_sub;
    
    % 预分配
    Segs = zeros(rows, Cache.num_intervals);
    w = Cache.weights;
    
    for k = 1:m_pts
        if w(k) ~= 0
            % 提取每段的第 k 个积分点
            cols = k:m_pts:size(Mat, 2);
            Segs = Segs + Mat(:, cols) * w(k);
        end
    end
    
    % 乘以区间长度 h
    Segs = bsxfun(@times, Segs, Cache.h_vec');
end

function kf = get_kf_mixed_nc_fast(coeffs, kf_v, kf_f, Cache)
    % 快速计算残差积分项 (逻辑同 Jacobian)
    u_vals = Cache.Phi_Int * coeffs; 
    
    x_col = Cache.x_nodes;   
    t_row = Cache.T_flat';   
    u_row = u_vals';         
    
    % Volterra 部分
    Mat_V = kf_v(x_col, t_row, u_row);
    Seg_V = compute_segment_integrals_fast(Mat_V, Cache);
    
    % 同样的 Volterra 累加修复逻辑
    kf_v_val = zeros(Cache.num_targets, 1);
    for row = 2:Cache.num_targets
        kf_v_val(row) = sum(Seg_V(row, 1:row-1));
    end
    
    % Fredholm 部分
    Mat_F = kf_f(x_col, t_row, u_row);
    Seg_F = compute_segment_integrals_fast(Mat_F, Cache);
    kf_f_val = sum(Seg_F, 2);
    
    kf = kf_v_val + kf_f_val;
end

function [dKv, dKf] = get_dk_vals(coeffs, dkf_v, dkf_f, Cache)
    u_vals = Cache.Phi_Int * coeffs;
    x_col = Cache.x_nodes;
    t_row = Cache.T_flat';
    u_row = u_vals';
    
    dKv = dkf_v(x_col, t_row, u_row);
    dKf = dkf_f(x_col, t_row, u_row);
end

function Cache = init_integration_cache(x_nodes, K, n)
    if K <= 1, m = 1; elseif K <= 2, m = 2; elseif K==3, m = 3; else, m=4; end   
    [nc_nodes, nc_weights] = get_nc_weights(m);
    
    grid_points = x_nodes(:);
    h_vec = diff(grid_points);
    num_intervals = length(h_vec);
    interval_starts = grid_points(1:end-1);
    
    T_matrix = nc_nodes * h_vec' + interval_starts'; 
    T_flat = T_matrix(:);
    
    Phi_Int = evaluate_basis_matrix(T_flat, n, K);
    
    Cache.x_nodes = grid_points;
    Cache.T_flat = T_flat;
    Cache.weights = nc_weights;
    Cache.h_vec = h_vec;
    Cache.num_sub = length(nc_nodes);
    Cache.Phi_Int = Phi_Int;
    Cache.num_intervals = num_intervals;
    Cache.num_targets = length(grid_points);
end

function Phi = evaluate_basis_matrix(t, n, K)
    % 修复了 bsxfun 维度的版本
    t = t(:);
    tk_map = CPPF(n);
    centers = tk_map(1:n);
    centers = centers(:)'; % 强制转为行向量
    
    num_pts = length(t);
    num_basis = n + K + 1;
    Phi = zeros(num_pts, num_basis);
    
    for p = 0:K
        Phi(:, p+1) = t.^p;
    end
    
    if n > 0
        D = bsxfun(@minus, t, centers);
        Mask = (D > 0); 
        if K == 0
            Phi(:, K+2:end) = double(Mask);
        else
            Phi(:, K+2:end) = (D.^K) .* Mask;
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