function [ua_iter, tk_xj, cg_it] = NLVIE_PGFM_NC(fx, dkut, n, iter, tol, K, kf_f, init)
% NLVIE_PGFM_NC: Nonlinear Volterra Integral Equation Solver
%
% 参照 NLFIE_PGFM_NC 风格重构，保持了变量命名和代码结构的一致性。
% 针对 Volterra 方程引入了高效的区间掩码（Masking）积分算法。

%% 1. 介绍和思路
% 伪Franklin方法中的dkut_xt的维度以及迭代过程中所使用的矩阵维度选择
% dkut_xt=(n+K+1)*(n), Pg_ns1=(n)*(n+K+1) -> Pg_ns1'=(n+K+1)*(n)

%% 2. 逆Broyden秩1方法初始化
% cp=collocation points
pseudo_Franklin_basis = @(x, cp, K) 0.*((0<=x) < cp) + ((x - cp).^K).*(cp<=x & x <= 1);

Pg_ns1 = zeros(n+K+1, n);
Pg_ns2 = zeros(n+K+1, n+K+1);

% 生成配置点和扩展节点
tk_xj = CPPF(n); 
% 强制转换为列向量，确保维度一致性
tk_xj = tk_xj(:);

tk_xj_new = [linspace(0, tk_xj(1), ceil(K/2)+1), tk_xj(2:n-1)', linspace(tk_xj(n), 1, ceil((K+1)/2)+1)]';
tk_xj_new = tk_xj_new(:);

% 构造基函数矩阵 (向量化优化版)
for i = 0:K
    Pg_ns1(i+1, :) = tk_xj.^(i); 
    Pg_ns2(i+1, :) = tk_xj_new.^(i); 
end

if n > 0
    % Franklin Basis
    centers = tk_xj(1:n);
    centers = centers(:)'; % 确保为行向量 (1 x n)
    
    % Pg_ns1 (Source)
    X1 = tk_xj; % 列向量 (n x 1)
    D1 = bsxfun(@minus, X1, centers); % (n x n)
    Pg_ns1(K+2:end, :) = (D1.^K)' .* ((D1 > 0 & X1 <= 1)');
    
    % Pg_ns2 (Target)
    X2 = tk_xj_new; % 列向量 (M x 1)
    D2 = bsxfun(@minus, X2, centers); % (M x n)
    Pg_ns2(K+2:end, :) = (D2.^K)' .* ((D2 > 0 & X2 <= 1)');
end

Pg_ns1 = Pg_ns1';
Pg_ns2 = Pg_ns2';

%% 3. 预计算 Volterra 积分缓存 (关键效率优化)
% 为了与 NLFIE 结构保持一致，这里将积分所需的静态数据打包进 Cache
% 避免在主循环的 get_kf_nc 中重复计算网格
Cache = init_volterra_cache(tk_xj_new, n, K);

%% 3. Jacobian 矩阵初始化 (深度优化版)
Init_ux = zeros(n+K+1, 1); 
Init_ux(1:K+1) = init;

ua_iter = zeros(n+K+1, iter+1);
ua_iter(:, 1) = Init_ux;

u1 = Pg_ns1 * ua_iter(:, 1); % 获得关于t的初值函数矩阵 (n x 1)

% --- 优化 1: 向量化计算初始 Jacobian (dkut_xt) ---
% 避免双重循环，利用矩阵广播一次性计算所有元素
% 构造网格: X (Target points), T (Source points)
[T_mat, X_mat] = meshgrid(tk_xj, tk_xj_new); 

% 构造 U 矩阵: u1 对应 T_mat 的列变化 (Source u)
% u1 是 (n x 1)，我们需要将其转置并复制以匹配 (M x n)
U_mat = repmat(u1(:)', length(tk_xj_new), 1);
Volterra_Mask = (T_mat <= X_mat + 1e-12);
try
    J_raw = dkut(X_mat, T_mat, U_mat);
    dkut_xt = J_raw .* Volterra_Mask;
catch
    % 如果 dkut 不支持向量化，回退到循环但保持对角线优化
    dkut_xt = zeros(length(tk_xj_new), length(tk_xj));
    for i = 1:length(tk_xj_new)
        % 仅循环计算 t <= x 的部分
        valid_idx = tk_xj <= tk_xj_new(i) + 1e-12;
        if any(valid_idx)
            dkut_xt(i, valid_idx) = dkut(tk_xj_new(i), tk_xj(valid_idx), u1(valid_idx));
        end
    end
end

% --- 优化 2: 高效求逆 ---
% 初始 Broyden 矩阵 BdJx 是 (I - K)' 的逆近似
% 计算 Jacobian 近似矩阵 J_approx
J_approx = Pg_ns2 - dkut_xt * Pg_ns1 ./ length(u1);

BdJx = cell(1, iter+1);

BdJx{1} = pinv(J_approx);

% 预先计算 f(x) 以提高效率
fx_val = fx(tk_xj_new);
if isrow(fx_val), fx_val = fx_val'; end

%% 5. 主循环函数
cg_it = iter; % 默认收敛步数

for i = 2:iter+1  
    u2 = (Pg_ns2 * ua_iter(:, i-1));
    S2_x = u2;
    
    % 计算积分项 (调用与 NLFIE 同名的接口，但传入 Cache)
    kf = get_kf_nc(tk_xj_new, ua_iter(:, i-1), kf_f, Cache);
    
    g_x = S2_x - fx_val - kf;
    delta_a = BdJx{i-1} * (-g_x);
    
    ua_iter(:, i) = ua_iter(:, i-1) + delta_a;
    
    %%%%% Relative Tolerance
    current_norm = norm(ua_iter(:, i));
    update_norm = norm(delta_a);
    if update_norm / (current_norm + 1e-10) < tol
        disp(['Converged in ', num2str(i), ' iterations.']);
        cg_it = i;
        break;
    end
    
    % Broyden 更新
    u2_new = Pg_ns2 * ua_iter(:, i);
    kf_new = get_kf_nc(tk_xj_new, ua_iter(:, i), kf_f, Cache);
    
    gx_new = u2_new - fx_val - kf_new;
    delta_gx = gx_new - g_x;
    
    % Sherman-Morrison 公式更新逆 Jacobian
    term1 = delta_a - BdJx{i-1} * delta_gx;
    term2 = (delta_a') * BdJx{i-1};
    denom = term2 * delta_gx;
    
    if abs(denom) < 1e-14
        BdJx{i} = BdJx{i-1};
    else
        BdJx{i} = BdJx{i-1} + (term1 * term2) / denom;
    end
end

% 截断未使用的迭代空间
ua_iter = ua_iter(:, 1:cg_it);

end

%% 辅助函数：Volterra 积分计算 (核心优化)
function kf = get_kf_nc(x_nodes, coeffs, kernel_func, Cache)
    % NLFIE 风格的积分接口，但在内部利用 Cache 进行 Volterra 快速计算
    
    % 1. 快速评估 u(t) 在所有积分点的值 (矩阵乘法)
    U_flat = Cache.Phi_Integration * coeffs;
    
    % 2. Reshape 为 3D 矩阵以便广播计算 [1, SubPoints, Intervals]
    U_3D = reshape(U_flat, 1, Cache.num_sub_points, Cache.num_intervals);
    
    % 3. 计算核函数 K(x, t, u)
    % MATLAB 自动广播: (Targets,1,1) vs (1,Sub,Int) -> (Targets,Sub,Int)
    Kernel_Val = kernel_func(Cache.X_3D, Cache.T_3D, U_3D);
    
    % 4. 积分加权求和 (Newton-Cotes Weights)
    % Sum over quadrature points (Dim 2)
    Segment_Integrals = sum(Kernel_Val .* Cache.W_3D, 2); 
    
    % 5. 应用区间长度 Jacobian 和 Volterra 掩码
    % 仅累加完全在 x 左侧的区间 (Interval_Mask)
    % Sum over intervals (Dim 3)
    kf = sum(Segment_Integrals .* Cache.H_3D .* Cache.Interval_Mask, 3);
    
    if isrow(kf), kf = kf'; end
end

%% 辅助函数：缓存初始化
function Cache = init_volterra_cache(x_nodes, n, K)
    % 确定积分阶数 m
    if K <= 1, m = 1; elseif K <= 2, m = 2; elseif K == 3, m = 3; else, m = 4; end       
    [nc_nodes, nc_weights] = get_nc_weights(m);
    
    x_targets = x_nodes(:); 
    num_sub = length(nc_nodes);
    num_tar = length(x_targets);
    
    % 构建全局积分网格 (必须包含0和所有x_nodes)
    unique_nodes = unique([0; x_targets]);
    sorted_grid = sort(unique_nodes);
    
    h_vec = diff(sorted_grid);
    num_int = length(h_vec);
    
    interval_starts = sorted_grid(1:end-1);
    interval_ends   = sorted_grid(2:end);
    
    % 生成所有积分点 T
    T_matrix = nc_nodes * h_vec' + interval_starts'; 
    T_flat = T_matrix(:); 
    
    % 预计算基函数矩阵 Phi (复用 evaluate_basis 逻辑)
    Phi = evaluate_basis_matrix(T_flat, n, K);
    
    % 打包缓存数据
    Cache.num_sub_points = num_sub;
    Cache.num_intervals  = num_int;
    Cache.Phi_Integration = Phi;
    
    % 预 Reshape 矩阵以便广播
    Cache.T_3D = reshape(T_matrix, 1, num_sub, num_int);
    Cache.X_3D = reshape(x_targets, num_tar, 1, 1);
    Cache.W_3D = reshape(nc_weights, 1, num_sub, 1);
    Cache.H_3D = reshape(h_vec, 1, 1, num_int);
    
    % 预计算 Volterra Mask (区间右端点 <= 目标点x)
    Interval_Ends_3D = reshape(interval_ends, 1, 1, num_int);
    Cache.Interval_Mask = (Interval_Ends_3D <= Cache.X_3D + 1e-12);
end

%% 辅助函数：基函数矩阵生成 (bsxfun 修复版)
function Phi = evaluate_basis_matrix(t, n, K)
    t = t(:); % 强制列向量
    num_coeffs = n + K + 1;
    Phi = zeros(length(t), num_coeffs);
    
    % 多项式部分
    for p = 0:K
        Phi(:, p+1) = t.^p; 
    end
    
    % Franklin 部分
    if n > 0
        tk_map = CPPF(n); 
        centers = tk_map(1 : n);
        centers = centers(:)'; % 强制行向量
        
        % D = (M x 1) - (1 x N) = (M x N)
        D = bsxfun(@minus, t, centers);
        Mask = (D > 0) & (t <= 1); 
        Phi(:, K+2:end) = (D.^K) .* Mask;
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