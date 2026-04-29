function [ua_coeffs, tk_xj] = LFIE_PGFM_NC(fx, n, K, kf_handle)
% LFIE_PGFM_NC: 线性 Fredholm 积分方程求解器 (矩阵化极速版)
% 
% 优化说明:
% 1. 纯矩阵运算: 将积分过程转化为 V = K_matrix * (Weights .* Phi_matrix)。
%    彻底移除了基函数系数 j 的循环，利用 CPU 的 BLAS 库进行极速计算。
% 2. 预计算优化: 积分节点、权重、基函数值均一次性生成，无重复计算。
% 3. 内存友好: 即使 N 较大，也能通过紧凑的矩阵操作高效运行。
%
% 方程: u(x) - int_0^1 K(x,t)u(t) dt = f(x)

%% 1. 节点与网格生成
tk_xj = CPPF(n); 
% 配置点 (转置为列向量)
tk_xj_new = [linspace(0, tk_xj(1), ceil(K/2)+1), ...
             tk_xj(2:n-1), ...
             linspace(tk_xj(n), 1, ceil((K+1)/2)+1)].';
         
num_coeffs = n + K + 1;

%% 2. 快速构建基函数矩阵 Phi
% Phi(i, j) 表示第 j 个基函数在第 i 个配置点的值
% Size: [Num_Points, Num_Coeffs]
Phi = evaluate_basis_matrix(tk_xj_new, tk_xj, K);

%% 3. 矩阵化构建 Fredholm 积分矩阵 V
% V(i, j) = int_0^1 K(x_i, t) * phi_j(t) dt
V = get_kf_nc_fredholm_optimized(tk_xj_new, K, n, kf_handle, num_coeffs);

%% 4. 求解线性方程组
% (Phi - V) * C = F
F_vec = fx(tk_xj_new);
System_Matrix = Phi - V;

% 使用通用求解器 (Fredholm 矩阵通常是稠密的，'\' 是最优选择)
ua_coeffs = System_Matrix \ F_vec;

end

%% ---------------------------------------------------------
%  辅助函数：Fredholm 积分矩阵生成 (核心优化)
%% ---------------------------------------------------------
function V_matrix = get_kf_nc_fredholm_optimized(x_nodes, K, n, kf_handle, num_coeffs)
    % 1. 准备积分参数
    if K <= 1, m = 1; elseif K <= 2, m = 2; elseif K==3, m = 3; else, m=4; end       
    [nc_nodes, nc_weights] = get_nc_weights(m);
    num_sub = length(nc_nodes);
    
    % 2. 构建积分网格
    grid_points = unique([0; x_nodes(:); 1]);
    grid_points = sort(grid_points);
    h_vec = diff(grid_points);
    interval_starts = grid_points(1:end-1);
    num_intervals = length(h_vec);
    
    % 3. 生成全局积分点 T_flat 和 权重 W_flat
    T_matrix = nc_nodes * h_vec' + interval_starts';
    T_flat = T_matrix(:); % [TotalPts, 1]
    
    W_flat = repmat(nc_weights, num_intervals, 1) .* repelem(h_vec, num_sub, 1);
    
    % 4. 计算基函数矩阵 Phi_quad
    Phi_quad = evaluate_basis_matrix(T_flat, CPPF(n), K);
    
    % 5. 预乘权重
    Phi_W = bsxfun(@times, Phi_quad, W_flat);
    
    % 6. 计算核函数矩阵 K_matrix (修复维度缺失bug)
    % -----------------------------------------------------------
    % 原始计算: 可能返回 scalar, (N*1), 或 (1*Q)
    K_val = kf_handle(x_nodes, T_flat');
    
    % 强制扩展: 加上一个 (N * Q) 大小的零矩阵
    % 利用 MATLAB 的隐式扩展功能，自动补全缺失的维度
    K_matrix = K_val + zeros(length(x_nodes), length(T_flat));
    % -----------------------------------------------------------
    
    % 7. 核心：矩阵乘法
    V_matrix = K_matrix * Phi_W;
end

%% ---------------------------------------------------------
%  辅助函数：基函数矩阵生成 (向量化通用版)
%% ---------------------------------------------------------
function Phi = evaluate_basis_matrix(t, tk_map, K)
    t = t(:);
    num_pts = length(t);
    num_centers = length(tk_map);
    num_coeffs = num_centers + K + 1;
    
    Phi = zeros(num_pts, num_coeffs);
    
    % 多项式部分
    for p = 0:K
        Phi(:, p+1) = t.^p;
    end
    
    % 截断幂函数部分 (利用 Broadcasting 加速)
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