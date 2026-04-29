function [ua_coeffs, tk_xj] = LVIE_PGFM_NC(fx, n, K, kf_handle)
% LVIE_PGFM_NC: 线性 Volterra 积分方程求解器 (掩码矩阵极速版)
%
% 优化说明:
% 1. 输入预计算: 权重和节点只计算一次，不再在循环中重复生成。
% 2. 移除所有内层循环: 使用 "逻辑掩码矩阵" (Logical Mask) 处理 Volterra 三角积分限制。
%    通过 K_masked = K .* Mask，将三角积分转化为标准的全矩阵乘法。
%
% 方程: u(x) - int_0^x K(x,t)u(t) dt = f(x)

%% 1. 节点与网格生成
tk_xj = CPPF(n); 
tk_xj_new = [linspace(0, tk_xj(1), ceil(K/2)+1), ...
             tk_xj(2:n-1), ...
             linspace(tk_xj(n), 1, ceil((K+1)/2)+1)].';
         
num_coeffs = n + K + 1;

%% 2. 快速构建基函数矩阵 Phi
Phi = evaluate_basis_matrix(tk_xj_new, tk_xj, K);

%% 3. 预计算积分权重 (Calculate Once)
% 将静态参数计算移出循环
if K <= 1, m = 1; elseif K <= 2, m = 2; else, m = 4; end   
[nc_nodes, nc_weights] = get_nc_weights(m); 

%% 4. 混合分块构建 Volterra 积分矩阵 V
V = get_kf_nc_masked(tk_xj_new, K, n, kf_handle, num_coeffs, nc_nodes, nc_weights);

%% 5. 求解
F_vec = fx(tk_xj_new);
System_Matrix = Phi - V;
ua_coeffs = System_Matrix \ F_vec;

end

%% ---------------------------------------------------------
%  辅助函数：掩码矩阵积分生成 (No Loops)
%% ---------------------------------------------------------
function V_matrix = get_kf_nc_masked(x_nodes, K, n, kf_handle, num_coeffs, nc_nodes, nc_weights)
    num_sub = length(nc_nodes); % m+1
    
    num_targets = length(x_nodes);
    h_vec = diff(x_nodes); 
    interval_starts = x_nodes(1:end-1);
    num_intervals = length(h_vec);
    
    V_matrix = zeros(num_targets, num_coeffs);
    
    % 分块大小 (Chunk Size)
    chunk_size = 256; 
    
    for start_idx = 1:chunk_size:num_intervals
        end_idx = min(start_idx + chunk_size - 1, num_intervals);
        current_indices = start_idx:end_idx;
        num_curr = length(current_indices);
        
        % --- 1. 预计算当前分块积分参数 ---
        curr_h = h_vec(current_indices);       
        curr_starts = interval_starts(current_indices);
        
        % T_chunk: [m+1, NumCurr] 每一列是一个区间的积分点
        T_chunk = nc_nodes * curr_h' + curr_starts';
        T_flat = T_chunk(:); 
        
        % Phi_flat: [TotalPts, Coeffs]
        Phi_flat = evaluate_basis_matrix(T_flat, CPPF(n), K);
        
        % 全局权重向量 W_flat (含 Jacobian)
        % 既然 nc_weights 是固定的，我们利用 Kronecker 积的思想快速构造
        % W = [w*h1; w*h2; ...]
        % 使用 repelem 扩展 h，repmat 扩展 w
        W_flat = repmat(nc_weights, num_curr, 1) .* repelem(curr_h, num_sub, 1);
        
        % 预乘权重: Phi_W
        Phi_W = bsxfun(@times, Phi_flat, W_flat);
        
        % 分界线
        cutoff_idx = end_idx + 1;
        
        % --- Group A: 下游目标点 (Far Targets) - 标准矩阵乘法 ---
        if cutoff_idx <= num_targets
            far_indices = cutoff_idx:num_targets;
            x_far = x_nodes(far_indices);
            
            % K_far: [NumFar, TotalPts]
            K_far = kf_handle(x_far, T_flat'); 
            
            % Update: Matrix Multiply
            V_matrix(far_indices, :) = V_matrix(far_indices, :) + K_far * Phi_W;
        end
        
        % --- Group B: 本地目标点 (Local Targets) - 掩码矩阵乘法 ---
        % 我们不再循环 j，而是构造一个 Mask 矩阵
        
        local_start = start_idx + 1;
        local_end = min(end_idx, num_targets); % 只有部分 target 落在当前 chunk 内
        
        if local_start <= local_end
            local_indices = (local_start:local_end)';
            x_local = x_nodes(local_indices);
            
            % 1. 计算全量核矩阵 K_local: [NumLocal, TotalPts]
            K_local = kf_handle(x_local, T_flat');
            
            % 2. 构造因果掩码 (Causality Mask)
            % 我们需要: Mask(i, k) = 1 if interval(k) < target(i)
            % 积分点 k 属于 interval j = ceil(k / num_sub)
            % Volterra 条件: interval_index < target_index (因为 x_i 积分到 x_i)
            % 注意: local_indices 是全局 target index
            % current_indices 是全局 interval index
            
            % 使用 bsxfun 构造 Block Mask: [NumLocal, NumCurr]
            % Row i (target), Col j (interval) -> 1 if Target > Interval
            Block_Mask = bsxfun(@gt, local_indices, current_indices);
            
            % 扩展 Mask 到所有积分点: [NumLocal, TotalPts]
            % 每个 interval 有 num_sub 个点，Mask 值相同
            Full_Mask = repelem(Block_Mask, 1, num_sub);
            
            % 3. 应用掩码
            K_masked = K_local .* Full_Mask;
            
            % 4. 矩阵乘法累加
            % 这一步一次性完成了所有三角区域的积分求和
            V_matrix(local_indices, :) = V_matrix(local_indices, :) + K_masked * Phi_W;
        end
    end
end

%% ---------------------------------------------------------
%  辅助函数
%% ---------------------------------------------------------
function Phi = evaluate_basis_matrix(t, tk_map, K)
    t = t(:);
    num_pts = length(t);
    num_centers = length(tk_map);
    num_coeffs = num_centers + K + 1;
    Phi = zeros(num_pts, num_coeffs);
    for p = 0:K, Phi(:, p+1) = t.^p; end
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