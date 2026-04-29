function [ua_iter, tk_xj,cg_it] = NLFIE_PGFM_NC(fx,dkut, n, iter,tol,K,kf_f,init)
% nonlinear Fredholm integral equation
% fx 为给定函数, 接受x参数 定义为函数句柄
% kuf 为已知积分核, 接受x,t,u三个参数 定义为函数句柄,包含被解函数的函数
% dkut 表示kuf对被解函数u(t)的偏导
% n 为阶数，iter,tol表示迭代此时和容差
% ux_iter 为 每次迭代后的值
% cg_it表示收敛时的阶次
%% 介绍和思路
% 伪Franklin方法中的dkut_xt的维度以及迭代过程中所使用的矩阵维度怎么选择
% dkut_xt*Pg_ns1中使用的数值积分所用到的n个配置点，因此dkut_xt=(n+K+1)*(n),
% Pg_ns1'=(n)*(n+K+1)--> Pg_ns1=(n+K+1)*(n)
% Pg_ns2=zeros(n+K+1,n+K+1)
%% 逆Broyden秩1方法
% cp=collocation points
tk_xj = CPPF(n);
tk_xj_new = [linspace(0, tk_xj(1), ceil(K/2)+1), tk_xj(2:n-1), linspace(tk_xj(n), 1, ceil((K+1)/2)+1)];

exp_grid = (0:K)';
Pg_ns1_poly = tk_xj .^ exp_grid; 
Pg_ns2_poly = tk_xj_new .^ exp_grid;
cp = tk_xj(1:n); 
diff1 = tk_xj - cp'; 
mask1 = diff1 >= 0; % 对应 cp <= x
Pg_ns1_basis = (diff1 .^ K) .* mask1;

diff2 = tk_xj_new' - cp; % 维度: length(tk_xj_new) x n
mask2 = diff2 >= 0;
Pg_ns2_basis = (diff2 .^ K) .* mask2;

% 合并并转置
Pg_ns1 = [Pg_ns1_poly; Pg_ns1_basis]';
Pg_ns2 = [Pg_ns2_poly; Pg_ns2_basis']';

% 3. 初始化迭代变量
Init_ux = zeros(n+K+1, 1);
Init_ux(1:K+1) = init;
ua_iter = zeros(n+K+1, iter+1);
ua_iter(:, 1) = Init_ux;

% 获得关于 t 的初值函数矩阵
u1 = Pg_ns1 * ua_iter(:, 1); 
cp_dkut = tk_xj(1:n-K); % 对应原循环中 j 的索引范围
[X_grid, CP_grid] = meshgrid(cp_dkut, tk_xj_new);
U1_grid = repmat(u1(1:length(cp_dkut))', length(tk_xj_new), 1);
dkut_xt = dkut(CP_grid, X_grid, U1_grid);

% 5. 计算逆矩阵
BdJx = cell(1, n);
BdJx{1} = pinv(Pg_ns2 - dkut_xt * Pg_ns1(1:size(dkut_xt, 2), :) ./ length(u1));
%% 主函数
for i=2:iter+1  
    u2=(Pg_ns2*ua_iter(:,i-1));
    % u11=(Pg_ns1*ua_iter(:,i-1));
    S2_x=u2;
    kf=get_kf_nc(tk_xj_new, ua_iter(:, i-1), K, n, kf_f);
    g_x=S2_x-fx(tk_xj_new)'-kf;
    delta_a=BdJx{i-1}*(-g_x);
    ua_iter(:,i)=ua_iter(:,i-1)+delta_a;
    %%%%% Relative Tolerance
    current_norm = norm(ua_iter(:,i));
    if delta_a / (current_norm + 1e-10) < tol
        disp(['Converged in ', num2str(i), ' iterations.']);
        cg_it = i;
        break;
    end
    %%%% Absolute Error
    % if norm(ua_iter(:,i)-ua_iter(:,i-1))<tol
    %     disp(['Converged in ', num2str(i), ' iterations.']);
    %     cg_it=i;
    %     break;
    % end
    % u_11new=Pg_ns1*ua_iter(:,i);
    u2_new=Pg_ns2*ua_iter(:,i);
    kf_new=get_kf_nc(tk_xj_new, ua_iter(:, i), K, n, kf_f);
    gx_new=u2_new-fx(tk_xj_new)'-kf_new;
    delta_gx=gx_new-g_x;%(n+1*1)
    BdJx{i}=BdJx{i-1}+((delta_a-BdJx{i-1}*delta_gx)*(delta_a')*BdJx{i-1})/((delta_a')*BdJx{i-1}*delta_gx);

end
end

function kf = get_kf_nc(x_nodes, coeffs, K, n, kernel_func)
    % NLFIE 专用：直接基于输入节点 x_nodes 的牛顿-科特斯积分
    % 特点：积分网格完全由 x_nodes 决定，不依赖 n 重新生成
    
    % 1. 确定积分阶数 m (自适应匹配 PGFM 阶数)
    if K <= 1
        m = 1; 
    elseif K <= 2
        m = 2; 
    elseif K==3 
        m = 3;
    else
        m=4;
    end       
    [nc_nodes, nc_weights] = get_nc_weights(m);
    num_sub_points = length(nc_nodes);
    
    % 2. 准备目标点 (Target Points)
    % 这些是方程左边的点，保持原顺序，不排序
    x_targets = x_nodes(:); 
    num_targets = length(x_targets);
    
    % 3. 构建积分网格 (Integration Grid)
    % [用户需求]: 网格位置与 x_node 一致
    % 注意：Fredholm 积分通常是 [0, 1]。
    % 如果传入的 x_nodes 已经包含了 0 和 1，则直接使用。
    % 如果不包含，这里强制添加 0 和 1 以确保积分覆盖全域 (根据需求可注释掉)
    
    % 方案 A: 严格使用 x_nodes (风险：如果 x_nodes 不覆盖 [0,1]，积分范围会变小)
    % grid_points = unique(x_nodes);
    
    % 方案 B: 使用 x_nodes 并确保覆盖 [0,1] (推荐)
    grid_points = unique([0; x_nodes(:); 1]);
    
    grid_points = sort(grid_points); % 必须排序以计算步长
    h_vec = diff(grid_points);       % 每个子区间的长度
    num_intervals = length(h_vec);
    interval_starts = grid_points(1:end-1);
    
    % 4. 构建全局采样点 T (隐式扩展)
    % T_matrix: [num_sub_points, num_intervals]
    % 这里的采样点 t 是在每个由 x_nodes 定义的子区间内部生成的
    T_matrix = nc_nodes * h_vec' + interval_starts'; 
    
    % 5. 计算采样点处的解 U
    % 注意: evaluate_u 仍然需要 n 来确定基函数的形状(中心点位置)
    % 除非您想把基函数中心也改成由 x_nodes 决定 (这涉及更底层的基函数定义修改)
    T_flat = T_matrix(:);
    U_flat = evaluate_u_fast_opt(T_flat, coeffs, K, n);
    
    % 恢复 3D 形状 [1, SubPoints, Intervals]
    T_3D = reshape(T_matrix, 1, num_sub_points, num_intervals);
    U_3D = reshape(U_flat,   1, num_sub_points, num_intervals);
    X_3D = reshape(x_targets, num_targets, 1, 1);

    % 6. 计算核函数 (利用广播机制)
    % X_3D (N x 1 x 1) 与 T_3D (1 x S x I) 自动扩展
    Kernel_Val = kernel_func(X_3D, T_3D, U_3D);
    
    % 7. 积分加权求和
    W_3D = reshape(nc_weights, 1, num_sub_points, 1);
    
    % 第一步：子区间内积分 (加权求和) -> [N, 1, I]
    Segment_Integrals = sum(Kernel_Val .* W_3D, 2);
    Segment_Integrals = squeeze(Segment_Integrals); 
    
    % 保护：防止单点计算时 squeeze 转置
    if num_targets == 1, Segment_Integrals = Segment_Integrals(:).'; end
    
    % 第二步：乘以区间长度 Jacobian -> [N, I]
    Segment_Integrals = Segment_Integrals .* h_vec';
    
    % 8. 全局求和 (Fredholm)
    % 对所有小区间求和得到 int_0^1
    kf = sum(Segment_Integrals, 2); 
end

%% 辅助函数 (保持不变，必须包含)
function u_val = evaluate_u_fast_opt(t, coeffs, K, n)
    % 注意：这里的 CPPF(n) 定义了基函数的中心点位置
    % 如果您希望基函数本身也不依赖 n，需要传入基函数中心点向量
    tk_map = CPPF(n); 
    Phi = zeros(length(t), length(coeffs));
    for p = 0:K, Phi(:, p+1) = t.^p; end
    centers = tk_map(1 : length(coeffs)-K-1);
    if ~isempty(centers)
        D = t - centers; 
        Phi(:, K+2:end) = (D.^K) .* (D > 0 & t <= 1);
    end
    u_val = Phi * coeffs;
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
