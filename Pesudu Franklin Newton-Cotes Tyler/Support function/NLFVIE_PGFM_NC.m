function [ua_iter,tk_xj,cg_it] = NLFVIE_PGFM_NC(fx, dvkut,dfkut, n, iter,tol,K, kernel_func_v, kernel_func_f,init)
% nonlinear Fredholm integral equation
% fx 为给定函数, 接受x参数 定义为函数句柄
% kuf 为已知积分核, 接受x,t,u三个参数 定义为函数句柄,包含被解函数的函数
% dkut表示kuf对被解函数u(t)的偏导
% n 为阶数，iter,tol表示迭代此时和容差
% ux_iter 为 每次迭代后的值
% cg_it表示收敛时的阶次
% load('true_coff.mat','Array_fx1');
%% 预分配小波参数
pseudo_Franklin_basis=@(x,cp,K)0.*(0<=x& x< cp)+((x - cp).^K).*(cp<=x&x <= 1);
% K=1;
Pg_ns1=zeros(n+K+1,n+2);
Pg_ns2=zeros(n+K+1,n+K+1);
% Pg_ns3=zeros(n+K+1,n+K);
tk_xj=CPPF(n);
tk_xj2=[0,tk_xj,1];
tk_xj_new=[linspace(0,tk_xj(1),ceil(K/2)+1),tk_xj(2:n-1),linspace(tk_xj(n),1,ceil((K+1)/2)+1)];
% tk_xj_new2=(tk_xj_new(1:n+K)+tk_xj_new(2:n+K+1))./2;
% tk_xj_new=CPPF(n+K+1);
for i=0:K
    Pg_ns1(i+1,:)=tk_xj2.^(i); % 0,x,x^2
    Pg_ns2(i+1,:)=tk_xj_new.^(i); % 0,x,x^2
    % Pg_ns3(i+1,:)=tk_xj_new2.^(i);
end
for i=K+1:n+K
    Pg_ns1(i+1,:)=pseudo_Franklin_basis(tk_xj2,tk_xj(i-K),K); 
    Pg_ns2(i+1,:)=pseudo_Franklin_basis(tk_xj_new,tk_xj(i-K),K); 
    % Pg_ns3(i+1,:)=pseudo_Franklin_basis(tk_xj_new2,tk_xj(i-K),K); 
end
Pg_ns1=Pg_ns1';
Pg_ns2=Pg_ns2';
% Pg_ns3=Pg_ns3';
dvkut_xt=zeros(n+K+1,n);
dfkut_xt=zeros(n+K+1,n);
Init_ux = zeros(n+K+1, 1); % 定义迭代初值
Init_ux(1:K+1)=init;
% Init_ux(3)=0.5;
% Init_ux(2)=1;
% Init_ux(1)=-1;
ua_iter=zeros(n+K+1,iter+1);
ua_iter(:,1)=Init_ux;
u1=Pg_ns1*ua_iter(:,1); % 获得关于t的初值函数矩阵
for cc=1:n
    if cc == 1
    dvkut_xt(1:length(tk_xj_new),cc)=dvkut(tk_xj_new,tk_xj(cc),u1(cc+1)); % 表示k(x,t)的矩阵，
    else
    dvkut_xt((ceil(K/2)+cc):length(tk_xj_new),cc)=dvkut(tk_xj_new((ceil(K/2)+cc):length(tk_xj_new)),tk_xj(cc),u1(cc+1));
    end
    dfkut_xt(:,cc)=dfkut(tk_xj_new,tk_xj(cc),u1(cc+1));
end
dvkut_xt(1,1)=0;
BdJx=cell(1,n);
% Pg_ns1(2:n+1,:) 对应于 t_1, ..., t_n 这些点
Pg_ns1_nodes = Pg_ns1(2:n+1, :); 
% h_vec 对应于 t_1, ..., t_n 的区间长度
h_vec = tk_xj2(2:n+1) - tk_xj2(1:n); 
BdJx{1}=pinv(Pg_ns2 - dvkut_xt * (Pg_ns1_nodes .* h_vec')-dfkut_xt * (Pg_ns1_nodes .* h_vec'));
%% 主函数
for i=2:iter+1  
    u2=(Pg_ns2*ua_iter(:,i-1));
    % u_t=(Pg_ns1*ua_iter(:,i-1));
    % u_t2=(Pg_ns3*ua_iter(:,i-1));
    kf = get_kf_mixed_nc(tk_xj_new, ua_iter(:, i-1), K, n, kernel_func_v, kernel_func_f);
    g_x=u2-fx(tk_xj_new)'-kf;
    delta_a=BdJx{i-1}*(-g_x);
    ua_iter(:,i)=ua_iter(:,i-1)+delta_a;
    %%%%% Relative Tolerance
    current_norm = norm(ua_iter(:,i));
    if delta_a / (current_norm + 1e-14) < tol
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
    % u_tnew=Pg_ns1*ua_iter(:,i);
    u2_new=Pg_ns2*ua_iter(:,i);
    % u_t2_new=Pg_ns3*ua_iter(:,i);
    kf_new=get_kf_mixed_nc(tk_xj_new, ua_iter(:, i), K, n, kernel_func_v, kernel_func_f);
    gx_new=u2_new-fx(tk_xj_new)'-kf_new;
    delta_gx=gx_new-g_x;%(n+1*1)
    % muk=1+(delta_gx'*BdJx{i-1}*delta_gx)./(delta_a'*delta_gx);
    % BdJx{i}=BdJx{i-1}+(muk*(delta_a*delta_a')-(delta_a*delta_gx')*BdJx{i-1}-BdJx{i-1}*delta_gx*delta_a')...
    % /((delta_a')*delta_gx);% 秩2方法
    BdJx{i}=BdJx{i-1}+((delta_a-BdJx{i-1}*delta_gx)*(delta_a')*BdJx{i-1})/((delta_a')*BdJx{i-1}*delta_gx);% Sherman–Morrison formula
    % BdJx{i}=BdJx{i-1}+(delta_a-BdJx{i-1}*delta_gx)*(delta_gx')/(norm(delta_gx)^2);
    % BdJx{i}=BdJx{i-1}+((delta_a-BdJx{i-1}*delta_gx)*(delta_a-BdJx{i-1}*delta_gx)')/((delta_a-BdJx{i-1}*delta_gx)'*delta_gx);
    % BdJx{i}=BdJx{i-1}+((delta_gx-BdJx{i-1}*delta_a)*(delta_a'))./((delta_a')*delta_a);%不求逆
end
end
%% 问题1
function kf = get_kf_mixed_nc(x_nodes, coeffs, K, n, kernel_func_v, kernel_func_f)
% GET_KF_MIXED_NC 计算混合 Volterra-Fredholm 积分项 (高性能优化版)
%
% 输入:
%   x_nodes:       积分节点/配点
%   coeffs:        当前解的系数向量
%   K:             多项式阶数
%   n:             配置点数量
%   kernel_func_v: Volterra 核函数句柄 @(x,t,u)...
%   kernel_func_f: Fredholm 核函数句柄 @(x,t,u)...
%
% 输出:
%   kf:            对应的积分项向量 (Volterra + Fredholm)

    %% 1. 参数准备 (保持原逻辑)
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
    
    %% 2. 构建积分网格
    x_nodes = x_nodes(:);    
    % 确定积分区间端点 (假设积分下限为 0)
    grid_points = x_nodes;
    h_vec = diff(grid_points);     % 区间步长向量
    interval_starts = grid_points(1:end-1);    
    % 生成全区间采样点 T_flat (1 x TotalSamples)
    % 维度: (m+1) x num_intervals -> 展平
    T_matrix = nc_nodes * h_vec' + interval_starts'; 
    T_flat = T_matrix(:); 
    
    %% 3. 计算 u(t) (公共部分)
    % 一次性计算所有积分点上的 u 值，供两个核函数共用
    U_flat = evaluate_u_fast_opt(T_flat, coeffs, K, n);
    
    % 准备广播用的向量方向 (转置为行向量以匹配 x_nodes 的列向量)
    x_col = x_nodes;        % N x 1
    t_row = T_flat(:)';     % 1 x M
    u_row = U_flat(:)';     % 1 x M
    
    %% 4. 计算 Volterra 部分 (积分上限为 x)
    % [Step V1] 计算核矩阵
    % 结果维度: N x M (Targets x TotalSamples)
    Mat_V = kernel_func_v(x_col, t_row, u_row);
    
    % [Step V2] 分段积分 (Strided Accumulation)
    % 结果维度: Targets x Intervals
    Seg_Int_V = compute_segment_integrals(Mat_V, nc_weights, h_vec, num_sub_points);
    
    % [Step V3] Volterra 累积求和
    % 利用 cumsum 模拟积分上限的变化，并提取对角线
    Cumul_V = cumsum(Seg_Int_V, 2);
    
    if abs(x_nodes(1)) < 1e-14
        % 如果第一个点是0，对应积分值为0
        kf_vals_v = [0; diag(Cumul_V(2:end, :))];
    else
        kf_vals_v = diag(Cumul_V);
    end
    
    %% 5. 计算 Fredholm 部分 (积分上限固定为 1)
    % [Step F1] 计算核矩阵
    Mat_F = kernel_func_f(x_col, t_row, u_row);
    
    % [Step F2] 分段积分
    Seg_Int_F = compute_segment_integrals(Mat_F, nc_weights, h_vec, num_sub_points);
    
    % [Step F3] Fredholm 全局求和
    % 对每一行(每个目标点 x)的所有区间积分值求和
    kf_vals_f = sum(Seg_Int_F, 2);
    
    %% 6. 结果合并
    kf = kf_vals_v + kf_vals_f;
    
end

%% 内部辅助函数：计算分段积分 (将循环逻辑封装)
function Segment_Integrals = compute_segment_integrals(Kernel_Matrix, weights, h_vec, m_pts)
    [num_targets, ~] = size(Kernel_Matrix);
    num_intervals = length(h_vec);
    
    Segment_Integrals = zeros(num_targets, num_intervals);
    
    % 步长索引累加 (Strided Accumulation)
    for i = 1:m_pts
        vals = Kernel_Matrix(:, i:m_pts:end);
        if weights(i) ~= 0
            Segment_Integrals = Segment_Integrals + vals * weights(i);
        end
    end
    
    % 乘以区间长度 (Jacobian)
    Segment_Integrals = bsxfun(@times, Segment_Integrals, h_vec');
end

%% 辅助函数：获取权重 (保持不变)
function [nodes, weights] = get_nc_weights(m)
    nodes = linspace(0, 1, m + 1)';
    switch m
        case 1, weights = [1; 1] / 2;
        case 2, weights = [1; 4; 1] / 6;
        case 3, weights = [1; 3; 3; 1] / 8;
        case 4, weights = [7; 32; 12; 32; 7] / 90;
    end
end

%% 辅助函数：快速基函数求值 (保持不变)
function u_val = evaluate_u_fast_opt(t, coeffs, K, n)
    t = t(:);            
    coeffs = coeffs(:);  
    
    tk_map = CPPF(n); 
    num_pts = length(t);
    num_coeffs = length(coeffs);
    
    Phi = zeros(num_pts, num_coeffs);
    
    for p = 0:K
        Phi(:, p+1) = t.^p;
    end
    
    centers = tk_map(1 : num_coeffs-K-1);
    
    if ~isempty(centers)
        D = bsxfun(@minus, t, centers(:)'); 
        if K == 0
            Phi(:, K+2:end) = double(D > 0);
        else
            Phi(:, K+2:end) = (max(0, D)).^K;
        end
    end
    
    u_val = Phi * coeffs;
end