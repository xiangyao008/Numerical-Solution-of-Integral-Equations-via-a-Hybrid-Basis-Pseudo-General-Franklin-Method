function [Array_ax, tk_xj] = LFIE_HAAR_S1(fx, kf, n)
% LFIE_HAAR_S1_Optimized: 线性 Fredholm 积分方程 Haar 小波求解器 (矩阵化优化版)
%
% 输入:
%   fx - 目标函数句柄 f(x)
%   kf - 积分核句柄 K(x,t)
%   n  - 逼近阶数 (建议为 2 的幂次)
%
% 输出:
%   Array_ax - 未知系数向量 (包含 C1)
%   tk_xj    - 离散配置点

%% 1. 初始化与参数生成
% 生成配置点 (假设 CP 函数存在且返回列向量)
tk_xj = CP(n); 
if isrow(tk_xj), tk_xj = tk_xj.'; end % 确保为列向量 [n x 1]

% 生成 Haar 小波参数 (假设 generate_haar 存在)
j_level = ceil(log2(n));
[alpha, beta, gamma] = generate_haar(j_level); 

%% 2. 矩阵化构建关键组件
% 2.1 计算 s2 (常数项)
% 利用向量化计算 K(0, t_j) 的均值
K_0_t = kf(0, tk_xj); % [1 x n] 或 [n x 1]
s2 = sum(K_0_t) / n;

% 2.2 构建基函数矩阵 P_ns [n x n]
% P_ns(j, i) 对应原代码逻辑：第 j 个点在第 i 个基函数下的值
% 通过矩阵运算替代原来的 for i=1:n 循环
P_ns = compute_haar_matrix(tk_xj, alpha, beta, gamma, 1);
% 注意：原代码最后做了转置 P_ns = P_ns'，此处我们在函数内部处理或保持一致
% 原定义：行是基函数参数 i。原代码 P_ns(i,:) = integral... 然后转置。
% 现在的 compute_haar_matrix 返回的是 [n_points x n_basis]，即原代码转置后的状态。

% 2.3 构建核函数矩阵 K_mat [n x n]
% 利用隐式扩展 K(x_i, t_j)，消除双重循环
K_mat = kf(tk_xj, tk_xj.'); 

%% 3. 构建线性方程组
% 3.1 计算 s1 向量
% 原逻辑: s1 = (((P_ns')*(kf(0,tk_xj)'))./n)';
% P_ns 现在是 [n x n] (点 x 基), K_0_t 是 [n x 1] (如果 kf 返回列)
% 目标是积分 int K(0,t) * Phi_i(t) dt
if isrow(K_0_t), K_0_t = K_0_t.'; end
s1 = (K_0_t.' * P_ns) ./ n; % 结果为 [1 x n] 行向量

% 3.2 计算 C1 (常数)
f_0 = fx(0);
C1 = f_0 / (1 - s2);

% 3.3 计算 C2 矩阵项
% 原逻辑: C2=((ones(n,1)-(Constant_kf*ones(n,1))./n).*s1_new).*(1/(1-s2));
% term_k: int K(x,t) dt 近似值 -> K_mat * ones / n
term_k = sum(K_mat, 2) ./ n; % 按行求和除以 n, 得到 [n x 1]
factor_vec = (1 - term_k) * (1 / (1 - s2)); % [n x 1]
C2_mat = factor_vec * s1; % [n x 1] * [1 x n] -> [n x n] 矩阵

% 3.4 组装系统矩阵 LHS * a = RHS
% 原方程: (P_ns - (K * P_ns)/n + C2) * a = ...
LHS = P_ns - (K_mat * P_ns) ./ n + C2_mat;

% 3.5 组装右端项 RHS
% RHS = f(x) - C1 * (1 - int K(x,t) dt)
fx_vals = fx(tk_xj); % [n x 1]
RHS = fx_vals - C1 * (1 - term_k);

%% 4. 求解与后处理
% 使用左除 '\' 替代 pinv，提高速度和精度
% 如果矩阵接近奇异，MATLAB 会发出警告
Array_ax_raw = LHS \ RHS;

% 计算 OUTPUT_c1
OUTPUT_c1 = (s1 * Array_ax_raw + f_0) / (1 - s2);

% 组合最终输出
Array_ax = [OUTPUT_c1; Array_ax_raw];

end

%% ---------------------------------------------------------
%  辅助函数：矩阵化 Haar 小波积分计算
%% ---------------------------------------------------------
function Phi = compute_haar_matrix(x, alpha, beta, gamma, s)
    % 输入:
    %   x: 配置点向量 [n_pts x 1]
    %   alpha, beta, gamma: Haar 参数向量 [1 x n_basis]
    %   s: 阶数 (通常为 1)
    % 输出:
    %   Phi: [n_pts x n_basis] 矩阵
    
    % 确保维度正确以利用广播 (Broadcasting)
    x = x(:); % 列向量
    alpha = alpha(:).'; % 行向量
    beta = beta(:).';
    gamma = gamma(:).';
    
    % 预计算阶乘
    fact_s = factorial(s);
    
    % 利用逻辑矩阵进行分段计算 (核心向量化部分)
    % 第一段: alpha <= x < beta
    mask1 = (x >= alpha) & (x < beta);
    term1 = mask1 .* ((x - alpha).^s ./ fact_s);
    
    % 第二段: beta <= x < gamma
    mask2 = (x >= beta) & (x < gamma);
    term2 = mask2 .* (((x - alpha).^s - 2*(x - beta).^s) ./ fact_s);
    
    % 第三段: gamma <= x <= 1
    mask3 = (x >= gamma) & (x <= 1);
    term3 = mask3 .* (((x - alpha).^s - 2*(x - beta).^s + (x - gamma).^s) ./ fact_s);
    
    % 组合结果
    Phi = term1 + term2 + term3;
end