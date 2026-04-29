function [Array_fx, tk_xj] = LFIE_HAAR_S2(fx, kf, n)
% LFIE_HAAR_S2_Optimized: 二阶 Haar 小波积分方程求解器 (矩阵化极速版)
% 
% 优化说明:
% 1. 移除了所有 for 循环，利用 MATLAB 矩阵广播机制实现全向量化。
% 2. 修正了原代码中潜在的维度不一致风险，统一使用列向量运算。
% 3. 使用左除 '\' 替代 pinv，提升线性方程组求解的精度与速度。

%% 1. 初始化与网格生成
% 强制配置点为列向量 [n x 1]
tk_xj = CP(n); 
if isrow(tk_xj), tk_xj = tk_xj.'; end 

% 生成 Haar 参数
j_level = ceil(log2(n));
[alpha, beta, gamma] = generate_haar(j_level); 
s = 2; % 阶数固定为 2

%% 2. 基础矩阵构建
% 2.1 核函数矩阵 K_mat [n x n]
% 利用隐式扩展计算 K(x_i, t_j)
K_mat = kf(tk_xj, tk_xj.'); 

% 2.2 基函数矩阵 Phi [n x n] 和 边界值向量 Phi_x1 [1 x n]
% Phi(j, k) -> 第 k 个基函数在第 j 个点的值
Phi = compute_haar_matrix(tk_xj, alpha, beta, gamma, s);
Phi_x1 = compute_haar_matrix(1, alpha, beta, gamma, s); % x=1 处的基函数值 (行向量)

%% 3. 计算积分标量参数 (s2 - s5)
% 预计算 K(0, t) 和 K(1, t) 的行向量 [1 x n]
k_0_t = kf(0, tk_xj); if iscolumn(k_0_t), k_0_t = k_0_t.'; end
k_1_t = kf(1, tk_xj); if iscolumn(k_1_t), k_1_t = k_1_t.'; end

% 计算标量 s2, s3, s4, s5
inv_n = 1/n;
s2 = sum(k_0_t) * inv_n;
s3 = sum(k_1_t) * inv_n;
s4 = (k_0_t * tk_xj) * inv_n; % 向量点积
s5 = (k_1_t * tk_xj) * inv_n;

% 计算分母常数 s8
s8 = 1 - s2 + s4*(1-s3) - s5*(1-s2);

%% 4. 计算积分向量参数 (s6, s7)
% s6, s7 为 [1 x n] 行向量，表示 K(0,t) 和 K(1,t) 与基函数的加权积分
s6 = (k_0_t * Phi) * inv_n;
s7 = (k_1_t * Phi) * inv_n;

%% 5. 组装线性系统矩阵 (LHS)
% 5.1 准备辅助向量 (列向量 [n x 1])
% Term: 1 - int K(x,t)*1 dt
vec_1_min_K1 = 1 - (K_mat * ones(n,1)) * inv_n;
% Term: x - int K(x,t)*t dt
% 注意: 原代码此处可能有维度歧义，此处修正为标准的算子作用于函数 t (即 tk_xj)
vec_t_min_Kt = tk_xj - (K_mat * tk_xj) * inv_n;

% 5.2 准备辅助行向量 (Row Terms [1 x n])
% 对应原代码中 s6_arr 和 s7_arr 相关的复杂项
% Row 1: 对应 C1 系数的乘子部分
row_term_1 = s6 * (1 - s5) - s4 * (Phi_x1 - s7);
% Row 2: 对应 C2 系数的乘子部分
row_term_2 = -(1 - s3) * s6 - (1 - s2) * (Phi_x1 - s7);

% 5.3 组合系统矩阵 A
% A = (Phi - K*Phi) + (vec1 * row1)/s8 + (vect * row2)/s8
% 利用向量外积 (Column * Row) 生成矩阵，替代 repmat
Mat_Core = Phi - (K_mat * Phi) * inv_n;
Mat_Bnd1 = (vec_1_min_K1 * row_term_1) / s8;
Mat_Bnd2 = (vec_t_min_Kt * row_term_2) / s8;

Array_ax_mat = Mat_Core + Mat_Bnd1 + Mat_Bnd2;

%% 6. 组装右端项 (RHS)
% 计算 RHS 中的常数系数
f0 = fx(0);
f1 = fx(1);
rhs_const_1 = (f0*(1-s5) + f1*s4) / s8;
rhs_const_2 = (-f0*(1-s3) + f1*(1-s2)) / s8;

% RHS = f(x) - c1 * vec_1_min_K1 - c2 * vec_t_min_Kt
Constant_xj = fx(tk_xj) - rhs_const_1 * vec_1_min_K1 - rhs_const_2 * vec_t_min_Kt;

%% 7. 求解与后处理
% 求解线性方程组 (使用 LU 分解优于伪逆)
Array_fx_core = Array_ax_mat \ Constant_xj;

% 恢复系数 C1, C2
% 利用之前计算的 row_term 直接计算，无需重新书写公式
% C1 formula part: ... + (row_term_1 * a)
OUTPUT_c1 = rhs_const_1 + (row_term_1 * Array_fx_core) / s8;
% C2 formula part: ... + (row_term_2 * a)
OUTPUT_c2 = rhs_const_2 + (row_term_2 * Array_fx_core) / s8;

% 组合最终输出 [C1; a1...an; C2]
Array_fx = [OUTPUT_c1; Array_fx_core; OUTPUT_c2];

end

%% ---------------------------------------------------------
%  辅助函数：向量化 Haar 矩阵生成
%% ---------------------------------------------------------
function Phi = compute_haar_matrix(x, alpha, beta, gamma, s)
    % 输入 x 可以是标量或列向量
    % 返回 Phi 矩阵 [length(x) x length(alpha)]
    
    x = x(:);       % [Nx x 1]
    alpha = alpha(:).'; % [1 x Nb]
    beta  = beta(:).';
    gamma = gamma(:).';
    
    fact_s = factorial(s);
    
    % 利用广播计算差值矩阵
    D_alpha = bsxfun(@minus, x, alpha);
    D_beta  = bsxfun(@minus, x, beta);
    D_gamma = bsxfun(@minus, x, gamma);
    
    % 逻辑掩码
    mask1 = (x >= alpha) & (x < beta);
    mask2 = (x >= beta)  & (x < gamma);
    mask3 = (x >= gamma) & (x <= 1);
    
    % 向量化计算三段函数值
    term1 = mask1 .* (D_alpha.^s);
    
    term2 = mask2 .* (D_alpha.^s - 2 * D_beta.^s);
    
    term3 = mask3 .* (D_alpha.^s - 2 * D_beta.^s + D_gamma.^s);
    
    Phi = (term1 + term2 + term3) ./ fact_s;
end