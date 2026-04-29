function [coeffs_numeric, u_series] = SolveMixedNumeric(fx_handle, kf_handle, K)
    % SolveMixedNumeric 数值极速版 (针对混合 Volterra-Fredholm 方程)
    % 
    % 问题形式: u(x) = f(x) + int_0^x int_0^1 K(t, y, u(y)) dy dt
    %
    % 特点:
    % 1. 使用 Picard 迭代法在固定网格上快速逼近解。
    % 2. 针对双重积分利用矩阵运算加速。
    % 3. 输出 K 阶多项式拟合系数。
    
    % --- 1. 参数设置 ---
    N = 51;                 % 固定离散点数，用于快速估算
    x_grid = linspace(0, 1, N)'; 
    y_grid = x_grid;        % 内层积分网格
    h = 1/(N-1);
    
    % Picard 迭代设置
    max_iter = 15;          
    tol = 1e-4;
    
    % 积分权重 (梯形公式)
    weights = ones(N, 1) * h;
    weights(1) = h/2;
    weights(end) = h/2;
    
    % --- 2. 迭代求解 ---
    u = fx_handle(x_grid); % 初始猜测 u0 = f(x)
    
    for iter = 1:max_iter
        u_prev = u;
        
        % A. 内层积分: F(t) = int_0^1 K(t, y, u(y)) dy
        % 计算每个 t (即 x_grid) 对应的 F(t)
        F_t = zeros(N, 1);
        
        % 循环计算每一行 t 的积分 (此时 u 已知)
        for i = 1:N
            t_val = x_grid(i);
            k_vals = kf_handle(t_val, y_grid, u);
            F_t(i) = sum(k_vals .* weights);
        end
        
        % B. 外层积分: V(x) = int_0^x F(t) dt
        % Volterra 积分：对 F(t) 进行累积梯形积分
        cum_trapz = cumsum((F_t(1:end-1) + F_t(2:end)) * h / 2);
        V_x = [0; cum_trapz];
        
        % C. 更新 u
        u = fx_handle(x_grid) + V_x;
        
        % 简单收敛检查
        if norm(u - u_prev) < tol
            break;
        end
    end
    
    % --- 3. 多项式拟合与输出 ---
    % 使用 polyfit 将离散点拟合为 K 阶多项式
    % polyfit 返回降幂 [c_K, ..., c_0]
    p_coeffs_desc = polyfit(x_grid, u, K);
    
    % 调整为升幂排列 [c_0, c_1, ..., c_K] 以适配 PGFM 习惯
    coeffs_numeric = fliplr(p_coeffs_desc);
    
    % 生成函数句柄
    u_series = @(x) polyval(p_coeffs_desc, x);
    
    fprintf('初值数值计算完成 (Picard Iter: %d, FitOrder=%d)\n', iter, K);
end