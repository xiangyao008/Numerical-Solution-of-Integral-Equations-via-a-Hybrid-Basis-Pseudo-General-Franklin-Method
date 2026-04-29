function init_coeffs = Get_Volterra_Init_Coeffs(fx_handle, kf_handle, dkf_handle, a, b, K)
% Get_Volterra_Init_Coeffs
% 使用最速下降法计算 Volterra 方程的多项式近似系数，作为主求解器的初值。
%
% 输入:
%   fx_handle, kf_handle, dkf_handle : f(x), K(x,t,u), dK/du
%   a, b : 求解区间
%   K    : 多项式阶数
%
% 输出:
%   init_coeffs : (K+1)x1 列向量，顺序为 [x^0, x^1, ..., x^K]
%                 即: u(x) = c(1) + c(2)*x + ... + c(K+1)*x^K

    % 1. 网格设置 (用于优化初值的粗网格)
    M = 64; 
    x_grid = linspace(a, b, M)';
    h = (b - a) / (M - 1);
    
    % 2. 预计算 f(x) 和锚定初值
    f_vec = fx_handle(x_grid);
    a0 = f_vec(1); % 强制固定常数项 u(a) = f(a)，消除起点误差
    
    % 3. 准备基函数矩阵 Phi (对应 x^1, x^2, ..., x^K)
    % 注意：这里不包含 x^0，因为 a0 已固定
    Phi = zeros(M, K);
    for j = 1:K
        Phi(:, j) = (x_grid - a).^j;
    end
    
    % 待优化系数 (对应 x^1 ... x^K)
    coeffs_opt = zeros(K, 1); 
    
    % 4. 最速下降迭代参数
    max_iter = 200; 
    tol = 1e-3; % 初值不需要极其精确，1e-3 足够
    learning_rate = 0.5;
    
    % 5. 迭代求解
    for iter = 1:max_iter
        % 当前解 u(x)
        u_vec = a0 + Phi * coeffs_opt;
        
        % 计算积分项
        I_vec = compute_integral_simple(x_grid, u_vec, kf_handle, h);
        
        % 残差 R(x)
        Residual = u_vec - f_vec - I_vec;
        Loss = 0.5 * sum(Residual.^2) * h;
        
        if Loss < tol, break; end
        
        % 计算梯度 (简化版，仅利用数值投影)
        % dJ/dc = Phi' * R * h (近似梯度方向)
        Gradient = (Phi' * Residual) * h;
        
        % 步长更新 (简单的回溯线搜索)
        coeffs_new = coeffs_opt - learning_rate * Gradient;
        
        % 检查新一步是否发散，如果发散则减小步长
        u_new = a0 + Phi * coeffs_new;
        I_new = compute_integral_simple(x_grid, u_new, kf_handle, h);
        Loss_new = 0.5 * sum((u_new - f_vec - I_new).^2) * h;
        
        if Loss_new > Loss
            learning_rate = learning_rate * 0.5;
        else
            coeffs_opt = coeffs_new;
            learning_rate = min(1.0, learning_rate * 1.1);
        end
    end
    
    % 6. 转换系数格式 (适配 NLVIE_PGFM_NC 的输入要求)
    % 您的求解器循环是: for i=0:K, ... tk_xj.^(i)
    % 这意味着系数应该是升幂排列: [Const, x, x^2, ...]
    
    % 目前得到的 coeffs_opt 是对应 (x-a)^j 的。
    % 如果 a=0 (通常情况)，直接组合即可。
    % 如果 a!=0，需要二项式展开 (这里假设 a=0 简化处理，或提醒用户)
    
    if a == 0
        init_coeffs = [a0; coeffs_opt];
    else
        % 简单的平移处理: u = a0 + c1(x-a) + ... 
        % 建议直接返回基于 (x-a) 的系数，并在主程序中处理，
        % 或者使用 polyfit 将数值解转换回标准幂基底。
        [p_std, ~, mu] = polyfit(x_grid, a0 + Phi*coeffs_opt, K);
        % polyfit 返回降幂 [x^K ... x^0]，且可能带缩放
        % 这里为了稳健，我们直接返回数值拟合的标准系数
        init_coeffs = flip(polyfit(x_grid, a0 + Phi*coeffs_opt, K))';
    end
end

function I_val = compute_integral_simple(x, u, kf, h)
    % 简单的梯形积分用于梯度计算
    M = length(x);
    I_val = zeros(M, 1);
    for i = 2:M
        ti = x(1:i); ui = u(1:i); xi = x(i);
        K_val = kf(xi, ti, ui);
        w = ones(i,1)*h; w(1)=h/2; w(end)=h/2;
        I_val(i) = sum(w .* K_val);
    end
end