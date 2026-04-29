function [coeffs_numeric, u_func] = SolveFredholmTaylorNumeric(fx_handle, kf_handle, a, b, K)
    % SolveFredholmTaylor 数值极速版 (Nyström + Picard)
    %
    % 原理: 
    %   1. 将积分方程离散化为 N 个点的代数方程。
    %   2. 使用皮卡迭代 (Picard Iteration) 更新数值解。
    %   3. 最后对收敛的离散点进行多项式拟合。
    %
    % 输入:
    %   fx_handle : f(x)
    %   kf_handle : K(x,t,u)
    %   a, b      : 积分区间
    %   K         : 最终拟合的阶数
    
    % --- 1. 参数设置 (追求效率的固定配置) ---
    N = 51;                 % 固定离散点数
    x_grid = linspace(a, b, N);
    h = (b - a) / (N - 1);
    
    % 预计算梯形公式的权重向量
    % weights = [0.5, 1, 1, ..., 1, 0.5] * h
    w = ones(1, N);
    w(1) = 0.5;
    w(end) = 0.5;
    w = w * h;
    
    MaxIter = 20;           % 最大迭代次数
    Tol = 1e-5;             % 收敛容差
    
    % --- 2. 初始猜测 ---
    % 初始假设 u(x) = f(x) (对应积分项为0的情况)
    u_curr = fx_handle(x_grid);
    
    fprintf('启动 Fredholm 数值求解 (Range=[%.1f, %.1f], N=%d)...\n', a, b, N);
    
    % --- 3. 皮卡迭代 (Picard Iteration) ---
    % 方程形式: u(x) = f(x) + int(K(x,t,u(t)))
    
    for iter = 1:MaxIter
        u_prev = u_curr;
        u_next = zeros(1, N);
        
        % 遍历每一个 x_i 点，计算其对应的积分值
        for i = 1:N
            xi = x_grid(i);
            
            % 核心加速点：一次性计算所有 t 点的核函数值
            % K_vals 向量对应 K(xi, t_1...t_N, u_1...u_N)
            try
                % 尝试向量化调用 (如果 kf_handle 支持)
                K_vals = kf_handle(xi, x_grid, u_prev);
            catch
                % 如果 kf_handle 不支持向量化，降级为循环 (增强鲁棒性)
                K_vals = zeros(1, N);
                for j = 1:N
                    K_vals(j) = kf_handle(xi, x_grid(j), u_prev(j));
                end
            end
            
            % 数值积分 = 权重 * 核函数值
            integral_val = sum(w .* K_vals);
            
            % 更新 u(xi)
            u_next(i) = fx_handle(xi) + integral_val;
        end
        
        % 更新解向量
        u_curr = u_next;
        
        % 检查收敛性
        err = max(abs(u_curr - u_prev));
        if err < Tol
            % fprintf('  -> 收敛于第 %d 次迭代 (Err=%.1e)。\n', iter, err);
            break;
        end
    end
    
    if iter == MaxIter
        warning('达到最大迭代次数，结果可能未完全收敛 (Last Err=%.1e)。', err);
    end
    
    % --- 4. 结果拟合与输出 ---
    % 将离散点拟合为 K 阶多项式
    % polyfit 返回降幂系数 [c_K, ..., c_0]
    p_coeffs_desc = polyfit(x_grid, u_curr, K);
    
    % 转为升幂排列 [c_0, c_1, ..., c_K]
    coeffs_numeric = fliplr(p_coeffs_desc);
    
    % 构建返回的函数句柄
    u_func = @(x) polyval(p_coeffs_desc, x);
    
    fprintf('计算完成 (FitOrder=%d)。\n', K);
end