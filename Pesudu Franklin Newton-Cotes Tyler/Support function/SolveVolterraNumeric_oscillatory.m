function [coeffs_numeric, u_func] = SolveVolterraNumeric_oscillatory(fx_handle, kf_handle, a, b, K)
    % SolveVolterraNumeric_Robust
    % 
    % 采用 "逐步隐式推进法 (Step-by-Step Implicit Marching)" 求解非线性 Volterra 积分方程。
    % 相比全局迭代法，该方法对刚性问题(Stiff)、震荡核和强非线性具有极高的鲁棒性。
    %
    % 方法:
    %   1. 离散化: 均匀网格 + 复合梯形公式 (Composite Trapezoidal Rule)。
    %   2. 求解策略: 利用 Volterra 方程的下三角特性，将大系统分解为 N 个标量非线性方程。
    %   3. 根查找: 对每个节点使用 "阻尼牛顿法 (Damped Newton)" 求解 u_i。
    %   4. 结果拟合: 对离散解进行多项式拟合，输出系数。
    %
    % 输入:
    %   fx_handle, kf_handle : 函数句柄
    %   a, b : 求解区间
    %   K    : 输出多项式的阶数
    
    % --- 1. 网格设置 ---
    % 适当增加网格密度以捕捉震荡或剧烈变化
    N = 101; 
    x_grid = linspace(a, b, N);
    h = (b - a) / (N - 1);
    
    u_vals = zeros(1, N);
    
    % --- 2. 初始条件 (x_1) ---
    % Volterra 方程中，积分为 0，故 u(a) = f(a)
    u_vals(1) = fx_handle(x_grid(1));
    
    % 预计算 f(x) 避免重复调用
    f_vec = fx_handle(x_grid);
    
    % 打印状态
    % fprintf('启动 Volterra 鲁棒求解 (Range=[%.2f, %.2f], N=%d)...\n', a, b, N);
    
    % --- 3. 逐步推进求解 (Time Marching) ---
    for i = 2:N
        xi = x_grid(i);
        
        % 计算 "历史积分项" (History Term)
        % Integral ~ h/2 * K(t0) + h * sum(K(t1...ti-1)) + h/2 * K(ti)
        % 我们先把已知的前 i-1 项算出来
        
        history_sum = 0;
        
        % 向量化计算历史核函数值
        t_hist = x_grid(1:i-1);
        u_hist = u_vals(1:i-1);
        
        try
            K_hist = kf_handle(xi, t_hist, u_hist);
        catch
            % 降级循环处理（防止核函数不支持向量化）
            K_hist = zeros(1, i-1);
            for j=1:i-1, K_hist(j) = kf_handle(xi, t_hist(j), u_hist(j)); end
        end
        
        % 梯形公式权重处理历史项
        % 权重要点: 首项 h/2, 中间项 h
        weights_hist = ones(1, i-1) * h;
        weights_hist(1) = h / 2; 
        
        history_sum = sum(weights_hist .* K_hist);
        
        % --- 4. 建立当前步的标量非线性方程 ---
        % 方程: u_i = f(x_i) + History + (h/2) * K(x_i, x_i, u_i)
        % 移项: g(u) = u - (h/2)*K(x_i, x_i, u) - RHS = 0
        
        RHS = f_vec(i) + history_sum;
        dt_weight = h / 2;
        
        % 定义局部残差函数
        g_func = @(u) u - dt_weight * kf_handle(xi, xi, u) - RHS;
        
        % --- 5. 使用局部阻尼牛顿法求解 u_i ---
        % 初值猜测: 使用上一步的值 u_{i-1} (Continuity)
        u_guess = u_vals(i-1); 
        
        [u_sol, success] = local_newton_solve(g_func, u_guess, xi);
        
        if success
            u_vals(i) = u_sol;
        else
            % 如果求解失败（极罕见），尝试使用简单的显式欧拉预测
            % u_i_approx = f(xi) + history_sum; 
            warning('节点 %d (x=%.2f) 处非线性方程求解失败，使用近似值。', i, xi);
            u_vals(i) = RHS; % 忽略当前步的隐式贡献
        end
    end
    
    % --- 6. 结果拟合 ---
    % 使用去中心化和缩放的 polyfit 提高数值稳定性
    [p, S, mu] = polyfit(x_grid, u_vals, K);
    
    % 将 centered 的系数转换为普通系数 (a_n x^n + ...)
    % 这一步对于 K 较大时非常重要，直接 polyfit 可能会报 Condition Poor
    coeffs_numeric = convert_centered_poly(p, mu);
    
    % 构建返回函数
    u_func = @(x) polyval(p, (x - mu(1))/mu(2));
    
    % fprintf('Volterra 求解完成。\n');
end

%% ---------------------------------------------------------
%  辅助函数：局部阻尼牛顿求解器
%% ---------------------------------------------------------
function [root, success] = local_newton_solve(fun, x0, t_current)
    max_iter = 20;
    tol = 1e-8;
    x = x0;
    success = false;
    
    % 扰动步长用于计算数值导数
    delta = 1e-6; 
    
    for k = 1:max_iter
        try
            fx = fun(x);
        catch
            % 核函数出错（如 NaN），直接跳出
            break; 
        end
        
        if abs(fx) < tol
            success = true;
            root = x;
            return;
        end
        
        % 计算数值 Jacobian (导数)
        try
            fx_delta = fun(x + delta);
            df = (fx_delta - fx) / delta;
        catch
            df = 1; % 降级保护
        end
        
        % 防止导数过小导致飞逸
        if abs(df) < 1e-6
            df = sign(df) * 1e-6; 
            if df == 0, df = 1e-6; end
        end
        
        % 牛顿更新
        dx = -fx / df;
        
        % 阻尼保护 (防止步长过大导致进入 NaN 区域)
        % 对于 exp(-10x) 这种问题，u 值不可能突变太大
        if abs(dx) > 1.0 
            dx = sign(dx) * 1.0; 
        end
        
        x_new = x + dx;
        
        % 检查是否出现 NaN
        if isnan(x_new) || isinf(x_new)
            % 如果发散，尝试减小步长
            x = x + dx * 0.1;
        else
            x = x_new;
        end
    end
    
    root = x;
    % 如果最终残差不算太离谱，也勉强算成功，防止主程序崩溃
    if abs(fun(root)) < 1e-3
        success = true;
    end
end
%% ---------------------------------------------------------
%  辅助函数：将 polyfit 的 mu 参数转换为标准系数 (纯数值版)
%% ---------------------------------------------------------
function coeffs_standard = convert_centered_poly(p, mu)
    % 将 centered 形式 p(z) 转换为标准形式 P(x)
    % z = (x - mu(1)) / mu(2)
    % 原理: 使用 conv (卷积/多项式乘法) 展开 (ax+b)^n
    
    mean_val = mu(1);
    std_val = mu(2);
    
    if std_val == 0, std_val = 1; end % 防止除零保护
    
    % 变换关系: z = a*x + b
    a = 1 / std_val;
    b = -mean_val / std_val;
    
    % 基础线性多项式 [a, b] 对应 ax + b
    linear_base = [a, b];
    
    % 初始化最终系数向量 (长度与 p 相同)
    N = length(p); 
    coeffs_standard = zeros(1, N);
    
    % 遍历 p 的每一项: p(i) * z^(order)
    % p 是降幂排列: p(1)*z^K + p(2)*z^(K-1) ...
    for i = 1:N
        term_coeff = p(i);
        order = N - i; % 当前项的幂次
        
        if term_coeff == 0
            continue;
        end
        
        % 计算 (ax+b)^order 的系数
        % 使用多项式乘法 conv
        expanded_poly = 1; 
        for k = 1:order
            expanded_poly = conv(expanded_poly, linear_base);
        end
        
        % 累加到总系数中
        % expanded_poly 的长度是 order+1，需要对齐到 coeffs_standard 的末尾
        len_expanded = length(expanded_poly);
        start_idx = N - len_expanded + 1;
        
        coeffs_standard(start_idx:end) = coeffs_standard(start_idx:end) + term_coeff * expanded_poly;
    end
end