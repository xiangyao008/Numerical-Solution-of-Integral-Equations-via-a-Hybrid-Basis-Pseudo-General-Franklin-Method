function [coeffs_numeric, u_series] = Solve_mix_TaylorNumeric(fx_handle, kv_handle, kf_handle, K)
    % Solve_mix_Taylor 数值迭代版
    %
    % 算法原理: 皮卡迭代法 (Picard Iteration)
    % 1. 将 Fredholm 项视为已知源项的一部分进行解耦。
    % 2. 使用步进法求解剩余的 Volterra 部分。
    % 3. 循环迭代直至收敛。
    
    % --- 1. 参数设置 ---
    N = 51;                 % 固定网格点数 (步长 0.02)
    x_grid = linspace(0, 1, N);
    h = x_grid(2) - x_grid(1);
    MaxIter = 10;           % 固定迭代次数 (通常 5-10 次即可收敛)
    
    % --- 2. 初始化 ---
    % 初始猜测：假设 u(x) 仅由 f(x) 构成
    u_curr = arrayfun(fx_handle, x_grid);
    
    fprintf('启动混合方程数值求解 (Picard Iteration, N=%d)...\n', N);
    
    % 预计算梯形公式的权重向量 (用于 Fredholm 积分 [0,1])
    % weights = [0.5, 1, 1, ..., 1, 0.5]
    w_fred = ones(1, N);
    w_fred(1) = 0.5; 
    w_fred(end) = 0.5;
    
    % --- 3. 迭代求解主循环 ---
    for iter = 1:MaxIter
        u_prev = u_curr;
        
        % Step A: 计算 Fredholm 积分项 (基于上一轮的 u_prev)
        % I_fred(x) = int_0^1 Kf(x, t, u_prev(t)) dt
        I_fred_vals = zeros(1, N);
        
        for i = 1:N
            xi = x_grid(i);
            % 计算被积函数向量: Kf(xi, all_t, all_u)
            kf_vals = kf_handle(xi, x_grid, u_prev);
            % 全区间积分 (梯形公式)
            I_fred_vals(i) = sum(w_fred .* kf_vals) * h;
        end
        
        % Step B: 求解 Volterra 方程 (基于当前的 Fredholm 项)
        % 等效方程: u(x) = [f(x) + I_fred(x)] + int_0^x Kv(x,t,u) dt
        % 令 effective_source(x) = f(x) + I_fred(x)
        
        u_new = zeros(1, N);
        % 起始点: Volterra 积分在 0 处为 0
        u_new(1) = fx_handle(0) + I_fred_vals(1);
        
        for i = 2:N
            xi = x_grid(i);
            
            % --- Volterra 历史积分 (0 到 x_{i-1}) ---
            t_hist = x_grid(1:i-1);
            u_hist = u_new(1:i-1);
            kv_vals = kv_handle(xi, t_hist, u_hist);
            
            % 局部梯形权重
            w_vol = ones(1, i-1);
            w_vol(1) = 0.5; 
            w_vol(end) = 0.5;
            I_vol_hist = sum(w_vol .* kv_vals) * h;
            
            % --- 当前步预估 (Predictor) ---
            % 等效源项值
            source_term = fx_handle(xi) + I_fred_vals(i);
            
            % 简单预估：假设增量与上一步相同
            % u_pred = source_term + I_vol_hist + h * Kv(xi, t_{i-1}, u_{i-1})
            kv_prev = kv_handle(xi, x_grid(i-1), u_new(i-1));
            u_pred_val = source_term + I_vol_hist + h * kv_prev; 
            
            % --- 当前步校正 (Corrector) ---
            % u_i = source + I_vol_hist + 0.5*h*Kv_prev + 0.5*h*Kv_curr
            kv_curr_est = kv_handle(xi, xi, u_pred_val);
            
            % 修正 I_vol_hist 的末项权重 (从 0.5 变为 1.0 的过程隐含在加法中)
            % 正确的梯形增量: 0.5*h * (Kv_prev + Kv_curr)
            step_increment = 0.5 * h * (kv_prev + kv_curr_est);
            
            u_new(i) = source_term + I_vol_hist + step_increment;
        end
        
        % 更新解
        u_curr = u_new;
        
        % (可选) 简单的收敛性检查
        diff_norm = max(abs(u_curr - u_prev));
        if diff_norm < 1e-6
            % fprintf('  -> 收敛于第 %d 次迭代。\n', iter);
            break; 
        end
    end
    
    % --- 4. 多项式拟合与输出 ---
    % 将最终的数值解拟合为 K 阶多项式
    p_coeffs_desc = polyfit(x_grid, u_curr, K);
    coeffs_numeric = fliplr(p_coeffs_desc); % 转为升幂排列
    
    % 生成函数句柄
    u_series = @(x) polyval(p_coeffs_desc, x);
    
    fprintf('计算完成 (Iter=%d, MaxDiff=%.2e)。\n', iter, diff_norm);
end