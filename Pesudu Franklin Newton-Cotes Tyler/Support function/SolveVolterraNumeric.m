function [coeffs_numeric, u_func] = SolveVolterraNumeric(fx_handle, kf_handle, a, b, K)
    % SolveVolterraNumeric (Robust Version)
    % 
    % 改进点:
    % 1. 增加松弛因子 (alpha) 防止发散。
    % 2. 增加 NaN/Inf 检测，遇到数值爆炸自动回退。
    % 3. 如果迭代完全失败，直接返回 f(x) 的拟合系数作为保底。
    
    % --- 1. 参数设置 ---
    N = 101;                 % 稍微增加点数以应对刚性
    x_grid = linspace(a, b, N);
    h = (b - a) / (N - 1);
    
    MaxIter = 30;           
    Tol = 1e-4;             % 初始化不需要太高的精度
    Alpha = 0.1;            % [关键] 松弛因子 (0 < Alpha <= 1)。越小越稳定，但收敛越慢。
                            % 对于刚性问题，设小一点 (如 0.1 - 0.5) 防止震荡。
    
    % --- 2. 初始猜测 ---
    % 初始假设 u(x) = f(x)
    u_curr = fx_handle(x_grid);
    if any(isnan(u_curr)) || any(isinf(u_curr))
        u_curr = ones(size(x_grid)); % 极端保底
    end
    
    % 保存一份纯 f(x) 的值作为最后防线的保底
    u_safe_fallback = u_curr; 
    
    fprintf('启动 Volterra 预估 (Range=[%.1f, %.1f], Relax=%.1f)...\n', a, b, Alpha);
    
    is_converged = false;
    
    % --- 3. 阻尼 Picard 迭代 ---
    for iter = 1:MaxIter
        u_prev = u_curr;
        u_calc = zeros(1, N);
        
        u_calc(1) = fx_handle(x_grid(1)); % 边界条件
        
        has_error = false;
        
        % 积分计算循环
        for i = 2:N
            xi = x_grid(i);
            idx_range = 1:i;
            
            t_sub = x_grid(idx_range);
            u_sub = u_prev(idx_range);
            
            % 计算核函数 (增加 Try-Catch 防止核函数内部出错)
            try
                K_vals = kf_handle(xi, t_sub, u_sub);
            catch
                has_error = true; break;
            end
            
            % 梯形公式积分
            w = ones(1, i); w(1)=0.5; w(end)=0.5; w = w * h;
            integral_val = sum(w .* K_vals);
            
            u_calc(i) = fx_handle(xi) + integral_val;
        end
        
        if has_error || any(isnan(u_calc)) || any(isinf(u_calc))
            warning('Volterra 迭代中检测到 NaN/Inf，停止迭代，回退到 f(x)。');
            u_curr = u_safe_fallback; % 回退
            break;
        end
        
        % [关键] 阻尼更新: New = (1-alpha)*Old + alpha*Calculated
        u_new = (1 - Alpha) * u_prev + Alpha * u_calc;
        
        % 检查收敛
        diff_val = max(abs(u_new - u_prev));
        u_curr = u_new;
        
        if diff_val < Tol
            is_converged = true;
            break;
        end
    end
    
    if ~is_converged
        fprintf('  注意: 预估求解未完全收敛或触发保护 (Diff=%.1e)，使用当前最佳猜测。\n', diff_val);
    end
    
    % --- 4. 结果拟合与防抖动 ---
    
    % 检查最终结果是否合理
    if max(abs(u_curr)) > 1e6 
        % 如果结果大得离谱（通常意味着发散），直接强制使用 f(x)
        warning('预估结果数值过大，可能已发散。重置为 f(x)。');
        u_curr = u_safe_fallback;
    end

    % [关键] 对于刚性问题，全局多项式拟合可能会在区间末端剧烈震荡
    % 这里我们尽量使用低阶拟合，或者只拟合前段
    
    % 尝试拟合
    try
        p_coeffs_desc = polyfit(x_grid, u_curr, K);
    catch
        % 如果拟合失败，返回常数 (u(0))
        p_coeffs_desc = zeros(1, K+1);
        p_coeffs_desc(end) = u_curr(1);
    end
    
    % 转为升幂排列 [c_0, c_1, ..., c_K]
    coeffs_numeric = fliplr(p_coeffs_desc);
    
    % 再次检查系数是否含有 NaN
    if any(isnan(coeffs_numeric))
        coeffs_numeric = zeros(1, K+1);
        coeffs_numeric(1) = u_safe_fallback(1); % 至少给个常数项
    end
    
    % 构建返回函数
    u_func = @(x) polyval(p_coeffs_desc, x);
    
    % fprintf('Volterra 预估完成。\n');
end