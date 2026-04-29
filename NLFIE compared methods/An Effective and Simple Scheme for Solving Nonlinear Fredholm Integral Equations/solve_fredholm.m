function u_approx_handle = solve_fredholm(fx, kf, a, b, N)
% SOLVE_FREDHOLM 求解非线性 Fredholm 积分方程
% 形式: u(x) = fx(x) + int( kf(x,t,u(t)) dt )
%
% 输入:
%   fx: 源项函数句柄 @(x)
%   kf: 核函数句柄 @(x,t,u) (包含非线性项和系数 lambda)
%   a, b: 积分区间 [a, b]
%   N: Gauss-Legendre 节点数量 (即论文中的 k+1)
%
% 输出:
%   u_approx_handle: 近似解的函数句柄，调用方式 y = u_approx_handle(x_query)

    % 1. 获取标准区间 [-1, 1] 上的 Gauss-Legendre 节点和权重
    [x_std, w_std] = lgwt(N, -1, 1);
    
     % 2. 变换到物理区间 [a, b] [cite: 63-65]
    % t = ((b-a)/2)*x_std + (b+a)/2
    map_t = @(tau) ((b-a)/2)*tau + (b+a)/2;
    nodes = map_t(x_std); 
    weights = w_std * (b-a)/2; 
    
    % 3. 构建非线性方程组求解节点上的 u 值
    % 未知量 u_nodes 是一个 Nx1 的向量，代表 u(nodes)
    
    % 初始猜测: 假设积分项为0，则 u approx f(x)
    u0 = fx(nodes);
    
    % fsolve 选项
    options = optimoptions('fsolve', 'Display', 'off', ...
        'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12, ...
        'Algorithm', 'trust-region-dogleg');
    
    % 定义方程组残差函数 F(u) = 0
    sys_fun = @(u) system_residual(u, nodes, weights, fx, kf);
    
    [u_sol, ~, exitflag] = fsolve(sys_fun, u0, options);
    
    if exitflag <= 0
        warning('SOLVER:NoConvergence', 'fsolve did not converge perfectly.');
    end
    
    % 4. 构造返回的插值/重构函数
    % 对于任意点 x, u(x) = fx(x) + sum( weights_i * kf(x, t_i, u_i) )
    u_approx_handle = @(x_eval) construct_solution(x_eval, u_sol, nodes, weights, fx, kf);

end

function F = system_residual(u, nodes, weights, fx, kf)
    % 构建代数方程组: u_p - fx(x_p) - sum( w_i * kf(x_p, x_i, u_i) ) = 0
    N = length(u);
    F = zeros(N, 1);
    
    % 预计算 fx(nodes)
    f_val = fx(nodes);
    
    for p = 1:N
        x_p = nodes(p);
        
        % 计算求积 sum_{i} w_i * kf(x_p, t_i, u_i)
        integral_val = 0;
        for i = 1:N
            % kf(x, t, u)
            val = kf(x_p, nodes(i), u(i)); 
            integral_val = integral_val + weights(i) * val;
        end
        
        F(p) = u(p) - f_val(p) - integral_val;
    end
end

function u_val = construct_solution(x_eval, u_sol, nodes, weights, fx, kf)
    % Nyström 插值公式重构解
    % 确保输入 x_eval 即使是向量也能正确计算
    u_val = zeros(size(x_eval));
    N = length(nodes);
    
    for k = 1:numel(x_eval)
        x = x_eval(k);
        
        integral_term = 0;
        for i = 1:N
            val = kf(x, nodes(i), u_sol(i));
            integral_term = integral_term + weights(i) * val;
        end
        
        u_val(k) = fx(x) + integral_term;
    end
end