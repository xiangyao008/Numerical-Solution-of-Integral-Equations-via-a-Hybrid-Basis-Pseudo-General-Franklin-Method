    % 主程序：用于设置问题参数并调用求解器
    % 遵循科研代码规范：参数定义与核心逻辑分离
    clc; clear; close all;
    
    %% 1. 问题定义 (Problem Definition)
    % Example 5.1: int_0^x e^{x+t} u(t) dt = x e^x
    % Exact Solution: u(x) = exp(-x)
    
    % 核函数 k(x,t)
    kf = @(x, t) exp(x + t);
    
    % 右端项 f(x)
    fx = @(x) x .* exp(x);
    
    % 真解 u(x) (Reference Function)
    RF = @(x) exp(-x);
    
    % 伯努利多项式阶数 (Degree of Bernoulli Polynomials)
    N = 13; 
    
    %% 2. 调用通用求解器 (Solver Execution)
    fprintf('正在求解线性Volterra积分方程 (第一类)...\n');
    fprintf('阶数 N = %d\n', N);
    
    [u_approx_func, run_time] = solve_volterra_first_kind(kf, fx, N);
    
    %% 3. 结果验证 (Verification)
    % 使用高密度验证点进行误差分析 (Verify on 1024 points)
    x_test = linspace(0, 1, 1024);
    
    u_exact_val = RF(x_test);
    u_num_val   = u_approx_func(x_test);
    
    abs_error = abs(u_exact_val - u_num_val);
    max_error = max(abs_error);
    l2_error  = norm(abs_error) / sqrt(length(x_test));
    
    fprintf('求解完成。耗时: %.4f 秒\n', run_time);
    fprintf('最大绝对误差 (L_inf): %.2e\n', max_error);
    fprintf('均方根误差 (L_2):     %.2e\n', l2_error);
    
    %% 4. 绘图 (Visualization)
    figure('Color', 'w', 'Name', 'Bernoulli Solver Verification');
    
    subplot(2, 1, 1);
    plot(x_test, u_exact_val, 'k-', 'LineWidth', 1.5); hold on;
    plot(x_test, u_num_val, 'r--', 'LineWidth', 1.5);
    legend('Exact Solution (RF)', 'Numerical Solution (Bernoulli)', 'Location', 'best');
    title(['Solution Comparison (N=', num2str(N), ')']);
    xlabel('x'); ylabel('u(x)');
    grid on;
    
    subplot(2, 1, 2);
    semilogy(x_test, abs_error, 'b-', 'LineWidth', 1.2);
    title('Absolute Error Distribution');
    xlabel('x'); ylabel('|u_{exact} - u_{num}|');
    grid on;
    xlim([0, 1]);

%% ========================================================================
%  通用求解器函数 (General Solver Function)
%  ========================================================================
function [u_approx_handle, t_cost] = solve_volterra_first_kind(kf, fx, N)
    % SOLVE_VOLTERRA_FIRST_KIND 求解第一类线性Volterra积分方程
    %
    % 输入:
    %   kf - 核函数句柄 @(x,t)
    %   fx - 右端项句柄 @(x)
    %   N  - 伯努利基底阶数
    % 输出:
    %   u_approx_handle - 近似解的函数句柄 @(x)
    %   t_cost          - 计算耗时
    
    tic;
    
    % 1. 配置点生成 (Collocation Points)
    % 使用切比雪夫节点或等距节点，这里沿用论文的Newton-Cotes节点定义
    % x_i = (2i + 1)/(2(N+1)), i = 0...N
    indices = 0:N;
    x_nodes = (2 * indices + 1) ./ (2 * (N + 1));
    x_nodes = x_nodes(:); % 列向量
    
    % 2. 构建系统矩阵 A 和右端向量 F
    % 方程离散形式: sum_{j=0}^N c_j * [ int_0^{x_i} k(x_i, t) B_j(t) dt ] = f(x_i)
    % A_ij = int_0^{x_i} k(x_i, t) B_j(t) dt
    
    n_nodes = length(x_nodes);
    A = zeros(n_nodes, N + 1);
    F = fx(x_nodes);
    
    % 获取伯努利数的预计算向量，加速多项式求值
    B_nums_vec = get_bernoulli_nums_vector(N);
    
    % 组装矩阵 (Assembly)
    % 注意：为了保证通用性，这里对矩阵元素采用了数值积分 (integral)
    % 对于特定形式的核函数，此处可以用Operational Matrix P优化，但通用性会下降。
    for i = 1:n_nodes
        xi = x_nodes(i);
        
        % 如果 xi 接近 0，积分区间为 0，这行方程可能奇异或为 0=0。
        % 第一类方程在 x=0 处通常 f(0)=0，无法提供系数信息。
        % 但由于我们选取的节点 x_nodes > 0，避免了这个问题。
        
        for j = 0:N
            % 定义被积函数 integrand = k(xi, t) * B_j(t)
            % 注意：积分变量是 t
            integrand = @(t) kf(xi, t) .* bernoulli_poly_fast(j, t, B_nums_vec);
            
            % 计算积分值 A_ij
            % 使用 'ArrayValued', true 以支持向量化核函数
            A(i, j+1) = integral(integrand, 0, xi, 'AbsTol', 1e-12, 'RelTol', 1e-12);
        end
    end
    
    % 3. 求解线性方程组 A * C = F
    % 第一类Volterra方程离散后往往是病态的，使用伪逆或正则化可能更稳健，
    % 但在 N 较小且核函数光滑时，直接求解通常可行。
    C = A \ F;
    
    % 4. 构造解的函数句柄
    u_approx_handle = @(x) evaluate_bernoulli_basis(C, x, B_nums_vec);
    
    t_cost = toc;
end

%% ========================================================================
%  辅助函数 (Helper Functions)
%  ========================================================================

function val = evaluate_bernoulli_basis(C, x, B_nums_vec)
    % 计算伯努利级数的值: u(x) = sum(C_j * B_j(x))
    val = zeros(size(x));
    N = length(C) - 1;
    for j = 0:N
        val = val + C(j+1) * bernoulli_poly_fast(j, x, B_nums_vec);
    end
end

function val = bernoulli_poly_fast(n, x, B_all)
    % 快速计算第 n 阶伯努利多项式 B_n(x)
    % B_all 包含预先计算好的伯努利数 [B_0, B_1, ...]
    
    val = zeros(size(x));
    % B_n(x) = sum_{k=0}^n (n choose k) * B_{n-k} * x^k
    for k = 0:n
        % 获取 B_{n-k}，对应索引 (n-k) + 1
        bk = B_all( (n-k) + 1 );
        coef = nchoosek(n, k) * bk;
        val = val + coef * (x.^k);
    end
end

function B_vec = get_bernoulli_nums_vector(max_order)
    % 生成伯努利数向量 B_0 到 B_max_order
    % 避免使用符号工具箱，提高兼容性
    B_vec = zeros(1, max_order + 1);
    B_vec(1) = 1; % B_0
    
    if max_order >= 1
        B_vec(2) = -0.5; % B_1
    end
    
    for n = 2:max_order
        % 递推公式: sum_{k=0}^n (n+1 choose k) * B_k = 0
        s = 0;
        for k = 0:(n-1)
            s = s + nchoosek(n+1, k) * B_vec(k+1);
        end
        B_vec(n+1) = -s / (n+1);
    end
end