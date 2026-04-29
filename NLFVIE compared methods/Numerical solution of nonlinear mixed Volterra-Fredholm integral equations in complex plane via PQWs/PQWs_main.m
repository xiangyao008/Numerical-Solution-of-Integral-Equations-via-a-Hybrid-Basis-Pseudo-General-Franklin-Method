%% Nonlinear Volterra-Fredholm Solver (Spline Projection Method)
% 复现论文: Example 1
% 核心逻辑: 循环测速 + 双网格全局收敛阶分析 (m vs m/2)
% **保持用户原始核心函数不变**

clc; clear; close all;

%% 1. 问题定义与参数设置
T = 1.0;
alpha = 1.0;
beta  = 1.0;

% --- Example 1 ---
% u(x) = x^2 - 2
u_exact_func = @(x) x.^2 - 2;
% 源项 f(x)
f_func = @(x) -1/30*x.^6 + 1/3*x.^4 - x.^2 + 5/3*x - 5/4;

% --- 算法参数 ---
% m 控制投影空间的维数 (粗网格节点数)
m_fine = 8;        % Fine Grid Nodes
m_coarse = m_fine/2;      % Coarse Grid Nodes
max_iter = 50;
tol = 1e-12;
method_type = 'improved'; % 这里我们测试改进方法

vp = 1024;          % 验证/积分网格点数
x_check = linspace(0, T, vp)'; % 列向量
f_val_check = f_func(x_check); % 预计算源项

num_runs = 10;      % 测速循环次数

%% 2. 循环测速 (Performance Test for Fine Grid m)
fprintf('-------------------------------------------\n');
fprintf('Spline Projection Solver (Fine m=%d, Coarse m=%d)\n', m_fine, m_coarse);
fprintf('测试方法: %s\n', method_type);

time_records = zeros(num_runs, 1);

% [预热] Warm-up
fprintf('正在进行预热运行...\n');
solve_system(m_fine, method_type, x_check, f_val_check, alpha, beta, max_iter, tol);

% [循环] Loop Execution
fprintf('正在进行 %d 次循环测速 (m=%d)...\n', num_runs, m_fine);
for i = 1:num_runs
    t_tick = tic;
    
    % --- 求解细网格 ---
    % 直接调用您的原始函数
    u_fine = solve_system(m_fine, method_type, x_check, f_val_check, alpha, beta, max_iter, tol);
    
    % --- 计算误差向量 ---
    u_true = u_exact_func(x_check);
    err_vec_fine = abs(u_fine - u_true);
    
    time_records(i) = toc(t_tick);
end

% 计算平均时间
avg_time = mean(time_records);
fprintf('网格 m=%d 平均耗时: %.6f 秒\n', m_fine, avg_time);

%% 3. 精度与全局收敛阶分析 (Accuracy & Convergence)

% [计算粗网格] Coarse Grid Calculation
fprintf('正在计算粗网格对照组 (m=%d)...\n', m_coarse);
u_coarse = solve_system(m_coarse, method_type, x_check, f_val_check, alpha, beta, max_iter, tol);
err_vec_coarse = abs(u_coarse - u_true);

% --- 全局误差范数与收敛阶计算 (去除首尾边界点) ---

% 1. 数据切片：去除第1个和最后1个点 (Interior Points Only)
% 避免边界插值可能存在的微小误差干扰整体阶数分析
inner_err_fine   = err_vec_fine(2:end-1);
inner_err_coarse = err_vec_coarse(2:end-1);
x_plot = x_check(2:end-1);

% 2. 计算离散 L-infinity 范数 (内部最大绝对误差)
E_inf_fine   = max(abs(inner_err_fine));
E_inf_coarse = max(abs(inner_err_coarse));

% 3. 计算离散 L2 范数 (使用 RMS 近似)
E_rms_fine   = rms(inner_err_fine);
E_rms_coarse = rms(inner_err_coarse);

% 4. 计算全局收敛阶 (Global Convergence Order)
% Order = log2( ||E_{m/2}|| / ||E_m|| )
global_order_inf = log2(E_inf_coarse / E_inf_fine);
global_order_l2  = log2(E_rms_coarse / E_rms_fine);

%% 4. 结果格式化输出
fprintf('\n-------------------------------------------\n');
fprintf('Spline Projection Method 精度与收敛阶分析报告\n');
fprintf('-------------------------------------------\n');
fprintf('平均运行时间 (m=%d):\t %.6f 秒\n', m_fine, avg_time);
fprintf('验证点数量:\t\t %d (已剔除首尾边界点)\n', length(inner_err_fine));
fprintf('-------------------------------------------\n');
fprintf('Metric\t\t\t Error (m=%d)\t Error (m=%d)\t Order\n', m_coarse, m_fine);
fprintf('L_inf Norm (Max):\t %.4e\t %.4e\t %.4f\n', E_inf_coarse, E_inf_fine, global_order_inf);
fprintf('L_2 Norm (RMS):\t\t %.4e\t %.4e\t %.4f\n', E_rms_coarse, E_rms_fine, global_order_l2);
fprintf('-------------------------------------------\n');

% --- 绘图 ---
figure('Color', 'w', 'Position', [100, 100, 1000, 400], 'Name', 'Spline Projection Analysis');

% 子图 1: 解的对比
subplot(1, 2, 1);
plot(x_check, u_true, 'k-', 'LineWidth', 1.5); hold on;
plot(x_check, u_fine, 'r--', 'LineWidth', 1.5);
legend('Exact', ['Approx (m=', num2str(m_fine), ')'], 'Location', 'Best');
xlabel('x'); ylabel('u(x)');
title('Solution Comparison'); grid on;

% 子图 2: 误差分布
subplot(1, 2, 2);
semilogy(x_plot, inner_err_fine, 'b-', 'LineWidth', 1.2);
xlabel('x'); ylabel('|Error|');
title('Interior Error Distribution (Log Scale)');
grid on; xlim([0, 1]);

%% ========================================================================
%  【核心求解函数】 (完全保留原始代码)
%  =======================================================================
function u_curr = solve_system(m, method, x_fine, f_val, alpha, beta, max_iter, tol)
    T = x_fine(end);
    u_curr = f_val; % 初始猜测
    
    % 粗网格节点 (Projection Nodes)
    x_nodes = linspace(0, T, m + 1)';
    
    for k = 1:max_iter
        u_prev = u_curr;
        
        % 非线性项
        F1 = u_prev;      % Fredholm kernel linearity implies F1(u)=u
        F2 = u_prev.^2;   % Volterra kernel linearity implies F2(u)=u^2
        
        % --- 投影步骤 (Projection) ---
        F1_proj = project_function(x_fine, F1, x_nodes, method);
        F2_proj = project_function(x_fine, F2, x_nodes, method);
        
        % --- 积分步骤 ---
        % Fredholm: int_0^1 (x+t)*F1 dt
        % = x * int(F1) + int(t*F1)
        int_F1 = trapz(x_fine, F1_proj);
        int_tF1 = trapz(x_fine, x_fine .* F1_proj);
        term_fred = alpha * (x_fine * int_F1 + int_tF1);
        
        % Volterra: int_0^x (x-t)*F2 dt
        % = x * int_0^x(F2) - int_0^x(t*F2)
        cum_F2 = cumtrapz(x_fine, F2_proj);
        cum_tF2 = cumtrapz(x_fine, x_fine .* F2_proj);
        term_volt = beta * (x_fine .* cum_F2 - cum_tF2);
        
        % 更新
        u_curr = f_val + term_fred + term_volt;
        
        if max(abs(u_curr - u_prev)) < tol
            break;
        end
    end
end

%% ========================================================================
%  【投影算子模拟函数】 (完全保留原始代码)
%  ========================================================================
function f_proj = project_function(x_fine, f_fine, x_nodes, method)
    % 1. 在粗节点采样
    f_nodes = interp1(x_fine, f_fine, x_nodes, 'spline');
    
    if strcmp(method, 'periodic')
        % --- 模拟论文的周期性投影 ---
        % 强制边界条件一致 (模拟周期性空间的基函数)
        % 如果 u(0) != u(T)，这里会人为制造跳跃，引发 Gibbs 现象
        avg_val = (f_nodes(1) + f_nodes(end)) / 2;
        f_nodes(1) = avg_val;
        f_nodes(end) = avg_val;
        
        % 使用 spline 插值，但在节点数据已经被"污染"（强制周期化）的情况下
        % 这模拟了将非周期函数投影到周期基函数上的效果
        
        % 如果有 Curve Fitting Toolbox，推荐用 csape(x,y,'periodic')
        if exist('csape', 'file')
            pp = csape(x_nodes, f_nodes, 'periodic');
            f_proj = ppval(pp, x_fine);
        else
            % Fallback: 使用标准样条，但节点已被强制修改
            f_proj = interp1(x_nodes, f_nodes, x_fine, 'spline');
        end
        
    else
        % --- 改进方法 ---
        % 使用标准样条插值 (Not-a-knot 或 Natural)
        % 能够很好地逼近非周期边界
        f_proj = interp1(x_nodes, f_nodes, x_fine, 'spline');
    end
end