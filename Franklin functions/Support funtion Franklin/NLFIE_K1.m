function [ua_iter,tk_xj,cg_it] = NLFIE_K1(fx, n, iter,tol)
% nonlinear Fredholm integral equation
% fx 为给定函数, 接受x参数 定义为函数句柄
% kuf 为已知积分核, 接受x,t,u三个参数 定义为函数句柄,包含被解函数的函数
% dkut表示kuf对被解函数u(t)的偏导
% n 为阶数，iter,tol表示迭代此时和容差
% ux_iter 为 每次迭代后的值
% cg_it表示收敛时的阶次
%% 预分配参数
load('M_GKSF1_2_128.mat','NEW_FRANKLIN_function');
tk_xj=CP(n);
n=n+1;
tk_xj_new=[0,tk_xj];
BdJx=cell(1,n);
% BdJx{1}=eye(n);
Init_ux = zeros(n, 1);
ua_iter=zeros(n,iter+1);
ua_iter(:,1)=Init_ux;
hi_x=zeros(n,n-1);
for i=1:n
    for j=1:(n-1)
        hi_x(i,j)=NEW_FRANKLIN_function{j+1}(tk_xj_new(i));
    end
end
temp=ones(n,1);
hi_x_new=[temp,hi_x];
u1=(hi_x_new*ua_iter(:,1));
kf1=get_dkf(tk_xj_new,tk_xj,u1(2:n));
temp=repmat(kf1, 1, n);
BdJx{1}=pinv(hi_x_new-temp.*hi_x_new);
%% 主函数
for i=2:iter+1  
    u=(hi_x_new*ua_iter(:,i-1));% (n+1*1)
    S2_x=u;
    kf=get_kf(tk_xj_new,tk_xj,u(2:n));%(n*1)
    g_x=S2_x-fx(tk_xj_new)'-kf;%(n*1)
    delta_a=BdJx{i-1}*(-g_x);%(n*1)
    ua_iter(:,i)=ua_iter(:,i-1)+delta_a;%(n*1)
    u_new=(hi_x_new*ua_iter(:,i));%(n*1)
    kf_new=get_kf(tk_xj_new,tk_xj,u_new(2:n));
    gx_new=u_new-fx(tk_xj_new)'-kf_new;
    delta_gx=gx_new-g_x;%(n+1*1)
    BdJx{i}=BdJx{i-1}+((delta_a-BdJx{i-1}*delta_gx)*(delta_a')*BdJx{i-1})/((delta_a')*BdJx{i-1}*delta_gx);
    if norm(ua_iter(:,i)-ua_iter(:,i-1))<tol
        disp(['Converged in ', num2str(i), ' iterations.']);
        cg_it=i;
        break;
    end
end
end

function kf=get_kf(x,t,u)
kf=1/5.*cos(pi.*x').*sum(sin(pi.*t').* (u.^3))./(length(x)-1);
% t_new=t.*u;
%     [X, T] = ndgrid(x, t_new);
%     Kernel = (1/5 .* cos(pi*X) .* sin(pi*T) .* (T.^3))./length(x);
%     kf=sum(Kernel, 2);
end
function kf=get_dkf(x,t,u)
kf=3/5.*cos(pi.*x').*sum(sin(pi.*t').* (u.^2))./(length(x)-1);
% t_new=t.*u;
%     [X, T] = ndgrid(x, t_new);
%     Kernel = (1/5 .* cos(pi*X) .* sin(pi*T) .* (T.^3))./length(x);
%     kf=sum(Kernel, 2);
end
