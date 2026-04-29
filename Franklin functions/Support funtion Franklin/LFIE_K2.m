function [ua_iter,tk_xj] = LFIE_K2(fx, kf, n)
% nonlinear Fredholm integral equation
% fx 为给定函数, 接受x参数 定义为函数句柄
% kuf 为已知积分核, 接受x,t,u三个参数 定义为函数句柄,包含被解函数的函数
% dkut表示kuf对被解函数u(t)的偏导
% n 为阶数，iter,tol表示迭代此时和容差
% ux_iter 为 每次迭代后的值
% cg_it表示收敛时的阶次
%% 预分配参数
load('M_GKSF2_2_128.mat','NEW_FRANKLIN_function');
% load('M_GKSF2_2_128.mat','NEW_FRANKLIN_function');
tk_xj=CP(n);
n=n+2;
tk_xj_new=[0,tk_xj,1];
hi_x=zeros(n,n-2);
kf_xt=zeros(n,n-2);
for i=1:n
    for j=1:(n-2)
        hi_x(i,j)=NEW_FRANKLIN_function{j+2}(tk_xj_new(i));
        kf_xt(i,j)=kf(tk_xj_new(i),tk_xj(j));
    end
end
fx_x=fx(tk_xj_new)';
temp=ones(n,1);
hi_x_new=[temp,hi_x,tk_xj_new'];
%% 主函数
ua_iter=pinv(hi_x_new-kf_xt*hi_x_new(2:n-1,:)./length(tk_xj))*fx_x;
% AA=kf_xt*hi_x_new(2:n-1,:)./length(tk_xj);
% ua_iter=pinv(hi_x_new-(kf_xt(:,2:n)*hi_x_new(2:n,:)+kf_xt(:,1:n-1)*hi_x_new(1:n-1,:))./2./length(tk_xj))*fx_x;

end
