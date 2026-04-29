function [ua_iter,tk_xj] = LVIE_HAAR_S1(fx, kf, n)
% fx 为给定函数，接受x参数 定义为函数句柄
% kf 为已知积分核，接受x,t两个参数 定义为函数句柄
% n 为阶数
% OUTPUT_c1 为 常数，对应论文中的C1
% Array_fx 为 1*n，表示ai系数
%% 预分配小波参数
haar_wavelet_integral= @(x,alpha,beta,gamma,s)0.*((0<=x) < alpha)+((x - alpha).^s./ factorial(s)).*(alpha<=x&x< beta)+...
(((x - alpha).^s - 2*(x - beta).^s) ./ factorial(s)).* (beta<=x& x< gamma)+ ...
(((x - alpha).^s - 2*(x - beta).^s + (x - gamma).^s) ./ factorial(s)).*(gamma<=x&x <= 1);
j=ceil(log(n)/log(2));
s=1;
[alpha,beta,gamma]=generate_haar(j); 
%% 预分配参数
tk_xj=CP(n);
n=n+1;
tk_xj_new=[0,tk_xj];
P_ns=zeros(n-1,n);
kf_xt=zeros(n,n-1);
for i=1:n-1
    P_ns(i,:)=haar_wavelet_integral(tk_xj_new,alpha(i),beta(i),gamma(i),s); 
end
for i=1:n
    for j=1:(n-1)
        kf_xt(i,j)=kf(tk_xj_new(i),tk_xj(j));
    end
end
P_ns=P_ns';% P_ns的每一行表示n个配置点的i阶s次高阶HAAR小波的数值
fx_x=fx(tk_xj_new)';
temp=ones(n,1);
P_ns_new=[temp,P_ns];
kf_xt(1:length(tk_xj),1:length(tk_xj))=kf_xt(1:length(tk_xj),1:length(tk_xj)).*tril(ones(length(tk_xj)),-1);
%% 主函数
ua_iter=pinv(P_ns_new-kf_xt*P_ns_new(2:n,:)./length(tk_xj))*fx_x;
end

