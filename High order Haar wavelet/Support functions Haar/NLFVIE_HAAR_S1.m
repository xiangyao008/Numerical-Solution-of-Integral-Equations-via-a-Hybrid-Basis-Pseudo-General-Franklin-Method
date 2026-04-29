function [ua_iter,tk_xj,cg_it] = NLFVIE_HAAR_S1(fx,dvkut,dfkut, n, iter,tol)
% nonlinear Fredholm integral equation
% fx 为给定函数, 接受x参数 定义为函数句柄
% kuf 为已知积分核, 接受x,t,u三个参数 定义为函数句柄,包含被解函数的函数
% dkut表示kuf对被解函数u(t)的偏导
% n 为阶数，iter,tol表示迭代此时和容差
% ux_iter 为 每次迭代后的值
% cg_it表示收敛时的阶次
%% 预分配小波参数
haar_wavelet_integral= @(x,alpha,beta,gamma,s)0.*((0<=x) < alpha)+((x - alpha).^s./ factorial(s)).*(alpha<=x&x< beta)+...
(((x - alpha).^s - 2*(x - beta).^s) ./ factorial(s)).* (beta<=x& x< gamma)+ ...
(((x - alpha).^s - 2*(x - beta).^s + (x - gamma).^s) ./ factorial(s)).*(gamma<=x&x <= 1);
j=ceil(log(n)/log(2));
s=1;
[alpha,beta,gamma]=generate_haar(j);% 高阶haar小波基tk_xj=CP(n);
%% 预计算参数
tk_xj=CP(n);% 均布小波配点
tk_xj_new=[0,tk_xj]; % x=0的边界条件
n=n+1;
P_ns=zeros(n-1,n);
Init_ux = zeros(n, 1);
Init_ux(1,1)=-1;
Init_ux(n,1)=1;

% Init_ux(2,1)=0.5;
ua_iter=zeros(n,iter+1);
ua_iter(:,1)=Init_ux;
for i=1:n-1
    P_ns(i,:)=haar_wavelet_integral(tk_xj_new,alpha(i),beta(i),gamma(i),s); 
end
P_ns=P_ns';% P_ns的每一行表示n个配置点的i阶s次高阶HAAR小波的数值
temp=ones(n,1);
P_ns_new=[temp,P_ns];
u1=(P_ns_new*ua_iter(:,1));
BdJx=cell(1,n);
dvkut_xt=zeros(n,n-1);
dfkut_xt=zeros(n,n-1);
for i=1:length(u1)
    for j=1:(length(u1)-1)
        dvkut_xt(i,j)=dvkut(tk_xj_new(i),tk_xj(j),u1(j+1));
        dfkut_xt(i,j)=dfkut(tk_xj_new(i),tk_xj(j),u1(j+1));
    end
end
dvkut_xt(1:length(tk_xj),1:length(tk_xj))=dvkut_xt(1:length(tk_xj),1:length(tk_xj)).*tril(ones(length(tk_xj)),-1);
BdJx{1}=pinv(P_ns_new-dvkut_xt*(P_ns_new(2:n,:).*(tk_xj-tk_xj_new(1:n-1))')-dfkut_xt*(P_ns_new(2:n,:).*(tk_xj-tk_xj_new(1:n-1))'));
%% 主函数
for i=2:iter+1  
    u=(P_ns_new*ua_iter(:,i-1));% (n+1*1)
    S2_x=u;
    kf=get_kf(tk_xj_new,tk_xj_new,u);%(n*1)
    g_x=S2_x-fx(tk_xj_new)'-kf;%(n*1)
    delta_a=BdJx{i-1}*(-g_x);%(n*1)
    ua_iter(:,i)=ua_iter(:,i-1)+delta_a;%(n*1)
    u_new=(P_ns_new*ua_iter(:,i));%(n*1)
    kf_new=get_kf(tk_xj_new,tk_xj_new,u_new);
    gx_new=u_new-fx(tk_xj_new)'-kf_new;
    delta_gx=gx_new-g_x;%(n+1*1)
%     muk=1+(delta_gx'*BdJx{i-1}*delta_gx)./(delta_a'*delta_gx);
%     BdJx{i}=BdJx{i-1}+(muk*(delta_a*delta_a')-(delta_a*delta_gx')*BdJx{i-1}-BdJx{i-1}*delta_gx*delta_a')/((delta_a')*delta_gx);
    BdJx{i}=BdJx{i-1}+((delta_a-BdJx{i-1}*delta_gx)*(delta_a')*BdJx{i-1})/((delta_a')*BdJx{i-1}*delta_gx);
    if norm(ua_iter(:,i)-ua_iter(:,i-1))<tol
        disp(['Converged in ', num2str(i), ' iterations.']);
        cg_it=i;
        break;
    end
end
end
%% 问题1 Fredholm and Volterra NLVIE 
function kf=get_kf(x,t,u)
temp_Volt=tril(ones(length(x)-1),-1);
temp_Volt2=[temp_Volt;ones(1,length(x)-1)];
temp_Fred=[ones(length(x)-1);ones(1,length(x)-1)];
ktv=-t(1:length(x)-1)'.*(u(1:length(x)-1).^2).*(t(2:length(x))-t(1:length(x)-1))';
ktv1=-t(2:length(x))'.*(u(2:length(x)).^2).*(t(2:length(x))-t(1:length(x)-1))';
    kf_V1=(temp_Volt2*ktv+temp_Volt2*ktv1).*1/2;
ktv2=(u(1:length(x)-1).^2).*(t(2:length(x))-t(1:length(x)-1))';
ktv3=(u(2:length(x)).^2).*(t(2:length(x))-t(1:length(x)-1))';
    kf_V2=(temp_Volt2*ktv2+temp_Volt2*ktv3).*(x)'.*1/2;
    kf_V=kf_V1+kf_V2;
ktf=t(1:length(x)-1)'.*u(1:length(x)-1).*(t(2:length(x))-t(1:length(x)-1))';
ktf1=t(2:length(x))'.*u(2:length(x)).*(t(2:length(x))-t(1:length(x)-1))';
    kf_F1=(temp_Fred*ktf+temp_Fred*ktf1).*1/2;
ktf2=u(1:length(x)-1).*(t(2:length(x))-t(1:length(x)-1))';
ktf3=u(2:length(x)).*(t(2:length(x))-t(1:length(x)-1))';
    kf_F2=(temp_Fred*ktf2+temp_Fred*ktf3).*(x)'.*1/2;
    kf_F=kf_F1+kf_F2;
kf=kf_V+kf_F;
end