function [ua_iter,tk_xj,cg_it] = NLFVIE_HAAR_S2_Simpson(fx, dvkut,dfkut, n, iter,tol)
% nonlinear Fredholm integral equation
% fx 为给定函数, 接受x参数 定义为函数句柄
% kuf 为已知积分核, 接受x,t,u三个参数 定义为函数句柄,包含被解函数的函数
% dkut表示kuf对被解函数u(t)的偏导
% n 为阶数，iter,tol表示迭代此时和容差
% ux_iter 为 每次迭代后的值
% cg_it表示收敛时的阶次
% load('true_coff.mat','Array_fx1');
%% 预分配小波参数
haar_wavelet_integral= @(x,alpha,beta,gamma,s)0.*((0<=x) < alpha)+((x - alpha).^s./ factorial(s)).*(alpha<=x&x< beta)+...
(((x - alpha).^s - 2*(x - beta).^s) ./ factorial(s)).* (beta<=x& x< gamma)+ ...
(((x - alpha).^s - 2*(x - beta).^s + (x - gamma).^s) ./ factorial(s)).*(gamma<=x&x <= 1);
j=ceil(log(n)/log(2));
s=2;
[alpha,beta,gamma]=generate_haar(j);% 高阶haar小波基
%% 预计算参数
tk_xj=CP(n);% 均布小波配点
n=n+2;
tk_xj_new=[0,tk_xj,1]; % x=0的边界条件
tk_xj_new2=(tk_xj_new(1:n-1)+tk_xj_new(2:n))./2;
P_ns=zeros(n-2,n);
P_ns2=zeros(n-2,n-1);
Init_ux = zeros(n, 1);
Init_ux(n,1)=1;
Init_ux(1,1)=-1;
ua_iter=zeros(n,iter+1);
ua_iter(:,1)=Init_ux;
for i=1:n-2
    P_ns(i,:)=haar_wavelet_integral(tk_xj_new,alpha(i),beta(i),gamma(i),s); 
end
P_ns=P_ns';% P_ns的每一行表示n个配置点的i阶s次高阶HAAR小波的数值
for i=1:n-2
    P_ns2(i,:)=haar_wavelet_integral(tk_xj_new2,alpha(i),beta(i),gamma(i),s); 
end
P_ns2=P_ns2';
temp=ones(n,1);
P_ns_new=[temp,P_ns,tk_xj_new'];%将C1当作常数项1和a0系数的乘积，矩阵扩展为n+1*n+1
P_ns_new2=[temp(1:n-1),P_ns2,tk_xj_new2'];
u1=(P_ns_new*ua_iter(:,1));

BdJx=cell(1,n);
% 原来方法
% kf1=get_dkf(tk_xj_new,tk_xj,u1(2:n-1));
% temp=repmat(kf1, 1, n);
% BdJx{1}=pinv(P_ns_new-temp.*P_ns_new);
% 可能正确方法
dvkut_xt=zeros(n,n-2);
dfkut_xt=zeros(n,n-2);
for i=1:length(u1)
    for j=1:(length(u1)-2)
        dvkut_xt(i,j)=dvkut(tk_xj_new(i),tk_xj(j),u1(j+1));
        dfkut_xt(i,j)=dfkut(tk_xj_new(i),tk_xj(j),u1(j+1));
    end
end
dvkut_xt(1:length(tk_xj),1:length(tk_xj))=dvkut_xt(1:length(tk_xj),1:length(tk_xj)).*tril(ones(length(tk_xj)),-1);
BdJx{1}=pinv(P_ns_new-dvkut_xt*(P_ns_new(2:n-1,:).*(tk_xj-tk_xj_new(1:n-2))')-dfkut_xt*(P_ns_new(2:n-1,:).*(tk_xj-tk_xj_new(1:n-2))'));
%% 主函数
for i=2:iter+1  
    u=(P_ns_new*ua_iter(:,i-1));% (n+1*1)
    u2=(P_ns_new2*ua_iter(:,i-1));
    S2_x=u;
    kf=get_kf(tk_xj_new,tk_xj_new,tk_xj_new2,u,u2);%(n*1)
    g_x=S2_x-fx(tk_xj_new)'-kf;%(n*1)
    delta_a=-BdJx{i-1}*(g_x);%(n*1)
    ua_iter(:,i)=ua_iter(:,i-1)+delta_a;%(n*1)
    if norm(ua_iter(:,i)-ua_iter(:,i-1))<tol
        disp(['Converged in ', num2str(i), ' iterations.']);
        cg_it=i;
        break;
    end
    u_new=(P_ns_new*ua_iter(:,i));%(n*1)
    u2_new=(P_ns_new2*ua_iter(:,i));
    kf_new=get_kf(tk_xj_new,tk_xj_new,tk_xj_new2,u_new,u2_new);    
    gx_new=u_new-fx(tk_xj_new)'-kf_new;
    delta_gx=gx_new-g_x;%(n+1*1)
    % muk=1+(delta_gx'*BdJx{i-1}*delta_gx)./(delta_a'*delta_gx);
    % BdJx{i}=BdJx{i-1}+(muk*(delta_a*delta_a')-(delta_a*delta_gx')*BdJx{i-1}-BdJx{i-1}*delta_gx*delta_a')...
    % /((delta_a')*delta_gx);% 秩2方法
    BdJx{i}=BdJx{i-1}+((delta_a-BdJx{i-1}*delta_gx)*(delta_a')*BdJx{i-1})/((delta_a')*BdJx{i-1}*delta_gx);% Sherman–Morrison formula
    % BdJx{i}=BdJx{i-1}+(delta_a-BdJx{i-1}*delta_gx)*(delta_gx')/(norm(delta_gx)^2);
    % BdJx{i}=BdJx{i-1}+((delta_a-BdJx{i-1}*delta_gx)*(delta_a-BdJx{i-1}*delta_gx)')/((delta_a-BdJx{i-1}*delta_gx)'*delta_gx);
    % BdJx{i}=BdJx{i-1}+((delta_gx-BdJx{i-1}*delta_a)*(delta_a'))./((delta_a')*delta_a);%不求逆
end
end
%% 问题1
function kf=get_kf(x,t,t2,u,u2)
temp_Volt=tril(ones(length(x)-1),-1);
temp_Volt2=[temp_Volt;ones(1,length(x)-1)];
temp_Fred=[ones(length(x)-1);ones(1,length(x)-1)];
ktv=-t(1:length(x)-1)'.*(u(1:length(x)-1).^2).*(t(2:length(x))-t(1:length(x)-1))';
ktv_m=-t2'.*(u2.^2).*(t(2:length(x))-t(1:length(x)-1))';
ktv1=-t(2:length(x))'.*(u(2:length(x)).^2).*(t(2:length(x))-t(1:length(x)-1))';
    kf_V1=(temp_Volt2*ktv+4*temp_Volt2*ktv_m+temp_Volt2*ktv1).*1/6;
ktv2=(u(1:length(x)-1).^2).*(t(2:length(x))-t(1:length(x)-1))';
ktv2_m=(u2.^2).*(t(2:length(x))-t(1:length(x)-1))';
ktv3=(u(2:length(x)).^2).*(t(2:length(x))-t(1:length(x)-1))';
    kf_V2=(temp_Volt2*ktv2+4*temp_Volt2*ktv2_m+temp_Volt2*ktv3).*(x)'.*1/6;
    kf_V=kf_V1+kf_V2;
ktf=t(1:length(x)-1)'.*u(1:length(x)-1).*(t(2:length(x))-t(1:length(x)-1))';
ktf_m=t2'.*u2.*(t(2:length(x))-t(1:length(x)-1))';
ktf1=t(2:length(x))'.*u(2:length(x)).*(t(2:length(x))-t(1:length(x)-1))';
    kf_F1=(temp_Fred*ktf+4*temp_Fred*ktf_m+temp_Fred*ktf1).*1/6;
ktf2=u(1:length(x)-1).*(t(2:length(x))-t(1:length(x)-1))';
ktf2_m=u2.*(t(2:length(x))-t(1:length(x)-1))';
ktf3=u(2:length(x)).*(t(2:length(x))-t(1:length(x)-1))';
    kf_F2=(temp_Fred*ktf2+4*temp_Fred*ktf2_m+temp_Fred*ktf3).*(x)'.*1/6;
    kf_F=kf_F1+kf_F2;
kf=kf_V+kf_F;
end
