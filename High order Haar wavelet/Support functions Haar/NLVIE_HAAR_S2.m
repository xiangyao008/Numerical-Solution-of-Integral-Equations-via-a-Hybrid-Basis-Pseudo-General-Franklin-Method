function [ua_iter,tk_xj,cg_it] = NLVIE_HAAR_S2(fx, dkut, n, iter,tol)
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
tk_xj_new=[0,tk_xj,1]; % x=0的边界条件
n=n+2;
P_ns=zeros(n-2,n);
Init_ux = zeros(n, 1);
Init_ux(n,1)=0;
Init_ux(1,1)=0;
ua_iter=zeros(n,iter+1);
ua_iter(:,1)=Init_ux;
for i=1:n-2
    P_ns(i,:)=haar_wavelet_integral(tk_xj_new,alpha(i),beta(i),gamma(i),s); 
end
P_ns=P_ns';% P_ns的每一行表示n个配置点的i阶s次高阶HAAR小波的数值
temp=ones(n,1);
P_ns_new=[temp,P_ns,tk_xj_new'];%将C1当作常数项1和a0系数的乘积，矩阵扩展为n+1*n+1
u1=(P_ns_new*ua_iter(:,1));

BdJx=cell(1,n);
% 原来方法
% kf1=get_dkf(tk_xj_new,tk_xj,u1(2:n-1));
% temp=repmat(kf1, 1, n);
% BdJx{1}=pinv(P_ns_new-temp.*P_ns_new);
% 可能正确方法
dkut_xt=zeros(n,n-2);
for i=1:length(u1)
    for j=1:(length(u1)-2)
        dkut_xt(i,j)=dkut(tk_xj_new(i),tk_xj(j),u1(j+1));
    end
end
dkut_xt(1:length(tk_xj),1:length(tk_xj))=dkut_xt(1:length(tk_xj),1:length(tk_xj)).*tril(ones(length(tk_xj)),-1);
BdJx{1}=pinv(P_ns_new-dkut_xt*(P_ns_new(2:n-1,:).*(tk_xj-tk_xj_new(1:n-2))'));
% BdJx{1}=pinv(P_ns_new);
% BdJx{1}=ones(n,n);
% BdJx{1}=P_ns_new-(dkut_xt*P_ns_new(2:n-1,:)./length(tk_xj));
%% 主函数
for i=2:iter+1  
    u=(P_ns_new*ua_iter(:,i-1));% (n+1*1)
    S2_x=u;
    kf=get_kf(tk_xj_new,tk_xj_new,u);%(n*1)
    g_x=S2_x-fx(tk_xj_new)'-kf;%(n*1)
    delta_a=-BdJx{i-1}*(g_x);%(n*1)
    ua_iter(:,i)=ua_iter(:,i-1)+delta_a;%(n*1)
    if norm(ua_iter(:,i)-ua_iter(:,i-1))<tol
        disp(['Converged in ', num2str(i), ' iterations.']);
        cg_it=i;
        break;
    end
    u_new=(P_ns_new*ua_iter(:,i));%(n*1)
    kf_new=get_kf(tk_xj_new,tk_xj_new,u_new);
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
% function kf=get_kf(x,t,u)
% temp=tril(ones(length(x)-1),-1);
% temp2=[temp;ones(1,length(x)-1)];
% kt=(1./(1+u(1:length(x)-1).^2)).*(t(2:length(x))-t(1:length(x)-1))';
% kt1=(1./(1+u(2:length(x)).^2)).*(t(2:length(x))-t(1:length(x)-1))';
% kf=(temp2*kt+temp2*kt1)./2;
% end
%% 问题2
% function kf=get_kf(x,t,u)
% temp=tril(ones(length(x)-1),-1);
% temp2=[temp;ones(1,length(x)-1)];
% kt=(t(1:length(x)-1)'.^2).*(u(1:length(x)-1).^2).*(t(2:length(x))-t(1:length(x)-1))';
% kt1=(t(2:length(x))'.^2).*(u(2:length(x)).^2).*(t(2:length(x))-t(1:length(x)-1))';
% kf=(temp2*kt+temp2*kt1).*x'./2;
% end
%% 问题3
% function kf=get_kf(x,t,u)
% temp=tril(ones(length(x)-1),-1);
% temp2=[temp;ones(1,length(x)-1)];
% kt=(exp(-2.*t(1:length(x)-1)')).*(u(1:length(x)-1).^2).*(t(2:length(x))-t(1:length(x)-1))';
% kt1=(exp(-2.*t(2:length(x))')).*(u(2:length(x)).^2).*(t(2:length(x))-t(1:length(x)-1))';
% kf=(temp2*kt+temp2*kt1).*exp(-x)'./2;
% end
% function kf=get_kf(x,t,u) %计算法2，针对在0处kf无定义的情况
% kf=zeros(length(x),1);
% kf(1)=0;
%     for i=2:length(x)
%         temp=exp(-2.*t(1:length(x)-1)').*(u(1:length(x)-1).^2).*(t(2:length(x))-t(1:length(x)-1))';
%         temp2=exp(-2.*t(2:length(x))').*(u(2:length(x)).^2).*(t(2:length(x))-t(1:length(x)-1))';
%         temp3=(temp2+temp)./2;
%         kf(i)=exp(-x(i)).*sum(temp3(1:i-1));
%     end
% end
%% 问题4 Third kind of NLVIE 
function kf=get_kf(x,t,u)
temp=tril(ones(length(x)-1),-1);
temp2=[temp;ones(1,length(x)-1)];
kt=(u(1:length(x)-1).^2+u(1:length(x)-1)).*(t(2:length(x))-t(1:length(x)-1))';
kt1=(u(2:length(x)).^2+u(2:length(x))).*(t(2:length(x))-t(1:length(x)-1))';
kf=(temp2*kt+temp2*kt1).*(-1)./2;
end
%% 问题5 
% function kf=get_kf(x,t,u)
% temp=tril(ones(length(x)-1),-1);
% temp2=[temp;ones(1,length(x)-1)];
% kt=t(1:length(x)-1)'.*(u(1:length(x)-1).^2).*(t(2:length(x))-t(1:length(x)-1))';
% kt1=t(2:length(x))'.*(u(2:length(x)).^2).*(t(2:length(x))-t(1:length(x)-1))';
% kf=(temp2*kt+temp2*kt1).*(x)'./2;
% end