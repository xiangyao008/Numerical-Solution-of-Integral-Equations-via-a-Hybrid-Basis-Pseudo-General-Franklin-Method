function [ua_iter,tk_xj,cg_it] = NLVIE_K2(fx, dkut, n, iter,tol)
% nonlinear Fredholm integral equation
% fx 为给定函数, 接受x参数 定义为函数句柄
% kuf 为已知积分核, 接受x,t,u三个参数 定义为函数句柄,包含被解函数的函数
% dkut表示kuf对被解函数u(t)的偏导
% n 为阶数，iter,tol表示迭代此时和容差
% ux_iter 为 每次迭代后的值
% cg_it表示收敛时的阶次
load('M_GKSF3_2_128.mat','NEW_FRANKLIN_function');
tk_xj=CP(n);
n=n+2;
tk_xj_new=[0,tk_xj,1];
BdJx=cell(1,n);
% BdJx{1}=eye(n);
Init_ux = zeros(n, 1);
Init_ux(n,1)=1;
Init_ux(1,1)=1;
ua_iter=zeros(n,iter+1);
ua_iter(:,1)=Init_ux;
hi_x=zeros(n,n-2);
for i=1:n
    for j=1:(n-2)
        hi_x(i,j)=NEW_FRANKLIN_function{j+2}(tk_xj_new(i));
    end
end
temp=ones(n,1);
hi_x_new=[temp,hi_x,tk_xj_new'];
u1=(hi_x_new*ua_iter(:,1));
% 原方法
% kf1=get_dkf(tk_xj_new,tk_xj,u1(2:n-1));
% temp=repmat(kf1, 1, n);
% BdJx{1}=pinv(hi_x_new-temp.*hi_x_new);
% 可能正确方法
dkut_xt=zeros(n,n-2);
for i=1:length(u1)
    for j=1:(length(u1)-2)
        dkut_xt(i,j)=dkut(tk_xj_new(i),tk_xj(j),u1(j+1));
    end
end
dkut_xt(1:length(tk_xj),1:length(tk_xj))=dkut_xt(1:length(tk_xj),1:length(tk_xj)).*tril(ones(length(tk_xj)),-1);
BdJx{1}=pinv(hi_x_new-dkut_xt*(hi_x_new(2:n-1,:).*(tk_xj-tk_xj_new(1:n-2))'));
%% 主函数
for i=2:iter+1  
    u=(hi_x_new*ua_iter(:,i-1));% (n+1*1)
    S2_x=u;
    kf=get_kf(tk_xj_new,tk_xj_new,u);%(n*1)
    g_x=S2_x-fx(tk_xj_new)'-kf;%(n*1)
    delta_a=BdJx{i-1}*(-g_x);%(n*1)
    ua_iter(:,i)=ua_iter(:,i-1)+delta_a;%(n*1)
    if norm(ua_iter(:,i)-ua_iter(:,i-1))<tol
        disp(['Converged in ', num2str(i), ' iterations.']);
        cg_it=i;
        break;
    end
    u_new=(hi_x_new*ua_iter(:,i));%(n*1)
    kf_new=get_kf(tk_xj_new,tk_xj_new,u_new);
    gx_new=u_new-fx(tk_xj_new)'-kf_new;
    delta_gx=gx_new-g_x;%(n+1*1)
    BdJx{i}=BdJx{i-1}+((delta_a-BdJx{i-1}*delta_gx)*(delta_a')*BdJx{i-1})/((delta_a')*BdJx{i-1}*delta_gx);
    % muk=1+(delta_gx'*BdJx{i-1}*delta_gx)./(delta_a'*delta_gx);
    % BdJx{i}=BdJx{i-1}+(muk*(delta_a*delta_a')-(delta_a*delta_gx')*BdJx{i-1}-BdJx{i-1}*delta_gx*delta_a')/((delta_a')*delta_gx);
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