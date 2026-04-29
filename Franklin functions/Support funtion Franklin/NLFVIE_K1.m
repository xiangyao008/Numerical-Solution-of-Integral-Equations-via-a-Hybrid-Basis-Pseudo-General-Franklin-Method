function [ua_iter,tk_xj,cg_it] = NLFVIE_K1(fx,dvkut,dfkut, n, iter,tol)
% nonlinear Fredholm integral equation
% fx 为给定函数, 接受x参数 定义为函数句柄
% kuf 为已知积分核, 接受x,t,u三个参数 定义为函数句柄,包含被解函数的函数
% dkut表示kuf对被解函数u(t)的偏导
% n 为阶数，iter,tol表示迭代此时和容差
% ux_iter 为 每次迭代后的值
% cg_it表示收敛时的阶次
load('M_GKSF1_2_128.mat','NEW_FRANKLIN_function');
tk_xj=CP(n);
n=n+1;
tk_xj_new=[0,tk_xj];
BdJx=cell(1,n);
% BdJx{1}=eye(n);
Init_ux = zeros(n, 1);
Init_ux(1,1)=-2;
Init_ux(2,1)=2;
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
dvkut_xt=zeros(n,n-1);
dfkut_xt=zeros(n,n-1);
for i=1:length(u1)
    for j=1:(length(u1)-1)
        dvkut_xt(i,j)=dvkut(tk_xj_new(i),tk_xj(j),u1(j+1));
        dfkut_xt(i,j)=dfkut(tk_xj_new(i),tk_xj(j),u1(j+1));
    end
end
dvkut_xt(1:length(tk_xj),1:length(tk_xj))=dvkut_xt(1:length(tk_xj),1:length(tk_xj)).*tril(ones(length(tk_xj)),-1);
BdJx{1}=pinv(hi_x_new-dvkut_xt*(hi_x_new(2:n,:).*(tk_xj-tk_xj_new(1:n-1))')-dfkut_xt*(hi_x_new(2:n,:).*(tk_xj-tk_xj_new(1:n-1))'));
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
