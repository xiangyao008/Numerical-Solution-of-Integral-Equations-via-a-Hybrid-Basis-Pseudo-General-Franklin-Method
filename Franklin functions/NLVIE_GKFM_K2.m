%% An Effective and Simple Scheme for Solving Nonlinear Fredholm Integral Equations Nonlinear Fredholm Integral Equations
clc;close all;clear all;
n=64;iter=100;tol = 1e-6; 
%% 问题1
% fx=@(x)tan(x)-1/4*sin(2*x)-1/2*x;
% kf=@(x,t)1;
% kut=@(u)1./(1+u.^2);
% dkut=@(x,t,u)-2*u./((1+u.^2).^2);
% RF=@(x)tan(x);
%% 问题2
% fx=@(x)(1+11/9*x+2/3*x.^2-1/3*x.^3+2/9*x.^4).*log(x+1)-1/3*(x+x.^4).*(log(x+1).^2)...
%     -11/9*(x.^2)+5/18*(x.^3)-2/27*(x.^4);
% kf=@(x,t,u)x.*(t.^2).*(u.^2);
% kut=@(u)u.^2;
% dkut=@(x,t,u)x.*(t.^2).*(2*u);
% RF=@(x)log(x+1);
%% 问题3
% fx=@(x)exp(x)-x.*exp(-x);
% kf=@(x,t,u)exp(-x).*exp(-2.*t).*(u.^2);
% kut=@(u)u.^2;
% dkut=@(x,t,u)exp(-x).*exp(-2.*t).*(2*u);
% RF=@(x)exp(x);
%% 问题4 Example 5 Numerical Solution of Nonlinear Fredholm and Volterra Integrals by 
% Newton–Kantorovich and Haar Wavelets Methods Numerical Solution of Nonlinear Fredholm and
% Volterra Integrals by Newton–Kantorovich and Haar Wavelets Methods
fx=@(x)3/2-1/2*exp(-2*x);
kf=@(x,t,u)-(u.^2+u);
dkut=@(x,t,u)-(2*u+1);
RF=@(x)exp(-x);
%% 问题5 Example 5 On the numerical solution of linear and nonlinear volterra integral and integro-differential equations
% fx=@(x)x-x.^2-1/4.*x.^5+2/5.*x.^6-1/6.*x.^7;
% kf=@(x,t,u)x.*t.*(u.^2);
% dkut=@(x,t,u)x.*t.*(2.*u);
% RF=@(x)x-x.^2;
%% 求解n配置点的解和误差
[ua_iter1,tk_xj1,cg_it1]=NLVIE_K2(fx,dkut, n, iter,tol);
pointwise_error1 = PE2(RF,ua_iter1(:,cg_it1),256);
 %% 求解n/2配置点的解和误差
[ua_iter2,tk_xj2,cg_it2]=NLVIE_K2(fx,dkut, n/2, iter,tol);
pointwise_error2 = PE2(RF,ua_iter2(:,cg_it2),256);

%% 速度
SP=log(pointwise_error2./pointwise_error1)/log(2);