%% An Effective and Simple Scheme for Solving Nonlinear Fredholm Integral Equations Nonlinear Fredholm Integral Equations
clc;close all;clear all;
n=64;iter=100;tol = 1e-6; 
%% 问题1 Triangular functions (TF) method for the solution of nonlinear Volterra–Fredholm integral equations
%%%%%%%%%%%% Example 1
fx=@(x)-1/30*x.^6+1/3*x.^4-x.^2+5/3*x-5/4;
kf=@(x,t,u)(x-t).*(u.^2);
dvkut=@(x,t,u)(x-t).*2*u;
dfkut=@(x,t,u)(x+t);
RF=@(x)x.^2-2;
%% 求解n配置点的解和误差
[ua_iter1,tk_xj1,cg_it1]=NLFVIE_K2(fx,dvkut,dfkut, n, iter,tol);
pointwise_error1 = PE2(RF,ua_iter1(:,cg_it1),256);
 %% 求解n/2配置点的解和误差
[ua_iter2,tk_xj2,cg_it2]=NLFVIE_K2(fx,dvkut,dfkut, n/2, iter,tol);
pointwise_error2 = PE2(RF,ua_iter2(:,cg_it2),256);

%% 速度
SP=log(pointwise_error2./pointwise_error1)/log(2);