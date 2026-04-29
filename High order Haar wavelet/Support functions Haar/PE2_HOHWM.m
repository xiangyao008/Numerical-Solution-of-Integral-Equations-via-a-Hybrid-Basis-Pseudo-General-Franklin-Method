function [output] = PE2_HOHWM(RF,input_coefficient,n)
% 用来描述函数误差
% RF表示理论函数, 输入类型为函数句柄
% input_coefficient表示近似解的系数，输入为1*n大小的矩阵
% 输出output为一个实数，表示绝对误差
haar_wavelet_integral= @(x,alpha,beta,gamma,s)0.*((0<=x) < alpha)+((x - alpha).^s./ factorial(s)).*(alpha<=x&x< beta)+...
(((x - alpha).^s - 2*(x - beta).^s) ./ factorial(s)).* (beta<=x& x< gamma)+ ...
(((x - alpha).^s - 2*(x - beta).^s + (x - gamma).^s) ./ factorial(s)).*(gamma<=x&x <= 1);
j=ceil(log(length(input_coefficient)-2)/log(2));
s=2;
[alpha,beta,gamma]=generate_haar(j);
% nn=CP(n);
nn=linspace(0,1,n);
P_ns=zeros(length(input_coefficient)-2,n);
for i=1:length(input_coefficient)-2
    P_ns(i,:)=haar_wavelet_integral(nn,alpha(i),beta(i),gamma(i),s); % P_ns的每一行表示n个配置点的i阶s次高阶HAAR小波的数值
end
P_ns=P_ns';
temp=ones(n,1);
P_ns_new=[temp,P_ns,nn'];
output=abs(RF(nn)'-P_ns_new*input_coefficient);
end

