function [output] = PE1_PGFM(RF,input_coefficient,n,K)
% 用来描述函数误差
% RF表示理论函数, 输入类型为函数句柄
% input_coefficient表示近似解的系数，输入为1*n大小的矩阵
% 输出output为一个实数，表示绝对误差
pseudo_Franklin_basis=@(x,cp,K)0.*((0<=x) < cp)+((x - cp).^K).*(cp<=x&x <= 1);
% K=1;
nn=linspace(0,1,n);
cp_f=CPPF(length(input_coefficient)-K-1);
PG_ns=zeros(length(input_coefficient),n);
for i=0:K
    PG_ns(i+1,:)=nn.^(i); % P_ns的每一行表示n个配置点的i阶s次高阶HAAR小波的数值
end
for i=K+1:length(input_coefficient)-1
    PG_ns(i+1,:)=pseudo_Franklin_basis(nn,cp_f(i-K),K); % P_ns的每一行表示n个配置点的i阶s次高阶HAAR小波的数值
end
PG_ns=PG_ns';
output=abs(RF(nn)'-PG_ns*input_coefficient);
end

