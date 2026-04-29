function [output] = PE2(RF,input_coefficient,n)
% 用来描述函数误差
% RF表示理论函数, 输入类型为函数句柄
% input_coefficient表示近似解的系数，输入为1*n大小的矩阵
% 输出output为一个实数，表示绝对误差
load('M_GKSF3_2_128.mat','NEW_FRANKLIN_function');
% secondElement = NEW_FRANKLIN_function{2};% 将第二个元素移到末尾
% NEW_FRANKLIN_function(2:end-1) = NEW_FRANKLIN_function(3:end);  % 将第3到第128个元素向前移动一个位置
% NEW_FRANKLIN_function{end} = secondElement;  % 将原来的第二个元素放到末尾
% nn=CP(n);
nn=linspace(0,1,n);

hi_x=zeros(n,length(input_coefficient)-2);
% output=zeros(1,length(nn));
% for i=1:length(nn)
%     accumulator=0;
%     for j=1:length(input_coefficient)
%         accumulator = accumulator + input_coefficient(j)*NEW_FRANKLIN_function{j}(nn(i));
%     end
%     output(i)=abs(RF(nn(i))-accumulator);
% end
for i=1:n
    for j=1:length(input_coefficient)-2
        hi_x(i,j)=NEW_FRANKLIN_function{j+2}(nn(i));
    end
end
temp=ones(n,1);
hi_x_new=[temp,hi_x,nn'];
output=abs(RF(nn)'-hi_x_new*input_coefficient);
% output=output';
end

