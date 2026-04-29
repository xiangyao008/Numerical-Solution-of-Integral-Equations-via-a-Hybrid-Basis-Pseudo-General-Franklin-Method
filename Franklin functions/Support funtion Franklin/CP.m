function [output] = CP(n)
%CP collocation points 配置点的制作，返回1*n大小的数组
%n表示配置点数量
output=linspace(0.5/n, (n-0.5)/n, n);
end

