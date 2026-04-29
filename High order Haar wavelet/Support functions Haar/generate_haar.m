function [alpha,beta,gamma] = generate_haar(j)
m=2^j;
beta=zeros(1,m);
k=linspace(2,m,m-1);
NK=linspace(1,m,m); % new k include first points
NM=floor(log(2.*NK-1)/log(2));% new m
NM(1)=1;
beta(1)=1;
beta(2:m)=FA(k);
alpha=beta-(1/2)./(2.^(NM-1));
alpha(1)=0;
gamma=beta+(1/2)./(2.^(NM-1));
gamma(1)=1;
end

