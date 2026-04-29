function result = FA(n) %GFA whrn s=2
    s=2; 
   k = floor(log((s.*n - 1)) ./ log(s));
    % disp(['CalcuFFan(', num2str(n), ') is evaluating with k = ', num2str(k)]);
    result = (s .* n - 1 - s.^k) ./ (s.^k*(s-1));
end
