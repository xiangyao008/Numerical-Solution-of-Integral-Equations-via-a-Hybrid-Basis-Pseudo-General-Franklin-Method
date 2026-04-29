function C = get_nc_coeffs(n)
     switch n
        case 1 % 梯形公式 (Trapezoidal Rule)
            weights = [1, 1];
            divisor = 2;
            
        case 2 % 辛普森公式 (Simpson's 1/3 Rule)
            weights = [1, 4, 1];
            divisor = 6;
            
        case 3 % 辛普森 3/8 公式 (Simpson's 3/8 Rule)
            weights = [1, 3, 3, 1];
            divisor = 8;
            
        case 4 
            weights = [7, 32, 12, 32, 7];
            divisor = 90;
            
        case 5
            weights = [19, 75, 50, 50, 75, 19];
            divisor = 288;
            
        case 6
            weights = [41, 216, 27, 272, 27, 216, 41];
            divisor = 840;
            
        case 7
            weights = [751, 3577, 1323, 2989, 2989, 1323, 3577, 751];
            divisor = 17280;
            
        case 8
            weights = [989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989];
            divisor = 28350;
            
        otherwise
            error('仅支持 n=1 到 n=8 的系数，高阶系数请使用数值生成法。');
    end
    C = weights / divisor;
    
end