%% =============================================================================%%
%% 灰狼算法(DOI: 10.1016/j.advengsoft.2013.12.007)
%   coding:陈小斌
%   Encoding format：utf-8
%   N:种群大小
%   dim:问题的维度
%   x_max:解空间的上界
%   x_min:解空间的下界
%   iterate_max：最大迭代次数
%   fitnessFunc：测试函数
%% -----------------------------------------------------------------------------%%
function [alpha_value] = GWO(N,dim,x_max,x_min,iterate_max,fitnessFunc)
    % 初始化位置
    X = rand(N,dim) .* (x_max - x_min) + x_min;
    % 初始化 alpha、beta、delta 适应度值，求解最大值问题时将其初始化为 -inf，最小值问题时初始化为 inf
    alpha_value = inf;
    beta_value = inf;
    delta_value = inf;
    % 迭代计数器
    iterate = 1;
    while iterate < iterate_max + 1
        % 将越界位置进行相应调整
        X(X > x_max) = x_max;
        X(X < x_min) = x_min;
        
        
        % 计算每只灰狼的适应度，并更新 alpha、beta、delta 
        for i = 1:N
            fitness_value = fitnessFunc(X(i,:));
            % 更新 alpha
            if alpha_value > fitness_value
                alpha_value = fitness_value;
                alpha = X(i,:);
            end
            % 更新 beta
            if fitness_value > alpha_value && beta_value > fitness_value
                beta_value = fitness_value;
                beta = X(i,:);
            end
            % 更新 delta
            if fitness_value > alpha_value && fitness_value > beta_value && delta_value > fitness_value
                delta_value = fitness_value;
                delta = X(i,:);
            end
        end
        
        % 计算参数a
        a = 2 - 2 * (iterate-1) / iterate_max;
        
        % 计算 alpha 系数向量 A1 ,公式(3)
        A1 = 2 * a .* rand(N,dim) - a;
        % 计算 alpha 系数向量 C1 ,公式(4)
        C1 = 2 .* rand(N,dim);
        
        % 计算 D_alpha 公式(5-1)
        D_alpha = abs(C1 .*alpha - X);
        % 计算 X1 公式(6-1)
        X1 = alpha - A1 .* D_alpha;
        
        
        % 计算 beta 系数向量 A2 ,公式(3)
        A2 = 2 * a .* rand(N,dim) - a;
        % 计算 beta 系数向量 C2 ,公式(4)
        C2 = 2 .* rand(N,dim);
        
        % 计算 D_beta 公式(5-2)
        D_beta = abs(C2 .* beta - X);
        % 计算 X2 公式(6-2)
        X2 = beta - A2 .* D_beta;
        
        
        % 计算 delta 系数向量 A3 ,公式(3)
        A3 = 2 * a .* rand(N,dim) - a;
        % 计算 delta 系数向量 C3 ,公式(4)
        C3 = 2 .* rand(N,dim);
        
        % 计算 D_delta 公式(5-3)
        D_delta = abs(C3 .* delta - X);
        % 计算 X3 公式(6-3)
        X3 = delta - A3 .* D_delta;

        
        % 狼群状态转移，公式(7)
        X = (X1 + X2 + X3) ./ 3;
        
        iterate = iterate + 1;
    end
end