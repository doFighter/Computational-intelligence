%% =============================================================================%%
%% 狼群算法(DOI:10.3969/j.issn.1001-506X.2013.11.33)
%   coding:陈小斌
%   Encoding format：utf-8
%   N:种群大小
%   dim:问题的维度
%   x_max:解空间的上界
%   x_min:解空间的下界
%   iterate_max：最大迭代次数
%   fitnessFunc：测试函数
%% -----------------------------------------------------------------------------%%
function [result] = WPA(N,dim,x_max,x_min,iterate_max,fitnessFunc)
    alpha = 4;
    T_max = 20;
    omega = 500;
    S = 1000;
    beta = 6;
    % 初始化智能行为的步长，由于一般的问题种各维度求解区间一致，因此不用单独求解每个维度的步长，
    % 但是为了保证和论文原文的一致性，这里还是使用数组保存各维度的步长,
    step_a = ones(1,dim) .* (abs(x_max - x_min)/S);
    step_b = step_a * 2;
    step_c = step_a / 2;
    
    % 初始化种群位置
    X = x_min + (x_max - x_min) .* rand(N,dim);

    % 存储各位置的适应度
    fitness_Value = ones(N,1) * inf;
    
    
    iterate = 1;
    
    while iterate < iterate_max + 1
        % 探狼数目
        S_num = randi([ceil(N/(alpha+1)),floor(N/alpha)]);
        % 猛狼数目
        M_num = N - S_num -1;
        % 弱肉强食死亡狼数目
        R = randi([ceil(N/(2 * beta)),floor(N/beta)]);
        % 获取各适应度值
        for i=1:N
           fitness_Value(i) = fitnessFunc(X(i,:)); 
        end
        % 对狼所在位置按照适应度进行排列
        [~,Original_index] = sort(fitness_Value);
        % 获取头狼索引
        head_wolf_index = Original_index(1);
        % 获取探狼索引
        detective_wolf_index = Original_index(2:S_num+1);
        
        %% 游走行为
        for i = 1:T_max
            X(detective_wolf_index,:) = X(detective_wolf_index,:) + sin(2*pi*(1:dim)/dim) .* step_a;
            % 如果探狼已经出现优于头狼适应度位置的现象，则停止游走，并更新探狼和头狼
            
            for j = 1:S_num
                fitness_Value(detective_wolf_index(j)) = fitnessFunc(X(detective_wolf_index(j),:));
            end
            if min(fitness_Value) < fitness_Value(head_wolf_index)
                break;
            end
        end
        % 更新头狼位置
        % 对狼所在位置按照适应度进行排列
        [~,Original_index] = sort(fitness_Value);
        % 获取头狼索引
        head_wolf_index = Original_index(1);
        % 获取猛狼索引
        Fierce_wolf_index = Original_index(S_num+2:N);
        d_is = sum((X(Fierce_wolf_index,:) - X(head_wolf_index,:)) .^ 2,2);
        % 在原文中 d_near 的计算是在非对称区间维度进行求解，但是一般公开的测试函数都是对称求解区间，
        % 因此这里使用对称求解区间，稍微改动一下公式
        d_near = (x_max - x_min) / omega;
        
        %% 召唤行为
        while d_is > d_near
            X(Fierce_wolf_index,:) = X(Fierce_wolf_index,:) + step_b .* (X(head_wolf_index,:) - X(Fierce_wolf_index,:)) ./ abs(X(head_wolf_index,:) - X(Fierce_wolf_index,:));
            for j = 1:M_num
               fitness_Value(Fierce_wolf_index(j)) = fitnessFunc(X(Fierce_wolf_index(j),:));
            end
            if min(fitness_Value) < fitness_Value(head_wolf_index)
                break;
            end
            d_is = sum((X(Fierce_wolf_index,:) - X(head_wolf_index,:)) .^ 2,2);
        end
        
        % 更新头狼位置
        % 对狼所在位置按照适应度进行排列
        [~,Original_index] = sort(fitness_Value);
        % 获取头狼索引
        head_wolf_index = Original_index(1);
        % 获取其他狼位置
        other_wolf_index = Original_index(2:N);
        
        %% 围攻行为
        lamda = -1 + 2 * rand;
        X(other_wolf_index,:) = X(other_wolf_index,:) + lamda * step_c .* abs(X(head_wolf_index,:) - X(other_wolf_index,:));
        
        % 获取围攻行为后各适应度值
        for i=1:N
           fitness_Value(i) = fitnessFunc(X(i,:)); 
        end
        % 对狼所在位置按照适应度进行排列
        [~,Original_index] = sort(fitness_Value);
        
        %% 强者生存机制
        dead_wolf_index = Original_index(N - R + 1:N);
        X(dead_wolf_index,:) = x_min + (x_max - x_min) .* rand(R,dim);
        
        iterate = iterate + 1;
    end
    % 获取各适应度值
    for i=1:N
       fitness_Value(i) = fitnessFunc(X(i,:)); 
    end
    % 对狼所在位置按照适应度进行排列
    [~,Original_index] = sort(fitness_Value);
    % 获取头狼索引
    head_wolf_index = Original_index(1);
    result = fitnessFunc(X(head_wolf_index,:));
end