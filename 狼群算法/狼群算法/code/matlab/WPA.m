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
    h = dim;
    % 初始化智能行为的步长，由于一般的问题种各维度求解区间一致，因此不用单独求解每个维度的步长，
    % 但是为了保证和论文原文的一致性，这里还是使用数组保存各维度的步长,
    step_a = ones(1,dim) .* (abs(x_max - x_min)/S);
    step_b = step_a * 2;
    step_c = step_a / 2;
    
    % 初始化种群位置
    X = x_min + (x_max - x_min) .* rand(N,dim);

    % 存储各位置的适应度
    fitness_Value = ones(N,1) * inf;
    % 获取各适应度值
    for i=1:N
        fitness_Value(i) = fitnessFunc(X(i,:)); 
     end
    
    iterate = 1;
    
    while iterate < iterate_max + 1
        % 探狼数目
        S_num = randi([ceil(N/(alpha+1)),floor(N/alpha)]);
        % 猛狼数目
        M_num = N - S_num -1;
        % 弱肉强食死亡狼数目
        R = randi([ceil(N/(2 * beta)),floor(N/beta)]);
        

        % 对狼所在位置按照适应度进行排列
        [~,Original_index] = sort(fitness_Value);
        % 获取探狼索引
        detective_wolf_index = Original_index(2:S_num+1);
        % 获取猛狼索引
        Fierce_wolf_index = Original_index(S_num+2:N);
        
        %% 游走行为
        [X, fitness_Value] = WanderingBehavior(X, T_max, S_num, h, fitness_Value, detective_wolf_index, step_a, fitnessFunc);
        

        % 在原文中 d_near 的计算是在非对称区间维度进行求解，但是一般公开的测试函数都是对称求解区间，
        % D = size(x_max, 1);
        % d_near = sum((x_max - x_min)) /(D * omega);
        % 因此这里使用对称求解区间，稍微改动一下公式
        d_near = (x_max - x_min) / omega;
        %% 召唤行为
        [X, fitness_Value] = SummoningBehavior(X, M_num, d_near, fitness_Value, Fierce_wolf_index, step_b, fitnessFunc);

        %% 围攻行为
        [X, fitness_Value] = SiegeBehavior(X, N, fitness_Value, step_c, fitnessFunc);
        
        %% 适者生存机制
        [X, fitness_Value] = FittestSurvive(X, N, dim, R, x_max, x_min, fitness_Value, fitnessFunc)
        
        iterate = iterate + 1;
    end

    result = min(fitness_Value);
end


function [wolf_position, fitness_Value] = WanderingBehavior(wolf_position, T_max, S_num, h, fitness_Value, detective_wolf_index, step_a, fitnessFunc)
%myFun - Description
%
% Syntax: X = myFun(input)
%
% Long description
    %% 游走行为
    % 找到头狼位置
    head_wolf_index = find(fitness_Value==min(fitness_Value), 1);
    % 获取头狼适应度值
    head_wolf_value = fitness_Value(head_wolf_index);
    for j = 1:S_num
        % 游走行为是每只探狼往不同方向上行走并记录，使用最优的保存下来
        for i=1:T_max
            XJ_every_h = wolf_position(detective_wolf_index(j),:) + sin(2*pi*(1:h)'/h) .* step_a;
            % 对 h 个方向上的适应度进行评估，保留最好的适应度作为探狼 j 的位置
            for k=1:h
                if fitness_Value(detective_wolf_index(j)) > fitnessFunc(XJ_every_h(k,:));
                    wolf_position(detective_wolf_index(j),:) = XJ_every_h(k,:);
                    fitness_Value(detective_wolf_index(j)) = fitnessFunc(XJ_every_h(k,:));
                end
            end
            % 如果探狼已经出现优于头狼适应度位置的现象，则停止该探狼停止游走
            if min(fitness_Value) < head_wolf_value
                % 因为头狼是所有狼适应度最小的，因此如若此时出现适应度比头狼小，则必定是当前探狼
                % 更新头狼的适应度值，由于游走并不会使用到头狼位置，因此不予以保存
                head_wolf_value = min(fitness_Value);
                % 结束当前探狼的游走
                break;
            end
        end
    end
end


function [wolf_position, fitness_Value] = SummoningBehavior(wolf_position,M_num, d_near, fitness_Value, Fierce_wolf_index, step_b, fitnessFunc)
%myFun - Description
%
% Syntax: output = myFun(input)
%
% Long description
    
    % 找到头狼位置
    head_wolf_index = find(fitness_Value==min(fitness_Value), 1);
    head_walf_position = wolf_position(head_wolf_index, :);
    head_wolf_value = fitness_Value(head_wolf_index);
    
    %% 召唤行为
    for i=1:M_num
        d_is = sum(abs((wolf_position(Fierce_wolf_index(i),:) - head_walf_position)));
        %% 这两个 while 循环的合理性说实话是很难去界定的，理论来说，只有找到了更好的位置，才能跳出两个循环
        % 但倘若是没找到，则一样是死循环
        while d_is > d_near
            % 这个循环很奇妙，倘若猛狼i按照这也的召唤行为已知找不到比当前头狼位置好的位置，那就是个死循环
            % 在这个算法中你可以发现很多类似的地方，就是在算法内部狼其实走了很多步，因此时间复杂度也无法计算
            % 比较好的改变是将论文流程图中的循环改成判断
            %% 不合理的实现
            while fitness_Value(Fierce_wolf_index(i)) > head_wolf_value
                wolf_position(Fierce_wolf_index(i),:) = wolf_position(Fierce_wolf_index(i),:) + step_b .* (head_walf_position - wolf_position(Fierce_wolf_index(i),:)) ./ abs(head_walf_position - wolf_position(Fierce_wolf_index(i),:));
                fitness_Value(Fierce_wolf_index(i)) = fitnessFunc(wolf_position(Fierce_wolf_index(i),:));
            end
            % 跳出当前循环，说明探狼i已经找到更好位置(至少不输于当前头狼位置)
            head_walf_position = wolf_position(Fierce_wolf_index(i),:);
            head_wolf_value = fitness_Value(Fierce_wolf_index(i));

            % 其实判断 d_is>d_near 是个伪命题，因为当跳出 Yi>Ylead 循环之后，就意味着当前猛狼i成为新头狼
            % 而 d_is 是计算当前猛狼i和当前头狼的距离，两者是一模一样的， d_is 自然是零
            % d_is = sum(abs((wolf_position(Fierce_wolf_index(i),:) - head_walf_position)));
            break;
            d_is = sum(abs((wolf_position(Fierce_wolf_index(i),:) - head_walf_position)));
        end
    end
end


function [wolf_position, fitness_Value] = SiegeBehavior(wolf_position, N, fitness_Value, step_c, fitnessFunc)
%myFun - Description
%
% Syntax: output = myFun(input)
%
% Long description
    
    % 更新头狼位置
    % 对狼所在位置按照适应度进行排列
    [~,Original_index] = sort(fitness_Value);
    % 获取头狼索引
    head_wolf_index = Original_index(1);
    % 获取头狼位置
    head_walf_position = wolf_position(head_wolf_index, :);
    %==头狼和对狼所在位置按照适应度进行排列在结束召唤行为前已经完成更新==%
    % 获取其他狼位置(包含除头狼外的所有狼)
    other_wolf_index = Original_index(2:N);
    
    %% 围攻行为
    lamda = -1 + 2 * rand;
    wolf_position(other_wolf_index,:) = wolf_position(other_wolf_index,:) + lamda * step_c .* abs(head_walf_position - wolf_position(other_wolf_index,:));
    
    % 获取围攻行为后各适应度值
    for i=1:N
        fitness_Value(i) = fitnessFunc(wolf_position(i,:)); 
    end
end


function [wolf_position, fitness_Value] = FittestSurvive(wolf_position, N, dim, R, x_max, x_min, fitness_Value, fitnessFunc)
%myFun - Description
%
% Syntax: output = myFun(input)
%
% Long description
    % 对狼所在位置按照适应度进行排列
    [~,Original_index] = sort(fitness_Value);
    %% 计算淘汰狼的数目
    dead_wolf_index = Original_index(N - R + 1:N);
    % 淘汰狼重新生成
    wolf_position(dead_wolf_index,:) = x_min + (x_max - x_min) .* rand(R,dim);
    % 重新计算各适应度值
    for i=1:N
        fitness_Value(i) = fitnessFunc(wolf_position(i,:)); 
    end
end