%% =============================================================================%%
%% SSA:麻雀搜索算法 (doi:10.1080/21642583.2019.1708830)
%   coding:陈小斌
%   Encoding format：utf-8
%   :param N:   种群数目
%   :param dim: 求解维度
%   :param x_min:   各维度搜索下限
%   :param x_max:   各维度搜索上限
%   :param iterate_max: 最大迭代次数
%   :param fitness: 适应度评价函数
%   :return:
%         optimal_value:  对应评价函数最优适应度
%         optimal_site: 最优位置
%% -----------------------------------------------------------------------------%%


function [optimal_value, optimal_site] = SSA(N, dim, x_min, x_max, iterate_max, fitness)
    %   初始化麻雀位置
    x = x_min + (x_max - x_min) .* rand(N, dim);
    %   最小常数
    varepsilon = exp(-16);
    %   安全阈值
    ST = 0.8;
    %   生产者数量
    PD = int64(0.2 * N);
    %   侦察麻雀数量
    SD = int64(0.1 * N);
    %   存储各个体适应度值
    Fx = ones(N, 1);
    %   求解对应位置的适应度值
    for i = 1:N
        Fx(i) = fitness(x(i, :));
    end
        
    %   初始化迭代起点
    iterate = 1;
    
    while iterate < iterate_max+1
        %   对适应度值进行排列并获取按照适应度从大到小的下标排列
        [~, Fx_range_index] = sort(Fx);
        %   获取当前全局最优位置
        x_best = x(Fx_range_index(1), :);
        f_g = Fx((Fx_range_index(1)));
        %   获取当前全局最差位置
        x_worst = x(Fx_range_index(N), :);
        f_w = Fx(Fx_range_index(N));
        %   用于记录新位置
        x_new = x;
        %   按照论文算法框架图，所有生产者是统一执行位置更新，见公式(3)
        if rand() < ST
            for i =1:PD
                x_new(Fx_range_index(i), :) = x(Fx_range_index(i), :) .* exp(-Fx_range_index(i) ./ (rand(1, dim) .* iterate_max));
            end
        else
            %   论文中的公式描述有点问题，前半部分描述的是指定了对应麻雀对应维度，即为标量，而后面乘的却又是向量，赋给的又是标量，因此在这里实现了两种写法
            %   在对Sphere函数测试中，发现第一种方式效果更好
            %   第一种：按照向量写法，即每个位置只有一个alpha
            x_new(Fx_range_index(1: PD), :) = x(Fx_range_index(1: PD), :) .* rand(PD, 1);
            %   第二种：按照标量写法，即每个位置的对应维度都有一个随机alpha
            %x_new(Fx_range_index(: PD), :) = x(Fx_range_index(: PD), :) .* np.random.random((PD, dim));
        end
        
        %   拾取者位置更新，公式(4)
        for i =PD+1:N
            if Fx_range_index(i) > N/2
                %   公式(4)和公式(3)一样，Q应该是对应维度都不同
                x_new(Fx_range_index(i), :) = rand(1, dim) .* exp((x_worst - x(Fx_range_index(i), :))/(Fx_range_index(i)^2));
            else
                %   A表示一个1×d的矩阵，其中每个元素随机分配1或-1
                A = rand(1, dim);
                A(A>0.5) = 1;
                A(A<0.5) = -1;
                A_inv = A' * (1/(A * A'));
                x_new(Fx_range_index(i), :) = x_best + abs(x(Fx_range_index(i), :) - x_best) * A_inv;
            end
        end
        %   侦察麻雀更新，公式(5):需要说明的是，在原文中并未说明侦察麻雀是哪些，只是描述了占总数的10%~20%，原文选取10%。这里就直接按照索引进行更新
        for i = 1:SD
            f_i = fitness(x_new(i, :));
            if f_i > f_g
                x_new(i, :) = x_best + rand(1, dim) .* abs(x_new(i, :) - x_best);
            elseif f_i == f_g
                k = -1 + rand() * 2;
                x_new(i, :) = x_new(i, :) + k .* (abs(x_new(i, :) - x_worst)./(f_i - f_w + varepsilon));
            end
        end

        %   如果当前位置要好于历史位置，则更新
        for i = 1:N
            if fitness(x_new(i, :)) < fitness(x(i, :))
                x(i, :) = x_new(i, :);
            end
        end
            
        %   对更新后的位置进行合法性检查
        x(x<x_min) = x_min;
        x(x>x_max) = x_max;
        %   求解对应位置的适应度值
        for i = 1:N
            Fx(i) = fitness(x(i, :));
        end
           
        %   迭代器++
        iterate = iterate + 1;
    end
    %   对适应度值进行排列并获取按照适应度从大到小的下标排列
    [~, Fx_range_index] = sort(Fx);
    %   获取当前最优解
    optimal_value = Fx(Fx_range_index(1));
    %   获取当前全局最优位置
    optimal_site = x(Fx_range_index(1), :);
end
