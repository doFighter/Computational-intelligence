%% =============================================================================%%
%% FA:萤火虫算法(DOI: 10.1007/978-3-642-04944-6_14)
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

function [optimal_value, optimal_site] = FA(N, dim, x_min, x_max, iterate_max, fitness)
    % 初始化位置
    x = x_min + (x_max - x_min) * rand(N, dim);
    % 计算各位置的适应度值 I
    I = ones(N, 1);
    for i = 1:N
        I(i) = fitness(x(i, :));
    end
        
    % 对萤火虫按照适应度进行排序
    [~, I_range_index] = sort(I);
    % 定义灯光吸收系数
    % gamma = 1/(dim ** 2);
    gamma = 1;
    % 引力系数值
    beta0 = 1;
    % 初始化 alpha
    alpha = 0.2;

    % 迭代计数器
    iterate = 1;

    while iterate < iterate_max + 1
        for i = 1:N
            for j = 1:i
                % 如果前面的优于后面的，则往对应位置移动
                if I(I_range_index(j)) < I(I_range_index(i))
                    r_ij = sum((x(I_range_index(i), :) - x(I_range_index(j), :)) .^ 2);
                    x(I_range_index(i), :) = x(I_range_index(i), :) +beta0 * exp(-gamma * r_ij) .* (x(I_range_index(j), :) - x(I_range_index(i), :)) + alpha .* (rand(1, dim) - 0.5);
                    % 更新第i只萤火虫的解
                    I(I_range_index(i)) = fitness(x(I_range_index(i), :));
                end
            end
        end
        % 对萤火虫按照适应度进行排序
        [~, I_range_index] = sort(I);

        iterate = iterate + 1;
    end
    
    optimal_value = I(I_range_index(1));
    optimal_site = x(I_range_index(1), :);
end