%% =====================================================================%%
%% 双中心粒子群优化
% coding：陈小斌
% Github：doFighter
% Encoding format：utf-8
% N: 种群大小
% dim:求解问题的维度
% x_max:解空间上限
% x_min:解空间下限
% iterate_max:最大迭代次数
% fitness:评价函数
%% --------------------------------------------------------------------%%
function [rest] = FunDCPSO(N,dim,x_max,x_min,iterate_max,fitness)
    c = 2*ones(1,2); % 定义加速系数 c1,c2;这里直接放入一个矩阵中
    v_min = x_min*0.2;
    v_max = x_max*0.2;
    x = x_min + (x_max-x_min)*rand(N,dim); % 初始化种群位置；三行两列的矩阵，元素处于-10~10之间。三行寓意为种群，两列为维度.
    v = v_min + (v_max-v_min)*rand(N,dim);   % 初始化种群速度；速度一般在位置的取值范围的10%~20%，这里取10%
    pBest = x;  % 存储粒子的局部最优值,初始为 初始值
    gBest = x(1,:);  % 存储粒子的全局最优值
    for i = 2:N
       if fitness(gBest) > fitness(pBest(i,:))
          gBest = pBest(i,:); 
       end
    end
    iterate = 1;
    res = inf*ones(N,1);
    while iterate < iterate_max+1
        % 下面进行速度与位置的更新，当速度超出 -2~2 时，将其置为边界数，同理，当位置超出 -10~10 时，将其也置为边界数
        omega = 0.95 - 0.65*((iterate-1)/iterate_max);
        % 更新各粒子的历史最优位置
        v1 = omega * v;
        x1 = x + v1;
        x1(x1 > x_max) = x_max;
        x1(x1 < x_min) = x_min;
        for i=1:N
            if fitness(pBest(i, :)) > fitness(x1(i, :))
                pBest(i, :) = x1(i, :);
            end
        end
        v2 = c(1) * rand(N,dim) .* (pBest - x);
        x1 = x1 + v2;
        x1(x1 > x_max) = x_max;
        x1(x1 < x_min) = x_min;
        for i=1:N
            if fitness(pBest(i, :)) > fitness(x1(i, :))
                pBest(i, :) = x1(i, :);
            end
        end
        v3 = c(2) * rand(N,dim) .* (gBest - x);
        x1 = x1 + v3;
        x1(x1 > x_max) = x_max;
        x1(x1 < x_min) = x_min;
        for i=1:N
            if fitness(pBest(i, :)) > fitness(x1(i, :))
                pBest(i, :) = x1(i, :);
            end
            res(i) = fitness(pBest(i, :));
        end
        v = v1 + v2 + v3;
        v(v > v_max) = v_max;
        v(v < v_min) = v_min;
        x = x + v;
        x(x > x_max) = x_max;
        x(x < x_min) = x_min;
        % 更新全局最优位置
        x_GCP = 0;
        x_SCP = 0;
        for i=1:N-2
            x_GCP = x_GCP + pBest(i, :);
            x_SCP = x_SCP + x(i, :);
        end
        x_GCP = x_GCP / (N - 2);
        x_SCP = x_SCP / (N - 2);

        if min(res) < fitness(gBest)
            index = find(res == min(res));
            gBest = pBest(index(1), :);
        end
        if fitness(x_SCP) < fitness(gBest)
            gBest = x_SCP;
        end
        if fitness(x_GCP) < fitness(gBest)
            gBest = x_GCP;
        end
        
        % 迭代计数器加1
        iterate = iterate + 1;
    end
    rest = fitness(gBest);
end

