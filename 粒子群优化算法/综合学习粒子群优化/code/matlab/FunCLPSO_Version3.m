%% =====================================================================%%
%% 广泛学习粒子群优化：版本三——对粒子x取值进行约束
% coding：陈小斌
% Github：doFighter
% N: 种群大小
% dim:求解问题的维度
% x_max:解空间上限
% x_min:解空间下限
% iterate_max:最大迭代次数
% fitness:评价函数
%% --------------------------------------------------------------------%%
function [result] = FunCLPSO_Version3(N,dim, x_max,x_min,iterate_max,fitness)
    c = 2*ones(1,2); % 定义加速系数 c1,c2;这里直接放入一个矩阵中，这是经典PSO需要
    c1 = 1.49445;   % CLPSO 速度更新时用到的系数
    % 定义粒子编号数组
    P_index = (1:N);
    %获取各粒子广泛学习概率Pc
    Pc = zeros(N,1);
    for i=1:N
        Pc(i) = 0.05 + 0.45*(exp((10*(i-1)/(N-1))-1)/(exp(10)-1));
    end
    flag = zeros(N,1);   %用于记录粒子有多少次未曾进行速度迭代
    m = 7;              %刷新间隔
    %由于在文中未能看到上面的 rand 是随机生成亦或是初始化生成，这里使用随机生成
    v_min = x_min*0.2;
    v_max = x_max*0.2;
    x = x_min + (x_max-x_min)*rand(N,dim); % 初始化种群位置；三行两列的矩阵，元素处于-10~10之间。三行寓意为种群，两列为维度.
    v = v_min + (v_max-v_min)*rand(N,dim);   % 初始化种群速度；速度一般在位置的取值范围的10%~20%，这里取10%
    pBest = x;  % 存储粒子的局部最优值,初始为 初始值
    gBest = x(1,:);  % 存储粒子的全局最优值
    for i = 2:N
       if fitness(gBest) > fitness(x(i,:))
          gBest = x(i,:); 
       end
    end
    
    iterate = 1;
    while iterate < iterate_max+1
        % 下面进行速度与位置的更新，当速度超出 -2~2 时，将其置为边界数，同理，当位置超出 -10~10 时，将其也置为边界数
        omega = 0.9 - 0.5*((iterate-1)/iterate_max);
        for i=1:N
            if flag(i)>=m
                v(i,:) = omega*v(i,:) + c(1)*rand(1,dim).*(pBest(i,:)-x(i,:)) + c(2)*rand(1,dim).*(gBest-x(i,:));
                flag(i) = 0;
            end
            pBest_fi = pBest(i,:);
            rd = rand(1,dim);
            P_index = (1:N);
            P_index(i) = [];
            position = find(rd<Pc(i));
            for j=position
                pBest_f1 = pBest(i,:);
                pBest_f2 = pBest(i,:);
                
                f1 = P_index(ceil(rand*(N-1)));
                f2 = P_index(ceil(rand*(N-1)));
                pBest_f1(j) = pBest(f1,j);
                pBest_f2(j) = pBest(f2,j);
                if fitness(pBest_f1) < fitness(pBest_f2)
                    pBest_fi(j) = pBest_f1(j);
                else
                    pBest_fi(j) = pBest_f2(j);
                end
            end
            v(i,:) = omega*v(i,:) + c1*rand(1,dim).*(pBest_fi-x(i,:));
            
            vi = v(i,:);
            vi(vi>v_max) = v_max;
            vi(vi<v_min) = v_min;
            v(i,:) = vi;
            x(i,:) = x(i,:) + vi;
            xi = x(i,:);
            if isempty(xi(xi>x_max|xi<x_min))
                if fitness(xi) < fitness(pBest(i,:))
                    pBest(i,:) = xi;
                    flag(i) = 0;
                    if fitness(xi) < fitness(gBest)
                        gBest = xi;
                    end
                else
                    flag(i) = flag(i) + 1;
                end
            end
        end
        
        % 迭代计数器加1
        iterate = iterate + 1;
    end
    result = fitness(gBest);
end