%% =====================================================================%%
%% 二邻域粒子群优化：逻辑相邻(doi:10.1109/MHS.1995.494215)
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
function[rest]=FunPSO_LBEST_Logic(N,dim,x_max,x_min,iterate_max,fitness)
    %% ====================================================================
    % LBEST版本，在该版本中，对应粒子会受到邻近粒子的影响而导致搜索方向的改变
    % 在该版本的粒子群算法中，要通过以下两个公式便可(以邻近粒子数等于2为例)
    % 1.v(i,d)=v(i,d)+c1*rand(1,d)*(pBest(i,d)-x(i,d)+c2*rand(2,d)*(x(i-1,d)-x(i,d)+c3*rand(3,d)*(x(i+1,d)-x(i,d));
    % 2.x(i,d)=x(i,d)+v(i,d);
    %% ====================================================================
    % 第一步：初始化变量
    c = 2*ones(1,3); % 定义加速系数 c1,c2;这里直接放入一个矩阵中
    %由于在文中未能看到上面的 rand 是随机生成亦或是初始化生成，这里使用随机生成
    v_min = x_min*0.1;
    v_max = x_max*0.1;
    x = x_min + (x_max-x_min)*rand(N,dim); % 初始化种群位置；三行两列的矩阵，元素处于-10~10之间。三行寓意为种群，两列为维度.
    v = v_min + (v_max-v_min)*rand(N,dim);   % 初始化种群速度；速度一般在位置的取值范围的10%~20%，这里取10%
    pBest = x;  % 存储粒子的局部最优值
    iterate = 1;
    results = inf*ones(iterate_max,1);
    res = inf*ones(N,1);

    % 第二步：迭代求最优
    while iterate < iterate_max+1
        % 下面进行速度与位置的更新，当速度超出 -2~2 时，将其置为边界数，同理，当位置超出 -10~10 时，将其也置为边界数
        for i=1:N
            pre = i - 1;
            nex = i + 1;
            if pre < 1
                pre = N;
            end
            if nex > N
                nex = 1;
            end
            v(i,:) = v(i,:) + c(1)*rand(1,dim).*(pBest(i,:)-x(i,:)) + c(2)*rand(1,dim).*(pBest(pre,:)-x(i,:)) + c(3)*rand(1,dim).*(pBest(nex,:)-x(i,:));
        end
        v(v>v_max) = v_max;
        v(v<v_min) = v_min;
        x = x + v;
        x(x>x_max) = x_max;
        x(x<x_min) = x_min;
        
        
        for i = 1:N                     % 对每个粒子进行求解
            if fitness(x(i,:)) < fitness(pBest(i,:))   % 判断当前位置是否优于局部最优位置，若优于局部最优，则更新局部最优位置
                pBest(i,:) = x(i,:);
            end
            res(i) = fitness(pBest(i,:));
        end
        results(iterate) = min(res);    % 保存当前最优解
        
        % 迭代计数器加1
        iterate = iterate + 1;
    end
    rest = min(results);
end