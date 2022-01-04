%% =====================================================================%%
%% 完全受扰粒子群优化：局部版
% coding：陈小斌
% Github：doFighter
% N: 种群大小
% dim:求解问题的维度
% x_max:解空间上限
% x_min:解空间下限
% iterate_max:最大迭代次数
% fitness:评价函数
%% --------------------------------------------------------------------%%
function[result] = TDPSO(N,dim,x_max,x_min,iterate_max,fitness)
    % 第一步，生成混沌时间序列
    a = 1.4;
    b = 0.3;
    x = zeros(1,100);
    S = zeros(1,100);
    x(1) = 0;
    S(1) = 0;
    for i = 2 : 100
        x(i) = S(i-1) + 1 - a*x(i-1)^2;
        S(i) = b*x(i-1);
    end
    % 对混沌时间序列进行归一化操作
    S = S - min(S);
    S = S / max(S);
    % 声明两个学习因子c1，c2，在Parameter optimization of multi-pass turning using chaotic PSO 一文中为2.8,1.3
    c = [2.8,1.3];
    % 初始化粒子位置，首先在下面的所用的函数中，粒子的取值范围在[-100,100]之间
    x = x_min + (x_max-x_min)*chaos(N,dim,S);
    % 初始化粒子速度，每个粒子生成一个随机数，当随机数小于0.5时v取原值，大于等于0.5时取绝对值
    v_rand = rand(N,dim);
    v = (x_min - chaos(N,dim,S))./(x_max - chaos(N,dim,S));
    v(v_rand>=0.5) = abs(v(v_rand>=0.5));
    
    % 声明粒子历史最优解存储数组,初始化为粒子初始位置
    pBest = x;
    % 声明粒子的全局最优解,并拿到当前初始化的最优位置作为当前全局最优解
    res = ones(N,1);
    for i=1:N
        res(i) = fitness(x(i,:));
    end
    position = find(res == min(res));
    gBest = x(position(1),:);
    
    % 定以起始迭代位置
    iterate = 1;
    
    % 开始迭代
    while iterate < iterate_max+1
        % 定义惯性权重系数Omega
        omega = power(0.5,i) + 0.4;
        % 更新粒子群的位置和速度,在原文中根本没有提及是否要对速度以及位置进行限制，而且没有提及最小速度，只提及最大速度
        v = omega*v + c(1)*rand(N,dim).*(pBest-x) + c(2)*rand(N,dim).*(gBest-x);
        % 保险起见，我们按照PSO原本的约束限制位置
        x = x + v;
        x(x>x_max) = x_max;
        x(x<x_min) = x_min;
        % 当迭代次数超过总迭代次数的70%的时候执行下面语句:对粒子进行扰动
        if iterate > iterate_MAX*0.7
            % 查找v_max,v_max定义为所有粒子速度最大的那个值
            v_max = max(v);
            RVC = v./v_max;
            MAX_RVC = (max((RVC')))';
            position = find(MAX_RVC <= 0.5);
            cap = size(position);
            cap = cap(1);
            i = 1;
            while i<cap+1
                d = randperm(30,dim*0.5);
                for j=1:dim*0.5
                    flag = rand;
                    if flag>0.5
                        x(position(i),d(j)) = chaos(1,1,S) + x(position(i),d(j));
                    else
                        x(position(i),d(j)) = chaos(1,1,S) - x(position(i),d(j));
                    end
                end  
                i = i + 1;
            end
        end
        % 进行粒子历史最优位置更新以及全局最优位置更新
        for i=1:N
            if fitness(pBest(i,:))>fitness(x(i,:))
                pBest(i,:) = x(i,:);
            end
            res(i) = fitness(pBest(i,:));
        end
        if fitness(gBest)>min(res)
            pos = find(res == min(res));
            gBest = pBest(pos(1),:);
        end
        iterate = iterate + 1;
    end
    result = fitness(gBest);
end


% 编写 chaos 函数
function[res] = chaos(m,n,S)
res = S(ceil(rand(m,n)*100));
end