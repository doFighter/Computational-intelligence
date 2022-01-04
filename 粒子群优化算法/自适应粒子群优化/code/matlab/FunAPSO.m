%% =============================================================================%%
%% 自适应粒子群优化
%   coding：陈小斌
%   Github：doFighter
%   N:种群大小
%   dim:问题的维度
%   x_max:解空间的上界
%   x_min:解空间的下界
%   iterate_max：最大迭代次数
%   fitness：测试函数
%% -----------------------------------------------------------------------------%%
function [rest] = FunAPSO(N,dim,x_max,x_min,iterate_max,fitness)
    c = 2*ones(1,2);                            % 加速系数 c1,c2
    fitness_value = ones(N,1);                  % 对应粒子的适应度值
    curState = 1;                               % 当前状态(阶段) 初始化时都为1
    lastState = 1;                              % 上一状态(阶段)
    gBest_result = ones(iterate_max,1);         % 存放吗，每次迭代最优的适应度值


    v_min = 0.2*x_min; % 速度的下限，取解的范围的20%
    v_max = 0.2*x_max; % 速度的上限
    x = x_min + (x_max - x_min)*rand(N,dim); % 初始化粒子的位置
    v = v_min + (v_max - v_min)*rand(N,dim); % 初始化粒子的速度
    pBest = x; % 初始化 pBest，此时 pBest 为其本身
    % 初始化 gBest
    curBest_index = 1;                    % 全局最优粒子位置下标
    gWorst_index = 1;                   % 全局最差粒子位置下标
    fitness_value(1) = fitness(x(1,:));
    for i=2:N
        fitness_value(i) = fitness(x(i,:));
        if fitness(x(curBest_index,:)) > fitness_value(i)
            curBest_index = i;
        end
        if fitness(x(gWorst_index,:)) < fitness_value(i)
            gWorst_index = i;
        end
    end
    gBest = x(curBest_index,:);
 
    iterate = 1; % 由于 matlab 数组下标从 1 开始，为了方便后期处理，迭代次数的下标也从 1 开始
    % 迭代停止条件可以是到达某个可接受的精度或者是指定的迭代次数，这里使用指定迭代次数
    while iterate < iterate_max + 1
        gBest_result(iterate) = fitness(gBest);
        
        % 进化状态评估
        [w,c,lastState,gBest,pBest,x] = Evolutionary_States_Estimation(c,fitness,N,curBest_index,curState,lastState,pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index);         
        v = w*v + c(1)*rand(N,dim).*(pBest - x) + c(2)*rand(N,dim).*(gBest - x);

        % 对超出速度上下限的粒子进行速度矫正
        v(v<v_min) = v_min;
        v(v>v_max) = v_max;
        x = x + v;                                  % 速度更新

        % 对粒子位置超出界限的粒子在 all_in_range 数组中标记为否，也就是置为 0
        [r,~] = find(x>x_max|x<x_min);              % 获取超出界限的粒子
        r = unique(r);                              % 去重,也就是获得超出界限粒子的下标

        % 进行适应性评估，只针对不超过界限的粒子，亦即 all_in_range 为 1 的粒子
        for i =1:N
            if ismember(i,r)==0
                fitness_value(i) = fitness(x(i,:));      % 记录当前迭代在范围能进行适应度值计算的值
            end

            if fitness_value(i) < fitness(pBest(i,:))
                pBest(i,:) = x(i,:);
            end
            if fitness_value(i) < fitness(gBest)
                gBest = x(i,:);
            end
            if fitness_value(i) < fitness_value(curBest_index)
                curBest_index = i;
            end
            if fitness(pBest(i,:)) > fitness(pBest(gWorst_index,:))
                gWorst_index = i;
            end
        end
        iterate = iterate + 1;
    end
    rest = fitness(gBest);
end



% 进化状态评估函数
function [w,c,lastState,gBest,pBest,x] = Evolutionary_States_Estimation(c,fitness,N,curBest_index,curState,lastState,pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index)
    distance = zeros(N,1);
    for i=1:N
        sum_d = sum(((x(i,:) - x).^2),2);       % 粒子i与各粒子的距离平方和
        distance(i) = sum(sum_d.^0.5)/(N - 1);  % 粒子 i 的平均距离
    end
    d_g = distance(curBest_index);              % 全局最优粒子的平均距离
    d_max = max(distance);                      % 粒子群中最大的平均距离
    d_min= min(distance);                       % 粒子群中最小的平均距离
    
    % 计算进化因子
    if d_min == d_max
        f = 1;
    else
        f = (d_g - d_min)/(d_max - d_min); 
    end     
    
    % 根据进化因子判断进化状态并执行相对应的操作
    [w,c,lastState,gBest,pBest,x] = Adaptive_Parameters(c,fitness,f, curState, lastState,pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index);
end



% 自适应选择调整函数
function [w,c,lastState,gBest,pBest,x] = Adaptive_Parameters(c,fitness,f, curState, lastState,pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index)
    w = 1/(1 + 1.5*exp(-2.6*f));
    if f<=0.2                                   % 汇聚
    	c = Adaptive_C(c,0.5,0.5);
    	[gBest,pBest,x] = Elitist_learning(fitness,pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index);
    	curState=3;
    elseif f<=0.3                              % 模糊逻辑区域
        if lastState==3||lastState==4           % 汇聚
            c = Adaptive_C(c,0.5,0.5);
            [gBest,pBest,x] = Elitist_learning(fitness,pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index);
            curState=3;
        elseif lastState==2||lastState==1      %开发
            c = Adaptive_C(c,0.5,-0.5);
            curState=2;
        end
    elseif f<=0.4                              %开发
    	c = Adaptive_C(c,0.5,-0.5);
    	curState=2;
    elseif f<=0.6                              %模糊逻辑区域
        if lastState==2||lastState==3          %开发
        	c = Adaptive_C(c,0.5,-0.5);
        	curState=2;
        elseif lastState==1||lastState==4      %勘探
        	c = Adaptive_C(c,1.0,-1.0);
        	curState=1;
        end
    elseif f<=0.7                              %勘探
    	c = Adaptive_C(c,1.0,-1.0);
    	curState=1;
    elseif f<=0.8                              %模糊逻辑区域
    	if lastState==1||lastState==2          %勘探
    		c = Adaptive_C(c,1.0,-1.0);
    		curState=1;
        elseif lastState==4||lastState==3      %跳出
    		c = Adaptive_C(c,-1.0,1.0);
    		curState=4;
    	end
    else                                       %跳出
    	c = Adaptive_C(c,-1.0,1.0);
    	curState=4;
    end
    lastState=curState;
end


% 调整加速系数函数
function [c] = Adaptive_C(c,h1,h2)
    c(1) = c(1) + h1*(0.05 + 0.05*rand);            % (0.05 + 0.05*rand) 生成[0.05,0.1]之间的随机数
    c(2) = c(2) + h2*(0.05 + 0.05*rand);
    c(c>2.5) = 2.5;
    c(c<1.5) = 1.5;
    sum_c = sum(c);
    % 归一化操作
    if sum_c < 3
        c = (c.*3)/sum_c;
    elseif sum_c > 4
        c = (c.*4)/sum_c;
    end
end

% 精英学习策略
function [gBest,pBest,x] = Elitist_learning(fitness,pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index)
    mu = 0;
    sigma_max = 1;
    sigma_min = 0.1;
    sigma = sigma_max - (sigma_max - sigma_min)*iterate/iterate_max;
    p = gBest;
    d = 1 + round(rand*(dim - 1));                          % 随机选择一个维度进行扰动 [1,dim]
    p(d) = p(d) + (x_max - x_min)*normrnd(mu,sigma);        % normrnd(mu,sigma) 正态分布函数
    % 判断扰动过后是否超出范围，若超出，则在解的范围内随机取值
    if p(d) > x_max || p(d) < x_min
        p(d) = x_min + (x_max - x_min)*rand;
    end
    p_value = fitness(p);
    if p_value < fitness(gBest)
        gBest = p;
    else
        if p_value < fitness(pBest(gWorst_index,:))
            pBest(gWorst_index,:) = p;
            x(gWorst_index,:) = p;
        else
            x(gWorst_index,:) = p;
        end
    end
    
end

% gaussian 函数编写
% 可直接使用 matlab 提供的 gaussian 随机生成函数，测试效果一样

% function [res] = gaussian(mu,sigma)
%     u1 = rand;
%     u2 = rand;
%     x=sqrt(-2.0*log(u1))*cos(2.0*pi*u2);
%     res = sigma*x+mu;
% end

