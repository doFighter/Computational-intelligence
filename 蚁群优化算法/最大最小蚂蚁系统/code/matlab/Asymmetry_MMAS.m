%% =====================================================================%%
%% 最大最小蚂蚁系统：非对称版
% coding：陈小斌
% Github:doFighter
%%  输入：
% x: x轴坐标
% y: y轴坐标
%%  输出：
% minimal_path：最短路径序列
% minimal_length：最短路径长度
%% --------------------------------------------------------------------%%
function [minimal_path,minimal_length] = Asymmetry_MMAS(x,y,iterate_max)
    % 获取城市数目
    city_num = length(x);
    % 设置蚂蚁数目
%     ants = 20;
    ants = city_num;
    % 路径的相对重要性参数
    alpha = 1;
    % 能见度的相对重要性参数
    beta = 2;
    % 信息素持久性
    rho = 0.98;
    % 初始化各路径之间的信息素,对各路径赋予一个足够小的常数
    tau = ones(city_num) * 1e-5;
    pBest = 0.05;
    % 各城市之间的距离
    distance = zeros(city_num);
    for i = 1:city_num
        for j = i+1:city_num
            distance(i,j) = sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);
            distance(j,i) = distance(i,j);
        end
    end
    distance_diag = ones(1,city_num) .* 1e-5;
    distance_diag = diag(distance_diag);
    distance = distance_diag + distance;
    % 路径的能见度，使用距离的倒数
    eta = 1 ./ distance;
    % 记录最短路径
    minimal_path = zeros(1,city_num);
    % 记录最短路径的长度
    minimal_length = inf;
    iterate = 0;
    while iterate < iterate_max
        % 生成一个禁忌表，禁忌表大小为ants行，city_num列
        tabu = zeros(ants,city_num);
        % 将所有蚂蚁分布在不同的城市起点
        random_city = randperm(city_num);
        for i = 1:ants
            city_index = randperm(length(random_city),1);
            city = random_city(city_index);
            random_city(city_index) = [];
            tabu(i,city) = 1;
        end
        % 蚂蚁通过相应的公式选择对应路径进行移动，并求解对应蚂蚁走过的路径长度
        ants_track_length = zeros(1,ants);
        for i = 2:city_num
            for j = 1:ants
                allowed = find(tabu(j,:) == 0);
                start_city = find(tabu(j,:) == i-1);
                visited_probability = ((tau(start_city,allowed).^alpha) .* (eta(start_city,allowed).^beta)) ./ (sum((tau(start_city,allowed)) .* (eta(start_city,allowed).^beta)));
                visited_probability = cumsum(visited_probability);
                % 轮盘赌方式选择路径
                q = rand;
                visit_city_index = find(visited_probability > q,1);
                visit_city = allowed(visit_city_index);
                tabu(j,visit_city) = i;
                ants_track_length(j) = ants_track_length(j) + distance(tabu(j,:)==i-1,tabu(j,:)==i);
            end
        end
        for i = 1:ants
            ants_track_length(i) = ants_track_length(i) + distance(tabu(i,:)==city_num,tabu(i,:)==1);
            if min(ants_track_length) < minimal_length
                minimal_ant = i;
                minimal_length = min(ants_track_length);
                minimal_path = tabu(find(ants_track_length == min(ants_track_length),1),:);
            end
        end
        % 每次迭代时，信息素会挥发，信息素余量为 rho倍
        tau = rho .* tau;
        % 只增加最优路径上的信息素
        for j = 2:city_num
            tau(tabu(minimal_ant,:)==j-1,tabu(minimal_ant,:)==j) = tau(tabu(minimal_ant,:)==j-1,tabu(minimal_ant,:)==j) + 1/minimal_length;
        end
        tau(tabu(minimal_ant,:)==city_num,tabu(minimal_ant,:)==1) = tau(tabu(minimal_ant,:)==city_num,tabu(minimal_ant,:)==1) + 1/minimal_length;
        % 计算信息素上限
        tau_max = 1/((1-rho)*minimal_length);
        % 计算信息素下限
        avg = city_num / 2;
        tau_min = (tau_max * (1 - pBest^(1/city_num)))/((avg - 1) * pBest^(1/city_num));
        % 限制信息素上下限
        tau(tau<tau_min) = tau_min;
        tau(tau>tau_max) = tau_max;

        
        iterate = iterate + 1;
    end
end


