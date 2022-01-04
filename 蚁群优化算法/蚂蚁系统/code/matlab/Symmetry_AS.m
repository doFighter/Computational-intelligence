%% =====================================================================%%
%% 蚂蚁系统：对称版
% coding：陈小斌
% Github：doFighter
%%  输入：
% x: x轴坐标
% y: y轴坐标
%%  输出：
% minimal_path：最短路径序列
% minimal_length：最短路径长度
%% --------------------------------------------------------------------%%
function [minimal_path,minimal_length] = Symmetry_AS(x,y,iterate_max)
    % 获取城市数目
    city_num = length(x);
    % 设置蚂蚁数目
    ants = 20;
    % 路径的相对重要性参数
    alpha = 1;
    % 能见度的相对重要性参数
    beta = 1;
    % 信息素持久性
    rho = 0.5;
    % 蚂蚁铺设的足迹数量相关的常数
    Q = 100;
    % 初始化各路径之间的信息素,对各路径赋予一个足够小的常数
    tau = ones(city_num) * 1e-5;
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
                visited_probability = ((tau(start_city,allowed).^alpha) .* (eta(start_city,allowed).^beta)) ./ (sum((tau(start_city,allowed).^alpha) .* (eta(start_city,allowed).^beta)));
                visit_city_index = find(visited_probability == max(visited_probability),1);
                visit_city = allowed(visit_city_index);
                tabu(j,visit_city) = i;
                ants_track_length(j) = ants_track_length(j) + distance(tabu(j,:)==i-1,tabu(j,:)==i);
            end
        end
        for i = 1:ants
            ants_track_length(i) = ants_track_length(i) + distance(tabu(i,:)==city_num,tabu(i,:)==1);
            if min(ants_track_length) < minimal_length
                minimal_length = min(ants_track_length);
                minimal_path = tabu(find(ants_track_length == min(ants_track_length),1),:);
            end
        end
        % 求解对应蚂蚁所走过的路径上的信息素增量
        delta_tau = Q ./ ants_track_length;
        % 每次迭代时，信息素会挥发，信息素余量为 rho倍
        tau = rho .* tau;
        % 更新对应路径上的信息素
        for i = 1:ants
            for j = 2:city_num
                tau(tabu(i,:)==j-1,tabu(i,:)==j) = tau(tabu(i,:)==j-1,tabu(i,:)==j) + delta_tau(i);
                tau(tabu(i,:)==j,tabu(i,:)==j-1) = tau(tabu(i,:)==j-1,tabu(i,:)==j);
            end
            tau(tabu(i,:)==city_num,tabu(i,:)==1) = tau(tabu(i,:)==city_num,tabu(i,:)==1) + delta_tau(i);
            tau(tabu(i,:)==1,tabu(i,:)==city_num) = tau(tabu(i,:)==city_num,tabu(i,:)==1);
        end
        
        iterate = iterate + 1;
    end
end

