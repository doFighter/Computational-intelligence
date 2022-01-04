%% =====================================================================%%
%% 蚂蚁系统：非对称版
% coding：陈小斌
% Github：doFighter
%%  输入：
% x: x轴坐标
% y: y轴坐标
%%  输出：
% minimal_path：最短路径序列
% minimal_length：最短路径长度
%% --------------------------------------------------------------------%%
function [minimal_path,minimal_length] = Asymmetry_ACS(x,y,iterate_max)
    % 获取城市数目
    city_num = length(x);
    % 设置蚂蚁数目
    ants = 20;
    % 全局信息素更新参数
    alpha = 0.1;
    % 能见度的相对重要性参数
    beta = 2;
    % 局部信息素更新参数
    rho = 0.1;
    % 选择阈值（通过q0在开发及有偏探索中进行选择）
    q0 = 0.9;
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
    % 求解最近邻启发式产生的旅行长度
    L_nn = 0;
    L_nn_Tabu = zeros(1,city_num);
    L_nn_Tabu(1) = 1;
    for i = 2:city_num
        allowed = find(L_nn_Tabu == 0);
        start_city = find(L_nn_Tabu == i-1);
        visit_city_index = find(eta(start_city,allowed) == max(eta(start_city,allowed)),1);
        visit_city = allowed(visit_city_index);
        L_nn_Tabu(visit_city) = i;
        L_nn = L_nn + distance(L_nn_Tabu==i-1,L_nn_Tabu==i);
    end
    L_nn = L_nn + distance(L_nn_Tabu==city_num,L_nn_Tabu==1);

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
                % 搜索路径状态选择
                q = rand;
                if q <= q0
                    % 使用贪婪开发，即选择信息素最强的路径
                    visited_probability = ((tau(start_city,allowed)) .* (eta(start_city,allowed).^beta));
                    visit_city_index = find(visited_probability == max(visited_probability),1);
                else
                    % 使用有偏探索，在一定概率上会选择较差的路径
                    visited_probability = ((tau(start_city,allowed)) .* (eta(start_city,allowed).^beta)) ./ (sum((tau(start_city,allowed)) .* (eta(start_city,allowed).^beta)));
                    visited_probability = cumsum(visited_probability);
                    q = rand;
                    visit_city_index = find(visited_probability > q,1);
                end
                visit_city = allowed(visit_city_index);
                tabu(j,visit_city) = i;
                tau(start_city,visit_city) = (1 - rho) .* tau(start_city,visit_city) + rho / (city_num * L_nn);
                ants_track_length(j) = ants_track_length(j) + distance(tabu(j,:)==i-1,tabu(j,:)==i);
            end
        end
        for i = 1:ants
            ants_track_length(i) = ants_track_length(i) + distance(tabu(i,:)==city_num,tabu(i,:)==1);
            tau(tabu(i,:)==city_num,tabu(i,:)==1) = (1 - rho) .* tau(tabu(i,:)==city_num,tabu(i,:)==1) + rho / (city_num * L_nn);
            if min(ants_track_length) < minimal_length
                minimal_length = min(ants_track_length);
                minimal_path = tabu(ants_track_length == minimal_length,:);
            end
        end

        % 信息素全局更新规则
        tau = (1-alpha).*tau;
        minimal_ants = size(minimal_path,1);
        for i = 1:minimal_ants
            for j = 2:city_num
                tau(minimal_path(i,:)==j-1,minimal_path(i,:)==j) = tau(minimal_path(i,:)==j-1,minimal_path(i,:)==j) + alpha / minimal_length;
            end
            tau(minimal_path(i,:)==city_num,minimal_path(i,:)==1) = tau(minimal_path(i,:)==city_num,minimal_path(i,:)==1) + alpha / minimal_length;  tau(minimal_path(i,:)==1,minimal_path(i,:)==city_num) = tau(minimal_path(i,:)==city_num,minimal_path(i,:)==1);
        end
        iterate = iterate + 1;
    end
end

