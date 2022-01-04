%% =====================================================================%%
%% 路径绘画函数
% coding：陈小斌
% Github：doFighter
%%  输入：
% x: x轴坐标
% y: y轴坐标
% minimal_path：最短路径序列
%% --------------------------------------------------------------------%%
function [] = DrawPath(x,y,minimal_path)
    % 画图，首先需要最短路径的对应城市访问顺序
    city_num = length(x);
    sequence_x = zeros(1,city_num+1);
    sequence_y = zeros(1,city_num+1);
    for i = 1:city_num
        city_index = find(minimal_path == i);
        sequence_x(i) = x(city_index);
        sequence_y(i) = y(city_index);
    end
    city_index = find(minimal_path == 1);
    sequence_x(city_num+1) = x(city_index);
    sequence_y(city_num+1) = y(city_index);
    % scatter(x,y,'k',"filled");
    plot(sequence_x,sequence_y,'g-o','MarkerFaceColor','k');
end

