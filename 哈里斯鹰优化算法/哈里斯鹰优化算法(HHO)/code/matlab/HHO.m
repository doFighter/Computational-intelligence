%% =============================================================================%%
%% HHO:哈里斯鹰优化算法 (doi:10.1016/j.future.2019.02.028)
%   Encoding format：utf-8
%   :param N:   种群数目
%   :param dim: 求解维度
%   :param x_min:   各维度搜索下限
%   :param x_max:   各维度搜索上限
%   :param iterate_max: 最大迭代次数
%   :param fitness: 适应度评价函数
%   :return:
%         Rabbit_Energy:  对应评价函数在指定迭代下最优适应度
%         every_time_Rabbit_Energy：每次迭代的最优适应度
%         Rabbit_Location: 最优位置
%% -----------------------------------------------------------------------------%%

function [Rabbit_Energy, every_time_Rabbit_Energy, Rabbit_Location]=HHO(N, dim, x_max, x_min, iterate_max, fitness)
    Rabbit_Location=zeros(1,dim);
    Rabbit_Energy=inf;
    
    %Initialize the locations of Harris' hawks
    X = x_min + (x_max - x_min) .* rand(N, dim);
    
    % 记录每次迭代最优值
    every_time_Rabbit_Energy = ones(iterate_max,1);
    
    iterate = 0; % Loop counter

    while iterate<iterate_max
        for i=1:size(X,1)
            % Check boundries
            FU=X(i,:)>x_max;
            FL=X(i,:)<x_min;
            X(i,:)=(X(i,:).*(~(FU+FL)))+x_max.*FU+x_min.*FL;
            % fitness of locations
            fitness_value=fitness(X(i,:));
            % Update the location of Rabbit
            if fitness_value<Rabbit_Energy
                Rabbit_Energy=fitness_value;
                Rabbit_Location=X(i,:);
            end
        end

        E1=2*(1-(iterate/iterate_max)); % factor to show the decreaing energy of rabbit
        % Update the location of Harris' hawks
        for i=1:size(X,1)
            E0=2*rand()-1; %-1<E0<1
            Escaping_Energy=E1*(E0);  % escaping energy of rabbit

            if abs(Escaping_Energy)>=1
                %% Exploration:
                % Harris' hawks perch randomly based on 2 strategy:

                q=rand();
                rand_Hawk_index = floor(N*rand()+1);
                X_rand = X(rand_Hawk_index, :);
                if q<0.5
                    % perch based on other family members
                    X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*X(i,:));
                elseif q>=0.5
                    % perch on a random tall tree (random site inside group's home range)
                    X(i,:)=(Rabbit_Location(1,:)-mean(X))-rand()*((x_max-x_min)*rand+x_min);
                end

            elseif abs(Escaping_Energy)<1
                %% Exploitation:
                % Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                %% phase 1: surprise pounce (seven kills)
                % surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r=rand(); % probablity of each event

                if r>=0.5 && abs(Escaping_Energy)<0.5 % Hard besiege
                    X(i,:)=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X(i,:));
                end

                if r>=0.5 && abs(Escaping_Energy)>=0.5  % Soft besiege
                    Jump_strength=2*(1-rand()); % random jump strength of the rabbit
                    X(i,:)=(Rabbit_Location-X(i,:))-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:));
                end

                %% phase 2: performing team rapid dives (leapfrog movements)
                if r<0.5 && abs(Escaping_Energy)>=0.5 % Soft besiege % rabbit try to escape by many zigzag deceptive motions

                    Jump_strength=2*(1-rand());
                    X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:));

                    if fitness(X1)<fitness(X(i,:)) % improved move?
                        X(i,:)=X1;
                    else % hawks perform levy-based short rapid dives around the rabbit
                        X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:))+rand(1,dim).*Levy(dim);
                        if (fitness(X2)<fitness(X(i,:))) % improved move?
                            X(i,:)=X2;
                        end
                    end
                end

                if r<0.5 && abs(Escaping_Energy)<0.5 % Hard besiege % rabbit try to escape by many zigzag deceptive motions
                    % hawks try to decrease their average location with the rabbit
                    Jump_strength=2*(1-rand());
                    X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X));

                    if fitness(X1)<fitness(X(i,:)) % improved move?
                        X(i,:)=X1;
                    else % Perform levy-based short rapid dives around the rabbit
                        X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X))+rand(1,dim).*Levy(dim);
                        if (fitness(X2)<fitness(X(i,:))) % improved move?
                            X(i,:)=X2;
                        end
                    end
                end
                %%
            end
        end
        iterate=iterate+1;
        every_time_Rabbit_Energy(iterate) = Rabbit_Energy;
    end
end



% ___________________________________
function o=Levy(d)
    beta=1.5;
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u=randn(1,d)*sigma;
    v=randn(1,d);
    step=u./abs(v).^(1/beta);
    o=step;
end
