#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/18 15:44
# @Author : doFighter
import random
import numpy as np

def GA_CodingWithFloat(N, dim, x_min, x_max, iterate_max, fitness):
    """
    Floating point coding：浮点型编码版本
    :param N:染色体种群数目
    :param dim:每条染色体携带基因数目，即问题求解维度
    :param x_min:搜索空间下限
    :param x_max:搜索空间上限
    :param iterate_max:最大迭代次数
    :param fitness:适应度函数
    :return:
        1. 返回最优适应度值
        2. 返回最优适应度位置
    """
    # 初始化交叉概率Pc
    Pc = 0.8
    # 初始化突变概率Pm
    Pm = 0.18
    # 交叉因子
    Pa = 0.01
    # 一个极小值
    eta = 1e-4
    # 初始化染色体，染色体数量为N，染色体上的基因个数为dim
    DNAs = x_min + (x_max - x_min) * np.random.rand(N, dim)
    # 染色体基于下标
    GeneIndex = range(0, dim)
    # 初始化适应度保存空间
    fitnessValue = np.ones(N)
    # 1.进行适应度评价
    for i in range(N):
        fitnessValue[i] = fitness(DNAs[i, :])
    # 开始进入主体循环
    iterate = 0
    while iterate < iterate_max:
        # 进行选择操作
        # 2 轮盘赌构建中间种群
        # 2.1 由于轮盘赌是根据概率进行的，当最优化问题求解的是最小值时，则概率并不是直接用适应度直接计算
        fitnessSum = max(fitnessValue) + eta
        probability = fitnessSum - fitnessValue
        probability = probability / sum(probability)
        probability = np.cumsum(probability)
        # 2.2 根据轮盘赌构建中间种群
        intermediatePopulation = np.zeros([N, dim])
        for i in range(N):
            rd = np.random.rand()
            index = np.where(probability > rd)
            intermediatePopulation[i, :] = DNAs[index[0][0], :].copy()

        # 3 重组操作和变异操作(即交叉操作和突变操作)
        # 3.1 计算重组操作次数
        CrossoverNum = int(N / 2)
        DNAList = list(range(0, N))
        for i in range(CrossoverNum):
            # 随机获取两个染色体
            CrossoverIndex = random.sample(DNAList, 2)
            # 在这里，染色体进行交叉时并不会重复选择
            for k in CrossoverIndex:
                DNAList.remove(k)

            if np.random.rand() < Pc:
                # 进行交换(两点式交换)，由于python是浅拷贝，所以不需要再次变换
                DNA1 = intermediatePopulation[CrossoverIndex[0], :].copy()
                DNA2 = intermediatePopulation[CrossoverIndex[1], :].copy()
                intermediatePopulation[CrossoverIndex[0], :] = Pa * DNA2 + (1-Pa) * DNA1
                intermediatePopulation[CrossoverIndex[1], :] = Pa * DNA1 + (1-Pa) * DNA2


        # 进行变异操作
        for i in range(N):
            k = 0.8
            rd = np.random.rand()
            if rd < Pm:
                rd1 = round(np.random.rand())
                if rd1 == 1:
                    intermediatePopulation[i, :] = intermediatePopulation[i, :] + k * (
                                x_max - intermediatePopulation[i, :]) * np.random.rand()
                else:
                    intermediatePopulation[i, :] = intermediatePopulation[i, :] - k * (
                                intermediatePopulation[i, :] - x_min) * np.random.rand()

        DNAs = intermediatePopulation.copy()

        # 进行适应度评价
        for i in range(N):
            fitnessValue[i] = fitness(DNAs[i, :])

        iterate += 1
    optimalIndex = np.where(fitnessValue == min(fitnessValue))
    return fitnessValue[optimalIndex[0][0]], DNAs[optimalIndex[0][0], :]


if __name__ == "__main__":
    def Sphere(xx):
        """
        Sphere Function
        :param xx: 疑似最优解
        :return:适应度值
        """
        d = len(xx)
        ele_sum = 0
        for i in range(d):
            ele_sum += xx[i] ** 2
        return ele_sum


    # 函数测试
    [result, position] = GA_CodingWithFloat(20, 30, -10, 10, 1000, Sphere)
    print(result)
    print(position)




