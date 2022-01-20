#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/14 14:06
# @Author  : 陈小斌
# @Github  : doFighter
# @File    : FA.py
# @Software: PyCharm

import numpy as np


def FA(N, dim, x_min, x_max, iterate_max, fitness):
    """
    FA:萤火虫算法 DOI: 10.1007/978-3-642-04944-6_14
    :param N:   种群数目
    :param dim: 求解维度
    :param x_min:   各维度搜索下限
    :param x_max:   各维度搜索上限
    :param iterate_max: 最大迭代次数
    :param fitness: 适应度评价函数
    :return:
        I[I_range_index[0]]:  对应评价函数最优适应度
        x[I_range_index[0], :]: 最优位置
    """
    # 初始化位置
    x = x_min + (x_max - x_min) * np.random.random([N, dim])
    # 计算各位置的适应度值 I
    I = np.ones([N, 1])
    for i in range(N):
        I[i] = fitness(x[i, :])
    # 对萤火虫按照适应度进行排序
    range_list = sorted(enumerate(I), key=lambda I_zip: I_zip[1])
    I_range_index = [ele[0] for ele in range_list]
    # 定义灯光吸收系数
    # gamma = 1/(dim ** 2)
    gamma = 1
    # 初始化 alpha
    alpha = 0.2

    # 迭代计数器
    iterate = 0

    while iterate < iterate_max:
        for i in range(N):
            for j in range(i):
                # 如果前面的优于后面的，则往对应位置移动
                if I[I_range_index[j]] < I[I_range_index[i]]:
                    r_ij = sum((x[I_range_index[i], :] - x[I_range_index[j], :]) ** 2)
                    x[I_range_index[i], :] = x[I_range_index[i], :] + np.exp(-gamma * r_ij) * (x[I_range_index[j], :] - x[I_range_index[i], :]) + alpha * (np.random.rand(dim) - 0.5)
                    # 更新第i只萤火虫的解
                    I[I_range_index[i]] = fitness(x[I_range_index[i], :])

        # 对萤火虫按照适应度进行排序
        range_list = sorted(enumerate(I), key=lambda I_zip: I_zip[1])
        I_range_index = [ele[0] for ele in range_list]

        iterate = iterate + 1
    return I[I_range_index[0]], x[I_range_index[0], :]


def Sphere(xx):
    """
    Sphere Function
    :param xx: 疑似最优解
    :return:适应度值
    """
    d = len(xx)
    sum = 0
    for i in range(d):
        sum += xx[i] ** 2
    return sum


# 函数测试
for i in range(3):
    result, site = FA(20, 30, -10, 10, 1000, Sphere)
    print(result)
    print(site)
