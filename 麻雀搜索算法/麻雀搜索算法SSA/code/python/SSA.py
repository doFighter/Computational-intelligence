#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 21:34
# @Author  : 陈小斌
# @Github  : doFighter
# @File    : SSA.py
# @Software: PyCharm

import numpy as np


def SSA(N, dim, x_min, x_max, iterate_max, fitness):
    """
    SSA:麻雀搜索算法 doi:10.1080/21642583.2019.1708830
    :param N:   种群数目
    :param dim: 求解维度
    :param x_min:   各维度搜索下限
    :param x_max:   各维度搜索上限
    :param iterate_max: 最大迭代次数
    :param fitness: 适应度评价函数
    :return:
        Fx[Fx_range_index[0]]:  对应评价函数最优适应度
        x_best: 最优位置
    """
    # 初始化麻雀位置
    x = x_min + (x_max - x_min) * np.random.random([N, dim])
    # 最小常熟
    varepsilon = np.exp(-16)
    # 安全阈值
    ST = 0.8
    # 生产者数量
    PD = int(0.2 * N)
    # 侦察麻雀数量
    SD = int(0.1 * N)
    # 存储各个体适应度值
    Fx = np.ones([N, 1])
    # 求解对应位置的适应度值
    for i in range(N):
        Fx[i] = fitness(x[i, :])
    # 初始化迭代起点
    iterate = 0

    while iterate < iterate_max:

        # 对适应度值进行排列并获取按照适应度从大到小的下标排列
        range_list = sorted(enumerate(Fx), key=lambda Fx_zip: Fx_zip[1])
        Fx_range_index = [ele[0] for ele in range_list]
        # 获取当前全局最优位置
        x_best = x[Fx_range_index[0], :].copy()
        f_g = Fx[[Fx_range_index[0]]]
        # 获取当前全局最差位置
        x_worst = x[Fx_range_index[N-1], :].copy()
        f_w = Fx[Fx_range_index[N-1]]
        # 用于记录新位置
        x_new = x.copy()
        # 按照论文算法框架图，所有生产者是统一执行位置更新，见公式(3)
        if np.random.rand() < ST:
            for i in range(PD):
                x_new[Fx_range_index[i], :] = x[Fx_range_index[i], :] * \
                    np.exp(-Fx_range_index[i] /
                           (np.random.random([1, dim]) * iterate_max))
        else:
            # 论文中的公式描述有点问题，前半部分描述的是指定了对应麻雀对应维度，即为标量，而后面乘的却又是向量，赋给的又是标量，因此在这里实现了两种写法
            # 在对Sphere函数测试中，发现第一种方式效果更好
            # 第一种：按照向量写法，即每个位置只有一个alpha
            x_new[Fx_range_index[: PD], :] = x[Fx_range_index[: PD],
                                               :] * np.random.random([PD, 1])
            # 第二种：按照标量写法，即每个位置的对应维度都有一个随机alpha
            # x_new[Fx_range_index[: PD], :] = x[Fx_range_index[: PD], :] * np.random.random([PD, dim])

        # 拾取者位置更新，公式(4)
        for i in range(PD, N):
            if Fx_range_index[i] > N/2:
                # 公式(4)和公式(3)一样，Q应该是对应维度都不同
                x_new[Fx_range_index[i], :] = np.random.random(
                    [1, dim]) * np.exp((x_worst - x[Fx_range_index[i], :])/(Fx_range_index[i]**2))
            else:
                A = np.array([np.random.choice([1, -1]) for i in range(dim)])
                A_inv = np.dot(A.T, 1/np.dot(A, A.T))
                x_new[Fx_range_index[i], :] = x_best + \
                    np.abs(x[Fx_range_index[i], :] - x_best) * A_inv

        # 侦察麻雀更新，公式(5):需要说明的是，在原文中并未说明侦察麻雀是哪些，只是描述了占总数的10%~20%，原文选取10%。这里就直接按照索引进行更新
        for i in range(SD):
            f_i = fitness(x_new[i, :])
            if f_i > f_g:
                x_new[i, :] = x_best + \
                    np.random.random([1, dim]) * np.abs(x_new[i, :] - x_best)
            elif f_i == f_g:
                k = -1 + np.random.random() * 2
                x_new[i, :] = x_new[i, :] + k * \
                    (np.abs(x_new[i, :] - x_worst)/(f_i - f_w + varepsilon))

        # 对当前位置进行检查，判断是否合法
        x[x > x_max] = x_max
        x[x < x_min] = x_min
        # 如果当前位置要好于历史位置，则更新
        for i in range(N):
            if fitness(x_new[i, :]) < fitness(x[i, :]):
                x[i, :] = x_new[i, :].copy()

        # 求解对应位置的适应度值
        for i in range(N):
            Fx[i] = fitness(x[i, :])
        # 迭代器++
        iterate = iterate + 1

    # 对适应度值进行排列并获取按照适应度从大到小的下标排列
    range_list = sorted(enumerate(Fx), key=lambda Fx_zip: Fx_zip[1])
    Fx_range_index = [ele[0] for ele in range_list]
    # 获取当前全局最优位置
    x_best = x[Fx_range_index[0], :].copy()

    return Fx[Fx_range_index[0]], x_best


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
for i in range(10):
    result, site = SSA(20, 30, -10, 10, 100, Sphere)
    print(result)
    print(site)
