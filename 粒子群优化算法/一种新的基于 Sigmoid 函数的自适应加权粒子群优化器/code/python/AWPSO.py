#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/11 10:56
# @Author : doFighter
import numpy as np


def AWPSO(N, dim, x_min, x_max, iterate_max, fitness):
    """
    OPSO算法:带有惯性权重的粒子群优化算法，惯性权重随迭代次数线性递减
    :param N: 粒子数目
    :param dim: 问题维度
    :param x_min: 搜索空间下限
    :param x_max: 搜索空间上限
    :param iterate_max: 最大迭代次数
    :param fitness: 评价函数
    :return: 返回当前搜索到的最佳适应值
    """

    # 计算求解区间
    m = x_max - x_min
    v_max = x_max * 0.2
    v_min = x_min * 0.2
    x = x_min + (x_max - x_min) * np.random.random([N, dim])
    v = v_min + (v_max - v_min) * np.random.random([N, dim])
    pBest = x
    # 获取初始时全局最优位置
    gBest = pBest[0, :]
    for i in range(1, N):
        if fitness(gBest) > fitness(pBest[i, :]):
            gBest = pBest[i, :]
    pBest_res = np.ones([N])
    iterate = 0
    while iterate < iterate_max:
        omega = 0.9 - 0.5 * (iterate / iterate_max)
        # 按照笔记中第一种方式计算距离
        g_pi = np.sum(pBest - x, axis=1)
        g_gi = np.sum(gBest - x, axis=1)
        # 按照第二种方式计算距离
        # g_pi = np.sum(pBest - x, axis=1) / dim
        # g_gi = np.sum(gBest - x, axis=1) / dim
        c_g_pi = Sigmoid(g_pi, m)
        c_g_gi = Sigmoid(g_gi, m)
        v = omega * v + c_g_pi * np.random.random([N, dim]) * (pBest - x) + c_g_gi * np.random.random([N, dim]) * (gBest - x)
        # 对速度或位置超出规定的做相应的纠正
        v[v > v_max] = v_max
        v[v < v_min] = v_min
        x = x + v
        x[x > x_max] = x_max
        x[x < x_min] = x_min
        # 更新各粒子的历史最优位置
        for i in range(N):
            if fitness(pBest[i, :]) > fitness(x[i, :]):
                pBest[i, :] = x[i, :]
            pBest_res[i] = fitness(pBest[i, :])
        # 更新全局最优位置
        if pBest_res.min() < fitness(gBest):
            index = np.where(pBest_res == pBest_res.min())
            gBest = pBest[index[0][0], :]

        iterate += 1
    res = fitness(gBest)
    return res


def Sigmoid(D, m):
    """
    Sigmoid：加速系数调整函数
    :param D: 各维度距离，可直接传入矩阵
    :param m: 求解区间
    :return:
        res:最终的计算结果，数据形状与D一致
    """
    # 初始化参数
    a = 0.000035 * m
    b = 0.5
    c = 0
    d = 1.5
    # 执行 sigmoid 函数
    res = b / (1 + np.e ** (-a * (D - c))) + d
    return res
