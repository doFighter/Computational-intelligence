#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/8/16 9:24
# @Author : doFighter
import numpy as np


def chaos(m, n, S):
    """
    混沌时间序列随机数生成器
    :param m: 行数指标
    :param n: 列数指标
    :param S: 混沌时间序列
    :return: 返回生成的随机数组
    """
    index = np.ceil(np.random.rand(m, n) * 99).astype(int)
    res = S[index]
    return res


def TDPSO(N, dim, x_min, x_max, iterate_max, fitness):
    """
    完全受扰粒子群优化算法
    :param N: 种群数目
    :param dim: 问题维度
    :param x_min: 解空间下限
    :param x_max: 解空间上限
    :param iterate_max: 最大迭代次数
    :param fitness: 适应度值
    :return: 最优适应值
    """
    # 第一步，生成混沌时间序列
    a = 1.4
    b = 0.3
    x = np.zeros(100)
    S = np.zeros(100)
    x[0] = 0
    S[0] = 0
    for i in range(1, 100):
        x[i] = S[i - 1] + 1 - a * x[i - 1] ** 2
        S[i] = b * x[i - 1]
    # 对混沌时间序列进行归一化操作
    S = S - min(S)
    S = S / max(S)

    c = [2.8, 1.3]
    x = x_min + (x_max - x_min) * chaos(N, dim, S)
    v_rand = np.random.rand(N, dim)
    v = (x_min - chaos(N, dim, S)) / (x_max - chaos(N, dim, S))
    v[v_rand >= 0.5] = abs(v[v_rand >= 0.5])
    pBest = x
    # 获取初始时全局最优位置
    gBest = pBest[0, :]
    for i in range(1, N):
        if fitness(gBest) > fitness(pBest[i, :]):
            gBest = pBest[i, :]
    iterate = 0
    pBest_res = np.ones([N])
    while iterate < iterate_max:
        omega = np.power(0.5, iterate + 1) + 0.4
        v = omega * v + c[0] * np.random.rand(N, dim) * (
            pBest - x) + c[1] * np.random.rand(N, dim) * (gBest - x)
        x += v
        x[x > x_max] = x_max
        x[x < x_min] = x_min
        if iterate > iterate_max * 0.7:
            v_max = np.max(v, 0)
            # 在python中会出现数字太小，而导致除法运算的警告，因此在本算法中，对绝对值过小的数统一赋值
            v_max[(v_max > 0) & (v_max < np.exp(-60))] = np.exp(-60)
            v_max[(v_max < 0) & (v_max > -np.exp(-60))] = -np.exp(-60)
            # v_max[abs(v_max) < np.exp(-30)] = np.exp(-30)
            RVC = v / v_max
            MAX_RVC = (np.max(RVC.T, 0)).T
            position = np.array(np.where(MAX_RVC <= 0.5))[0]
            cap = np.size(position)
            for i in range(cap):
                d = np.random.choice(30, int(dim * 0.5), replace=False)
                for j in range(int(dim * 0.5)):
                    flag = np.random.rand()
                    if flag > 0.5:
                        x[position[i], d[j]] = chaos(
                            1, 1, S)[0] + x[position[i], d[j]]
                    else:
                        x[position[i], d[j]] = chaos(
                            1, 1, S)[0] - x[position[i], d[j]]
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
    result = fitness(gBest)
    return result


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
    result = TDPSO(20, 30, -10, 10, 1000, Sphere)
    print(result)
