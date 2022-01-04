# coding：陈小斌
# GitHub：doFighter
import numpy as np


def OPSO(N, dim, x_min, x_max, iterate_max, fitness):
    """
    OPSO算法:带有惯性权重的粒子群优化算法，惯性权重随迭代次数线性递减((doi:10.1109/ICEC.1998.699146))
    :param N: 粒子数目
    :param dim: 问题维度
    :param x_min: 搜索空间下限
    :param x_max: 搜索空间上限
    :param iterate_max: 最大迭代次数
    :param fitness: 评价函数
    :return: 返回当前搜索到的最佳适应值
    """
    c1 = 2
    c2 = 2
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
        v = omega * v + c1 * \
            np.random.random([N, dim]) * (pBest - x) + c2 * \
            np.random.random([N, dim]) * (gBest - x)
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


def Levy(xx):
    d = len(xx)
    w = np.zeros(d)
    for ii in range(d):
        w[ii] = 1 + (xx[ii] - 1) / 4

    term1 = (np.sin(np.pi * w[1])) ** 2
    term3 = (w[d - 1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[d - 1])) ** 2)

    sum1 = 0
    for ii in range(d - 1):
        wi = w[ii]
        new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
        sum1 = sum1 + new

    y = term1 + sum1 + term3
    return y


# 函数测试
for i in range(10):
    result = OPSO(20, 30, -10, 10, 1000, Sphere)
    print(result)
