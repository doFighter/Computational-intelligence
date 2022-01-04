# coding：陈小斌
# GitHub：doFighter
import numpy as np


def CLPSO_Version2(N, dim, x_min, x_max, iterate_max, fitness):
    """
    CLPSO版本二：广泛学习粒子群优化,较版本一增加了当某个粒子所有维度都不进行学习时，就任意选择一个维度进行学习
    :param N: 粒子种群大小
    :param dim: 问题解的维度
    :param x_min: 求解问题的解空间下限
    :param x_max: 求解问题的解空间上限
    :param iterate_max: 最大迭代次数
    :param fitness: 评价函数，即求解函数
    :return: result，最优解
    """
    c = 2 * np.ones([2, 1])
    c1 = 1.49445
    Pc = np.zeros([N, 1])
    for i in range(N):
        Pc[i] = 0.05 + 0.45 * \
            (np.exp((10 * (i - 1) / (N - 1) - 1) / np.exp(10) - 1))
    flag = np.zeros([N, 1])
    m = 7
    v_max = 0.2 * x_max
    v_min = 0.2 * x_min
    x = x_min + (x_max - x_min) * np.random.random([N, dim])
    v = v_min + (v_max - v_min) * np.random.random([N, dim])
    pBest = x
    gBest = x[0, :].copy()
    for i in range(1, N):
        if fitness(gBest) > fitness(x[i, :]):
            gBest = x[i, :].copy()
    iterate = 0
    while iterate < iterate_max:
        omega = 0.9 - 0.5 * (iterate / iterate_max)
        for i in range(N):
            if flag[i] >= m:
                v[i, :] = omega * v[i, :] + c[0] * np.random.random([1, dim]) * (pBest[i, :] - x[i, :]) + c[
                    1] * np.random.random([1, dim]) * (gBest - x[i, :])
                flag[1] = 0
            pBest_fi = pBest[i, :]
            rd = np.random.random(dim)
            position = np.where(rd < Pc[i])
            position = position[0]
            if len(position) == 0:
                pBest_fi[int(np.ceil(np.random.rand() * (dim - 1)))] = pBest[int(np.ceil(
                    np.random.rand() * (N - 1))), int(np.ceil(np.random.rand() * (dim - 1)))]
            else:
                for j in position:
                    pBest_f1 = pBest[i, :].copy()
                    pBest_f2 = pBest[i, :].copy()
                    f1 = int(np.ceil(np.random.random() * (N - 1)))
                    f2 = int(np.ceil(np.random.random() * (N - 1)))
                    pBest_f1[j] = pBest[f1, j]
                    pBest_f2[j] = pBest[f2, j]
                    if fitness(pBest_f1) < fitness(pBest_f2):
                        pBest_fi[j] = pBest_f1[j]
                    else:
                        pBest_fi[j] = pBest_f2[j]
            v[i, :] = omega * v[i, :] + c1 * \
                np.random.random([1, dim]) * (pBest_fi - x[i, :])
            vi = v[i, :]
            vi[vi > v_max] = v_max
            vi[vi < v_min] = v_min
            x[i, :] = x[i, :] + vi
            xi = x[i, :]
            if len(xi[(xi > x_max) | (xi < x_min)]) == 0:
                if fitness(xi) < fitness(pBest[i, :]):
                    pBest[i, :] = xi.copy()
                    flag[i] = 0
                if fitness(xi) < fitness(gBest):
                    gBest = xi.copy()
            else:
                flag[i] += 1
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
result = CLPSO_Version2(20, 30, -10, 10, 1000, Sphere)
print(result)
