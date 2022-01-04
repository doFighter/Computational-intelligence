# coding：陈小斌
# GitHub：doFighter
import numpy as np


def PSO_LBEST_distance(N, dim, x_min, x_max, iterate_max, fitness):
    """
    二邻域结构：在本代码中，该邻域结构是距离相邻，即真正意义上的空间结构相邻(doi:10.1109/MHS.1995.494215)
    :param N: 粒子种群大小
    :param dim: 问题解的维度
    :param x_min: 解空间的下限
    :param x_max: 解空间的上限
    :param iterate_max: 迭代上限
    :param fitness: 评价函数
    :return: 迭代中适应度最优的值
    """
    c = 2 * np.ones([3, 1])
    v_max = 0.2 * x_max
    v_min = 0.2 * x_min
    x = x_min + (x_max - x_min) * np.random.random([N, dim])
    v = v_min + (v_max - v_min) * np.random.random([N, dim])
    pBest = x
    results = np.inf * np.ones([iterate_max, 1])
    res = np.inf * np.ones([N, 1])
    iterate = 0
    while iterate < iterate_max:
        for i in range(N):
            # 获取粒子群结构中空间相邻的二领域结构
            temp = np.delete(pBest, i, axis=0)
            distance = np.sum((pBest[i, :] - temp) ** 2, 1)
            sort_distance = np.sort(distance)
            position = np.where(distance <= sort_distance[1])
            position = position[0]
            pre = position[0]
            nex = position[1]

            v[i, :] = v[i, :] + c[0] * np.random.random([1, dim]) * (pBest[i, :] - x[i, :]) + c[1] * np.random.random(
                [1, dim]) * (pBest[pre, :] - x[i, :]) + c[2] * np.random.random([1, dim]) * (pBest[nex, :] - x[i, :])
        v[v > v_max] = v_max
        v[v < v_min] = v_min
        x += v
        x[x > x_max] = x_max
        x[x < x_min] = x_min
        for i in range(N):
            if fitness(x[i, :]) < fitness(pBest[i, :]):
                pBest[i, :] = x[i, :]
            res[i] = fitness(pBest[i, :])
        results[iterate] = min(res)
        iterate += 1
    return min(results)


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
result = PSO_LBEST_distance(20, 30, -10, 10, 10000, Sphere)
print(result)
