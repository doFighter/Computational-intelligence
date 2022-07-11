# coding：陈小斌
# GitHub：doFighter
import numpy as np


def Adaptive_C(c, h1, h2):
    """
    加速系数调整函数
    :param c:加速系数，一维数组，有两个元素
    :param h1:控制 c[0] 加速系数的增量因子
    :param h2:控制 c[1] 加速系数的增量因子
    :return:返回调整后的加速系数
    """
    c[0] = c[0] + h1 * (0.05 + 0.05 * np.random.random())
    c[1] = c[1] + h2 * (0.05 + 0.05 * np.random.random())
    c[c > 2.5] = 2.5
    c[c < 1.5] = 1.5
    sum_c = sum(c)
    # 归一化操作
    if sum_c < 3:
        c = (c * 3) / sum_c
    elif sum_c > 4:
        c = (c * 4) / sum_c
    return c


def Elitist_learning(fitness, pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index):
    """
    精英学习策略
    :param fitness: 适应度函数
    :param pBest: 粒子历史最优位置
    :param gBest: 全局历史最优位置
    :param dim: 问题的维度
    :param x_max: 问题求解空间上限
    :param x_min: 问题求解空间下限
    :param iterate: 当前迭代次数
    :param iterate_max: 最大迭代次数
    :param x: 粒子当前所处位置
    :param gWorst_index: 适应度最差的粒子的下标
    :return: [gBest, pBest, x]
    """
    mu = 0
    sigma_max = 1
    sigma_min = 0.1
    sigma = sigma_max - (sigma_max - sigma_min) * iterate / iterate_max
    # 需要注意，在 python 中，数组赋值是浅拷贝，即赋给对象的是该变量的引用，当变量变化时，则之前的数据也会发生改变，所以在这里应该十分小心
    p = gBest.copy()
    d = round(np.random.random() * (dim - 1))
    p[d] = p[d] + (x_max - x_min) * np.random.normal(mu, sigma)
    # 判断是否超出求解范围，若超出，则在解的范围内随机取值
    if (p[d] > x_max) | (p[d] < x_min):
        p[d] = x_min + (x_max - x_min) * np.random.random()
    p_value = fitness(p)
    if p_value < fitness(gBest):
        gBest = p.copy()
    else:
        if p_value < fitness(pBest[gWorst_index, :]):
            pBest[gWorst_index, :] = p.copy()
            x[gWorst_index, :] = p.copy()
        else:
            x[gWorst_index, :] = p.copy()
    return [gBest, pBest, x]


def Adaptive_Parameters(c, fitness, f, curState, lastState, pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index):
    """
    自适应调整参数，按照粒子群算法所处的不同阶段对粒子中的参数进行自适应
    :param c:   加速系数
    :param fitness: 评价函数
    :param f:   进化因子
    :param curState:    当前状态
    :param lastState:   上一状态
    :param pBest:   粒子的历史最优位置
    :param gBest:   粒子的全局最优位置
    :param dim: 所求解问题的维度
    :param x_max:   求解问题解空间的上限
    :param x_min:   求解问题解空间的下限
    :param iterate: 当前迭代次数
    :param iterate_max: 最大迭代次数
    :param x:   粒子当前所处位置
    :param gWorst_index:    最差粒子的下标
    :return:    [w, c, lastState, gBest, pBest, x]
    """
    w = 1 / (1 + 1.5 * np.exp(-2.6 * f))
    if f <= 0.2:
        c = Adaptive_C(c, 0.5, 0.5)
        [gBest, pBest, x] = Elitist_learning(
            fitness, pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index)
        curState = 3
    elif f <= 0.3:
        if (lastState == 3) | (lastState == 4):
            c = Adaptive_C(c, 0.5, 0.5)
            [gBest, pBest, x] = Elitist_learning(
                fitness, pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index)
            curState = 3
        elif (lastState == 2) | (lastState == 1):
            c = Adaptive_C(c, 0.5, -0.5)
            curState = 2
    elif f <= 0.4:
        c = Adaptive_C(c, 0.5, -0.5)
        curState = 2
    elif f <= 0.6:
        if (lastState == 2) | (lastState == 3):
            c = Adaptive_C(c, 0.5, -0.5)
            curState = 2
        elif (lastState == 1) | (lastState == 4):
            c = Adaptive_C(c, 1.0, -1.0)
            curState = 1
    elif f <= 0.7:
        c = Adaptive_C(c, 1.0, -1.0)
        curState = 1
    elif f <= 0.8:
        if (lastState == 1) | (lastState == 2):
            c = Adaptive_C(c, 1.0, -1.0)
            curState = 1
        elif (lastState == 4) | (lastState == 3):
            c = Adaptive_C(c, -1.0, 1.0)
            curState = 4
    else:
        c = Adaptive_C(c, -1.0, 1.0)
        curState = 4
    lastState = curState
    return [w, c, lastState, gBest, pBest, x]


def Evolutionary_States_Etimation(c, fitness, N, curBest_index, curState, lastState, pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index):
    """
    进化状态评估函数
    :param c: 加速因子
    :param fitness: 评价函数
    :param N: 粒子数目，即种群大小
    :param curState: 当前状态
    :param lastState: 上一状态
    :param pBest: 粒子历史最优位置
    :param gBest: 全局最优位置
    :param dim: 问题解的维度
    :param x_max: 问题解空间上限
    :param x_min: 问题解空间下限
    :param iterate: 当前迭代次数
    :param iterate_max: 总迭代次数
    :param x: 粒子当前所处位置
    :param gWorst_index: 最差适应度粒子下标
    :return: [w, c, lastState, gBest, pBest, x]
    """
    distance = np.zeros([N, 1])
    for i in range(N):
        sum_d = np.sum(((x[i, :] - x) ** 2), 1)
        distance[i] = np.sum(sum_d ** 0.5) / (N - 1)
    d_g = distance[curBest_index]
    d_max = max(distance)
    d_min = min(distance)
    if d_min == d_max:
        f = 1
    else:
        f = (d_g - d_min) / (d_max - d_min)
    [w, c, lastState, gBest, pBest, x] = Adaptive_Parameters(
        c, fitness, f, curState, lastState, pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index)
    return [w, c, lastState, gBest, pBest, x]


def APSO(N, dim, x_min, x_max, iterate_max, fitness):
    """
    APSO:自适应粒子群优化算法
    :param N: 种群大小
    :param dim: 问题解的维数
    :param x_max: 解空间上限
    :param x_min: 解空间下限
    :param iterate_max: 最大迭代次数
    :param fitness: 评价函数
    :return: rest，所查找到的最优适应值
    """
    c = np.ones([2, 1]) * 2
    fitness_value = np.ones([N, 1])
    curState = 1
    lastState = 1
    gBest_result = np.ones([iterate_max, 1])
    v_min = 0.2 * x_min
    v_max = 0.2 * x_max
    x = x_min + (x_max - x_min) * np.random.random([N, dim])
    v = v_min + (v_max - v_min) * np.random.random([N, dim])
    pBest = x
    curBest_index = 0
    gWorst_index = 0
    fitness_value[0] = fitness(x[0, :])
    for i in range(1, N):
        fitness_value[i] = fitness(x[i, :])
        if fitness(x[curBest_index, :]) > fitness_value[i]:
            curBest_index = i
        if fitness(x[gWorst_index, :]) < fitness_value[i]:
            gWorst_index = i
    gBest = x[curBest_index, :]
    iterate = 0
    while iterate < iterate_max:
        gBest_result[iterate] = fitness(gBest)
        # 进化状态评估
        [w, c, lastState, gBest, pBest, x] = Evolutionary_States_Etimation(
            c, fitness, N, curBest_index, curState, lastState, pBest, gBest, dim, x_max, x_min, iterate, iterate_max, x, gWorst_index)
        v = w * v + c[0] * np.random.random([N, dim]) * (
            pBest - x) + c[1] * np.random.random([N, dim]) * (gBest - x)
        v[v < v_min] = v_min
        v[v > v_max] = v_max
        x = x + v
        r = np.argwhere((x > x_max) | (x < x_min))
        r = np.unique(r[:, 0])
        for i in range(N):
            if i not in r:
                fitness_value[i] = fitness(x[i, :])
            if fitness_value[i] < fitness(pBest[i, :]):
                pBest[i, :] = x[i, :]
            if fitness_value[i] < fitness(gBest):
                gBest = x[i, :]
            if fitness_value[i] < fitness_value[curBest_index]:
                curBest_index = i
            if fitness(pBest[i, :]) > fitness(pBest[gWorst_index, :]):
                gWorst_index = i
        iterate += 1
    rest = fitness(gBest)
    return rest


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
result = APSO(20, 30, -10, 10, 1000, Sphere)
print(result)
