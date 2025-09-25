import numpy as np

def value_iteration_with_discount_test1(grid, gamma=0.9, theta=1e-6, max_iterations=10000):
    """
    值迭代使用折扣因子 gamma，但最终输出真实最小路径和（整数）
    """
    m, n = grid.shape
    V = np.full((m, n), np.inf)
    # grid代表地图
    # V代表从当前位置到终点的最小路径和
    # V的初始值为无穷大，代表不确定能否到达终点
    # print("1. 初始化grid和V")
    # print("grid = ", grid)
    # print("V = ", V)

    # V代表从当前位置到终点的最小折扣路径和
    # 初始时，终点的值为grid的值
    # 因为从终点到终点的路径和就是grid的值
    # print("2. 设置终点的值")
    V[-1, -1] = grid[-1, -1]
    # print("V = ", V)

    # 值迭代过程（折扣 gamma）
    # print("3. 开始值迭代")
    for it in range(max_iterations):
        # 快照
        prev_V = V.copy()
        # 最后一行
        for i in reversed(range(m)):
        # for i in (range(m)):
            # 最后一列
            for j in reversed(range(n)):
                # 跳过终点
                if (i, j) == (m-1, n-1):
                    continue
                candidates = []
                if i+1 < m:
                    # 下方的值
                    candidates.append(V[i+1, j])
                if j+1 < n:
                    # 右方的值
                    candidates.append(V[i, j+1])
                if candidates:
                    # 状态转移，取下方和右方的最小值，加上当前格子的值，并乘以折扣因子
                    V[i, j] = grid[i, j] + gamma * min(candidates)
        # 打印每次迭代的结果
        # print(f"迭代 {it+1}: V = \n{V}")
        if np.max(np.abs(prev_V - V)) < theta:
            # 如果变化小于阈值，认为收敛，停止迭代
            print("收敛，停止迭代,iteration =", it+1)
            break


    discounted_V0 = V[0,0]

    # 最终输出真实整数最小路径和（不折扣）
    V_real = np.full((m, n), np.inf)
    V_real[-1, -1] = grid[-1, -1]
    for i in reversed(range(m)):
        for j in reversed(range(n)):
            if (i, j) == (m-1, n-1):
                continue
            candidates = []
            if i+1 < m:
                candidates.append(V_real[i+1, j])
            if j+1 < n:
                candidates.append(V_real[i, j+1])
            if candidates:
                V_real[i, j] = grid[i, j] + min(candidates)

    real_min_cost = V_real[0,0]

    return discounted_V0, real_min_cost, V, V_real

def value_iteration_with_discount_test2(grid, gamma=0.9, theta=1e-6, max_iterations=10000):
    """
    值迭代使用折扣因子 gamma，但最终输出真实最小路径和（整数）
    """
    m, n = grid.shape
    V = np.full((m, n), np.inf)
    # grid代表地图
    # V代表从当前位置到终点的最小路径和
    # V的初始值为无穷大，代表不确定能否到达终点
    # print("1. 初始化grid和V")
    # print("grid = ", grid)
    # print("V = ", V)

    # V代表从当前位置到终点的最小折扣路径和
    # 初始时，终点的值为grid的值
    # 因为从终点到终点的路径和就是grid的值
    # print("2. 设置终点的值")
    V[-1, -1] = grid[-1, -1]
    # print("V = ", V)

    # 值迭代过程（折扣 gamma）
    # print("3. 开始值迭代")
    for it in range(max_iterations):
        # 快照
        prev_V = V.copy()
        # 最后一行
        # for i in reversed(range(m)):
        for i in (range(m)):
            # 最后一列
            for j in reversed(range(n)):
            # for j in (range(n)):
                # 跳过终点
                if (i, j) == (m-1, n-1):
                    continue
                candidates = []
                if i+1 < m:
                    # 下方的值
                    candidates.append(V[i+1, j])
                if j+1 < n:
                    # 右方的值
                    candidates.append(V[i, j+1])
                if candidates:
                    # 状态转移，取下方和右方的最小值，加上当前格子的值，并乘以折扣因子
                    V[i, j] = grid[i, j] + gamma * min(candidates)
        # 打印每次迭代的结果
        # print(f"迭代 {it+1}: V = \n{V}")
        if np.max(np.abs(prev_V - V)) < theta:
            # 如果变化小于阈值，认为收敛，停止迭代
            print("收敛，停止迭代,iteration =", it+1)
            break


    discounted_V0 = V[0,0]

    # 最终输出真实整数最小路径和（不折扣）
    V_real = np.full((m, n), np.inf)
    V_real[-1, -1] = grid[-1, -1]
    for i in reversed(range(m)):
        for j in reversed(range(n)):
            if (i, j) == (m-1, n-1):
                continue
            candidates = []
            if i+1 < m:
                candidates.append(V_real[i+1, j])
            if j+1 < n:
                candidates.append(V_real[i, j+1])
            if candidates:
                V_real[i, j] = grid[i, j] + min(candidates)

    real_min_cost = V_real[0,0]

    return discounted_V0, real_min_cost, V, V_real

def value_iteration_with_discount_test3(grid, gamma=0.9, theta=1e-6, max_iterations=10000):
    """
    值迭代使用折扣因子 gamma，但最终输出真实最小路径和（整数）
    """
    m, n = grid.shape
    V = np.full((m, n), np.inf)
    # grid代表地图
    # V代表从当前位置到终点的最小路径和
    # V的初始值为无穷大，代表不确定能否到达终点
    # print("1. 初始化grid和V")
    # print("grid = ", grid)
    # print("V = ", V)

    # V代表从当前位置到终点的最小折扣路径和
    # 初始时，终点的值为grid的值
    # 因为从终点到终点的路径和就是grid的值
    # print("2. 设置终点的值")
    V[-1, -1] = grid[-1, -1]
    # print("V = ", V)

    # 值迭代过程（折扣 gamma）
    # print("3. 开始值迭代")
    for it in range(max_iterations):
        # 快照
        prev_V = V.copy()
        # 最后一行
        for i in reversed(range(m)):
        # for i in (range(m)):
            # 最后一列
            # for j in reversed(range(n)):
            for j in (range(n)):
                # 跳过终点
                if (i, j) == (m-1, n-1):
                    continue
                candidates = []
                if i+1 < m:
                    # 下方的值
                    candidates.append(V[i+1, j])
                if j+1 < n:
                    # 右方的值
                    candidates.append(V[i, j+1])
                if candidates:
                    # 状态转移，取下方和右方的最小值，加上当前格子的值，并乘以折扣因子
                    V[i, j] = grid[i, j] + gamma * min(candidates)
        # 打印每次迭代的结果
        # print(f"迭代 {it+1}: V = \n{V}")
        if np.max(np.abs(prev_V - V)) < theta:
            # 如果变化小于阈值，认为收敛，停止迭代
            print("收敛，停止迭代,iteration =", it+1)
            break


    discounted_V0 = V[0,0]

    # 最终输出真实整数最小路径和（不折扣）
    V_real = np.full((m, n), np.inf)
    V_real[-1, -1] = grid[-1, -1]
    for i in reversed(range(m)):
        for j in reversed(range(n)):
            if (i, j) == (m-1, n-1):
                continue
            candidates = []
            if i+1 < m:
                candidates.append(V_real[i+1, j])
            if j+1 < n:
                candidates.append(V_real[i, j+1])
            if candidates:
                V_real[i, j] = grid[i, j] + min(candidates)

    real_min_cost = V_real[0,0]

    return discounted_V0, real_min_cost, V, V_real

def value_iteration_with_discount_test4(grid, gamma=0.9, theta=1e-6, max_iterations=10000):
    """
    值迭代使用折扣因子 gamma，但最终输出真实最小路径和（整数）
    """
    m, n = grid.shape
    V = np.full((m, n), np.inf)
    # grid代表地图
    # V代表从当前位置到终点的最小路径和
    # V的初始值为无穷大，代表不确定能否到达终点
    # print("1. 初始化grid和V")
    # print("grid = ", grid)
    # print("V = ", V)

    # V代表从当前位置到终点的最小折扣路径和
    # 初始时，终点的值为grid的值
    # 因为从终点到终点的路径和就是grid的值
    # print("2. 设置终点的值")
    V[-1, -1] = grid[-1, -1]
    # print("V = ", V)

    # 值迭代过程（折扣 gamma）
    # print("3. 开始值迭代")
    for it in range(max_iterations):
        # 快照
        prev_V = V.copy()
        # 最后一行
        # for i in reversed(range(m)):
        for i in (range(m)):
            # 最后一列
            # for j in reversed(range(n)):
            for j in (range(n)):
                # 跳过终点
                if (i, j) == (m-1, n-1):
                    continue
                candidates = []
                if i+1 < m:
                    # 下方的值
                    candidates.append(V[i+1, j])
                if j+1 < n:
                    # 右方的值
                    candidates.append(V[i, j+1])
                if candidates:
                    # 状态转移，取下方和右方的最小值，加上当前格子的值，并乘以折扣因子
                    V[i, j] = grid[i, j] + gamma * min(candidates)
        # 打印每次迭代的结果
        # print(f"迭代 {it+1}: V = \n{V}")
        if np.max(np.abs(prev_V - V)) < theta:
            # 如果变化小于阈值，认为收敛，停止迭代
            print("收敛，停止迭代,iteration =", it+1)
            break


    discounted_V0 = V[0,0]

    # 最终输出真实整数最小路径和（不折扣）
    V_real = np.full((m, n), np.inf)
    V_real[-1, -1] = grid[-1, -1]
    for i in reversed(range(m)):
        for j in reversed(range(n)):
            if (i, j) == (m-1, n-1):
                continue
            candidates = []
            if i+1 < m:
                candidates.append(V_real[i+1, j])
            if j+1 < n:
                candidates.append(V_real[i, j+1])
            if candidates:
                V_real[i, j] = grid[i, j] + min(candidates)

    real_min_cost = V_real[0,0]

    return discounted_V0, real_min_cost, V, V_real


# 最短路径和
grid = np.array([[1,3,1,2],
                 [1,5,1,3],
                 [4,2,1,1],
                 [2,1,2,1]], dtype=float)

grid = np.random.rand(30, 70) * 10
# grid = np.round(grid)

gamma = 0.9
discounted_cost, real_cost, V_discount, V_real = value_iteration_with_discount_test1(grid, gamma=gamma)
print("真实最小路径和:", real_cost)

gamma = 0.9
discounted_cost, real_cost, V_discount, V_real = value_iteration_with_discount_test2(grid, gamma=gamma)
print("真实最小路径和:", real_cost)

gamma = 0.9
discounted_cost, real_cost, V_discount, V_real = value_iteration_with_discount_test3(grid, gamma=gamma)
print("真实最小路径和:", real_cost)

gamma = 0.9
discounted_cost, real_cost, V_discount, V_real = value_iteration_with_discount_test4(grid, gamma=gamma)
print("真实最小路径和:", real_cost)
