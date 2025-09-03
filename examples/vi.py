import numpy as np

def value_iteration_with_discount(grid, gamma=0.9, theta=1e-6, max_iterations=10000):
    """
    值迭代使用折扣因子 gamma，但最终输出真实最小路径和（整数）
    """
    m, n = grid.shape
    V = np.full((m, n), np.inf)
    V[-1, -1] = grid[-1, -1]

    # 值迭代过程（折扣 gamma）
    for it in range(max_iterations):
        prev_V = V.copy()
        for i in reversed(range(m)):
            for j in reversed(range(n)):
                if (i, j) == (m-1, n-1):
                    continue
                candidates = []
                if i+1 < m:
                    candidates.append(V[i+1, j])
                if j+1 < n:
                    candidates.append(V[i, j+1])
                if candidates:
                    V[i, j] = grid[i, j] + gamma * min(candidates)
        if np.max(np.abs(prev_V - V)) < theta:
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


# 测试
grid = np.array([[1,3,1,2],
                 [1,5,1,3],
                 [4,2,1,1],
                 [2,1,2,1]], dtype=float)

grid = np.random.rand(50, 50) * 10
grid = np.round(grid)

gamma = 0.9
discounted_cost, real_cost, V_discount, V_real = value_iteration_with_discount(grid, gamma=gamma)

print("折扣后的 V0:", discounted_cost)
print("真实最小路径和:", real_cost)
print("折扣后的 V 表:\n", V_discount)
print("真实 V 表:\n", V_real)
