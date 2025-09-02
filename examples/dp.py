import numpy as np

def min_path_sum(grid):
    """
    动态规划求解从左上到右下的最短路径和（只能向右或向下走）
    grid: 2D numpy array
    """
    m, n = grid.shape
    dp = np.zeros((m, n), dtype=int)
    dp[0, 0] = grid[0, 0]
    # 初始化第一行和第一列
    for i in range(1, m):
        dp[i, 0] = dp[i-1, 0] + grid[i, 0]
    for j in range(1, n):
        dp[0, j] = dp[0, j-1] + grid[0, j]
    # 填充DP表
    for i in range(1, m):
        for j in range(1, n):
            dp[i, j] = min(dp[i-1, j], dp[i, j-1]) + grid[i, j]
    return dp[m-1, n-1], dp



def value_iteration_min_path(grid, gamma=1.0, theta=1e-6, max_iterations=10000):
    """
    用值迭代求解从左上到右下的最短路径和（只能向右或向下走）
    """
    m, n = grid.shape
    V = np.full((m, n), np.inf)
    V[-1, -1] = grid[-1, -1]  # 终点的值就是自身代价

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
                    V[i, j] = grid[i, j] + min(candidates)
        if np.max(np.abs(prev_V - V)) < theta:
            print(f"值迭代在第 {it+1} 次迭代后收敛")
            break
    return V[0, 0], V

if __name__ == "__main__":
    grid = np.array([
        [1, 3, 1, 2],
        [1, 5, 1, 3],
        [4, 2, 1, 1],
        [2, 1, 2, 1]
    ])
    min_sum, V_table = value_iteration_min_path(grid)
    print("最短路径和（值迭代）:", min_sum)
    print("值函数表:")
    print(V_table)
    min_sum, dp_table = min_path_sum(grid)
    print("最短路径和:", min_sum)
    print("DP表:")
    print(dp_table)