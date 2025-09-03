from typing import NamedTuple, List, Tuple
import jax
from jax import jit, vmap, random, tree_util
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

# 配置JAX
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# 定义动态规划状态类
class GridDPState(NamedTuple):
    grid: jnp.ndarray  # 网格，包含每个位置的代价
    
    @classmethod
    def create(cls, grid, dtype=jnp.float32):
        return cls(grid=jnp.array(grid, dtype=dtype))

# JAX实现的动态规划求解最短路径
@jit
def min_path_dp_jax(state: GridDPState):
    grid = state.grid
    m, n = grid.shape
    
    # 初始化DP表
    dp = jnp.zeros((m, n), dtype=grid.dtype)
    dp = dp.at[0, 0].set(grid[0, 0])
    
    # 初始化第一行和第一列
    for j in range(1, n):
        dp = dp.at[0, j].set(dp[0, j-1] + grid[0, j])
    for i in range(1, m):
        dp = dp.at[i, 0].set(dp[i-1, 0] + grid[i, 0])
    
    # 填充DP表
    for i in range(1, m):
        for j in range(1, n):
            dp = dp.at[i, j].set(jnp.minimum(dp[i-1, j], dp[i, j-1]) + grid[i, j])
    
    return dp[m-1, n-1], dp

# 生成多个随机网格
def generate_random_grids(num_grids, size, seed=42):
    """生成多个随机网格"""
    rng_key = random.PRNGKey(seed)
    keys = random.split(rng_key, num_grids)
    
    # 定义生成单个网格的函数
    def generate_single_grid(key):
        return random.randint(key, shape=(size, size), minval=1, maxval=10)
    
    # 使用 vmap 并行应用到所有键
    grids = vmap(generate_single_grid)(keys)
    return grids

# 可视化路径
def plot_grid_path(grid, dp):
    m, n = grid.shape
    path = [(m-1, n-1)]
    i, j = m-1, n-1
    
    while (i, j) != (0, 0):
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if dp[i-1, j] < dp[i, j-1]:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    
    path.reverse()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='viridis')
    y, x = zip(*path)
    plt.plot(x, y, 'r-', linewidth=2, marker='o')
    plt.title("最短路径可视化")
    plt.colorbar(label="格子代价")
    plt.show()
    
    return path

def benchmark_dp(num_grids=10, grid_size=100, run_numpy=True, visualize=True):
    print(f"生成 {num_grids} 个 {grid_size}x{grid_size} 的随机网格...")
    grids = generate_random_grids(num_grids, grid_size)
    
    # 创建状态并定义批处理函数
    states = tree_util.tree_map(GridDPState.create, grids)
    batch_dp = jit(vmap(min_path_dp_jax))
    
    # JIT预热
    print("\n预热JAX JIT编译...")
    start_jit = time.time()
    min_sums, dp_tables = batch_dp(states)
    min_sums.block_until_ready()
    jit_time = time.time() - start_jit
    print(f"JIT编译时间: {jit_time:.6f} 秒")
    
    # 实际计时
    print(f"\n并行求解 {num_grids} 个 {grid_size}x{grid_size} 网格...")
    start_time = time.time()
    min_sums, dp_tables = batch_dp(states)
    min_sums.block_until_ready()
    jax_time = time.time() - start_time
    print(f"JAX总用时: {jax_time:.6f} 秒")
    print(f"JAX平均每网格: {jax_time/num_grids:.6f} 秒")
    
    results = {
        'jax': {
            'min_sums': min_sums,
            'dp_tables': dp_tables,
            'time': jax_time,
            'jit_time': jit_time
        }
    }
    
    # NumPy对比测试
    if run_numpy:
        print("\n使用NumPy版本(单网格)...")
        np_grid = np.array(grids[0])
        start_time = time.time()
        np_min_sum, np_dp = np_min_path_sum(np_grid)
        np_time = time.time() - start_time
        print(f"NumPy单网格用时: {np_time:.6f} 秒")
        print(f"NumPy结果: {np_min_sum:.2f}, JAX结果: {float(min_sums[0]):.2f}")
        
        results['numpy'] = {
            'min_sum': np_min_sum,
            'dp_table': np_dp,
            'time': np_time
        }
        
        # 估算加速比
        estimated_np_total = np_time * num_grids
        speedup = estimated_np_total / jax_time
        print(f"\n估计加速比: {speedup:.2f}x")
    
    # 可视化
    if visualize:
        print("\n可视化第一个网格的最短路径...")
        path = plot_grid_path(np.array(grids[0]), np.array(dp_tables[0]))
    
    # 打印部分结果
    print("\n部分结果:")
    for i in range(min(5, num_grids)):
        print(f"网格 {i+1} 最短路径和: {float(min_sums[i]):.2f}")
    
    return results

# NumPy版本的DP函数，用于对比
def np_min_path_sum(grid):
    m, n = grid.shape
    dp = np.zeros((m, n), dtype=grid.dtype)
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

if __name__ == "__main__":
    # 小规模测试
    print("运行小规模测试...")
    benchmark_dp(num_grids=10, grid_size=50)
    
    # 中规模测试
    print("\n运行中规模测试...")
    benchmark_dp(num_grids=50, grid_size=100)
    
    # 大规模测试
    print("\n运行大规模测试...")
    benchmark_dp(num_grids=100, grid_size=200, visualize=False)