import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

print("JAX devices:", jax.devices())

def benchmark_numpy(N, runs=3):
    x = np.random.rand(N, N).astype(np.float32)
    y = np.random.rand(N, N).astype(np.float32)
    times = []
    for _ in range(runs):
        start = time.time()
        np.dot(x, y)
        end = time.time()
        times.append(end - start)
    return np.mean(times)

def benchmark_jax(N, runs=3):
    x = jnp.array(np.random.rand(N, N).astype(np.float32))
    y = jnp.array(np.random.rand(N, N).astype(np.float32))
    # 第一次编译预热，不计入
    jnp.dot(x, y).block_until_ready()
    times = []
    for _ in range(runs):
        start = time.time()
        jnp.dot(x, y).block_until_ready()
        end = time.time()
        times.append(end - start)
    return np.mean(times)

# 不同矩阵规模
sizes = [500, 1000, 2000, 3000, 4000]

times_numpy = []
times_jax = []

for N in sizes:
    print(f"\nBenchmarking N={N}")
    t_np = benchmark_numpy(N)
    t_jax = benchmark_jax(N)
    print(f"NumPy: {t_np:.4f} s | JAX (GPU): {t_jax:.4f} s")
    times_numpy.append(t_np)
    times_jax.append(t_jax)

# 可视化
plt.plot(sizes, times_numpy, marker="o", label="NumPy (CPU)", color="gray")
plt.plot(sizes, times_jax, marker="s", label="JAX (GPU)", color="green")

plt.xlabel("Matrix size (N x N)")
plt.ylabel("Average time (s)")
plt.title("Matrix Multiplication Performance: NumPy vs JAX")
plt.legend()
plt.grid(True)
plt.show()
