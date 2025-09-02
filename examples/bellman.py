import numpy as np

def visualize_frozen_lake(env, optimal_policy=None):
    """可视化 FrozenLake 环境及最优策略"""
    desc = env.unwrapped.desc
    print("FrozenLake-v1 环境:")
    print("S: 起点, F: 冰面(安全), H: 洞(危险), G: 目标\n")
    for row in desc:
        row_str = []
        for cell in row:
            if isinstance(cell, (bytes, bytearray)):
                row_str.append(cell.decode('utf-8'))
            else:
                row_str.append(str(cell))
        print(" ".join(row_str))
    print("\n智能体需要从起点S导航到目标G，同时避开洞H。")
    print("可用动作: 左(0), 下(1), 右(2), 上(3)")
    if hasattr(env.unwrapped, 'is_slippery'):
        if not env.unwrapped.is_slippery:
            print("当前设置: 冰面不滑 (确定性环境)")
        else:
            print("当前设置: 冰面会滑 (概率性环境)")
    if optimal_policy is not None:
        grid_size = int(np.sqrt(len(optimal_policy)))
        policy_arrows = ['←', '↓', '→', '↑']
        policy_visualization = np.array([policy_arrows[a] for a in optimal_policy]).reshape(grid_size, grid_size)
        print("\n最优策略路径:")
        for i, row in enumerate(desc):
            row_str = ""
            row_chars = []
            for cell in row:
                if isinstance(cell, (bytes, bytearray)):
                    row_chars.append(cell.decode('utf-8'))
                else:
                    row_chars.append(str(cell))
            for j, cell in enumerate(row_chars):
                if cell == 'S' or cell == 'G' or cell == 'H':
                    row_str += f" {cell} "
                else:
                    row_str += f" {policy_visualization[i, j]} "
            print(row_str)

def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=10000):
    """Bellman值迭代"""
    env_unwrapped = env.unwrapped
    nS = env.observation_space.n
    nA = env.action_space.n
    V = np.zeros(nS)
    for i in range(max_iterations):
        prev_V = np.copy(V)
        for s in range(nS):
            A = np.zeros(nA)
            for a in range(nA):
                for prob, next_state, reward, done in env_unwrapped.P[s][a]:
                    A[a] += prob * (reward + gamma * prev_V[next_state] * (not done))
            V[s] = max(A)
        if np.max(np.abs(prev_V - V)) < theta:
            break
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env_unwrapped.P[s][a]:
                A[a] += prob * (reward + gamma * V[next_state] * (not done))
        policy[s] = np.argmax(A)
    return V, policy

def q_learning(env, num_episodes=2000, alpha=0.8, gamma=0.99, epsilon=0.1, max_steps=100):
    """Q-Learning 算法"""
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    for episode in range(num_episodes):
        state, _ = env.reset()
        for t in range(max_steps):
            # ε-greedy 策略
            if np.random.rand() < epsilon:
                action = np.random.choice(nA)
            else:
                action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            if done:
                break
    # 从Q表提取策略
    policy = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    return V, policy

if __name__ == "__main__":
    try:
        import gymnasium as gym
        print("使用 Gymnasium")
    except ImportError:
        import gym
        print("使用旧版 Gym (不推荐)")

    env = gym.make('FrozenLake-v1', is_slippery=False)
    print("\n=== Bellman值迭代 ===")
    vi_V, vi_policy = value_iteration(env)
    print("最优价值函数(VI):")
    print(vi_V.reshape(4, 4))
    print("\n最优策略(VI):")
    print(vi_policy.reshape(4, 4))
    visualize_frozen_lake(env, vi_policy)

    print("\n=== Q-Learning ===")
    ql_V, ql_policy = q_learning(env, num_episodes=50000, alpha=0.5, epsilon=0.5, max_steps=200)
    print("最优价值函数(VI):")
    print(ql_V.reshape(4, 4))
    print("\n最优策略(Q-Learning):")
    print(ql_policy.reshape(4, 4))
    visualize_frozen_lake(env, ql_policy)

    print("\n对比说明：")
    print("""
- Bellman值迭代(Value Iteration)属于动态规划，需要已知环境模型(P、R)，直接计算最优值函数和策略，收敛快，适合小型离散环境。
- Q-Learning属于无模型强化学习，通过与环境交互采样学习Q值，不需要知道转移概率，适合实际不可知模型的场景。
- 在FrozenLake-v1这类小型确定性环境下，两者最终策略通常一致，但Q-Learning收敛速度依赖采样次数和探索策略。
""")