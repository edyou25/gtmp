import jax
from jax import jit, vmap, random
import hydra
import omegaconf
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

from chrono import Timer
from gtmp.files import get_configs_path, get_data_path
from gtmp.planners import GTMPState, gtmp_plan, gtmp_akima_plan
from gtmp.objectives.occupancy_map import OccupancyMap
from gtmp.metrics import compute_metrics

import logging
logging.getLogger("jax").setLevel(logging.WARNING) # debug




jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_occupancy")
def main(cfg: omegaconf.DictConfig):
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)

    # Environment
    occ = 1. - jnp.load((get_data_path() / 'real_map' / str(cfg.environment.map_file)).as_posix())
    limits = jnp.array(cfg.environment.limits)
    q = jnp.array(cfg.environment.start_state)
    goals = jnp.array(cfg.environment.goal_state)[None, :]
    occ_map = OccupancyMap.from_prob(occ, limits=limits, threshold=0.1, infinite_cost=True)
    # occ 变量类型
    print(f"Occupancy map shape: {occ.shape}, dtype: {occ.dtype}")
    # planner
    planner_state = GTMPState.create(
        q=q,
        goals=goals,
        bounds=limits,
        transition_field=occ_map,
        occ_map=occ_map,
        get_velocity=False,
        **cfg.planner.params
    )
    # print(type(planner_state))
    # # planner_state attr
    # for attr in dir(planner_state):
    #     if not attr.startswith('_'):
    #         print(f"{attr}: {type(getattr(planner_state, attr))}")
    


    # warmup
    cfg.planner.name = 'akima'
    if cfg.planner.name == 'straight':
        gtmp = jit(vmap(gtmp_plan, in_axes=(0, None)))
    elif cfg.planner.name == 'akima':
        gtmp = jit(vmap(gtmp_akima_plan, in_axes=(0, None)))
    # plan
    keys = jax.random.split(rng_key, cfg.num_plans)
    start = time.time()
    paths = gtmp(keys, planner_state)
    jax_compile_time = time.time() - start
    print(f"JIT Time taken: {time.time() - start} seconds")
    with Timer() as timer:    
        paths = gtmp(keys, planner_state)
    metrics = compute_metrics(paths)
    
    print("--- GTMP with Occupancy Map ---")
    print(f"Time taken: {timer.elapsed} seconds")
    print(f"Collision-free percentage: {metrics[0]}")
    print(f"Averaged path length: {metrics[1]}")
    print(f"Path Diversities: {metrics[2]}")
    print(f"Min Cosin: {metrics[3]}")
    print(f"Mean Cosin: {metrics[4]}")

    # fig, ax = plt.subplots()
    # X = jnp.linspace(*limits[0], occ.shape[0])
    # Y = jnp.linspace(*limits[1], occ.shape[1])
    # X, Y = jnp.meshgrid(X, Y)
    # # ax.contourf(X, Y, occ.T, cmap='Greys')
    # ax.contourf(X, Y, occ_map.map.T, cmap='Greys')
    # free_path = paths.path[~paths.collision]
    # # path_vel = paths.path_vel[~paths.collision]   # getting path velocities here from Akima splines (which are constant vel per segments)
    # coll_path = paths.path[paths.collision]
    # for i in range(free_path.shape[0]):
    #     ax.plot(free_path[i, :, 0], free_path[i, :, 1], 'bo--', linewidth=1, markersize=1, alpha=0.7)
    # for i in range(coll_path.shape[0]):
    #     ax.plot(coll_path[i, :, 0], coll_path[i, :, 1], 'ro--', linewidth=1, alpha=0.3)
    # ax.plot(q[0], q[1], 'ro', markersize=5)
    # ax.plot(goals[0, 0], goals[0, 1], 'go', markersize=5)
    # ax.set_axis_off()
    # ax.set_aspect('equal')
    # fig.tight_layout(pad=0)
    # plt.show()


    if cfg.planner.name == 'straight':
        single_plan = gtmp_plan
    elif cfg.planner.name == 'akima':
        single_plan = gtmp_akima_plan

    # 用普通 for 循环批量生成路径
    with Timer() as timer_for:

        all_paths = []
        for key in jax.random.split(rng_key, cfg.num_plans):
            path = single_plan(key, planner_state)
            all_paths.append(path)

    # 将 all_paths 转为和批量输出一致的 GTMPOutput 结构
    from gtmp.planners import GTMPOutput
    # import jax

    # 合并各字段
    paths_obj = GTMPOutput(
        path=jax.numpy.stack([p.path for p in all_paths]),
        path_vel=None if all_paths[0].path_vel is None else jax.numpy.stack([p.path_vel for p in all_paths]),
        goal_idx=jax.numpy.array([p.goal_idx for p in all_paths]),
        collision=jax.numpy.array([p.collision for p in all_paths]),
        dream_points=None,
        splines=None,
        V=None
    )

    metrics = compute_metrics(paths_obj)
    print(type(paths))
    print(type(all_paths))
    print("--- For loop results ---")
    print(f"Time taken: {timer_for.elapsed} seconds")
    print(f"Collision-free percentage: {metrics[0]}")
    print(f"Averaged path length: {metrics[1]}")
    print(f"Path Diversities: {metrics[2]}")
    print(f"Min Cosin: {metrics[3]}")
    print(f"Mean Cosin: {metrics[4]}")


    fig, ax = plt.subplots()
    X = jnp.linspace(*limits[0], occ.shape[0])
    Y = jnp.linspace(*limits[1], occ.shape[1])
    X, Y = jnp.meshgrid(X, Y)
    # ax.contourf(X, Y, occ.T, cmap='Greys')
    ax.contourf(X, Y, occ_map.map.T, cmap='Greys')
    free_path = paths.path[~paths.collision]
    # path_vel = paths.path_vel[~paths.collision]   # getting path velocities here from Akima splines (which are constant vel per segments)
    coll_path = paths.path[paths.collision]
    for i in range(free_path.shape[0]):
        ax.plot(free_path[i, :, 0], free_path[i, :, 1], 'bo--', linewidth=1, markersize=1, alpha=0.7)
    for i in range(coll_path.shape[0]):
        ax.plot(coll_path[i, :, 0], coll_path[i, :, 1], 'ro--', linewidth=1, alpha=0.3)
    ax.plot(q[0], q[1], 'ro', markersize=5)
    ax.plot(goals[0, 0], goals[0, 1], 'go', markersize=5)
    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.tight_layout(pad=0)
    plt.savefig("paths.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    def visualize_expansion_for_paths(paths, q, goals,time_token=0.001, merge_rate=10):
        import matplotlib.pyplot as plt
        import imageio
        from io import BytesIO
        images = []
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        num_paths = paths.path.shape[0]
        layer_colors = ['orange', 'purple', 'green', 'blue', 'red', 'cyan', 'magenta', 'yellow']
        for i in range(num_paths):
            ax1.clear()
            ax2.clear()
            # ax1: 当前路径+扩展点
            ax1.contourf(X, Y, occ_map.map.T, cmap='Greys')
            ax1.plot(q[0], q[1], 'ro', markersize=5, label='Start')
            ax1.plot(goals[0, 0], goals[0, 1], 'go', markersize=5, label='Goal')
            ax1.plot(paths.path[i, :, 0], paths.path[i, :, 1], 'bo--', linewidth=1, markersize=2, alpha=0.7, label='Path')
            if hasattr(paths, "dream_points") and paths.dream_points is not None:
                dp = paths.dream_points[i] if paths.dream_points.ndim == 4 else paths.dream_points
                for layer in range(dp.shape[0]):
                    ax1.scatter(dp[layer, :, 0], dp[layer, :, 1], label=f"Layer {layer}", alpha=0.5, color=layer_colors[layer % len(layer_colors)])
            # ax1.legend()
            ax1.set_axis_off()
            ax1.set_aspect('equal')
            ax1.set_title(f"Current Path {i+1}")

            # ax2: 每merge_rate条路径合并显示
            ax2.contourf(X, Y, occ_map.map.T, cmap='Greys')
            ax2.plot(q[0], q[1], 'ro', markersize=5, label='Start')
            ax2.plot(goals[0, 0], goals[0, 1], 'go', markersize=5, label='Goal')
            start_idx = (i-1)* merge_rate
            end_idx = i * merge_rate
            if end_idx > num_paths:
                end_idx = num_paths
                start_idx = num_paths-merge_rate
                for j in range(start_idx, end_idx):
                    ax2.plot(paths.path[end_idx, :, 0], paths.path[end_idx, :, 1], 'bo--', linewidth=1, markersize=2, alpha=0.7, label='Path')
                    if hasattr(paths, "dream_points") and paths.dream_points is not None:
                        dp = paths.dream_points[j] if paths.dream_points.ndim == 4 else paths.dream_points
                        for layer in range(dp.shape[0]):
                            ax2.scatter(dp[layer, :, 0], dp[layer, :, 1], label=f"Layer {layer}", alpha=0.5, color=layer_colors[layer % len(layer_colors)])
            else:
                for j in range(start_idx, end_idx):
                    ax2.plot(paths.path[j, :, 0], paths.path[j, :, 1], 'bo--', linewidth=1, markersize=2, alpha=0.7, label='Path')
                    if hasattr(paths, "dream_points") and paths.dream_points is not None:
                        dp = paths.dream_points[j] if paths.dream_points.ndim == 4 else paths.dream_points
                        for layer in range(dp.shape[0]):
                            ax2.scatter(dp[layer, :, 0], dp[layer, :, 1], label=f"Layer {layer}", alpha=0.5, color=layer_colors[layer % len(layer_colors)])
            ax2.set_axis_off()
            ax2.set_aspect('equal')
            ax2.set_title(f"Paths {start_idx+1} ~ {end_idx}")
            fig.tight_layout(pad=0)
            plt.pause(time_token)
            # 保存当前帧到内存
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            images.append(imageio.v2.imread(buf))
            buf.close()
        plt.ioff()
        plt.show()
        imageio.mimsave("test.gif", images, duration=time_token)
    if hasattr(paths, "dream_points") and paths.dream_points is not None:
        visualize_expansion_for_paths(paths, q, goals)

    jax_exec_time = timer.elapsed
    jax_total_time = jax_exec_time  + jax_compile_time
    for_time = timer_for.elapsed

    labels = ['JAX Compile', 'JAX Exec', 'JAX Total', 'For Loop']
    values = [jax_compile_time, jax_exec_time, jax_total_time, for_time]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=['orange', 'blue', 'green', 'red'])
    plt.ylabel('Time (seconds)')
    plt.title('JAX vs For Loop Timing')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("timing_bar.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
