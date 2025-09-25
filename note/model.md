梳理 urdf.py 里的 `load_model`，机器人模型（`URDFSystem`）包含以下几类核心信息：

## 结构骨架
- **`link_names`**：URDF 中每个 link 的名字列表。
- **`link_parents`**：每个 link 的父节点索引（root 为 -1），定义整棵 kinematic tree。
- **`link_types`**：用关节类型串起来的字符串（如 `'1ff1...'`），在 kinax 里代表 link/关节的类别组合。
- **`joint_ids`**：能运动的关节（revolute/prismatic/continuous/floating）的索引列表，后续做规划或控制时只关注这些自由度。

## 几何与惯量
封装在 `link` 字段的 `Link` 对象里，按 link 顺序存储：
- `transform.pos` 与 `transform.rot`：每个 link 的参考位姿（来自 `<inertial origin>`），表示质心在父坐标系中的位置和朝向。
- `inertia.i`：惯性张量（$3\times3$ 矩阵），`inertia.mass`：质量。
- 其它如 `invweight`、`constraint_*` 为后续动力学/约束解算准备。

## 运动学与约束
- `dof.motion`：把每个关节的运动子空间堆叠起来（转动轴或滑动方向），根节点用零向量占位，固定关节也加“空的”自由度确保索引一致。
- `dof.limit`：对应的上下界（角度/位移），缺省就用 $[-\infty, \infty]$。
- `dof.damping`、`dof.invweight` 等：关节阻尼、求解权重等参数。
- `dof.armature`、`dof.stiffness` 在这里设成 `None`，留给更高级的动力学扩展。

## 全局参数
- `dt`：默认仿真步长 `0.01` 秒。
- `gravity`：重力向量 `[0, 0, -9.81]`。

这套信息（结构拓扑、几何惯量、关节运动与约束）组成了 `URDFSystem`，可以驱动 `kinax.forward` 做正运动学，也能为规划器或动力学模拟提供物理属性。如果你补充 URDF 里的质量/惯性数据，`load_model` 就会自动把更准确的参数写进这个系统。