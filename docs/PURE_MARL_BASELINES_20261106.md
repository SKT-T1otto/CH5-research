# 纯 MARL baseline 实验设计与准备（2026-11-06）

本版本准备 MADDPG、MATD3、MAPPO、HAPPO（HARL）和 MASAC（BenchMARL）的独立实验文件。
不启动训练、不恢复检查点、不安装外部库，不构造 D2 控制器。已有 D2、B2/B3、safe-RL、
reach-avoid 的算法实现、配置、来源证据与历史检查点保持各自身份。

## 当前完成范围

已经提供公共环境、HARL 环境接口、可选 TorchRL 环境接口、原生策略评估包装器、统一物理指标记录器、
原生配置快照、场景拆分校验、25 个实验任务的生成器，以及仅允许准备工作的 CLI。
这些是可审阅的实验准备文件，**尚不是通过验收的原生训练入口**。

HARL/BenchMARL 尚未安装。后续还需完成 native runner 环境注册、固定开发集回调、
TorchRL spec 验证、原生采样与更新机制检查、带来源身份的检查点导出/评估加载，
以及当前源码的执行准入证据。不能把本轮接口检查当作五种算法的训练正确性或性能验收。

## 固定来源与算法机制

| 算法 | 来源 | 保留的机制 |
| --- | --- | --- |
| MADDPG | HARL | 原生确定性 actor、集中式 Q、经验回放、目标网络、探索噪声 |
| MATD3 | HARL | 双 Q、目标动作平滑及噪声截断、延迟策略更新 |
| MAPPO | HARL | 原生 on-policy 采样、PPO clipping、GAE、价值归一化及优化流程 |
| HAPPO | HARL | 原生智能体顺序更新及重要性修正，不改成 MAPPO 的并行更新 |
| MASAC | BenchMARL | 原生随机策略、双 Q、熵项、自动温度优化、经验回放与目标更新 |

固定提交：

- HARL：`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`
- BenchMARL：`65d649d80e0bdcbdbe2c5d6a3f02dbfed8f0bec1`

原始 YAML 与逐文件 SHA256 在 `configs/pure_marl/upstream/`。不把本地 B2 的同名 MADDPG
实现代替 HARL MADDPG，也不把 HASAC 改名为 MASAC。外部配置和代码不从旧目录导入。
HARL 的许可证声明保留在其 `setup.py` 快照；BenchMARL 保留原 LICENSE。

网络宽度、学习率、优化器、batch、目标更新系数及更新频率沿用各自固定版本的原生配置。
明确的公共覆盖项是 gamma=.95、单采样环境、统一交互预算、actor 不共享参数，
MAPPO/HAPPO collection horizon=400，以及统一评估入口。MASAC critic 也设为独立参数。
这些覆盖项全部写入 run_spec，而不是声称所有超参数与原论文逐字相同。
不同算法的更新数据利用率保持各自身份，后续必须另外记录更新次数、采样数及耗时。

## 纯 MARL 的控制与信息边界

每个 actor 直接输出 `[-1,1]^3`，按该 AUV 的 XY/Z 最大加速度转为实际控制指令：

```text
a_cmd_xy = clip(u_xy, -1, 1) * a_xy_max
a_cmd_z  = clip(u_z,  -1, 1) * a_z_max
```

关闭环境 PD 航点先验，残差缩放设为 1；不调用 D2 分配、引导安装、repair residual gate、
安全投影或规则避障。保留原任务发现后搜索器冻结、动力学限幅、壁面处理、障碍碰撞终止、
目标运动、真实检测、固定交接、执行器捕获与任务超时语义。冻结是公共任务规则。

保留 4 agents、每 agent 28D 输入、3D 输出。28D 使用新的 `public28_without_planner_v1` 身份：

| 0-based 列 | 处理 |
| --- | --- |
| 0:6 | 原位置与速度 |
| 6:12、15、17、19 | 规划航点差向量、方向、距离、方向速度与航点进度清零 |
| 12:15 | 只从发现时局部测量/已交付目标载荷重建；不使用规划拦截点 |
| 16、18、20、22:28 | 原速度标量、传感范围内障碍距离、冻结标志、角色与已知阶段 |
| 21 | 只保留交接后的真实捕获 hold 进度；交接前置零 |

收到的目标测量可以陈旧；本版本不加入拦截规划或完整地图信息，也不声称 28D 具有严格 Markov 充分性。
集中式训练状态为四个策略观测的拼接 112D；Q 的 state/action 合计为 124D。
MAPPO/HAPPO 的 V 输入不需要硬凑成 124D。执行时每个 actor 仅接收自身 28D。
各环境实例隔离 Python/NumPy/CPU Torch RNG，场景重置不会重置学习器随机流。

## 同场景与奖励

物理环境使用当前 `configs/environment.json`，协议和最终奖励适配使用固定哈希的
`configs/b2.json`：`collision_terminal_v1`、线段/AABB 碰撞、团队失败覆盖、
Phase1C-v2 奖励及 `team_mean_v1`，不重设奖励系数。超时沿原任务协议视为 terminal，
bootstrap=0；不擅自改成 time-limit truncation。

**纯控制策略不接规划器输出，但模拟器内置地图/航点仍参与原奖励与传感过程。**
这是保留原奖励函数所需的环境计量，不应描述成“整个模拟器完全没有规划计算”。
原方法的 D2 安装航点与纯 MARL 的环境航点参考也可能不同，因此主结论以共同任务/物理指标为准，
回报曲线仅作辅助，不声称两类方法的航点 shaping 回报完全等价。
若后续要彻底去掉环境航点奖励，应单独定义新奖励协议，并同步重训全部方法。

默认拟议实验：5 算法 × 5 训练种子（2729/3729/4729/5729/6729）=25 jobs。
每 job 2,400,000 个整队交互步，4 个智能体同步推进一次只计 1 步；并不乘 4。
该预算同时整除原生收集批次，避免 MASAC 因 6000-frame 批次多采末尾预算。
这是待开展实验的设计预算，本次不会执行。

| 集合 | 场景数 | 生成种子 | 用途 |
| --- | --- | --- | --- |
| train | 1000 | 12729 | 训练，按固定清单轮转 |
| dev | 100 | 22729 | 固定中间诊断集，每 120000 整队步评估 |
| test | 300 | 32729 | 配置冻结后，统一预算最终检查点评估一次 |

训练种子只改变模型与学习随机性。物理清单、每场景创新种子与场景顺序对所有算法相同。
三组场景同时检查 ID、scenario_seed 与物理输入 fingerprint 不重叠。
也可传入已有当前重建的 train/dev/test 清单；必须三份一起提供且满足相同协议和 400-step horizon。
你的原方法也应使用这三份清单与同一整队交互预算，不能把不同场景的旧结果直接拼表。

## 统一评估接口

策略接口：`reset()`、`act(local_observations, deterministic=True)`。
HARLPolicy 包装原生 on-policy/off-policy actors；BenchMARLPolicy 包装原生 MASAC collection policy。
这里只接收已构造的原生策略，不实现历史检查点加载或新训练器。

公共 `EpisodeRecorder` 可同时观察纯 MARL 和现有方法：

```python
from chapter3_bser.baselines.metrics import EpisodeRecorder

recorder = EpisodeRecorder(scene, reference_config)
recorder.start(runtime.env.unwrapped)  # 原方法显式 reset 后，step=0
while not runtime.terminal:
    transition = runtime.advance(policy, explore=False)
    recorder.observe(runtime.env.unwrapped, transition['rewards'])
record = recorder.finish(runtime.env.unwrapped)
record.update(algorithm='YOUR_METHOD', training_seed=seed)
```

原方法保留自己的控制和观测。记录器只读物理状态与最终奖励，不额外扫描、不调用规划或改任务状态。
统一记录成功、碰撞、发现、发现/交接/完成步、团队回报、逐艇路径长度、控制平方积分。
控制平方积分不是实际物理能耗，记录中有明确标识。
失败完成时间惩罚为 400+100=500 步，防止提前撞毁被统计成“完成更快”。

汇总先在每个训练种子内平均相同测试场景，再对训练种子等权平均。
提供种子间标准差与条件于固定测试集的 seed-bootstrap 95% 区间。
拒绝重复记录、场景集不完整、训练种子缺组及协议/horizon 不一致的混合结果。
仅报告成功样本耗时会有幸存者偏差，因此主要时间指标包含失败惩罚。

`evaluation.evaluate` 本轮仅允许显式 `execute=True, mechanism=True` 的 ≤8 episodes ×32 steps 接口验证。
正式评估需要后续原生 backend 验收和当前源码执行证据，本轮没有正式评估开关。

## 文件与命令

- `configs/pure_marl/suite_v1.json`：统一实验设计。
- `configs/pure_marl/requirements-*.txt`：独立环境依赖候选，不自动安装。
- `chapter3_bser/baselines/`：环境、backend 接口、评估、配置与准备 CLI。
- `checks/pure_marl_contracts.py`：有限接口验证；不训练五种算法。
- `tools/verify_pure_marl.py`：独立来源、旧实现不可变性与准备检查。
- `provenance/pure_marl_*`：独立来源演进，本轮不继承旧版本的训练准入结论。

在项目的 AUV Python 环境中查看计划：

```powershell
python -m chapter3_bser.baselines.workflow plan
```

只生成场景和 run_spec，不创建模型或训练进程：

```powershell
python -m chapter3_bser.baselines.workflow prepare --output work/collision_terminal/pure_marl_design_20261106_v1 --execute
```

复用当前已导出的公共场景：

```powershell
python -m chapter3_bser.baselines.workflow prepare --output work/collision_terminal/pure_marl_design_new --train-manifest PATH_TRAIN --dev-manifest PATH_DEV --test-manifest PATH_TEST --execute
```

所有输出必须为新路径；现有目录直接拒绝。无 `--execute` 的 prepare/report 仅打印计划。
本 CLI 没有 train 命令。原生库未验收，不能直接拿 run_spec 冒充可执行的 HARL/BenchMARL 配置。
后续训练入口必须消费同一场景清单、记录原生算法配置/commit/软件环境/预算/梯度更新计数，
并保留完整检查点来源与防覆盖规则。

本轮可验证纯控制映射与接口数据，但不作性能、收敛、安全保证或历史 E0 验收声明。
