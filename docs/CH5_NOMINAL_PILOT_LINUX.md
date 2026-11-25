# 第五章标准 MADDPG 的 64 回合 Linux 试验

在已有的 AUV conda 环境中使用 CPU，每个种子一个线程。发布分支为 `ch5-nominal-pilot`。第一次从已有仓库取出专用工作目录。

```bash
cd ~/AUV-Thesis/CH5-research
git fetch origin ch5-nominal-pilot
git worktree add --detach ../CH5-nominal-pilot FETCH_HEAD
cd ../CH5-nominal-pilot
conda activate AUV
bash scripts/ch5_nominal_pilot.sh check
bash scripts/ch5_nominal_pilot.sh run --execute
```

再次启动时，进入 `~/AUV-Thesis/CH5-nominal-pilot`，激活 AUV 后重用 `run --execute`。种子按 2729、3733、4751 顺序执行。每完成一个训练回合便原子提交回合边界，每个验证场景单独提交。中途退出后，未提交的训练回合或未提交的验证场景从头运行，已提交的经验只计一次。源码、环境版本、运行标识或已提交文件不匹配会报错。输出默认为 `outputs/chapter5/nominal_maddpg_pilot64/`。指定其他目录时，为后续命令传入相同的 `--root`。

如需分开调度，可以用 `run --seed 2729 --execute` 等命令给每个种子一个进程。三个进程使用同一个输出根，各自持有种子锁。查看已完成的验证点时执行下列命令。

```bash
bash scripts/ch5_nominal_pilot.sh report
```

## 已登记的设计

- 训练池为 1,000 个普通难度场景，各种子使用独立初始化、场景排列、采样和 OU 随机流，每个种子最多 64 个完整回合。每回合原期限 400 个团队步，合计最多 76,800 个已提交训练步。
- 均匀 Replay 容量 500,000，每批 128 个不同团队转移。累计到 256 条有效转移后，在 4 的整倍数步触发两次联合更新。一次联合更新训练四个 Critic、所有本批有合法执行行的 Actor，再更新目标网络。三种子合计理论上限 38,022 次联合更新。
- OU 初始尺度 0.20，下限 0.02，每训练团队步衰减 0.00002。均值 0、回归系数 0.15、过程时间增量 1；这里的时间增量按决策步定义，任务物理步长为 0.2 秒。回合初重置噪声记忆，保留随机流和步数。已比对第四章名义 MADDPG 登记配置及 `configs/ch4/nominal_maddpg_training.json`，其 SHA-256 见本次协议。
- 各种子在第 0、16、32、48、64 回合的冻结权重上评价相同的 32 个场景，每个检查点关闭探索和更新。共 480 个验证回合，最多另用 192,000 个物理步。每个验证点先从磁盘完整恢复训练状态，再校验评价前后的状态摘要。恢复包含 Replay、在线和目标网络、Adam、场景顺序、随机流与探索状态。
- 第 0 回合用于配对参照。训练候选从 16、32、48、64 中按成功数高、碰撞数低、原任务未折扣团队回报均值高、回合早的顺序选择。固定 32 场使计数与发生率排序等价。此登记覆盖第 1 步原配置的检查点选择字段，保留原始任务奖励与终局。
- 完成三个种子后生成 `report.json`。每个种子的 `training_episodes` 保存逐回合训练回报、终局、通信位数、动作门开放步数、限幅附近动作、Critic MSE 与发现交接阶段。`validation_checkpoints` 列出五个检查点的成功、碰撞、超时、评价回报、步数与哈希。`candidate_episode` 给出选中回合，对应的权重和 Replay 在 `seed_<种子>/boundaries/<四位回合>/state.pt` 中。
- 三个种子的经验计数、预算、磁盘恢复与评价检查全部通过且均有学习更新，并有至少两个种子的已训练候选严格超过各自初始化策略的验证成功数，才得到 `PASS`。未完成为 `PENDING`。全部完成但筛选未过为 `STOP`。完整性或损坏错误直接中断，保留现场，不放行。
- `PASS` 后的后续观察点为累计 200、500、1,000 回合，已完成的 64 回合计入其中。这个入口在 64 回合停止。若为 `STOP`，先结合每艇执行门、动作幅值/饱和度、探索尺度、Critic MSE 和发现/交接/接触记录定位失败阶段。
