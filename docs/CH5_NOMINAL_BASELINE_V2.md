# 普通工况 MADDPG 任务基线

本轮从零初始化三个种子 2729、3733、4751。第一个预算终点为各 200 回合，总计至多 240,000 个训练团队步。检查点保存完整训练状态，每回合可恢复。第 0 回合及每 20 回合评价固定的 32 场，200 回合阶段共 1,056 个评价回合，另计至多 422,400 个评价团队步。后续使用同一入口累计到 500 或 1,000 回合。算法为原标准四 Actor、四单 Critic MADDPG。

导航采用已登记 causal_d2_v1、own_history_v1、本艇局部图和实际通信。此前 64 回合已经使用 b2.json 的 waypoint_arrival、inactive_actor、hold_residual 修复及 CH5 初始端点 guard，本次没有新增导航修复。任务期限 400 步及奖励、物理和终止规则沿用原版本。

本轮重新登记 Actor 学习率 0.0003、OU 尺度每团队步衰减 0.000002；原值为 0.001 和 0.00002。噪声仍从 0.20 降到最低 0.02，达到下限的时刻由约 9,000 步延后到约 90,000 步。200 回合最多 80,000 步/种子。批次 128、容量 500,000、预热 256、每 4 步两次联合更新、Critic 学习率 0.0005 均沿用原登记。两项新超参数有待检验，本次设计不能单独归因其作用。

## Linux 启动

使用独立克隆，避开此前失效的 worktree 关联。已有同名独立克隆时可在其内部 fetch 并快进到远端版本。

```bash
git clone --branch ch5-nominal-pilot https://github.com/SKT-T1otto/CH5-research.git ~/AUV-Thesis/CH5-baseline-v2
cd ~/AUV-Thesis/CH5-baseline-v2
conda activate AUV
bash scripts/ch5_nominal_baseline.sh check && bash scripts/ch5_nominal_baseline.sh train --until 200 --execute
```

运行器逐回合及逐验证场输出进度。再次执行 train 使用同一个默认输出根，自动从最近完整回合续跑。可指定 --seed 单独执行三个种子。输出根为 outputs/chapter5/nominal_baseline_v2。

另开终端，进入本项目、激活 AUV 后执行下列命令，生成已有回报及成功率曲线。

```bash
bash scripts/ch5_nominal_baseline.sh report
```

reports 下每份 JSON 都保留逐训练回合的原始回报、结局和验证摘要；PNG 分别显示训练回报的后向 20 回合均值、固定验证成功率、验证任务回报和碰撞率。训练带探索，验证关闭探索和更新。训练回报只作为诊断，验证同时看成功、碰撞、超时和任务回报。小 Critic MSE 或训练回报平台不能代替任务能力验收。

## 开发期冻结线

这些是本次提出的工程筛选标准，不是收敛定理或统计显著性界限。检查最近连续 5 个验证点，即跨度 80 回合，并要求最新已完成回合也已评价。

- 每个种子成功数的最大与最小之差不超过 2/32。
- 验证回报最大与最小之差不超过 max(1, abs(五点平均回报)) 的 10%。
- 五个点的每一点均至多 1/32 场碰撞。
- 至少两个种子五点都达到 29/32 成功，第三个种子五点都达到至少 26/32。
- 稳定在低成功率会得到 stable，但不能得到 ready_to_freeze。

200 回合未到门槛但还在改善，可以明确提高累计预算至 500；500 后仍可观察到 1,000。若曲线长期低位、持续退化或动作饱和伴随失败，应先诊断失败阶段与学习配置。入口到达指定预算自动停止，不会自动扩大预算。

```bash
bash scripts/ch5_nominal_baseline.sh train --until 500 --execute
```

## 独立准入

开发冻结线满足后，冻结每个种子最近五个验证点中最优的权重，再一次性打开已登记的 100 场准入集。三个种子都须达到至少 90/100 成功且至多 5/100 碰撞。100 场的原始结果和失败均保留。失败时保留冻结结果，后续新一轮开发需另登记未见准入场景，避免在这 100 场上反复调参。

```bash
bash scripts/ch5_nominal_baseline.sh freeze --execute
bash scripts/ch5_nominal_baseline.sh admit --execute
```

正式冻结需同时有开发曲线稳定和独立任务能力结果。200、500、1,000 回合都是预算观察点。100 场准入的成功比例用于这套登记场景的实测报告，尚不能保证所有普通工况达到相同比例。
