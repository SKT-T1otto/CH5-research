# 第五章 D2 和 D2+B2 对照

两组共用第三章 M20 未知多障碍移动目标任务、400 步期限、原动力学、奖励与成功判据。两组仅使用普通目标、队友状态、扫描、任务指令等计费 TDMA 报文。D2 ZERO 使用零残差，D2+B2 从头训练继承的 MADDPG 残差策略。短实验最多四次 B2 更新，用于验证接入。

`causal_d2_v1` 使用第三章的候选生成、检测目标函数、D2 贪心搜索和待命选择。搜索器可以从本艇视图规划，0 号艇可以根据已接收的队友位置与局部地图发送任务指令，执行艇根据本艇视图选择待命点。目标交接后执行艇使用原截获求解器，航行代价从本艇地图求取。报文接收顺序、可用队友范围、重规划和分配时钟属于第五章适配，原生第三章参考实验可另外运行并保留其来源。原有 pilot/formal 预设保留旧导航口径，新的 D2 对照使用单独预设。

## Linux

```bash
conda activate AUV
cd "$HOME/AUV-Thesis/CH5-research"
git fetch origin
git switch main
git merge --ff-only origin/main
bash scripts/ch5_linux.sh plan d2_short
RUN="outputs/chapter5/d2_short_$(date +%Y%m%d_%H%M%S)_$$"
bash scripts/ch5_linux.sh d2-short "$RUN" cpu
```

首次运行准备新的目录。若实验中断，保留该目录并用 `bash scripts/ch5_linux.sh all "$RUN" cpu` 从最近完成的边界继续。成功完成后在 `$RUN/reports/` 查看 `report.json` 和 `episodes.csv`。四个评估单元为两组策略各跑高可靠参考与组合劣化条件，每个单元使用同样两份固定测试场景。原始结果记录发现、碰撞、交接、捕获接触及最终成功，场景相同，报文事件随策略演化。

GitHub Actions 提供 Ubuntu CPU 运行及结果下载，实验文件按运行 ID 隔离，上传日志与结果表。

## 搜索阶段碰撞诊断

使用 D2-short 的第一个固定测试场景和同一物理种子。原生第三章 D2 ZERO 使用共享地图和原控制链。第五章 D2 ZERO 使用各艇局部地图和原高可靠 TDMA 条件。两组在普通 M20 任务中最多各观察前 25 个物理步。任务期限为 400 步，报告将观察窗口截止与任务结束分开记录。

```bash
RUN="outputs/chapter5/d2_diagnose_$(date +%Y%m%d_%H%M%S)_$$"
bash scripts/ch5_linux.sh d2-diagnose "$RUN"
```

- `plan.json`、`scenario.json` 和两组的 `initial_physics.json` 核对同一场景、种子、初始位置、流场和障碍。
- `result.json` 汇总碰撞艇、障碍、首次导航点分歧和首次实际加速度分歧。
- 每组的 `steps.jsonl` 保存积分前导航点、路径、零残差输入、实际加速度和碰撞。原生组另存窗口内实际输入共享地图的扫描。第五章组记录各艇自己的最近一次扫描及 `channel_events.json`。`maps/` 按内容哈希保存逐步可见地图快照。
- 新运行的 `handoff_event_step` 表示首次收到目标报文的步数，`last_target_received_step` 表示最近一次接收。此前已保存的原始结果使用各自记录时的字段定义。

本次诊断涉及 `chapter5_comm/d2_diagnostics.py`、`cli.py` 和 `scripts/ch5_linux.sh`。首次交接统计涉及 `chapter5_comm/runtime.py`、`chapter5_comm/experiment_report.py` 和 `checks/test_ch5_d2_controls.py`。

实现记录包括 `chapter5_comm/d2_navigation.py`、`runtime.py`、`pipeline.py`、`experiment_config.py`、`experiment_runner.py`、`experiment_state.py`、`experiment_report.py`，两份 `d2_*_experiment_v1.json` 配置、`scripts/ch5_linux.sh`、`checks/test_ch5_d2_controls.py`、`.github/workflows/ch5-d2-linux.yml` 及本说明。
