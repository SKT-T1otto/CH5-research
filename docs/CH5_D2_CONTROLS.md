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

实现记录包括 `chapter5_comm/d2_navigation.py`、`runtime.py`、`pipeline.py`、`experiment_config.py`、`experiment_runner.py`、`experiment_state.py`、`experiment_report.py`，两份 `d2_*_experiment_v1.json` 配置、`scripts/ch5_linux.sh`、`checks/test_ch5_d2_controls.py`、`.github/workflows/ch5-d2-linux.yml` 及本说明。
