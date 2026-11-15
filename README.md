# 当前第五章：鲁棒通信强化学习

本项目建立于 2026-11-13。它是第五章后续开发入口，继承第三章强化学习、地图和规划源码，在此基础上研究通信丢包、延迟、连续中断与信息过期下的协同决策。

代码仓库：[SKT-T1otto/CH5-research](https://github.com/SKT-T1otto/CH5-research)。仓库包含完整第三章强化学习、物理环境、地图/规划底座以及第五章通信模块、配置、标签/训练/评价流程、测试和来源记录；运行生成的缓存、输出与权重不纳入源码提交。

从 GitHub 获取并安装（建议 Python 3.10，保留文件原始换行）：

```bash
git clone --config core.autocrlf=false https://github.com/SKT-T1otto/CH5-research.git
cd CH5-research
python -m pip install -r requirements.txt
python -B cli.py ch5 plan
python -B -m unittest checks.test_ch5_neural_modules checks.test_ch5_data checks.test_ch5_training checks.test_ch5_runtime checks.test_ch5_pipeline
```

需要执行有限真实短测时，手动运行 `python -B cli.py ch5 smoke --execute --output outputs/chapter5/my_fresh_short_run`。不得将继承文档中的旧仓库长训练命令当作本项目默认流程；当前CH5入口和限制见下文及工程接入说明。

**当前主线为任务查询条件双流编码器＋几何条件连续时间解码器。** 已接入第三章 MADDPG、局部地图、实际计费通信、真实轨迹标签、有限训练和评价入口。使用方式和边界见 [工程接入说明](docs/CH5_INTEGRATION_20261113.md)。[网络v3](docs/CH5_NEURAL_MODULES_V3_20261113.md)与[系统v2](docs/CH5_REDESIGN_20261113.md)保留设计依据，接入状态以新说明和运行报告为准。

## 实际准备状态

- 已从 `AUV-Thesis-Active` 当前工作树复制 1217 份源码、配置及必要支持/验证资料，共 14,047,104 字节；包括明确选择的未跟踪源码，不是只按 Git HEAD 导出的干净提交。
- 每个复制文件的源路径、目的路径、SHA-256 与来源状态见 `provenance/ch3_parent_snapshot.json`。
- 父项目 README/AGENTS 原文保存在 `provenance/ch3_parent`，本项目适用自己的 AGENTS。
- 未复制 `.git`、模型权重、历史原始训练/评价结果或 `3090结果`。继承的短程验证资料是第三章来源证据，不是本项目新实验。
- 工程底座固定为 `configs/b2.json` 的 `D2_B2_repair_v2`、4×28维输入、连续3维残差动作、第三章 `OnlineUnknownMapTaskPlanner` 的四艇局部实例；这是明确的工程起点，论文最终最优基线仍需实验确认。
- `ch5_bounded_pipeline_integrated = true`
- `formal_long_training_started = false`
- `new_ch5_performance_verified = false`

## 目录职责

| 路径 | 职责 |
|---|---|
| `core`、`chapter3_bser` | 继承第三章的环境、地图、规划和学习底座；保存父版本身份 |
| `chapter5_comm` | 网络、全通道协议、局部运行时、真实标签、学习器和短流程 |
| `configs/chapter5` | 固定父配置身份的可执行短测试配置 |
| `checks` | 继承检查与后续 CH5 定向验证 |
| `outputs/chapter5` | 新建的短测、数据和检查点；失败记录也保留 |
| `docs/CH5_*` | 本项目的定位、创新对标和底座冻结事项 |
| `provenance/ch3_parent_snapshot.json` | 本次继承来源，不因后续代码演进而重写父快照哈希 |

阅读顺序：[网络主线v3](docs/CH5_NEURAL_MODULES_V3_20261113.md)、[协议/系统设计v2](docs/CH5_REDESIGN_20261113.md)、[保留/退出决策](docs/CH5_RETENTION_DECISIONS_20261113.md)、[实验方案](docs/CH5_EXPERIMENT_PLAN_20261113.md)及[底座冻结事项](docs/CH5_BASELINE_SELECTION.md)。v2实验矩阵需补充v3列出的网络消融；继承的其他文档和脚本不能作为CH5已完成的证明。

短流程入口（在本项目目录，使用可正常导入 PyTorch 的 AUV 环境）：

```powershell
D:/anaconda/anaconda/envs/AUV/python.exe -B cli.py ch5 plan
D:/anaconda/anaconda/envs/AUV/python.exe -B cli.py ch5 smoke --execute --output outputs/chapter5/my_fresh_short_run
```

`plan`不执行；`smoke`最多8个episode、每个外部截断32步。父环境仍为400步，短截断不报告完整任务成功率。输出目录必须是全新路径，禁止覆盖旧结果。

旧通信源码与数据从 `../../references/AUV-Thesis-CH5-CommRobust` 查阅时，准确定位请使用总入口 PATH_MAP.md；本项目运行代码不得导入该资料库。
