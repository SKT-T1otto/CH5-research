# 当前第五章：鲁棒通信强化学习

本项目建立于 2026-11-13。它是第五章后续开发入口，继承第三章强化学习、地图和规划源码，在此基础上研究通信丢包、延迟、连续中断与信息过期下的协同决策。

**当前方案已升级为全通道受限设计：目标、队友状态、地图共享及相应控制/ACK通道均受限。请从[第五章重设计v2](docs/CH5_REDESIGN_20261113.md)开始。** 文献、保留/退出决策、实验和实施路线已形成，运行时代码尚未实施该方案。

## 实际准备状态

- 已从 `AUV-Thesis-Active` 当前工作树复制 1217 份源码、配置及必要支持/验证资料，共 14,047,104 字节；包括明确选择的未跟踪源码，不是只按 Git HEAD 导出的干净提交。
- 每个复制文件的源路径、目的路径、SHA-256 与来源状态见 `provenance/ch3_parent_snapshot.json`。
- 父项目 README/AGENTS 原文保存在 `provenance/ch3_parent`，本项目适用自己的 AGENTS。
- 未复制 `.git`、模型权重、历史原始训练/评价结果或 `3090结果`。继承的短程验证资料是第三章来源证据，不是本项目新实验。
- 第三章最终 RL/地图配置尚未选定，来源校验与训练准入尚未重新通过。
- `ch5_integration_complete = false`
- `new_ch5_training_started = false`
- `new_ch5_performance_verified = false`

## 目录职责

| 路径 | 职责 |
|---|---|
| `core`、`chapter3_bser` | 继承第三章的环境、地图、规划和学习底座；保存父版本身份 |
| `chapter5_comm` | 后续 CH5 专属信道、消息、接收记忆、融合/调度与学习接口；当前仅接口设计说明 |
| `configs/chapter5` | 后续第五章基线与消融配置；当前不放虚构可运行配置 |
| `checks` | 继承检查与后续 CH5 定向验证 |
| `outputs/chapter5` | 本项目未来新实验；不得用旧结果冒充本项目结果 |
| `docs/CH5_*` | 本项目的定位、创新对标和底座冻结事项 |
| `provenance/ch3_parent_snapshot.json` | 本次继承来源，不因后续代码演进而重写父快照哈希 |

阅读顺序：[重设计方案](docs/CH5_REDESIGN_20261113.md)、[保留/退出决策](docs/CH5_RETENTION_DECISIONS_20261113.md)、[实验方案](docs/CH5_EXPERIMENT_PLAN_20261113.md)、[文件级实施路线](docs/CH5_IMPLEMENTATION_MAP_20261113.md)及[底座冻结事项](docs/CH5_BASELINE_SELECTION.md)。继承的其他文档和脚本保留原版本含义，不能把其旧标题或旧准入说明解释为 CH5 已完成。

旧通信源码与数据从 `../../references/AUV-Thesis-CH5-CommRobust` 查阅时，准确定位请使用总入口 PATH_MAP.md；本项目运行代码不得导入该资料库。
