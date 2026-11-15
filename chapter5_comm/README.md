# 第五章通信工程

当前可运行闭环为：第三章B2 MADDPG/局部在线地图 → 真实计费信道 → 双流量化编码与几何条件时间解码 → 原任务执行 → 事实轨迹标签 → 辅助学习/MADDPG/通信A2C → 新检查点恢复 → 短评价。

| 文件 | 职责 |
|---|---|
| neural_modules.py | 查询条件双流编码、int8量化、几何时间传输、路线效用头和预算调度头 |
| protocol.py | 16字节报头、共享声学串行队列、TDMA、预算、丢包/连续中断/延迟 |
| runtime.py | 四艇独立地图与知识、合法28维任务输入、真实收包、候选轨迹与原任务适配 |
| data.py | 事实目标轨迹、局部候选路线、扫掠相遇标签、动态状态标签及删失掩码 |
| training.py | 单包辅助预训练、第三章MADDPG回放、on-policy SMDP A2C、新检查点 |
| pipeline.py | 有限真实采集、三类学习更新、恢复和评价闭环 |
| workflow.py | plan/smoke/collect/labels/train/evaluate 命令；train同样受短测试硬上限约束 |

详细数据契约、命令和实验边界见 [接入说明](../docs/CH5_INTEGRATION_20261113.md)。当前流程不提供长训练和活动环境/replay恢复；不得将有限更新权重作为正式性能结果。
