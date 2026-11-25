# 第四章独立名义 MADDPG

本目录包含独立名义基线的配置、几何登记、网络构造和第3步只读任务运行器。网络模块构造四个分布式Actor、四个集中式单Critic及目标网络。学习更新、训练调度、Replay恢复与任务性能验收按后续步骤实施。

任务配置引用本工程的 `configs/b2.json` 并校验文件SHA256。新算法拥有独立身份 `ch4.maddpg.single_critic.v1`。相对水速名义点为主训练和准入域，原生环境为回归参照。主域固定线性阻力、公共流幅值0.10 m/s、四艇阻力和水平推力倍率1、推进记忆系数0.15、零离散延迟及N0条件。

场景清单位于 `configs/ch4/nominal_maddpg_scenes.json`，包含训练1000场、验证32场、名义准入100场、失配开发24场和失配确认候选256场。最后一组仅登记候选几何，最终确认样本量在失配开发之后锁定。原D2已查看场景纳入历史排重。

源码记录覆盖当前核心环境、继承任务装配代码、配置、新模块、实验脚本和已消费的历史JSON。来源完整性与既有第三章来源校验的结果分别记录。新记录保留旧来源校验失败的事实。

从第四章项目根目录执行核验。

```text
python -B -m unittest discover -s checks/chapter4 -p test_nominal_maddpg.py -v
python -B -m chapter4_interval_robust.nominal_maddpg --regenerate-sample
```

首次建库和来源登记使用以下命令，已有目标文件会被拒绝覆盖。

```text
python -B -m chapter4_interval_robust.nominal_maddpg --build-scenes --workers 4
python -B -m chapter4_interval_robust.nominal_maddpg --register-source --regenerate-sample --workers 4 --report verification/nominal_maddpg_steps12_20261125/registration.json
```

核验逐场重算全部物理指纹，并检查跨组和历史重复。`--regenerate-sample` 按每组首、中、末三个固定索引重新调用生成器，共复核15场。`--regenerate` 可额外执行全部场景的独立生成复核。

第1、2步报告记录当时的配置、静态任务契约、网络构造、场景物理指纹和源码一致性验收。其配置保留当时的零回合、零更新状态，第3步使用下述独立运行协议。

## 第3步只读运行验收

`configs/ch4/nominal_maddpg.json` 保留第1、2步原始登记状态。第3步的回合范围、归档校准引用和零更新条件另存于 `configs/ch4/nominal_maddpg_runtime.json`。

`NominalRuntime.advance()` 推进一次原生闭环任务，将执行门之后的归一化残差、原团队奖励、刷新导航后的下一观测和共享终止标记直接写入外部传入的 `NominalReplay`。返回值保持真实转移字段，物理命令、迟滞、推力、阶段和任务结果保存在 `last_record`。Replay采样采用独立随机流。

验收包含8个完整只读回合。零残差分别重放历史超时、碰撞、成功校准轨迹，并重复超时场景。冻结初始化策略在成功校准场景和首个验证场景上各自独立复位重放两次。每对重放的一侧额外抽样Replay，用于检查采样次数对任务轨迹的影响。校准样例按历史终止类型选取，仅作运行器验收。

```text
python -B -m unittest discover -s checks/chapter4 -p test_nominal_runtime.py -v
python -B -m chapter4_interval_robust.nominal_maddpg.validate_runtime
python -B -m chapter4_interval_robust.nominal_maddpg.validate_runtime --register-source --execute --workers 4
```

来源演化追加新记录，原始来源与验收文件保留各自身份。第3步报告验证运行器和真实转移数据流，优化器更新及名义性能仍由后续阶段验收。
