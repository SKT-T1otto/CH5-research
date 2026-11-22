# 第五章 Linux 实验运行手册

本版提供手动执行的完整分阶段流程：真实采集、双流编码/时间解码辅助训练、第三章MADDPG适配、冻结任务Actor后的通信A2C、验证集选模、最终故障扫描和多种子汇总。本地仅做有限CPU检查；Linux、CUDA和完整400步训练效果仍需服务器实际验收。脚本存在不代表实验已完成或创新已成立。

## 1. 克隆与环境

```bash
git clone --config core.autocrlf=false https://github.com/SKT-T1otto/CH5-research.git
cd CH5-research
git rev-parse HEAD

conda create -n CH5 python=3.10 pip -y
conda activate CH5
python -m pip install -r requirements.txt
python -m pip check
```

如已有符合依赖的环境，可直接激活它。依赖固定为NumPy2.2.6和PyTorch2.11.0；发布信息可核查 [PyTorch](https://pypi.org/project/torch/2.11.0/) 与 [NumPy](https://pypi.org/project/numpy/2.2.6/)。CUDA wheel及驱动须按服务器条件安装，保持版本符合requirements；本工程不自动升级系统驱动。

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

CPU是默认设备。`cuda`用于神经网络/优化，物理环境、局部地图和回放存储仍为CPU；不保证小网络在GPU更快。`auto`会选择可用CUDA；显式`cuda`不可用时直接报错。CPU与GPU结果分别记录，跨硬件不保证逐位一致。

## 2. 先做 Linux 短验收

```bash
bash scripts/ch5_linux.sh check cpu
# CUDA验收需自己另外执行：
# bash scripts/ch5_linux.sh check cuda
```

该命令先检查依赖并运行单元测试，再用独立check配置完成5段×最多24步、2次辅助更新、最多3次MADDPG更新及1次通信更新。父环境期限仍为400步，外部短截断不计完整任务成功率。输出在新的 `outputs/chapter5/linux_check_*`，包含源码身份、检查点和评价结果。服务器首次执行成功后，才形成该服务器的Linux运行证据。

旧 `cli.py ch5 smoke` 的8段×32步和更新硬上限保留；`cli.py ch5 train`仍是旧短流程，正式训练请使用下面的 `ch5-experiment`。

## 3. 预览、冻结并运行 pilot

```bash
bash scripts/ch5_linux.sh plan pilot

RUN="outputs/chapter5/pilot_$(date +%Y%m%d_%H%M%S)_$$"
bash scripts/ch5_linux.sh prepare pilot "$RUN"

# 先挑一个主方法/种子试跑，原样保存RUN，后续仍使用同一个RUN。
python -B cli.py ch5-experiment train --root "$RUN" \
  --seed 4317 --arm full --device cpu --execute
python -B cli.py ch5-experiment evaluate --root "$RUN" \
  --seed 4317 --arm full --device cpu --execute

# 检查资源与结果后，再运行其余种子/消融；已完成项会核验并跳过。
bash scripts/ch5_linux.sh all "$RUN" cpu
```

也可分别执行：

```bash
bash scripts/ch5_linux.sh train "$RUN" cpu
bash scripts/ch5_linux.sh evaluate "$RUN" cpu
bash scripts/ch5_linux.sh report "$RUN"
```

`prepare`只建立新目录和冻结计划；`plan`只读、不执行。完整suite默认顺序执行各种子/消融，不自动占满多块GPU。运行日志在 `suite_attempts/*/*.log`；某项失败保留日志，继续其他任务，最后返回非零状态并汇总缺失项。

## 4. 正式配置

pilot分析后使用新目录；不在旧RUN上更换预算或配置：

```bash
bash scripts/ch5_linux.sh plan formal
FORMAL="outputs/chapter5/formal_$(date +%Y%m%d_%H%M%S)_$$"
bash scripts/ch5_linux.sh prepare formal "$FORMAL"
bash scripts/ch5_linux.sh all "$FORMAL" cpu
# 服务器CUDA验收通过后可将cpu替换为cuda。
```

| 预算 | pilot | formal |
|---|---:|---:|
| 独立训练种子 | 3 | 5 |
| 独立训练组 | 8 | 8 |
| 每组采集episode | 8 | 32 |
| 辅助优化步 | 200 | 1500 |
| MADDPG适配episode | 50 | 800 |
| 通信策略episode | 50 | 800 |
| 每次验证episode | 5 | 20 |
| 通信阶段验证间隔 | 25 | 50 |
| 每种故障最终测试episode | 20 | 200 |
| 任务最大物理步 | 400 | 400 |

这些是可修改后另建RUN的研究预算，不保证收敛或效果；`plan`打印全部任务的最大交互步数，完整矩阵可能耗时很长。环境若真实碰撞/成功等终止会提前结束。MADDPG原gamma、网络/奖励/动作/成功规则不变，batch/warmup/update节奏显式记录；回放容量为pilot50000/formal100000，区别于父配置的500000，属于明确的资源超参数。

## 5. 训练组和故障组

八组均独立初始化并重新训练，不用评价时关开模块冒充重训消融：

- `full`：完整双流编码、地图条件时间传输、辅助损失和学习发包。
- `fixed_scheduler`：相同编码/解码和任务适配，最终固定轮转发包；用于隔离学习调度作用。
- `raw_only`：保留统一受限协议的原始状态、扫描、目标、命令和ACK，不发送神经包；不是完全零通信基线。
- `no_query`：编码始终使用unknown查询；请求仍计费，预算条件不赠送优势。
- `no_transport`：动态latent保持原值，关闭时间传输及无意义的时间一致性损失。
- `no_geometry`：时间向量场的几何输入置零；其他地图处理仍保留，准确检验几何对时间推进的作用。
- `no_time_loss`：保留时间传输，关闭组合一致性损失。
- `no_route_loss`：关闭路线相遇监督，保留状态监督和时间一致性。

各组阶段环境交互预算相同；raw_only不执行神经优化，优化计算量不同须披露。约束为同一发送机会、通信预算上限和帧协议，实际发送比特由决策产生，并不强制每组相同。第三章reward不新增通信人工评分。

七个测试工况：nominal、loss_high、delay_high、burst_long、budget_low、combined_unseen、reliable_reference。最后一组是低故障/充足预算诊断，不能与受限组作公平带宽排名。参数全部在配置JSON中。

外部CoDe、VA-MARO、VIL2C等论文算法没有在本套脚本中伪称复现；CVaR仅作为原回报尾部的描述统计，不代表风险训练或DRO。

## 6. 标签、选模与恢复

采集阶段包含原生成器seed699用于真实发现和目标消息覆盖，其余来自独立随机任务。机制场景只用于训练采集。真实标签不足会明确报错，保留输出；应重新设计新计划采集，不能补假标签或覆盖原结果。辅助训练使用随机小批量事实标签；当前仍是单包预训练，部署使用多包缓存，尚未实现缓存级精确重放。

任务Actor在固定通信下训练；随后codec/decoder/路线头和任务Actor冻结，通信A2C使用原任务物理步折扣回报。通信训练随机采样调度，任务Actor确定性执行。验证/测试均确定性执行。每个通信episode只更新一次，不复用旧策略轨迹。

验证集/测试集分别用独立固定seed命名空间，所有方法使用相同物理场景。按验证集平均原任务回报选checkpoint，最终测试不参与选模。链路RNG消耗随决策不同，因此是场景配对，不是严格逐包故障配对。

```bash
# 中断后：同一代码、同一RUN、同一seed/arm
python -B cli.py ch5-experiment train --root "$RUN" \
  --seed 4317 --arm full --device cpu --resume --execute

# 批量继续；已完成训练和已完成测试工况经核验后跳过。
bash scripts/ch5_linux.sh all "$RUN" cpu
```

新的实验schema保存网络、任务Actor/Critic/target、优化器、执行掩码回放、采样时钟、探索噪声和Python/NumPy/Torch/CUDA随机数。只恢复本RUN最后一个已提交的episode/辅助更新边界，不恢复活动环境。最近检查点之后的未提交工作可能重新计算，原attempt保留。采用新文件和独占job锁，未实现自动删除/压缩旧快照，需预留磁盘。

源码、脚本、依赖文件或实验配置发生变化后拒绝继续旧RUN，应新建RUN。不要用新入口恢复历史第三章权重或旧短测schema。训练和测试同一个job不能同时运行；不同job可由用户自行分配不同GPU，当前脚本只支持单进程单设备。

## 7. 输出与论文汇总

```text
RUN/plan.json                       冻结配置、父配置、源码身份、预算
RUN/jobs/seed_*/ARM/checkpoints/     不覆盖的新训练状态和SHA回执
RUN/jobs/seed_*/ARM/attempts/        逐阶段日志、标签、summary、失败记录
RUN/jobs/seed_*/ARM/evaluation/      各工况完成标记和逐episode原始结果
RUN/suite_attempts/                 批量启动日志与退出状态
RUN/reports/                       每次独立生成的CSV、JSON和Markdown
```

默认正式运行只保存逐episode摘要，采集另存原始轨迹/标签；避免保存所有400步网络张量造成过大输出。`save_episode_details=true`可在建立新计划前开启。结果按独立训练种子的均值聚合，给出回报、任务成功率、经验低尾回报、实际比特和墙钟时间；区间以训练种子bootstrap计算，小样本区间只作探索性证据，不将多episode冒充多训练种子。

缺失seed/组/工况明确列出，报告`complete=false`；完成不等于性能优越性，程序不会自动宣称论文创新或收敛通过。

## 8. D2 可靠通信碰撞回放（R123 扩场 0003、0009）

该入口读取第三章已解压的 `r123_expansion_20261120_123244_3078331` 和第五章已解压的 `d2_batch_r123_20261121_140145_3923224`。脚本按目录名查找，再校验计划、场景、逐回合数据及完成回执的 SHA-256。也可用 `--ch3-run`、`--batch-run` 直接传入两个已有运行目录。

```bash
conda activate AUV
cd "$HOME/AUV-Thesis/CH5-research"
RUN="outputs/chapter5/d2_collision_cases_$(date +%Y%m%d_%H%M%S)_$$"
echo "$RUN"
bash scripts/ch5_linux.sh d2-collision-cases --search-root "$HOME/AUV-Thesis" --root "$RUN" --execute
```

第三章部分从归档读取真实轨迹，按当时四艇的位置与传感器模型逐步重建共享占据地图，用日志中的每步地图版本和路径顶点哈希核对重建结果。只要第三章 tracking_both 和 task_sync 两条日志在窗口内的轨迹或决策记录不一致，本次双场景共同参照就会报错。

第五章每场重放两次，一次沿用历史批次种子，一次使用第三章评估种子。物理步最多为 2 × (66 + 37) = 206，训练更新为 0。采样时刻为应用物理动作前，逐步文件记录真实跟踪点、剩余路径、局部栅格、队友扫描时间、实际运动与碰撞记录。另在相同第五章轨迹上读取环境的全队共享地图供离线比较。图中的真实障碍轮廓和历史撞点只用于分析。

完成时核对 `report.json` 中的 `complete` 和 `verified`。各场的 `comparison.png` 使用统一尺度显示水平和垂直投影。`comparison.json` 记录首次分歧、最后一次动作及历史重现检查，`occupancy.npz` 保存全部时刻的三维占据概率，`grid.json` 保存坐标与阈值，三份 JSONL 保存按动作索引排列的原始比较记录。动作从 0 开始编号，动作 65 完成后是物理第 66 步。
