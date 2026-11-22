# 同预算历史占据证据闭环开发实验

本轮从 941426b 继续；仅测试确定性通信机制，训练更新为 0。两个开发场景从既有碰撞中选取，不能据此推断总体性能。结果均保存于 outputs/chapter5。

接口核查：runtime.SCAN 为 120 字节的 origin + 26 ranges + hit mask，协议头 16 字节，总计 1088 比特。接收端 integrate_obstacle_scan 仅积累实际送达的扫描。TaskPacketEncoder 已接收合法本地图 7D tokens、实际收到的 QUERY9 和 mask；GeometryTemporalDecoder 已接收送达 latent、age 与接收端几何。其 map_context 是估计的软上下文，install_decoded_state 不把它写成权威占据。两网络的特征、计费和标签接口可复用；本轮不加载权重，不声称网络训练收益。

新增 own_history_v1：每艇额外维护只由自身扫描更新的地图，不融合收到的地图，不转发队友证据。每个原 SCAN 轮转机会替换成 MAP_EVIDENCE，仍为 120 字节载荷。8 字节版本/条数/网格长度头，最多 14 个单元，每个 8 字节：公开网格索引 uint16、最后观测步 uint16、logodds×1024 int16、观测计数 uint16。剩余字节补零，完整帧实际计费 1088 比特。优先刷新此前发给该接收者且已变化的单元，再按占据 logodds、观测数、最近观测步及索引确定排序。没有场景号、艇号、动作号、危险格特例；排序不使用真实障碍或全队图。不是任务查询网络的替代训练结果。

接收端按 (来源, 单元, 最后观测步) 保存最新快照，旧版本忽略，重复快照不累加。规划图为自身扫描 logodds 与各来源最近送达快照之和，使用原占据截断和阈值；自己的后续扫描不能抹掉远端来源的独立累计值。源地图自身已有原 logodds 截断，因此这种充分统计量压缩与按时序混合所有射线不严格等价。该融合规则属于本轮明确标识的通信接口变更；没有改变 A*、D2、控制、物理、任务、奖励和成功判定。未更新的远端静态证据保留，丢包、移动障碍、超过 14 个单元时的刷新延迟仍是后续限制。

逐步 trajectory.json 记录四艇执行前位置/速度/目标/全路线/路径索引/航点及概率、执行后位置和碰撞；occupancy.npz 为每步四艇完整概率图；replans.json 为真实 A* 调用；channel.json 为含真实收发时间和比特的信道日志；evidence.json 为发送/接收的实际单元、版本和量化值。summary.json 是原任务终止结果。仿真运行至原任务终止（碰撞、成功或 400 步超时），不在改道时截断。

配对采用原始批次 SHA-256 固定的场景与 physical_seed、可靠链路、原消息轮转、D2 零残差。report.json 核对共同物理运行区间的完整发包序列（种类仅将 MAP_EVIDENCE 对应为 SCAN）、对象和大小。终止时间不同会使全回合实际总比特不同，因此必须同时报告共同区间预算和各组全回合开销，不能把相同额度说成相同总消耗。若某场 identical_common_schedule 为 false，应单列为未通过严格同预算时隙配对，不纳入该因果结论。

Linux 用法（conda AUV 已安装本项目依赖）：

```bash
cd ~/AUV-Thesis/CH5-research
git pull --ff-only origin main
conda activate AUV
python -m unittest checks.test_ch5_history_evidence checks.test_ch5_runtime checks.test_ch5_d2_controls
# 搜索原始批次，若有多个副本，请选择一个完整原始目录。
find ~/AUV-Thesis -type d -name d2_batch_r123_20261121_140145_3923224
export BATCH=/完整路径/d2_batch_r123_20261121_140145_3923224
# 两场短实验；先 0009，再 0003，每场两组，零训练。
nohup bash scripts/ch5_history_pair.sh short "$BATCH" > outputs/chapter5/history_short_launcher.log 2>&1 &
tail -f outputs/chapter5/history_short_launcher.log
# 查看已完成的配对结果（运行目录名写在日志首行）
cat outputs/chapter5/history_pair_short_*/report.json
# 后续独立启动 30 场配对；本轮未在本地执行此命令。
nohup bash scripts/ch5_history_pair.sh batch "$BATCH" > outputs/chapter5/history_batch_launcher.log 2>&1 &
tail -f outputs/chapter5/history_batch_launcher.log
# 待进程完成后，替换成日志首行给出的精确目录名。
export RUN=outputs/chapter5/history_pair_short_YYYYMMDD_HHMMSS
tar -czf "${RUN}.tar.gz" "$RUN" "${RUN}.log"
sha256sum "${RUN}.tar.gz"
```

每次新建结果目录，禁止覆盖历史结果。不要求第三章原始大批次或历史权重；只需上述第五章原始批次的 plan.json 与 scenarios.json。
