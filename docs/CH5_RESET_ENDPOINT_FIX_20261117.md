# 修复pilot初始栅格端点异常：2026-11-17

用户在Linux运行seed4317/full时，环境reset抛出 `current_pos is not a legal reachable planner point`。本地使用原pilot清单复现：第0个采集场景699可初始化，第1个场景1000004317000001失败。它的原始生成器通过物理场景可达性校验，但初始扫描融合到粗栅格后，Searcher2所在格被当前地图判为占据；旧初始化仍无条件调用sample_next_waypoint，因而退出。这不是GPU选择或用户命令错误。

新增CH5侧bootstrap端点guard：可规划起点完全沿用原采样和reserved顺序；暂不可规划起点保留自身坐标作为尚未解决的导航占位，并记录bootstrap_endpoint_deferrals。后续仍由原局部A*、传感更新及既有暂缓策略处理，不把该格强行改成自由，不移动起点、不重抽场景，不把不可达状态伪装为可达。NaN或越界坐标仍直接拒绝。

core/chapter3_bser代码、原400步期限、奖励公式、目标运动、动作和碰撞/成功规则不修改。新增子类只覆盖初始搜索航点边界，构造参数与继承在线地图一致；正常起点的航点、claim和占据输出与原实现逐值比较。

回归包含原生失败场景fixture：先确认原采样方法仍重现异常，再检查新reset成功且物理坐标、目标状态和障碍不变。Linux check命令加入该回归。Windows本地验证不能替代Linux实机验收。

本地结果：59项定向测试通过；seed4317原pilot的8个采集场景全部初始化通过，未重抽场景；原失败场景完成16个物理步，未训练，reward与原始记账逐步一致。790个受保护旧文件及212个继承底座文件字节不变。

## Linux更新与重试

先获取本次修复提交，并按聊天提供的新SHA checkout；不要强制丢弃本地修改。因为runtime/script源码指纹改变，旧RUN必须保留，并用新目录建立修复后的计划，不能直接对旧RUN执行resume。

```bash
conda activate AUV
cd "$HOME/AUV-Thesis/CH5-research"
git fetch origin
# git checkout --detach <聊天中提供的本次修复SHA>

python -B -m unittest checks.test_ch5_reset -v

RUN="outputs/chapter5/pilot_resetfix_$(date +%Y%m%d_%H%M%S)_$$"
bash scripts/ch5_linux.sh prepare pilot "$RUN"
python -u -B cli.py ch5-experiment train --root "$RUN" \
  --seed 4317 --arm full --device auto --resume --execute
```

训练正常完成后再执行evaluate。原目录 `outputs/chapter5/pilot_20261117_203346_3923224` 保留用于记录原失败；不删除或修改它的plan、检查点、attempt或结果。
