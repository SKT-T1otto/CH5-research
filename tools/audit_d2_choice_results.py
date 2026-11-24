"""Independently validate the sealed 24-case block, then count the preregistered scene-level gates."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess

import numpy as np

from chapter5_comm.data import COMMAND_ARMS, command_triplet_record
from chapter5_comm.pipeline import PROJECT
from chapter5_comm.protocol import Kind, Packet
from chapter5_comm.runtime import GOAL3
from tools.audit_d2_choice import checked_plan, emit, labels
from tools.audit_d2_choice_plan import AUDIT, STAGE2
from tools.executor_trace import read, sha


def seal_collector(folder):
    """Record the source in a new preregistered replication before its first case begins."""
    _, seal=checked_plan(folder)
    if (folder/'cases').exists() or (folder/'diagnostics').exists():
        raise ValueError('seal the collector before running a case')
    names=('chapter5_comm/data.py','tools/audit_d2_choice.py','tools/audit_d2_choice_plan.py','tools/audit_d2_choice_state.py')
    emit(folder/'collector_identity.json',dict(preregistration=seal,recorded_utc=datetime.now(timezone.utc).isoformat(),
         base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
         files={name:sha(PROJECT/name) for name in names}))


def eligible_from_wire(diag):
    prior, selected = {}, []
    packets = diag['native_commands']
    expected = [e['packet_id'] for e in diag['channel'] if e['kind'] == 'COMMAND']
    assert [x['packet'] for x in packets] == expected
    for row in packets:
        p = Packet.from_bytes(bytes.fromhex(row['wire_hex']))
        assert p.kind == Kind.COMMAND and p.sender == 0 and p.receiver in (1, 2) and p.sequence == row['packet']
        last = prior.get(p.receiver)
        if last is not None and last.payload != p.payload and len(selected) < 2:
            selected.append((p, last))
        prior[p.receiver] = p
    return selected


def check_run(raw, start, event):
    outcome, ch = raw['outcome'], raw['channel']
    end = outcome['terminal_step']
    assert [r['step'] for r in raw['tape']] == list(range(start,end)) and end <= 400
    assert outcome['reason'] in ('success','obstacle_collision','timeout')
    assert [outcome[k] for k in ('success','collision','timeout')]==[outcome['reason']==reason for reason in ('success','obstacle_collision','timeout')]
    assert not outcome['timeout'] or end==400
    assert sum(e['bits'] for e in ch) == outcome['full_tx_bits']
    assert sum(e['bits'] for e in ch if e['delivery_step'] is not None) == outcome['full_rx_bits']
    assert (outcome['first_finder'] is None) == (outcome['first_discovery_step'] is None)
    if outcome['first_finder'] is not None:
        assert outcome['first_finder'] in (0,1,2)
    h = outcome['first_handoff']
    if h is not None:
        assert outcome['first_discovery_step'] <= h['arrival_step'] <= end and h['receiver'] == 3
        assert any(e['packet_id'] == h['packet_id'] and e['kind'] == 'TARGET' and e['sender'] == h['sender']
                   and e['receiver'] == 3 and e['delivery_step'] == h['arrival_step'] for e in ch)
    if event is not None:
        packet = Packet.from_bytes(bytes.fromhex(event['sent_hex']))
        entry, = [e for e in ch if e['packet_id'] == packet.sequence]
        assert (entry['sender'], entry['receiver'], entry['sample_step'], entry['send_step'], entry['bits'],
                entry['scheduled_delivery_step'], entry['delivery_step'], entry['kind']) == (
                    packet.sender, packet.receiver, packet.sample_step, packet.send_step, 224,
                    event['scheduled_delivery'], event['arrival'], packet.kind.name)


def first_difference(runs, a, b, seq, receiver, field, atol):
    pkt = runs[a]['receipts'][seq]
    if pkt['arrival'] is None:
        return None
    begin = pkt['arrival']
    stop = min(runs[name]['outcome']['terminal_step'] for name in (a,b))
    aa, bb = ({r['step']:r[field][receiver] for r in runs[n]['tape']} for n in (a,b))
    return next((t for t in range(begin,stop) if any(abs(x-y)>atol for x,y in zip(aa[t],bb[t]))),None)


def validate_case(folder, i, plan, seal):
    case = read(folder/'cases'/f'{i:04d}.json')
    assert (case['case'], case['scene'], case['preregistration_sha256']) == (i,plan['scenes'][i]['scenario_id'],seal)
    assert sha(folder/case['diagnostics']) == case['diagnostics_sha256']
    with gzip.open(folder/case['diagnostics'], 'rt', encoding='utf8') as file:
        diag = json.load(file)
    assert diag['preregistration_sha256'] == seal
    obstacles=plan['scenes'][i]['obstacles']
    assert len(diag['true_obstacles'])==len(obstacles)
    for actual,registered in zip(diag['true_obstacles'],obstacles):
        for key in ('center','size'):
            np.testing.assert_array_equal(actual[key],np.asarray(registered[key],dtype=np.float32))
    base = diag['native']
    if (not case['opportunities'] and 'schema' not in diag and base['outcome']['reason']=='timeout'
            and not base['outcome']['timeout']):
        # The first recorder retained the authoritative reason but used Gym truncation for this zero-opportunity flag.
        assert case['native_outcome']==base['outcome'] and base['outcome']['terminal_step']==400
        case['native_outcome']['timeout']=base['outcome']['timeout']=True
        case['normalized_zero_opportunity_timeout']=True
    check_run(base, 0, None)
    selected = eligible_from_wire(base)
    assert len(selected) == case['opportunities'] == len(case['records'])
    assert len(base['native_commands']) == case['native_commands'] and case['suffixes'] == 3*len(selected)
    assert case['native_outcome'] == base['outcome'] and bool(case['zero_opportunity_reason']) == (not selected)
    for (packet, previous), row in zip(selected, case['records']):
        seq, rc, t = str(packet.sequence), packet.receiver, packet.send_step
        assert row['opportunity']['packet_id'] == packet.sequence and row['opportunity']['send_step'] == t
        assert row['sender_features']['candidate_goal'] == list(GOAL3.unpack(packet.payload))
        assert row['sender_features']['previous_command'] == dict(goal=list(GOAL3.unpack(previous.payload)),
                    send_step=previous.send_step, sample_step=previous.sample_step)
        tokens=row['sender_features']['tokens']
        for name, shape in (('map_tokens',(32,7)),('map_mask',(32,)),('dynamic_tokens',(5,8)),('dynamic_mask',(5,)),
                            ('receiver_query',(9,)),('query_available',())):
            array=np.asarray(tokens[name])
            assert array.shape==shape and array.dtype.kind in ('b','i','u','f') and np.isfinite(array).all()
            if name.endswith('_mask') or name=='query_available':
                assert array.dtype.kind=='b'
        command_triplet_record(row['opportunity'],row['sender_features'],row['arms'],row['route_encounter_labels'],row['offline_diagnostics_ref'])
        alt = diag['alternatives'][seq]
        assert alt['at_cut_receiver_map']['agent'] == rc and alt['at_cut_receiver_map']['step'] == t
        runs = dict(CMD=base, PAD=alt['PAD'], REPLAN_ONLY=alt['REPLAN_ONLY'])
        common_end = min(r['outcome']['terminal_step'] for r in runs.values())
        assert common_end == row['paired_observations']['common_end_exclusive']
        for name, run in runs.items():
            receipt = run['receipts'][seq]
            check_run(run, 0 if name=='CMD' else t, receipt)
            assert row['arms'][name] == dict(**receipt,outcome=run['outcome'])
            assert row['route_encounter_labels'][name] == labels(run, seq, receipt['arrival'])
            if receipt['arrival'] is not None:
                executed, = [r for r in run['tape'] if r['step'] == receipt['arrival']]
                assert receipt['next_executed']['waypoint'] == executed['waypoint'][rc]
                assert receipt['next_executed']['applied_acc'] == executed['applied_acc'][rc]
            changes = [e['packet_id'] for e in run['channel'] if e['kind'] in ('COMMAND_PAD','REPLAN_ONLY')]
            assert changes == ([] if name=='CMD' else [packet.sequence])
            end_pre = receipt['arrival'] if receipt['arrival'] is not None else common_end
            prefix = lambda rr: [r for r in rr['tape'] if t <= r['step'] < end_pre]
            assert prefix(run) == prefix(base)
            bits = dict(tx=sum(e['bits'] for e in run['channel'] if e['send_step']<common_end),
                        rx=sum(e['bits'] for e in run['channel'] if e['delivery_step'] is not None and e['delivery_step']<common_end))
            assert bits == row['paired_observations']['common_bits'][name]
        for a,b,key in (('CMD','REPLAN_ONLY','cmd_vs_replan'),('PAD','REPLAN_ONLY','pad_vs_replan'),('CMD','PAD','cmd_vs_pad')):
            for field in ('waypoint','applied_acc'):
                assert row['paired_observations'][key][field] == first_difference(runs,a,b,seq,rc,field,plan['accel_atol'])
    return case


def outcome_word(o):
    return '成' if o['success'] else '碰' if o['collision'] else '超'


def collect(cases):
    scenes = dict(pad_only=set(), cmd_only=set(), content=set())
    wins = {k:[] for k in scenes}
    counts = {name:Counter() for name in COMMAND_ARMS}
    exec_scenes, exec_rows = set(),0
    patterns = {}
    prefix_phases=Counter()
    stages = {field:[] for field in ('discovery', 'handoff', 'terminal')}
    for c in cases:
        for row in c['records']:
            arms = {k:v['outcome'] for k,v in row['arms'].items()}
            cmd,pad,rpl = (arms[k]['success'] for k in COMMAND_ARMS)
            changed = row['paired_observations']['cmd_vs_replan']['applied_acc'] is not None
            op = dict(case=c['case'], send_step=row['opportunity']['send_step'], receiver=row['opportunity']['receiver'])
            native=arms['CMD']
            if native['first_handoff'] and native['first_handoff']['arrival_step']<=op['send_step']:
                phase='already_handed_off'
            elif native['first_discovery_step'] is not None and native['first_discovery_step']<=op['send_step']:
                phase='found_awaiting_handoff'
            else:
                phase='before_first_discovery'
            prefix_phases[phase]+=1
            pattern=''.join(str(int(s)) for s in (cmd,pad,rpl))
            patterns.setdefault(pattern,[]).append(op)
            if changed:
                exec_scenes.add(c['case'])
                exec_rows+=1
            for key,hit in zip(scenes, (pad and not cmd,cmd and not pad,changed and cmd!=rpl)):
                if hit:
                    scenes[key].add(c['case'])
                    wins[key].append(op)
            for name,o in arms.items():
                counts[name].update({'success':o['success'],'collision':o['collision'],'timeout':o['timeout'],
                                     'discovery':o['first_finder'] is not None,'handoff':o['first_handoff'] is not None})
            for stage, extract in (('discovery', lambda o: (o['first_finder'],o['first_discovery_step'])),
                                   ('handoff',lambda o: None if not o['first_handoff'] else o['first_handoff']['arrival_step']),
                                   ('terminal',lambda o:(o['reason'],o['terminal_step']))):
                cv,pv,rv = (extract(arms[k]) for k in COMMAND_ARMS)
                if not cv == pv == rv:
                    stages[stage].append(dict(**op,CMD=cv,PAD=pv,REPLAN_ONLY=rv))
    disjoint = next(((sorted(a),sorted(b)) for a in combinations(sorted(scenes['pad_only']),2)
                    for b in combinations(sorted(scenes['cmd_only']-set(a)),2)), None)
    passed = disjoint is not None and len(scenes['content'])>=2
    return dict(scene_sets={k:sorted(v) for k,v in scenes.items()},opportunity_witnesses=wins,disjoint_witness=disjoint,
                independent_scenes=len(cases),passed=passed,arm_counts=counts,cmd_vs_replan_execution_cases=sorted(exec_scenes),
                cmd_vs_replan_execution_opportunities=exec_rows,stage_differences=stages,
                success_patterns_cmd_pad_replan=patterns,prefix_stage_diagnostic=prefix_phases)


def report(folder, result, cases):
    gates=result['scene_sets']
    show=lambda ids: '、'.join(f'{i:04d}' for i in ids) or '无'
    status='通过三项开发筛选线，可以开发改派学习器' if result['passed'] else '未同时通过三项开发筛选线，停止开发改派学习器'
    lines=['# 第 3 步 · 新场景中的改派选择与完整任务结局','',f'**{status}。**','',
        f"共登记 24 个普通难度 M20 开发场景，得到 {result['opportunities']} 个同前缀机会、{result['bound_suffixes']} 个命令分支结局。"
        f"无合格机会的场景为 {show(result['zero_opportunity'])}。每个机会按同一原终止条件评判，训练更新为 0。"
        '[预登记](preregistration.json) · [完整校验](validation.json) · [逐机会记录](triplets.jsonl)','',
        '| 预登记门槛 | 新场景数 | 场景编号 |','|---|---:|---|',
        f"| PAD 成功、CMD 失败，至少 2 场 | {len(gates['pad_only'])} | {show(gates['pad_only'])} |",
        f"| CMD 成功、PAD 失败，至少另 2 场 | {len(gates['cmd_only'])} | {show(gates['cmd_only'])} |",
        f"| CMD 与 REPLAN_ONLY 同一机会的执行和成功标志均不同，至少 2 场 | {len(gates['content'])} | {show(gates['content'])} |",'',
        ('前两项可以各取两个互不重叠的场景。' if result['disjoint_witness'] else '前两项未能各取两个互不重叠的场景。')+
        f"CMD 与 REPLAN_ONLY 之间实际施加的加速度出现差异的机会有 {result['cmd_vs_replan_execution_opportunities']} 个，"
        f"涉及 {len(result['cmd_vs_replan_execution_cases'])} 个场景。每个新场景是一个独立计数单位，同场两次机会可能共享原生命令主干。",'',
        '| 按机会计数 | 成功 | 碰撞 | 超时 | 发现 | 实际交接 |','|---|---:|---:|---:|---:|---:|']
    for name,c in result['arm_counts'].items():
        lines.append(f"| {name} | {c['success']} | {c['collision']} | {c['timeout']} | {c['discovery']} | {c['handoff']} |")
    stages=result['stage_differences']
    patterns=result['success_patterns_cmd_pad_replan']
    pattern_text='，'.join(f'{k} 有 {len(v)} 个' for k,v in sorted(patterns.items())) or '无配对机会'
    lines.extend(['','三位数字依次表示 CMD、PAD 和 REPLAN_ONLY 的最终成功标志。'+pattern_text+'。'])
    ph=result['prefix_stage_diagnostic']
    lines.extend(['',f"事后按共同发包前的任务进度复核，首次发现前有 {ph['before_first_discovery']} 个机会，"
        f"已发现、待交接有 {ph['found_awaiting_handoff']} 个，已交接有 {ph['already_handed_off']} 个。该阶段信息仅供离线解释。"])
    seen=set()
    for key in ('pad_only','cmd_only','content'):
        witnesses=result['opportunity_witnesses'][key]
        if not witnesses:
            continue
        w=witnesses[0]
        i,t=w['case'],w['send_step']
        if (i,t) in seen:
            continue
        seen.add((i,t))
        case=cases[i]
        row=next(r for r in case['records'] if r['opportunity']['send_step']==t)
        arms=row['arms']
        features=row['sender_features']
        diff=row['paired_observations']['cmd_vs_replan']['applied_acc']
        lines.extend(['',f"### 场景 {i:04d} 的端点与终局",'',
            f"第 {t} 步，0 号艇给 {w['receiver']} 号艇的候选端点由 {features['previous_command']['goal']} 改为 "
            f"{features['candidate_goal']}。三臂均于第 {arms['CMD']['arrival']} 步收到实际帧。"])
        if diff is not None:
            with gzip.open(folder/case['diagnostics'],'rt',encoding='utf8') as f:
                diagnostic=json.load(f)
            seq=str(row['opportunity']['packet_id'])
            streams=[diagnostic['native'],diagnostic['alternatives'][seq]['REPLAN_ONLY']]
            waypoints=[next(frame['waypoint'][w['receiver']] for frame in s['tape'] if frame['step']==diff) for s in streams]
            lines.append(f"CMD 与 REPLAN_ONLY 的实加速度在第 {diff} 步首次分化，该步实际航点分别为 {waypoints[0]} 与 {waypoints[1]}。")
        lines.extend(['','| 分支 | 实收后的首个执行航点 | 首次发现 | 执行艇首次收到目标 | 终局 |','|---|---|---|---|---|'])
        for name in COMMAND_ARMS:
            arm=arms[name]; o=arm['outcome']; hand=o['first_handoff']
            found='无' if o['first_finder'] is None else f"艇 {o['first_finder']}，步 {o['first_discovery_step']}"
            received='无' if hand is None else f"艇 {hand['sender']}→3，步 {hand['arrival_step']}"
            wp=None if arm['next_executed'] is None else arm['next_executed']['waypoint']
            lines.append(f"| {name} | {wp} | {found} | {received} | {outcome_word(o)}，步 {o['terminal_step']} |")
    lines.extend(['',f"发现者或首次发现步、首次实际交接步、终止原因或步数出现臂间差别的机会依次为 "
        f"{len(stages['discovery'])}、{len(stages['handoff'])}、{len(stages['terminal'])}。"
        '这些是阶段观察量。成功门槛只使用原任务的最终成功标志。','',
        '## 配对如何构成','',
        '预先锁定原 M20 普通难度生成器的 24 个物理实例，物理种子为 11240407 至 11240430。24 个物理指纹、编号和种子均与历史 30 场无重合。'
        '原生主干上仅取指挥艇向同一接收艇前后两次真实发送的端点字节发生变化的最早两次机会。'
        '首次分配跳过，覆盖零、一个和两个机会的场景。历史 0002 的动作差异与共同碰撞只用于核对判定口径，进入门槛的历史样本数为零。','',
        '每个运行进程使用 Windows CPU 和 Torch 单线程。每个机会在原发送入口切分，CMD 沿原生主干续跑。'
        'PAD 和 REPLAN_ONLY 从独立深拷贝续跑，均恢复同一随机状态。'
        '收包之后继续原声呐、通信、局部 A*、控制、奖励和 400 步期限。实际发送机会、帧头、载荷、采样时刻、身份及到达字节均被核对。'
        '首次执行航点取实际收包后第一次物理积分器使用的航点。执行差异取共同区间内接收艇实际加速度任一分量绝对差超过 0.00001 的首步。'
        'CMD 同时更新坐标与命令时间戳，两者各自的贡献仍待区分。','',
        '发送特征取切点处发送艇的地图与动态令牌、已收到的查询、上一次发给该艇的命令以及当步剩余预算。'
        '接收艇私图、真实障碍物、实际未来目标及四艇运动轨迹另存于逐场压缩诊断文件。'
        '原路线相遇标签由实收步的局部候选路线与随后的真实目标运动独立计算，保留有效范围及截尾标志。标签、全队状态和任务结局均未进入分支选择。','',
        '## 各场机会及结局','',
        '每组结局依次为 CMD、PAD、REPLAN_ONLY。成、碰、超分别指成功、碰撞、超时。括号内为原终止步。','',
        '| 场景 | 原生命令次数 | 原生主干 | 已干预的发送步、接收艇及三臂结局 |','|---|---:|---|---|'])
    for c in result['cases']:
        branches=[]
        for b in c['branches']:
            outcomes='／'.join(f"{b['outcome'][n]}({b['end'][n]})" for n in COMMAND_ARMS)
            branches.append(f"步 {b['send']}→艇 {b['receiver']}，{outcomes}")
        lines.append(f"| {c['case']:04d} | {c['native_commands']} | {c['native']}({c['native_terminal']}) | "+'<br>'.join(branches or ['无合格机会'])+' |')
    lines.extend(['','## 通信账目','',
        '可靠参考信道每艇容量 65,536 bit，每步补充 16,384 bit，速率 1 Mbit/s。干预帧含 16 字节帧头和 12 字节载荷，每次 224 bit。'
        '下表分别累加各机会的共同区间 [0, 最早终止步) 和各臂全回合真实事件，涵盖 ACK、查询与地图流量。同场两次机会的公共历史会在机会口径各计一次。','',
        '| 分支 | 共同发送 bit | 共同到达 bit | 全回合发送 bit | 全回合到达 bit |','|---|---:|---:|---:|---:|'])
    records=[r for c in cases for r in c['records']]
    for n in COMMAND_ARMS:
        values=[sum(r['paired_observations']['common_bits'][n][k] for r in records) for k in ('tx','rx')]
        values += [sum(r['arms'][n]['outcome'][k] for r in records) for k in ('full_tx_bits','full_rx_bits')]
        lines.append('| '+n+' | '+' | '.join(f'{v:,}' for v in values)+' |')
    lines.extend(['','## 记录器复核','',
        '执行艇收到目标后，截获规划缓存含有二进制键。审计用只读副本将其按原字节转为十六进制，真实规划器继续使用原键。'
        '正式校验以原终止原因核对全部分支，正常到期取 timeout。'])
    if (folder/'outcome_record_revision.json').exists():
        lines.extend(['',f"早期零机会原档使用了 Gym 截断位，依据相同原档的终止原因校正的场景编号为 "
            f"{show(result['normalized_zero_opportunity_timeout_cases'])}。本批全部三臂记录已用最终数据格式复核。"])
    if (folder/'worker_retries.json').exists():
        lines.extend(['',f"正式配对计 {result['bound_suffixes']} 个后缀。按每次导出失败都已跑完该场全部原登记后缀保守回算，"
            f"包含重试的后缀上界为 {result['suffix_bound_including_retries']}，预登记上限为 144。"
            '重试使用原种子，未增加场景或进入门槛统计的记录。[运行重试记录](worker_retries.json)'])
    lines.extend(['','固定语义涵盖可靠参考信道、零残差策略、现行 D2 与这批普通难度开发场景。'
        '本次只评估终局中是否出现值得学习的选择，场景数或机会数不能充当独立训练重复。',''])
    with (folder/'REPORT.md').open('x',encoding='utf8',newline='\n') as f:
        f.write('\n'.join(lines))


def summarize(folder, *, verify_only=False):
    plan, seal=checked_plan(folder)
    identity=read(folder/'collector_identity.json')
    assert identity['preregistration']==seal
    revision = read(folder/'audit_revision.json') if (folder/'audit_revision.json').exists() else None
    outcome_revision = read(folder/'outcome_record_revision.json') if (folder/'outcome_record_revision.json').exists() else None
    if revision:
        for name,h in revision['fixed_files'].items():
            expected=outcome_revision['final_runner_sha256'] if outcome_revision and name=='tools/audit_d2_choice.py' else h
            assert sha(PROJECT/name)==expected
    for name, recorded in identity['files'].items():
        if sha(PROJECT/name)==recorded:
            continue
        if name=='tools/audit_d2_choice.py' and revision:
            archived=folder/'runner_at_capture.py.txt'
            assert sha(archived)==recorded
            original=b'from tools.command_control_trace import fingerprint, fork, restore_rng, save_rng'
            amended=b'from tools.command_control_trace import fork, restore_rng, save_rng\nfrom tools.audit_d2_choice_state import fingerprint'
            source=archived.read_bytes()
            assert source.count(original)==1
            v2=source.replace(original,amended)
            latest=(PROJECT/name).read_bytes()
            if outcome_revision:
                assert hashlib.sha256(v2).hexdigest()==outcome_revision['original']
                # Only export bookkeeping and the terminal flag changed after V2.
                trunk=lambda s:s.split(b'def run_case(')[1].split(b'        ref = f')[0]
                observer=lambda s:s.split(b'class Observe:')[1].split(b'def sender_inputs')[0]
                assert trunk(v2)==trunk(latest)
                assert observer(v2).replace(b"timeout=bool(s['truncated'])",b"timeout=s['termination_reason']=='timeout'")==observer(latest)
            else:
                assert v2==latest
            continue
        # The captured data module differed only in placement of the legacy writer's return.
        # Verify the exact two-line restoration before revalidating every recorded triplet.
        archived=folder/'data_at_capture.py.txt'
        assert name=='chapter5_comm/data.py' and sha(archived)==recorded
        source=archived.read_bytes()
        final_return=b'    return destination\n'
        anchor=b'\n\nCOMMAND_TRIPLET_SCHEMA = '
        assert source.endswith(final_return) and source.count(anchor)==1
        restored=source[:-len(final_return)].replace(anchor, final_return+anchor, 1)
        assert restored==(PROJECT/name).read_bytes(), 'unaccounted collector change'
    cases=[validate_case(folder,i,plan,seal) for i in range(24)]
    by_case=[dict(case=c['case'], opportunities=c['opportunities'], native_commands=c['native_commands'],
        native=outcome_word(c['native_outcome']), native_terminal=c['native_outcome']['terminal_step'],native_outcome=c['native_outcome'],
        branches=[dict(send=r['opportunity']['send_step'],receiver=r['opportunity']['receiver'],
            outcome={k:outcome_word(r['arms'][k]['outcome']) for k in COMMAND_ARMS},
            end={k:r['arms'][k]['outcome']['terminal_step'] for k in COMMAND_ARMS},
            first_execution_difference=r['paired_observations']['cmd_vs_replan']['applied_acc']) for r in c['records']])
        for c in cases]
    verdict=collect(cases)
    old=read(STAGE2/'validation.json')
    assert old['passed'] and sum(a['success'] for a in old['arms'].values())==0
    assert old['first_difference_by_action_step']['REPLAN_ONLY_vs_CMD']['four_applied_accelerations']==54
    triplets=folder/'triplets.jsonl'
    encoded=''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n'
                    for c in cases for row in c['records']).encode('utf8')
    if verify_only:
        assert triplets.read_bytes()==encoded
    else:
        with triplets.open('xb') as f:
            f.write(encoded)
    total=sum(c['opportunities'] for c in cases)
    assert total<=48 and 3*total<=144
    native_steps=sum(c['native_outcome']['terminal_step'] for c in cases)
    alternative_steps=sum(r['arms'][name]['outcome']['terminal_step']-r['opportunity']['send_step']
                          for c in cases for r in c['records'] for name in ('PAD','REPLAN_ONLY'))
    retries=read(folder/'worker_retries.json')['failed_attempts'] if (folder/'worker_retries.json').exists() else []
    repeated=sum(cases[a['case']]['suffixes'] if a['stage']=='post_terminal_triplet_export'
                 else a['full_suffixes_before_failure'] for a in retries)
    assert 3*total+repeated<=plan['max_suffixes']
    result=dict(verified=True,preregistration_sha256=seal,training_updates=0,normal_new_scenes=24,
        zero_opportunity=[c['case'] for c in cases if not c['records']],opportunities=total,
        bound_suffixes=3*total,triplets_sha256=sha(folder/'triplets.jsonl'),historical_outcome_rows_in_gates=0,
        distinct_native_runs=24,independent_alternative_continuations=2*total,
        actual_native_transitions=native_steps,actual_alternative_transitions=alternative_steps,
        repeated_suffix_upper_bound=repeated,suffix_bound_including_retries=3*total+repeated,
        normalized_zero_opportunity_timeout_cases=[c['case'] for c in cases if c.get('normalized_zero_opportunity_timeout')],
        review_0002_first_execution_difference=54, capture_source_identity=identity['files'],
        validated_data_source_sha256=sha(PROJECT/'chapter5_comm/data.py'), cases=by_case,**verdict)
    if verify_only:
        assert read(folder/'validation.json')==json.loads(json.dumps(result,allow_nan=False))
    else:
        emit(folder/'validation.json',result)
        report(folder,result,cases)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=AUDIT)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--seal-collector',action='store_true',help='seal current code before simulating a new replication')
    mode.add_argument('--verify-only',action='store_true',help='read and verify the complete report receipts without writing or simulating')
    args=p.parse_args()
    if args.seal_collector:
        seal_collector(args.output)
    else:
        result=summarize(args.output,verify_only=args.verify_only)
        print('passed:',result['passed'],'opportunities:',result['opportunities'],'gate scenes:',result['scene_sets'])
