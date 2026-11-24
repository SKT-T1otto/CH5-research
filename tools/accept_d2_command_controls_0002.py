"""CPU acceptance of CMD, PAD and REPLAN_ONLY from one native D2 send boundary."""
import argparse
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import platform

import numpy as np
import torch

from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter5_comm.pipeline import new_output
from chapter5_comm.protocol import COMMAND_PADDING, HEADER, D2CommandOverride, Kind, Packet
from chapter5_comm.runtime import ACK, CH5Runtime, GOAL3
from tools.command_control_trace import (
    BeforeTransmission, CommandTrace, fingerprint, fork, matching, reference_signature, restore_rng, save_rng,
)
from tools.executor_trace import behavioral_summary, digest, plain, read, sha, write


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'checks/fixtures/d2_command_controls_0002.json'
SEND, RECEIVE = 52, 53
OLD_GOAL, NEW_GOAL = [3., 17., 6.5], [3., 15., 6.5]
ARMS = {'CMD': Kind.COMMAND, 'PAD': Kind.COMMAND_PAD, 'REPLAN_ONLY': Kind.REPLAN_ONLY}


def check(name, actual, expected):
    if actual != expected:
        raise ValueError(dict(field=name, actual=actual, expected=expected))


def verify_inputs(fixture, batch, aa):
    base = ROOT / 'configs/b2.json'
    check('b2 bytes', sha(base), fixture['base_sha256'])
    if batch:
        check('Linux plan bytes', sha(batch / 'plan.json'), fixture['plan_sha256'])
        for name, wanted in fixture['case_files'].items():
            check(name, sha(batch / 'case_0002/own_history_v1' / name), wanted)
        plan = read(batch / 'plan.json')
        check('frozen input', [plan['commit'], plan['base'], plan['scenes'][2], plan['channel'],
              dict(plan['seeds']['typed_mapping'])[2], plan['training_updates'], plan['deadline']],
              [fixture['source_commit'], read(base), fixture['scene'], fixture['channel'], fixture['seed'], 0, 400])
    if aa:
        for name, wanted in fixture['aa_files'].items():
            check('A/A ' + name, sha(aa / name), wanted)
        prior = read(aa / 'validation.json')
        check('previous A/A and Linux acceptance', [prior['same_machine']['passed'], prior['linux_history_passed']], [True, True])


def save_trace(folder, name, value):
    with (folder / (name + '.json.gz')).open('xb') as stream:
        with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as zipped:
            zipped.write(json.dumps(plain(value), separators=(',', ':'), allow_nan=False).encode())


def load_trace(folder, name):
    with gzip.open(folder / (name + '.json.gz'), 'rt', encoding='utf-8') as f:
        return json.load(f)


def capture(folder, fixture):
    policy = ZeroResidualPolicy()
    with ExitStack() as lifetime:
        root = CH5Runtime(read(ROOT / 'configs/b2.json'), fixture['scene'], seed=fixture['seed'],
                         channel_config=fixture['channel'], navigation='causal_d2_v1', map_transport='own_history_v1')
        lifetime.callback(root.close)
        check('physical device', root.physical.device.type, 'cpu')
        common = CommandTrace(root, SEND, prefix=True)
        with ExitStack() as taps:
            common.attach(taps)
            while root.step < SEND:
                common.advance(policy)
            try:
                root.begin_step()
            except BeforeTransmission:
                pass
            else:
                raise RuntimeError('no native send opportunity at step 52')
        before = common.pre_send
        check('shared native slot', [before[k] for k in ('kind', 'sender', 'receiver', 'sample', 'payload', 'sequence')],
              ['COMMAND', 0, 1, SEND, GOAL3.pack(*NEW_GOAL).hex(), SEND])
        state = save_rng()
        boundary = fingerprint(root)
        check('clean unwrapped send boundary', boundary, before['state'])
        save_trace(folder, 'common', dict(pre_send=before, events=common.events))
        print('common prefix complete', boundary, flush=True)
        copies = {}
        for name in ARMS:
            restore_rng(state)
            copies[name] = fork(root)
            lifetime.callback(copies[name].close)
        for name, rt in copies.items():
            restore_rng(state)
            check(name + ' before treatment', fingerprint(rt), boundary)
            check(name + ' initial replan watermark', rt._last_replan_request, {})
            if name != 'CMD':
                rt.command_override = D2CommandOverride(SEND, 1, ARMS[name])
            trace = CommandTrace(rt, SEND)
            with ExitStack() as taps:
                trace.attach(taps)
                # Resume cached native selection. All cut-step receipts/navigation are already complete.
                result = rt._bootstrap()
                check(name + ' accepted', result['accepted'], True)
                rt._begun_step = rt.step
                while not rt.terminal:
                    if rt.step >= rt.config['max_steps']:
                        raise RuntimeError('original deadline passed without termination')
                    trace.advance(policy)
            summary = plain(rt.summary())
            save_trace(folder, name, dict(pre_send=trace.pre_send, events=trace.events, summary=summary,
                       channel=rt.channel.events, evidence=rt.evidence_events))
            print(name, rt.step, summary['termination_reason'], flush=True)
        restore_rng(state)
        check('untouched shared state after three forks', fingerprint(root), boundary)
    return boundary


def one_event(events, kind, at):
    rows = matching(events, kind, step=at)
    check(f'{kind} count at {at}', len(rows), 1)
    return rows[0]['data']


def budget(events, stop):
    return dict(send_bits=sum(e['bits'] for e in events if e['send_step'] < stop),
                receive_bits=sum(e['bits'] for e in events if e['delivery_step'] is not None and e['delivery_step'] < stop))


def arm_receipt(name, run, boundary):
    ev, ch, term = run['events'], run['channel'], run['summary']
    pre = run['pre_send']
    check(name + ' native pre-send state', pre['state'], boundary)
    tx, rx = one_event(ev, 'tx', SEND), one_event(ev, 'rx', RECEIVE)
    sent = bytes.fromhex(tx['wire_hex'])
    expected_body = GOAL3.pack(*NEW_GOAL) if name == 'CMD' else COMMAND_PADDING
    wanted = Packet(ARMS[name], 0, 1, SEND, SEND, SEND, expected_body)
    check(name + ' actual wire', sent.hex(), wanted.to_bytes().hex())
    check(name + ' actual send/arrival', [rx['wire_hex'], tx['due'], rx['due'], tx['accepted']['bits'], len(sent) * 8],
          [sent.hex(), RECEIVE, RECEIVE, 224, 224])
    check(name + ' dispatch identity', [pre[k] for k in ('kind', 'sender', 'receiver', 'sequence', 'sample', 'payload')],
          [ARMS[name].name, 0, 1, SEND, SEND, expected_body.hex()])
    check(name + ' only scheduled substitution',
          [e['packet_id'] for e in ch if e['kind'] in ('COMMAND_PAD', 'REPLAN_ONLY')], [] if name == 'CMD' else [SEND])
    old, new = (one_event(ev, key, RECEIVE) for key in ('delivery_before', 'delivery_after'))
    check(name + ' common ACK path', new['acks'], (old['acks'] + [[0, SEND]])[-32:])
    check(name + ' arrived after natural plan', [old['last_plan'], old['goal']], [RECEIVE, OLD_GOAL])
    expected_command = [SEND, NEW_GOAL] if name == 'CMD' else old['command']
    check(name + ' command/watermarks/invalidated plan', [new['command'], new['command_sample'], new['last_plan'], new['last_replan_request']],
          [expected_command, SEND if name == 'CMD' else old['command_sample'],
           old['last_plan'] if name == 'PAD' else -100000, SEND if name == 'REPLAN_ONLY' else old['last_replan_request']])
    for key in ('goal', 'path', 'waypoint'):
        check(name + ' receive ' + key, new[key], old[key])
    at53 = [r for r in ev if r['step'] == RECEIVE and r['agent'] == 1 and
            r['event'] in ('astar', 'rx', 'delivery_before', 'delivery_after')]
    order = ['astar', 'rx', 'delivery_before', 'delivery_after']
    check(name + ' boat 1 event order at 53', [r['event'] for r in at53], order if name == 'PAD' else order + ['astar'])
    plans = [r['data'] for r in at53 if r['event'] == 'astar']
    goals = [OLD_GOAL] if name == 'PAD' else [OLD_GOAL, NEW_GOAL if name == 'CMD' else OLD_GOAL]
    check(name + ' queried goals', [row['goal'] for row in plans], goals)
    check(name + ' actual checked ACTION count', len(matching(ev, 'control')), term['actual_length'] - SEND)
    check(name + ' zero residual in every suffix control', all(not np.any(row['data']['applied_residual_action'])
          for row in matching(ev, 'control')), True)
    check(name + ' closed-loop terminal', [term['terminated'] or term['truncated'], term['task_deadline_steps']], [True, 400])
    collision = [[r['agent_id'], r['obstacle_id']] for r in term['collision_records']]
    window = []
    for row in matching(ev, 'trajectory'):
        t, d = row['step'], row['data']
        if t > 56:
            break
        a = d['agents'][1]
        ctl = one_event(ev, 'control', t)
        window.append(dict(action_step=t, next_state=t+1, goal=a['goal'], waypoint=a['waypoint'], route=a['route'],
            position=a['position'], next_position=d['next_positions'][1], next_velocity=ctl['post_velocity'][1],
            residual=ctl['applied_residual_action'][1], prior_acc=ctl['prior_acc'][1], applied_acc=ctl['acceleration'][1],
            collisions=[[c['agent_id'], c['obstacle_id']] for c in d['collisions']],
            astar=[dict(goal=e['data']['goal'], reachable=e['data']['answer']['reachable'],
                        reason=e['data']['answer']['failure_reason']) for e in matching(ev, 'astar', step=t, agent=1)]))
    full = budget(ch, term['actual_length'])
    check(name + ' actual channel accounting', full,
          dict(send_bits=term['channel']['sent_bits'], receive_bits=term['channel']['delivered_bits']))
    check(name + ' reliable reference channel', [term['channel']['dropped_frames'], term['channel']['rejected_frames']], [0, 0])
    outgoing = matching(ev, 'tx')
    incoming = matching(ev, 'rx')
    acknowledgments = []
    for row in outgoing:
        packet = Packet.from_bytes(bytes.fromhex(row['data']['wire_hex']))
        if packet.kind == Kind.ACK:
            acknowledgments.append(dict(step=row['step'], sender=packet.sender, receiver=packet.receiver,
                                        acknowledges_sequence=ACK.unpack(packet.payload)[0], bits=packet.bits))
    return dict(pre_send=pre, frame=tx['wire_hex'], received_frame=rx['wire_hex'], header_bytes=HEADER.size,
        payload_bytes=len(expected_body), sampled=SEND, sent=SEND, due=tx['due'], received=RECEIVE,
        receiver_before=old, receiver_after=new, order_at_53=[r['event'] for r in at53],
        actual_astar_at_53=[dict(goal=q['goal'], reachable=q['answer']['reachable'], reason=q['answer']['failure_reason'],
                                 cells=q['answer'].get('cells')) for q in plans],
        action_window=window, suffix_astar_calls=len(matching(ev, 'astar')), terminal_step=term['actual_length'],
        success=term['success'], collision=term['collision_episode'], timeout=term['truncated'],
        first_collision_step=term['first_collision_step'], collision_pairs=collision,
        terminal_reason=term['termination_reason'], full_episode=full, actual_suffix_acks=acknowledgments,
        suffix_wire=dict(tx_frames=len(outgoing), rx_frames=len(incoming),
            send_bits=sum(r['data']['accepted']['bits'] for r in outgoing),
            receive_bits=sum(8 * len(bytes.fromhex(r['data']['wire_hex'])) for r in incoming)),
        outstanding_frames=term['channel']['in_flight_frames'])


def paired_effects(left, right, stop):
    """First divergent action index in the shared suffix, with exact serialized observations."""
    fields = {
        'receiver_goal': ('trajectory', lambda d: d['agents'][1]['goal']),
        'receiver_route': ('trajectory', lambda d: d['agents'][1]['route']),
        'receiver_waypoint': ('trajectory', lambda d: d['agents'][1]['waypoint']),
        'four_applied_residuals': ('control', lambda d: d['applied_residual_action']),
        'four_prior_accelerations': ('control', lambda d: d['prior_acc']),
        'four_applied_accelerations': ('control', lambda d: d['acceleration']),
        'four_next_positions': ('control', lambda d: d['post_position']),
        'four_next_velocities': ('control', lambda d: d['post_velocity']),
        'four_rewards': ('transition', lambda d: d['rewards']),
        'complete_control': ('control', lambda d: d),
        'complete_navigation_and_collision': ('trajectory', lambda d: d),
        'complete_transition': ('transition', lambda d: d),
        'four_local_map_digests': ('local_maps', lambda d: d),
    }
    result = {}
    for name, (event, select) in fields.items():
        a, b = ([r for r in matching(run['events'], event) if r['step'] < stop] for run in (left, right))
        for rows in (a, b):
            check(name + ' action steps', [r['step'] for r in rows], list(range(SEND, stop)))
        result[name] = next((x['step'] for x, y in zip(a, b) if select(x['data']) != select(y['data'])), None)
    return result


def signature_check(label, events, expected, old_rows=None):
    got = reference_signature(events)
    if got != expected and old_rows is not None:
        prior = [dict(r, data=digest(r['data'])) if r['event'] == 'local_maps' else r for r in old_rows
                 if r['event'] != 'installed_command']
        rows = [r for r in events if r['event'] not in ('delivery_before', 'delivery_after')]
        for left, right in zip(rows, prior):
            if left != right:
                check(f"{label} t={right['step']} agent={right['agent']} event={right['event']}", left, right)
        check(label + ' count', len(rows), len(prior))
    check(label, got, expected)
    return got


def assess(folder, fixture, aa):
    common = load_trace(folder, 'common')
    native = load_trace(folder, 'CMD')
    prefix = signature_check('native shared prefix', common['events'], fixture['prefix'],
                             read(aa/'common.json')['events'] if aa else None)
    suffix = signature_check('native continued trajectory', native['events'], fixture['cmd_suffix'],
                             read(aa/'A.json')['events'] if aa else None)
    for key in ('summary', 'channel', 'evidence'):
        obj = behavioral_summary(native[key]) if key == 'summary' else native[key]
        check('native ' + key, digest(obj), fixture['cmd_' + key])
    boundary = common['pre_send']['state']
    check('A/A pre-send boundary', boundary, fixture['pre_send_fingerprint'])
    runs = {name: native if name == 'CMD' else load_trace(folder, name) for name in ARMS}
    arms = {name: arm_receipt(name, run, boundary) for name, run in runs.items()}
    for name in ARMS:
        check(name + ' equal receiver state before installing packet', arms[name]['receiver_before'], arms['CMD']['receiver_before'])
        check(name + ' equal tokens at send', arms[name]['pre_send']['tokens'], arms['CMD']['pre_send']['tokens'])
    stop = min(arm['terminal_step'] for arm in arms.values())
    for name, result in arms.items():
        result['common_interval'] = budget(runs[name]['channel'], stop)
    cmd = arms['CMD']
    wp54 = next(r['waypoint'] for r in cmd['action_window'] if r['action_step'] == 54)
    check('native archived landmarks', [wp54, cmd['first_collision_step'], cmd['collision_pairs'], cmd['terminal_reason'],
          cmd['terminal_step'], cmd['full_episode']], [[5., 15., 5.5], 56, [[1, 2]], 'obstacle_collision', 56,
          dict(send_bits=29056, receive_bits=28896)])
    effects = {a + '_vs_' + b: paired_effects(runs[a], runs[b], stop)
               for a, b in (('PAD', 'REPLAN_ONLY'), ('REPLAN_ONLY', 'CMD'))}
    return dict(schema=1, passed=True, scene=fixture['scene']['scenario_id'], seed=fixture['seed'],
                system=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, device='cpu', torch_threads=torch.get_num_threads(), training_updates=0,
                policy='ZeroResidualPolicy', navigation='causal_d2_v1', map_transport='own_history_v1',
                deadline=400, channel_config=fixture['channel'], fixture_sha256=sha(FIXTURE),
                shared_prefixes=1, independently_forked_arms=3, native_cut_sha256=boundary,
                native_regression=dict(prefix=prefix, suffix=suffix, summary=True, channel=True, evidence=True),
                first_difference_by_action_step=effects, common_interval_exclusive_end=stop, arms=arms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch', type=Path, help='also authenticate the original Linux batch')
    parser.add_argument('--aa', type=Path, help='also authenticate and compare the original Windows A/A logs')
    parser.add_argument('--assess-only', action='store_true', help='read existing traces and print acceptance without writing or simulating')
    args = parser.parse_args()
    torch.set_num_threads(1)
    fixture = read(FIXTURE)
    verify_inputs(fixture, args.batch, args.aa)
    if args.assess_only:
        result = assess(args.output, fixture, args.aa)
        print('PASS', result['native_cut_sha256'])
        return
    folder = new_output(args.output)
    try:
        capture(folder, fixture)
        result = assess(folder, fixture, args.aa)
        result.update(verified_original_batch=args.batch is not None, verified_original_aa=args.aa is not None)
    except Exception as error:
        write(folder/'failure.json', dict(passed=False, error=str(error)))
        raise
    write(folder/'validation.json', result)
    print('PASS', folder, flush=True)


if __name__ == '__main__':
    main()
