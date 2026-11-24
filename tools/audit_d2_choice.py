"""Run sealed development cases. Only a selected native COMMAND transmission is replaced."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter5_comm.data import COMMAND_ARMS, build_route_encounter_labels, command_triplet_record
from chapter5_comm.pipeline import PROJECT
from chapter5_comm.protocol import D2CommandOverride, Kind, Packet
from chapter5_comm.runtime import CH5Runtime, GOAL3
from tools.audit_d2_choice_plan import AUDIT, RULE
from tools.command_control_trace import fork, restore_rng, save_rng
from tools.audit_d2_choice_state import fingerprint
from tools.executor_trace import plain, read, sha, tap_method


def emit(path, data):
    """Publish a complete case result. Incomplete temporary files never qualify for resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    if path.exists():
        raise FileExistsError(path)
    with temp.open('xb') as stream:
        raw = json.dumps(data, allow_nan=False, ensure_ascii=False, separators=(',', ':')).encode()
        if path.suffix == '.gz':
            with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as zipped:
                zipped.write(raw)
        else:
            stream.write(raw)
    temp.rename(path)


def checked_plan(folder):
    src = folder/'preregistration.json'
    if sha(src) != (folder/'preregistration.sha256').read_text().strip():
        raise ValueError('preregistration seal mismatch')
    p = read(src)
    if p['eligible'] != RULE or p['max_suffixes'] != 144 or len(p['scenes']) != 24:
        raise ValueError('unexpected preregistration')
    if sha(PROJECT/'configs/b2.json') != p['base_sha256']:
        raise ValueError('base changed since registration')
    for name, wanted in p['fixed_code'].items():
        if sha(PROJECT/name) != wanted:
            raise ValueError('fixed source changed: '+name)
    return p, sha(src)


def received_state(rt, receiver):
    local = rt.locals[receiver]
    return plain(dict(command=local.command, sample=local.last_command_step, plan=local.last_plan,
                      goal=local.goal, waypoint=local.waypoint, acks=list(local.acks)))


def private_map(rt, receiver):
    m = rt.locals[receiver].mapper
    return dict(agent=receiver, step=rt.step, fused_logodds=m.occupancy_logodds.tolist(),
                observation_count=m.occupancy_observation_count.tolist(), last_observed=m.occupancy_last_observed_step.tolist())


class SendCut(Exception):
    def __init__(self, receiver, payload, sample):
        self.receiver, self.payload, self.sample = receiver, payload, sample


class Observe:
    def __init__(self, rt, *, native=False, prior_handoff=None):
        self.rt = rt
        self.native = native
        self.previous = {}
        self.watch = {}
        self.receipts = {}
        self.forecasts = {}
        self.tape = []
        self.handoff = prior_handoff
        self.native_commands = 0
        self.command_log = []

    def send(self, when, args, kwargs, result):
        rt = self.rt
        kind, sender, receiver, body, sample = args
        if when == 'before':
            prev = self.previous.get(receiver)
            if (self.native and len(self.watch) < 2 and rt.channel.sequence not in self.watch and
                    kind == Kind.COMMAND and sender == 0 and receiver in (1, 2) and prev and
                    body != GOAL3.pack(*prev['goal']) and rt.channel.remaining_bits(0) >= 224):
                raise SendCut(receiver, body, sample)
            seq = rt.channel.sequence
            if seq in self.watch:
                assert fingerprint(rt) == self.watch[seq]['prefix_sha256'], 'pre-send state changed on resume'
            return
        sent = result[1]
        if not sent['accepted']:
            return
        seq = sent['packet_id']
        if kind != Kind.COMMAND and seq not in self.watch:
            return
        raw = next(q[2] for q in rt.channel.queue if q[3]['packet_id'] == seq)
        pkt = Packet.from_bytes(raw)
        if pkt.kind == Kind.COMMAND:
            self.native_commands += 1
            self.previous[receiver] = dict(goal=list(GOAL3.unpack(pkt.payload)), sample_step=pkt.sample_step, send_step=pkt.send_step)
            self.command_log.append(dict(packet=seq, wire_hex=raw.hex()))
        if seq in self.watch:
            self.receipts[seq] = dict(prefix_sha256=self.watch[seq]['prefix_sha256'], sent_hex=raw.hex(),
                scheduled_delivery=sent['scheduled_delivery_step'], arrival=None, received_hex=None, next_executed=None)

    def deliver(self, when, args, kwargs, result):
        rt = self.rt
        pkt, due = args
        if when == 'before':
            if pkt.sequence in self.watch:
                return received_state(rt, pkt.receiver)
            return bool(rt.physical.executor_target_assigned)
        if pkt.sequence in self.watch:
            old = result[0]
            new = received_state(rt, pkt.receiver)
            assert new['acks'] == (old['acks'] + [[pkt.sender, pkt.sequence]])[-32:]
            if pkt.kind in (Kind.COMMAND_PAD, Kind.REPLAN_ONLY):
                assert (new['command'], new['sample']) == (old['command'], old['sample'])
                if pkt.kind == Kind.COMMAND_PAD:
                    assert old['plan'] == new['plan']
                elif pkt.sample_step > old['sample']:
                    assert new['plan'] == -100000
            self.receipts[pkt.sequence].update(arrival=rt.step, received_hex=pkt.to_bytes().hex(),
                receiver_effect=dict(old_command_sample=old['sample'], new_command_sample=new['sample'],
                                     last_plan_before=old['plan'], last_plan_after=new['plan'], ack_enqueued=True))
        if pkt.kind == Kind.TARGET and pkt.receiver == 3 and not result[0] and rt.physical.executor_target_assigned:
            assert self.handoff is None
            self.handoff = dict(sender=pkt.sender, receiver=3, packet_id=pkt.sequence, sample_step=pkt.sample_step,
                                send_step=pkt.send_step, arrival_step=rt.step, due=due)

    def save_arrival_forecasts(self):
        rt = self.rt
        for seq, packet in self.receipts.items():
            if packet['arrival'] != rt.step:
                continue
            rc = self.watch[seq]['receiver']
            state = save_rng()
            diagnostic = fork(rt)
            try:
                forecast = diagnostic.route_features(rc, horizon=8, candidate_count=4)
                self.forecasts[seq] = plain(dict(forecast=forecast, receiver_map=private_map(diagnostic, rc),
                    radius=float(diagnostic.physical._sensor_range[rc])+diagnostic.physical.detect_eps_bias))
            finally:
                diagnostic.close()
                restore_rng(state)

    def dynamics(self, when, args, kwargs, result):
        raw = self.rt.physical
        if when == 'before':
            return dict(step=self.rt.step, waypoint=raw._nav_targets.tolist(), position=raw._agent_pos.tolist(),
                        residual=plain(args[0]), target=raw.target_state.position.tolist())
        row = result[0]
        row.update(applied_acc=raw._agent_acc.tolist(), prior_acc=raw._last_prior_acc.tolist(),
                   next_position=raw._agent_pos.tolist(), next_velocity=raw._agent_vel.tolist())
        self.tape.append(row)

    def advance_to_cut_or_terminal(self, *, resume=False):
        rt, policy = self.rt, ZeroResidualPolicy()
        while not rt.terminal:
            assert rt.step < 400
            with ExitStack() as taps:
                taps.enter_context(tap_method(rt, '_send', self.send))
                taps.enter_context(tap_method(rt, '_deliver', self.deliver))
                if resume:
                    assert rt._bootstrap()['accepted']
                    rt._begun_step = rt.step
                    resume = False
                else:
                    rt.begin_step()
            self.save_arrival_forecasts()
            with tap_method(rt.physical, '_apply_agent_dynamics', self.dynamics):
                tr = rt.advance(model=policy, explore=False)
            row = self.tape[-1]
            assert row['step'] == rt.step-1 and not np.any(row['residual'])
            row.update(next_target=rt.physical.target_state.position.tolist(), rewards=plain(tr['rewards']))
            for seq, pkt in self.receipts.items():
                if pkt['arrival'] == row['step']:
                    rc = self.watch[seq]['receiver']
                    pkt['next_executed'] = dict(action_step=row['step'], waypoint=row['waypoint'][rc], applied_acc=row['applied_acc'][rc])

    def finish(self):
        rt = self.rt
        s = rt.summary()
        found = bool(s['found'])
        assert s['task_deadline_steps'] == 400 and rt.terminal and (s['terminated'] or s['truncated'])
        assert s['channel']['dropped_frames'] == s['channel']['rejected_frames'] == 0
        assert s['handoff_event_step'] == (self.handoff['arrival_step'] if self.handoff else None)
        outcome = plain(dict(success=bool(s['success']), collision=bool(s['collision_episode']), timeout=s['termination_reason']=='timeout',
            terminal_step=rt.step, reason=s['termination_reason'], first_discovery_step=s['found_step'] if found else None,
            first_finder=int(rt.physical.finder_idx) if found else None, first_handoff=self.handoff,
            first_collision_step=s['first_collision_step'], collisions=[[c['agent_id'], c['obstacle_id']] for c in s['collision_records']],
            success_step=s['success_step'], full_tx_bits=s['channel']['sent_bits'], full_rx_bits=s['channel']['delivered_bits']))
        return dict(outcome=outcome, tape=self.tape, receipts=self.receipts, forecasts=self.forecasts,
                    native_commands=self.command_log, channel=deepcopy(rt.channel.events))


def sender_inputs(native, cut):
    rt = native.rt
    local, rc = rt.locals[0], cut.receiver
    peer = local.peers.get(rc)
    query = local.queries.get(rc)
    return plain(dict(sender=0, receiver=rc, send_step=rt.step, sample_step=cut.sample, candidate_goal=GOAL3.unpack(cut.payload),
        previous_command=native.previous[rc], available_bits=rt.channel.remaining_bits(0),
        query_sample_step=query[0] if query else None, peer_sample_step=peer[0] if peer else None, tokens=rt.tokenize(0, rc)))


def labels(run, seq, at):
    if at is None:
        return None
    source = run['forecasts'][seq]
    route = source['forecast']
    frames = [r for r in run['tape'] if at <= r['step'] < at+8]
    assert frames and [r['step'] for r in frames] == list(range(at, at+len(frames)))
    targets = [frames[0]['target']] + [r['next_target'] for r in frames]
    return plain(dict(origin=route['route_origin'], horizon=8, forecast_step=at,
        **build_route_encounter_labels(route['route_positions'], targets, 8, source['radius'], route['route_valid_steps'])))


def effects(runs, rc, seq, atol):
    stop = min(run['outcome']['terminal_step'] for run in runs.values())
    costs = {}
    for name, run in runs.items():
        ch = run['channel']
        costs[name] = dict(tx=sum(e['bits'] for e in ch if e['send_step'] < stop),
            rx=sum(e['bits'] for e in ch if e['delivery_step'] is not None and e['delivery_step'] < stop))
    def compare(a, b):
        arrival = runs[a]['receipts'][seq]['arrival']
        if arrival is None:
            return dict(waypoint=None, applied_acc=None)
        end = min(runs[x]['outcome']['terminal_step'] for x in (a,b))
        aa, bb = ([r for r in runs[x]['tape'] if arrival <= r['step'] < end] for x in (a,b))
        assert [r['step'] for r in aa] == [r['step'] for r in bb] == list(range(arrival,end))
        return {key: next((l['step'] for l,r in zip(aa,bb)
                    if np.max(np.abs(np.subtract(l[key][rc],r[key][rc]))) > atol), None)
                for key in ('waypoint','applied_acc')}
    return dict(common_end_exclusive=stop, common_bits=costs,
        cmd_vs_replan=compare('CMD','REPLAN_ONLY'), pad_vs_replan=compare('PAD','REPLAN_ONLY'), cmd_vs_pad=compare('CMD','PAD'))


def finish_capture(folder, i, plan, seal, diag):
    """Build the primary record from a complete capture, also usable after an interrupted export."""
    assert diag['schema']=='d2_choice_full_capture_v3' and diag['preregistration_sha256']==seal
    assert diag['scene']==plan['scenes'][i]['scenario_id']
    ref=f'diagnostics/case_{i:04d}.json.gz'
    baseline=diag['native']
    records=[]
    for seq, bundle in diag['alternatives'].items():
        op=bundle['opportunity']
        runs=dict(CMD=baseline,PAD=bundle['PAD'],REPLAN_ONLY=bundle['REPLAN_ONLY'])
        arms={name:dict(**run['receipts'][seq],outcome=run['outcome']) for name,run in runs.items()}
        geometric={name:labels(run,seq,arms[name]['arrival']) for name,run in runs.items()}
        record=command_triplet_record(op,bundle['sender_features'],arms,geometric,ref)
        record['paired_observations']=effects(runs,op['receiver'],seq,plan['accel_atol'])
        records.append(record)
    result=dict(case=i,scene=diag['scene'],preregistration_sha256=seal,native_outcome=baseline['outcome'],
        native_commands=len(baseline['native_commands']),opportunities=len(records),suffixes=3*len(records),
        zero_opportunity_reason=None if records else 'no changed native endpoint sent to a previously commanded peer before termination',
        diagnostics=ref,diagnostics_sha256=sha(folder/ref),records=records,wall_seconds=diag['simulation_seconds'])
    emit(folder/'cases'/f'{i:04d}.json',result)
    print(f'case {i:02d} complete: {len(records)} opportunities, {result["wall_seconds"]} s',flush=True)


def run_case(folder, i, plan, seal):
    started = time.monotonic()
    scene = plan['scenes'][i]
    rt = CH5Runtime(plan['base'], scene, seed=plan['physical_seeds'][i], channel_config=plan['channel'],
                    navigation=plan['navigation'], map_transport=plan['map_transport'])
    candidates = []
    try:
        assert rt.physical.device.type == 'cpu'
        native = Observe(rt, native=True)
        obstacles = plain(rt.physical.ground_truth_obstacles)
        resume = False
        while not rt.terminal:
            try:
                native.advance_to_cut_or_terminal(resume=resume)
            except SendCut as cut:
                rc, step, seq = cut.receiver, rt.step, rt.channel.sequence
                boundary = fingerprint(rt)
                op = dict(scene=scene['scenario_id'], case=i, physical_seed=rt.seed, send_step=step, receiver=rc,
                          packet_id=seq, prefix_sha256=boundary, preregistration_sha256=seal)
                features = sender_inputs(native, cut)
                native.watch[seq] = op
                private = private_map(rt, rc)
                state = save_rng()
                alternatives = {}
                with ExitStack() as lifetime:
                    copies = {name: fork(rt) for name in ('PAD', 'REPLAN_ONLY')}
                    for copy in copies.values():
                        lifetime.callback(copy.close)
                    for name, copy in copies.items():
                        restore_rng(state)
                        copy.command_override = D2CommandOverride(step, rc, Kind.COMMAND_PAD if name == 'PAD' else Kind.REPLAN_ONLY)
                        monitor = Observe(copy, prior_handoff=deepcopy(native.handoff))
                        monitor.watch[seq] = op
                        monitor.advance_to_cut_or_terminal(resume=True)
                        alternatives[name] = monitor.finish()
                restore_rng(state)
                assert fingerprint(rt) == boundary
                candidates.append((seq, op, features, private, alternatives))
                resume = True
        baseline = native.finish()
        ref = f'diagnostics/case_{i:04d}.json.gz'
        diagnostics = dict(schema='d2_choice_full_capture_v3', scene=scene['scenario_id'], preregistration_sha256=seal,
                           true_obstacles=obstacles, native=baseline, alternatives={},
                           simulation_seconds=round(time.monotonic()-started,2))
        for seq, op, features, private, alternatives in candidates:
            diagnostics['alternatives'][seq] = dict(opportunity=op, sender_features=features, at_cut_receiver_map=private, **alternatives)
        emit(folder/ref, diagnostics)
        finish_capture(folder,i,plan,seal,diagnostics)
    finally:
        rt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=AUDIT)
    parser.add_argument('--cases', nargs='+', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan, seal = checked_plan(args.output)
    for case in args.cases:
        if case not in range(24):
            raise ValueError('unregistered case')
        saved = args.output/'cases'/f'{case:04d}.json'
        if saved.exists():
            old = read(saved)
            if old['case'] != case or old['preregistration_sha256'] != seal or sha(args.output/old['diagnostics']) != old['diagnostics_sha256']:
                raise ValueError('inconsistent completed case')
        elif (capture:=args.output/'diagnostics'/f'case_{case:04d}.json.gz').exists():
            with gzip.open(capture,'rt',encoding='utf8') as f:
                finish_capture(args.output,case,plan,seal,json.load(f))
        else:
            run_case(args.output, case, plan, seal)
