"""Closed-loop raw SCAN versus own historical evidence, zero residual, no training."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch

from chapter3_bser.learning.reference import ZeroResidualPolicy
from tools.executor_trace import digest, plain, tap_method
from .d2_diagnostics import PRESET, _initial_physics
from .d2_collision_inputs import BATCH_PLAN, checked_json
from .experiment_config import read_experiment, source_identity
from .experiment_runner import _git_commit
from .pipeline import PROJECT, new_output
from .runtime import CH5Runtime


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding='utf-8')


def evaluate(base, scene, seed, channel, transport, folder):
    runtime = CH5Runtime(base, scene, seed=seed, channel_config=channel,
                         navigation='causal_d2_v1', map_transport=transport)
    frames, replans, grids = [], [], []
    try:
        raw = runtime.physical
        initial = digest(_initial_physics(raw))

        def query(agent):
            def observe(when, args, kwargs, result):
                if when == 'after':
                    answer = result[1]
                    replans.append(dict(step=runtime.step, agent=agent, start=args[0], goal=args[1],
                        reachable=answer['reachable'], reason=answer['failure_reason']))
            return observe

        def dynamics(when, args, kwargs, result):
            if when == 'before':
                agents = []
                for i, local in enumerate(runtime.locals):
                    cell = local.mapper._cell_from_point_np(local.waypoint)
                    agents.append(dict(agent=i, position=raw._agent_pos[i].tolist(),
                        velocity=raw._agent_vel[i].tolist(), goal=local.goal.copy(), route=local.path.copy(),
                        route_index=local.path_index, waypoint=local.waypoint.copy(), waypoint_cell=cell,
                        waypoint_probability=float(local.mapper.occupancy_probability[cell]),
                        last_plan=local.last_plan))
                grids.append(np.stack([l.mapper.occupancy_probability.cpu().numpy().copy() for l in runtime.locals]))
                return dict(step=runtime.step, agents=agents)
            frame = result[0]
            frame.update(next_positions=raw._agent_pos.tolist(), collisions=raw.last_collision_records)
            frames.append(plain(frame))

        with ExitStack() as stack:
            for i, local in enumerate(runtime.locals):
                stack.enter_context(tap_method(local.mapper, 'grid_astar_path', query(i)))
            stack.enter_context(tap_method(raw, '_apply_agent_dynamics', dynamics))
            while not runtime.terminal:
                runtime.advance(model=ZeroResidualPolicy(), explore=False)
                if runtime.step % 25 == 0:
                    print(json.dumps(dict(arm=transport, step=runtime.step)), flush=True)
        summary = runtime.summary()
        save(folder/'trajectory.json', frames)
        save(folder/'replans.json', replans)
        save(folder/'channel.json', runtime.channel.events)
        save(folder/'evidence.json', runtime.evidence_events)
        np.savez_compressed(folder/'occupancy.npz', probability=np.asarray(grids))
        save(folder/'summary.json', summary)
        return dict(initial=initial, steps=runtime.step, terminal=runtime.terminal,
            collision=summary['collision_episode'], success=summary['success'],
            termination=summary['termination_reason'], channel=summary['channel'], events=runtime.channel.events)
    finally:
        runtime.close()


def execute(batch, output, cases):
    torch.set_num_threads(1)
    config, base = read_experiment(PRESET)
    plan = checked_json(batch/'plan.json', BATCH_PLAN)
    scenes = json.loads((batch/'scenarios.json').read_bytes())
    from hashlib import sha256
    if sha256((batch/'scenarios.json').read_bytes()).hexdigest() != '681c5afe276ad329c20cbb6b2f5c37aa1aad5b8afbdb504096a6712398324eb9':
        raise ValueError('scene manifest mismatch')
    allowed = (PROJECT/'outputs/chapter5').resolve()
    if not output.resolve().is_relative_to(allowed) or output.resolve() == allowed:
        raise ValueError('output must be a new child of outputs/chapter5')
    output = new_output(output)
    channel = dict(config['channel'], **config['evaluation_channels']['reliable_reference'])
    save(output/'plan.json', dict(cases=cases, base=base, scenes=[scenes[i] for i in cases],
        seeds={i:plan['cases'][i]['physical_seed'] for i in cases}, channel=channel,
        source=source_identity(), commit=_git_commit(), training_updates=0,
        payload_bits=960, frame_bits=1088, deadline=base['max_steps']))
    reports = []
    for i in cases:
        arms = {}
        for transport in ('raw_scan', 'own_history_v1'):
            arms[transport] = evaluate(base, scenes[i], plan['cases'][i]['physical_seed'], channel,
                                      transport, output/f'case_{i:04d}'/transport)
        a,b = arms.values()
        # Check all actual send opportunities in the common physical prefix.
        end = min(a['steps'], b['steps'])
        def schedule(row):
            return [(e['send_step'], e['sender'], e['receiver'],
                     'SCAN' if e['kind']=='MAP_EVIDENCE' else e['kind'], e['bits'])
                    for e in row['events'] if e['send_step'] < end]
        report = dict(case=i, common_steps=end, same_initial=a['initial']==b['initial'],
            identical_common_schedule=schedule(a)==schedule(b),
            common_sent_bits={k:sum(e['bits'] for e in v['events'] if e['send_step'] < end)
                              for k,v in arms.items()},
            common_delivered_bits={k:sum(e['bits'] for e in v['events']
                if e['delivery_step'] is not None and e['delivery_step'] < end) for k,v in arms.items()},
            arms={k:{key:v for key,v in val.items() if key!='events'} for k,val in arms.items()})
        reports.append(report)
        save(output/'report.json', reports)
        print(json.dumps(plain(report)), flush=True)
    save(output/'created_files.json', [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', default=[3,9])
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases) or any(i not in range(30) for i in args.cases):
        parser.error('unique case indices 0..29 required')
    execute(args.batch, args.output, args.cases)


if __name__ == '__main__':
    main()
