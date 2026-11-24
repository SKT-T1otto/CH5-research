"""Preregister the normal-difficulty development block before any task rollout."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.scenarios import validate_manifest
from chapter5_comm.pipeline import PROJECT, new_output
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from tools.executor_trace import read, sha, write


GENERATOR_SEED = 11240407
PROFILE = 'M20_MOVING_UNKNOWN_MULTI'
RULE = 'first_two_native_commands_with_changed_payload_since_last_accepted_tx_to_same_peer'
AUDIT = PROJECT / 'outputs/chapter5/audits/step3_d2_choice_dev24_20261124'
STAGE2 = PROJECT / 'outputs/chapter5/audits/case_0002_command_controls_20261124'


def preregister(folder, historical):
    old = read(historical)
    fixture = read(PROJECT / 'checks/fixtures/d2_command_controls_0002.json')
    if sha(historical) != fixture['plan_sha256'] or len(old['scenes']) != 30:
        raise ValueError('expected the archived 30-case batch plan')
    accepted = read(STAGE2 / 'validation.json')
    if not accepted['passed'] or accepted['seed'] != 2026111903:
        raise ValueError('missing accepted Step 2 control experiment')
    folder = new_output(folder)
    generated = build_scenario_manifests(count=24, generator_seed=GENERATOR_SEED, split='validation', profiles=[PROFILE])
    scenes = [dict(s, scenario_role='eval', scenario_split='eval', scenario_id=f'choice_dev_{GENERATOR_SEED}_{i:04d}')
              for i, s in enumerate(generated[PROFILE]['scenarios'])]
    validate_manifest(dict(scenarios=scenes), old['base'], evaluation=True)
    ids = {s['scenario_id'] for s in old['scenes']}
    fingerprints = {scenario_fingerprint(s) for s in old['scenes']}
    old_seeds = {s['scenario_seed'] for s in old['scenes']}
    assert all(s['scenario_id'] not in ids and scenario_fingerprint(s) not in fingerprints
               and s['scenario_seed'] not in old_seeds for s in scenes)
    assert len(scenes) == 24 and all(2 <= s['obstacle_count'] <= 4 and s['max_steps'] == 400 for s in scenes)
    plan = dict(schema='d2_choice_prereg_v1', recorded_utc=datetime.now(timezone.utc).isoformat(),
        stage2_receipt=sha(STAGE2 / 'validation.json'), original_30_plan=sha(historical),
        generator_seed=GENERATOR_SEED, generator_split='validation', purpose='development_screen',
        base_sha256=sha(PROJECT / 'configs/b2.json'), base=old['base'], channel=old['channel'],
        fixed_code={n: sha(PROJECT/n) for n in ('chapter5_comm/runtime.py', 'chapter5_comm/protocol.py',
            'chapter5_comm/d2_navigation.py', 'chapter5_comm/history_evidence.py', 'tools/command_control_trace.py',
            'core/scenarios/ch3_generator_impl.py', 'core/env/uav_env.py')},
        navigation='causal_d2_v1', map_transport='own_history_v1', policy='ZeroResidualPolicy',
        deadline=400, max_opportunities_per_scene=2, max_suffixes=144, eligible=RULE,
        selection='On the native CMD trunk, accept the earliest two opportunities from commander 0 to searchers 1/2 '
                  'whose coordinate bytes differ from its last actually transmitted COMMAND to that receiver. '
                  'Exclude first assignment per receiver. No receiver, delivery or future outcome filter. '
                  'No replacement of zero-opportunity scenes or early stop by outcomes.',
        controls=['CMD', 'PAD', 'REPLAN_ONLY'], bits_per_intervention=224,
        execution_difference='max absolute component difference > 1e-5 in receiver actual applied acceleration '
            'at an aligned physical action, at or after receipt, strictly before the earlier terminal state',
        accel_atol=1e-5, contrast_eligibility='native source at a shared pre-send cut, all arms counted even if terminal precedes arrival',
        gates=dict(pad_success_cmd_failure_distinct_scenes=2, cmd_success_pad_failure_in_two_other_scenes=2,
                   cmd_replan_execution_and_success_differ_distinct_scenes=2,
                   decision='all three required; otherwise stop reassignment learner'),
        secondary='First detector and detection state step, first actual TARGET handoff to executor, first executed waypoint, '
                  'collision, timeout, terminal step. None substitutes for binary mission success.',
        geometric_label=dict(origin='receiver_local_map_forecast_v1', horizon=8, candidates=4,
            at='actual arrival, after the native receipt/navigation and before applying the action, on an isolated diagnostic copy'),
        physical_seeds=[s['scenario_seed'] for s in scenes], physical_fingerprints=[scenario_fingerprint(s) for s in scenes],
        historical_physical_overlap=0, scenes=scenes)
    write(folder/'preregistration.json', plan)
    (folder/'preregistration.sha256').write_bytes((sha(folder/'preregistration.json')+'\n').encode())
    print('sealed 24 scenes', folder, sha(folder/'preregistration.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-plan', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=AUDIT)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    preregister(args.output, args.historical_plan)
