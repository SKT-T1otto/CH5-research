"""Backend-neutral deterministic evaluation and seed-level metric summaries."""
from collections import defaultdict
from typing import Protocol
import time
import numpy as np
from chapter3_bser.baselines.environment import PureMARLEnv
from chapter3_bser.learning.config import scenario_fingerprint


class EvaluationPolicy(Protocol):
    def reset(self) -> None: ...
    def act(self, observations: np.ndarray, *, deterministic: bool = True) -> np.ndarray: ...


def evaluate(policy: EvaluationPolicy, reference_config, scenarios, *, algorithm, training_seed,
             training_fingerprints, mechanism=False, execute=False):
    """No tensor restore and no model updates. Caller supplies native actors.

    Only explicit bounded interface checks are executable in this preparation
    release. Native backend and current-source acceptance precede formal runs.
    """
    if not execute or not mechanism:
        raise ValueError('preparation release: only explicit execute=True, mechanism=True interface checks are enabled')
    from chapter3_bser.learning.scenarios import validate_manifest
    validate_manifest(dict(scenarios=scenarios), reference_config, evaluation=True, mechanism=True)
    if not scenarios or any(s['scenario_role'] not in ('eval', 'smoke_eval') for s in scenarios):
        raise ValueError('evaluation requires an explicit non-training manifest')
    if mechanism and (len(scenarios) > 8 or reference_config['max_steps'] > 32):
        raise ValueError('mechanism evaluation is limited to 8 x 32')
    fingerprints = {scenario_fingerprint(s) for s in scenarios}
    if len(fingerprints) != len(scenarios) or fingerprints.intersection(training_fingerprints):
        raise ValueError('repeated or training-overlapping evaluation scenarios')
    env, records = PureMARLEnv(reference_config), []
    try:
        for scenario in scenarios:
            obs, _ = env.reset(scenario)
            policy.reset()
            started = time.perf_counter()
            policy_seconds, clipped = 0., 0
            while True:
                tick = time.perf_counter()
                action = policy.act(obs.copy(), deterministic=True)
                policy_seconds += time.perf_counter() - tick
                obs, _, _, terminated, truncated, info = env.step(action)
                clipped += info['action_clipped_components']
                if terminated or truncated:
                    break
                if env.unwrapped.step_count >= reference_config['max_steps']:
                    raise RuntimeError('simulator failed to emit finite-horizon terminal')
            row = env.summary()
            row.update(algorithm=algorithm, training_seed=int(training_seed),
                       scenario_fingerprint=scenario_fingerprint(scenario), deterministic=True,
                       wall_seconds=time.perf_counter() - started, policy_seconds=policy_seconds,
                       action_clipped_components=clipped,
                       penalized_completion_steps=(row['success_step'] if row['success'] else
                           reference_config['max_steps'] + env.unwrapped.failure_penalty_steps))
            records.append(row)
    finally:
        env.close()
    return records


def summarize(records):
    """Aggregate scenes within seed, then weight independent training seeds equally."""
    if not records:
        raise ValueError('no episode records')
    grouped, identities = defaultdict(list), set()
    scenario_sets = defaultdict(set)
    for row in records:
        if row.get('schema') != 'auv.pure_marl.episode.v1' or not row.get('terminated') or row.get('truncated'):
            raise ValueError('complete episodes in the common metric schema required')
        for metric in ('team_return', 'path_length', 'control_effort', 'penalized_completion_steps'):
            if not np.isfinite(row[metric]):
                raise ValueError('nonfinite evaluation metric')
        key = (row['algorithm'], row['training_seed'], row['scenario_fingerprint'])
        if key in identities:
            raise ValueError('duplicate algorithm/seed/scenario record')
        identities.add(key)
        grouped[key[:2]].append(row)
        scenario_sets[key[:2]].add(key[2])
    if any(s != next(iter(scenario_sets.values())) for s in scenario_sets.values()):
        raise ValueError('algorithms and training seeds must use the identical complete evaluation set')
    for field in ('task_protocol', 'collision_detection_revision', 'terminal_reward_revision', 'max_steps'):
        if len({row[field] for row in records}) != 1:
            raise ValueError('evaluation task/protocol/horizon mismatch: ' + field)
    per_algorithm = defaultdict(list)
    for (algorithm, seed), rows in sorted(grouped.items()):
        per_algorithm[algorithm].append(dict(training_seed=seed, episodes=len(rows),
            success_rate=float(np.mean([r['success'] for r in rows])),
            collision_rate=float(np.mean([r['collision_episode'] for r in rows])),
            discovery_rate=float(np.mean([r['found_step'] is not None for r in rows])),
            penalized_completion_steps=float(np.mean([r['penalized_completion_steps'] for r in rows])),
            team_return=float(np.mean([r['team_return'] for r in rows])),
            path_length=float(np.mean([r['path_length'] for r in rows])),
            control_effort=float(np.mean([r['control_effort'] for r in rows]))))
    seed_sets = [{s['training_seed'] for s in seeds} for seeds in per_algorithm.values()]
    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError('algorithms must have identical complete training seed sets')
    output = {}
    for algorithm, seeds in per_algorithm.items():
        stats = {}
        for metric in seeds[0].keys() - {'training_seed', 'episodes'}:
            values = np.asarray([s[metric] for s in seeds])
            ci = None
            if len(values) > 1:
                generator = np.random.default_rng(87291)
                boot = generator.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
                ci = np.quantile(boot, [.025, .975]).tolist()
            stats[metric] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else None,
                                 seed_bootstrap_95_interval=ci)
        output[algorithm] = dict(training_seeds=len(seeds), per_seed=seeds, metrics=stats)
    return dict(schema='auv.pure_marl.summary.v1', algorithms=output, performance_passed=False,
                confidence_scope='training-seed bootstrap conditional on the fixed evaluation scenarios')
