"""Validated experiment specifications composed from pinned upstream configs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from project_paths import ROOT
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.scenarios import validate_manifest

ALGORITHMS = {'maddpg': 'harl', 'matd3': 'harl', 'mappo': 'harl', 'happo': 'harl', 'masac': 'benchmarl'}
DEFAULT_SUITE = ROOT / 'configs/pure_marl/suite_v1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_suite(path=DEFAULT_SUITE):
    suite = read(path)
    if (suite['schema'] != 'auv.pure_marl.suite.v1' or suite['algorithms'] != ALGORITHMS
            or suite['execution_enabled'] is not False):
        raise ValueError('this release prepares the five specified baselines only')
    if (suite['environment']['action_id'] != 'direct_normalized_acceleration_v1'
            or suite['environment']['observation_id'] != 'public28_without_planner_v1'
            or suite['environment']['gamma'] != .95 or suite['environment']['n_agents'] != 4):
        raise ValueError('pure-MARL environment contract changed')
    reference_path = ROOT / suite['reference_config']['path']
    if sha(reference_path) != suite['reference_config']['sha256']:
        raise ValueError('reference configuration changed; review a new suite revision')
    reference = read(reference_path)
    if (reference['rl']['gamma'] != .95 or reference['reward_objective'] != 'team_mean_v1'
            or reference['task_protocol'] != 'collision_terminal_v1'
            or reference['max_steps'] != suite['environment']['max_steps']):
        raise ValueError('common reward/protocol/horizon differs from the reference')
    seeds = suite['training_seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('distinct nonnegative integer training seeds required')
    steps = suite['budget']['joint_environment_steps']
    if type(steps) is not int or steps < 1 or steps % reference['max_steps']:
        raise ValueError('on-policy budget must divide into whole collection horizons')
    upstream = ROOT / 'configs/pure_marl/upstream'
    lock = read(upstream / 'lock.json')
    for backend in set(ALGORITHMS.values()):
        entry = lock['sources'][backend]
        if len(entry['commit']) != 40:
            raise ValueError('full upstream commit required')
        for item in entry['files'].values():
            if sha(upstream / item['local']) != item['sha256']:
                raise ValueError('pinned upstream snapshot changed')
    return suite, reference, lock


def native_config(algorithm, seed, suite, output):
    """Overlay experiment conditions, leaving native update/loss hyperparameters intact."""
    folder = ROOT / 'configs/pure_marl/upstream'
    defaults = read(ROOT / 'configs/pure_marl/native_defaults.json')
    if algorithm not in ALGORITHMS:
        raise ValueError('unknown baseline')
    if ALGORITHMS[algorithm] == 'harl':
        config = copy.deepcopy(defaults[f'harl/harl/configs/algos_cfgs/{algorithm}.yaml'])
        config['seed'].update(seed_specify=True, seed=seed)
        config['train'].update(n_rollout_threads=1, num_env_steps=suite['budget']['joint_environment_steps'], model_dir=None)
        config['eval'].update(use_eval=False)  # Common evaluator owns fixed dev/test sets.
        config['render']['use_render'] = False
        config['algo'].update(gamma=.95, share_param=False)
        if algorithm in ('mappo', 'happo'):
            config['train']['episode_length'] = suite['environment']['max_steps']
        config['logger']['log_dir'] = str(output)
        return dict(algo_args=config, main_args=dict(algo=algorithm, env='auv_pure_marl_v1'),
                    env_args=dict(state_type='EP', n_agents=4))
    def snapshot(relative):
        value = copy.deepcopy(defaults['benchmarl/benchmarl/conf/' + relative])
        value.pop('defaults', None)
        return value
    algorithm_config = snapshot('algorithm/masac.yaml')
    algorithm_config['share_param_critic'] = False
    experiment = snapshot('experiment/base_experiment.yaml')
    experiment.update(gamma=.95, share_policy_params=False, max_n_frames=suite['budget']['joint_environment_steps'],
                      max_n_iters=None, off_policy_n_envs_per_worker=1, evaluation=False, render=False,
                      loggers=['csv'], save_folder=str(output), restore_file=None,
                      checkpoint_interval=suite['evaluation']['interval_joint_steps'],
                      checkpoint_at_end=True, keep_checkpoints_num=None)
    model = snapshot('model/layers/mlp.yaml')
    # Keep native MASAC losses, alpha learning, targets, minibatches, and UTD.
    return dict(algorithm=algorithm_config, experiment=experiment, model=model,
                critic_model=copy.deepcopy(model))


def load_splits(paths, reference):
    scenes, used = {}, {key: set() for key in ('ids', 'seeds', 'fingerprints')}
    if set(paths) != {'train', 'dev', 'test'}:
        raise ValueError('all three explicit manifests are required')
    for split in ('train', 'dev', 'test'):
        manifest = read(paths[split])
        rows = validate_manifest(manifest, reference, evaluation=split != 'train')
        values = dict(ids={s['scenario_id'] for s in rows}, seeds={s['scenario_seed'] for s in rows},
                      fingerprints={scenario_fingerprint(s) for s in rows})
        for key in used:
            if values[key] & used[key]:
                raise ValueError(f'{split} overlaps another split by {key}')
            used[key].update(values[key])
        scenes[split] = rows
    return scenes


def plan(path=DEFAULT_SUITE):
    suite, reference, lock = load_suite(path)
    jobs = [dict(job_id=f'{algo}_seed{seed}', algorithm=algo, backend=backend, seed=seed,
                 upstream_commit=lock['sources'][backend]['commit'])
            for algo, backend in ALGORITHMS.items() for seed in suite['training_seeds']]
    return dict(schema=suite['schema'], stage='preparation_only', executed=False, training_started=False,
                execution_enabled=False, jobs=jobs, budget=suite['budget'], environment=suite['environment'],
                installed={name: importlib.util.find_spec(name) is not None
                           for name in ('harl', 'benchmarl', 'torchrl', 'tensordict')},
                pending_native_acceptance=['install isolated pinned backends', 'native runner registration',
                    'TorchRL spec check', 'bounded native collector/update checks',
                    'checkpoint identity and independent evaluation roundtrip', 'current-source execution receipt'])
