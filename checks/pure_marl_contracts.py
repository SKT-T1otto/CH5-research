"""Focused baseline preparation checks; no native training or checkpoint load.

Explicit module name keeps immutable parent regression inventory separate.
Physical checks use two temporary environments x four steps only.
"""
import copy
import importlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import numpy as np
from project_paths import ROOT
from chapter3_bser.baselines.environment import PureMARLEnv, public_observation, MASKED_COLUMNS
from chapter3_bser.baselines.experiment import load_suite, native_config, plan, load_splits, write
from chapter3_bser.baselines.evaluation import evaluate, summarize
from chapter3_bser.baselines.harl_interface import HARLPolicy
from chapter3_bser.baselines.metrics import EpisodeRecorder
from chapter3_bser.baselines.workflow import main
from core.scenarios.ch3_generator_impl import build_scenario_manifests


def manifest(split='train', seed=981729, steps=4):
    role = 'train' if split == 'train' else 'eval'
    raw = build_scenario_manifests(count=1, generator_seed=seed,
        split='train' if role == 'train' else 'validation',
        profiles=['M20_MOVING_UNKNOWN_MULTI'])['M20_MOVING_UNKNOWN_MULTI']
    for scene in raw['scenarios']:
        scene.update(scenario_role=role, scenario_split=role, max_steps=steps)
        scene['scenario_id'] = split + '_' + scene['scenario_id']
    return raw


def record(algo='maddpg', seed=1, scenario='a', success=False):
    return dict(schema='auv.pure_marl.episode.v1', algorithm=algo, training_seed=seed,
        scenario_fingerprint=scenario, success=success, collision_episode=not success,
        found_step=1 if success else None, penalized_completion_steps=2 if success else 500,
        team_return=2. if success else -2., path_length=1., control_effort=.1,
        terminated=True, truncated=False, task_protocol='collision_terminal_v1',
        collision_detection_revision='segment_closed_aabb_v1', terminal_reward_revision='team_failure_override_v1', max_steps=400)


class PureMARLContracts(unittest.TestCase):
    def test_01_five_algorithms_and_proposed_budget(self):
        result = plan()
        self.assertEqual(len(result['jobs']), 25)
        self.assertFalse(result['training_started'])
        self.assertFalse(result['execution_enabled'])
        self.assertEqual(result['budget']['joint_environment_steps'] % 6000, 0)

    def test_02_native_maddpg_does_not_inherit_local_twin_critic(self):
        suite, _, _ = load_suite()
        config = native_config('maddpg', 2, suite, Path('unused'))['algo_args']
        defaults = json.loads((ROOT / 'configs/pure_marl/native_defaults.json').read_text())
        original = defaults['harl/harl/configs/algos_cfgs/maddpg.yaml']
        for key in ('batch_size', 'polyak', 'expl_noise', 'buffer_size', 'n_step'):
            self.assertEqual(config['algo'][key], original['algo'][key])
        self.assertNotIn('policy_freq', config['algo'])
        self.assertEqual(config['algo']['gamma'], .95)

    def test_03_matd3_and_happo_mechanisms_retained(self):
        suite, _, _ = load_suite()
        td3 = native_config('matd3', 2, suite, Path('unused'))['algo_args']
        self.assertEqual((td3['algo']['policy_freq'], td3['algo']['policy_noise'], td3['algo']['noise_clip']), (2, .2, .5))
        happo = native_config('happo', 2, suite, Path('unused'))['algo_args']
        self.assertFalse(happo['algo']['fixed_order'])
        self.assertEqual(happo['algo']['ppo_epoch'], 5)
        self.assertTrue(happo['algo']['use_gae'])
        self.assertFalse(happo['algo']['share_param'])

    def test_04_masac_retains_entropy_and_native_update_ratio(self):
        suite, _, _ = load_suite()
        native = native_config('masac', 2, suite, Path('unused'))
        self.assertEqual(native['algorithm']['num_qvalue_nets'], 2)
        self.assertFalse(native['algorithm']['fixed_alpha'])
        self.assertEqual(native['algorithm']['target_entropy'], 'auto')
        self.assertEqual(native['experiment']['off_policy_n_optimizer_steps'], 1000)
        self.assertEqual(native['experiment']['off_policy_collected_frames_per_batch'], 6000)
        self.assertIsNone(native['experiment']['restore_file'])

    def test_05_planner_and_uncommitted_target_information_removed(self):
        raw = np.arange(112, dtype=np.float32).reshape(4, 28)
        runtime = SimpleNamespace(task_found=False, finder_idx=-1, executor_delivered_target_state=None,
                                  executor_target_assigned=False)
        first = public_observation(raw, runtime)
        self.assertTrue(np.all(first[:, MASKED_COLUMNS] == 0))
        self.assertTrue(np.all(first[:, 12:15] == 0))
        changed = raw.copy(); changed[:, MASKED_COLUMNS] = -999.; changed[:, 12:15] = 777.
        np.testing.assert_array_equal(first, public_observation(changed, runtime))
        runtime.task_found, runtime.finder_idx = True, 1
        runtime.target_position_at_found = np.array([7., 8., 9.])
        runtime.executor_delivered_target_state = SimpleNamespace(position=np.array([3., 4., 5.]))
        self.assertTrue(np.all(public_observation(raw, runtime)[3, 12:15] == 0))
        runtime.executor_target_assigned, runtime._capture_hold_counter, runtime.target_capture_hold_steps = True, 2, 5
        public = public_observation(raw, runtime)
        np.testing.assert_array_equal(public[3, 12:15], np.array([3., 4., 5.]) - raw[3, :3])

    def test_06_physical_action_rng_isolation_and_finite_horizon(self):
        torch.set_num_threads(1)
        _, reference, _ = load_suite()
        reference['max_steps'] = 4
        scene = manifest()['scenarios'][0]
        # Independent caller RNG draws cannot affect the physical environment.
        torch.manual_seed(441)
        outside = torch.get_rng_state().clone()
        env1, env2 = PureMARLEnv(reference), PureMARLEnv(reference)
        try:
            obs1, state = env1.reset(scene)
            torch.testing.assert_close(outside, torch.get_rng_state(), rtol=0, atol=0)
            torch.rand(27); random.random(); np.random.rand(9)
            obs2, _ = env2.reset(scene)
            np.testing.assert_array_equal(obs1, obs2)
            self.assertEqual(state.shape, (112,))
            action = np.tile([.25, -.35, .15], (4, 1)).astype(np.float32)
            for index in range(4):
                a = env1.step(action)
                torch.rand(11)
                b = env2.step(action)
                for left, right in zip(a[:3], b[:3]):
                    np.testing.assert_array_equal(left, right)
                self.assertEqual(a[3:5], (index == 3, False))
                self.assertTrue(np.all(a[2] == a[2][0]))
                np.testing.assert_allclose(env1.unwrapped._last_prior_acc.numpy(), 0., atol=0)
                expected = np.column_stack([env1.unwrapped._a_xy_max.numpy()] * 2 + [env1.unwrapped._a_z_max.numpy()]) * action
                np.testing.assert_allclose(env1.unwrapped._agent_acc.numpy(), expected, atol=1e-6)
            result = env1.summary()
            self.assertEqual(result['termination_reason'], 'timeout')
            self.assertEqual(result['bootstrap_mask'], 0.)
            self.assertFalse(result['control_effort_is_physical_energy'])
            with self.assertRaises(RuntimeError):
                env1.step(action)
        finally:
            env1.close(); env2.close()

    def test_07_harl_native_evaluation_protocols(self):
        class OffPolicy:
            def get_actions(self, obs, add_noise):
                self.noise = add_noise
                return torch.ones((1, 3)) * .2
        class OnPolicy:
            def act(self, obs, rnn, mask, deterministic):
                self.deterministic = deterministic
                return torch.zeros((1, 3)), torch.as_tensor(rnn) + 1
        actors = [OffPolicy(), OffPolicy(), OnPolicy(), OnPolicy()]
        wrapper = HARLPolicy(actors)
        wrapper.reset()
        self.assertEqual(wrapper.act(np.zeros((4, 28), np.float32)).shape, (4, 3))
        self.assertFalse(actors[0].noise)
        self.assertTrue(actors[2].deterministic)
        self.assertEqual(float(wrapper.rnn[2].sum()), 128.)
        wrapper.reset()
        self.assertEqual(float(wrapper.rnn.sum()), 0.)

    def test_08_overlapping_splits_rejected_by_physical_fingerprint(self):
        _, reference, _ = load_suite()
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as temporary:
            paths = {k: Path(temporary) / f'{k}.json' for k in ('train', 'dev', 'test')}
            for index, (key, path) in enumerate(paths.items()):
                write(path, manifest(key, 981729 + index, steps=400))
            load_splits(paths, reference)
            dev = json.loads(paths['dev'].read_text())
            test = copy.deepcopy(dev)
            test['scenarios'][0].update(scenario_id='different_id', scenario_seed=991000)
            paths['test'].write_text(json.dumps(test))
            with self.assertRaisesRegex(ValueError, 'fingerprints'):
                load_splits(paths, reference)

    def test_09_seed_level_aggregation_and_failure_penalty(self):
        rows = [record(seed=1, scenario='a', success=True), record(seed=1, scenario='b'),
                record(seed=2, scenario='a', success=True), record(seed=2, scenario='b', success=True)]
        result = summarize(rows)['algorithms']['maddpg']
        self.assertEqual(result['metrics']['success_rate']['mean'], .75)
        self.assertEqual(result['metrics']['penalized_completion_steps']['mean'], 126.5)
        self.assertIsNotNone(result['metrics']['success_rate']['seed_bootstrap_95_interval'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summarize(rows + [rows[0]])

    def test_10_missing_scenarios_seeds_or_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError, 'evaluation set'):
            summarize([record(scenario='a'), record(seed=2, scenario='b')])
        with self.assertRaisesRegex(ValueError, 'training seed'):
            summarize([record(), record(seed=2), record(algo='masac')])
        changed = record(seed=2); changed['max_steps'] = 200
        with self.assertRaisesRegex(ValueError, 'horizon'):
            summarize([record(), changed])

    def test_11_no_implicit_execution_or_long_evaluation(self):
        _, reference, _ = load_suite()
        with self.assertRaisesRegex(ValueError, 'preparation release'):
            evaluate(None, reference, [], algorithm='maddpg', training_seed=1, training_fingerprints=[])
        with patch('builtins.print'):
            dry = main(['prepare', '--output', 'unused'])
        self.assertFalse(dry['executed'])
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            main(['train'])

    def test_12_output_overwrite_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as temporary:
            path = Path(temporary) / 'record.json'
            write(path, {'marker': 'original'})
            with self.assertRaises(FileExistsError):
                write(path, {'marker': 'replacement'})
            self.assertEqual(json.loads(path.read_text())['marker'], 'original')

    def test_13_optional_backend_module_import_does_not_install_or_require_backend(self):
        module = importlib.import_module('chapter3_bser.baselines.benchmarl_interface')
        self.assertIsNone(module._INTERFACES)

    def test_14_common_metric_recorder_requires_every_step(self):
        _, config, _ = load_suite()
        recorder = EpisodeRecorder(manifest()['scenarios'][0], config)
        runtime = SimpleNamespace(step_count=0, _agent_pos=torch.zeros((4, 3)))
        recorder.start(runtime)
        runtime.step_count = 2
        with self.assertRaisesRegex(ValueError, 'every physical transition'):
            recorder.observe(runtime, np.zeros(4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
