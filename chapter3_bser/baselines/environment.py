"""Direct-acceleration policy interface over the unchanged mission simulator.

Environment-native mapping/waypoints remain reward instrumentation. Neither D2
nor its guidance/repair runtime is constructed. Planner features are withheld
from both actors and the centralized training observation.
"""
from contextlib import contextmanager
import copy
import random
import numpy as np
import torch
from core.config.active_environment import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.env.task_protocol import PROTOCOL_FIELDS
from chapter3_bser.rewards.training_env import Phase1CV2TrainingEnv
from chapter3_bser.protocols.team_environment import PRRACTrainingEnv
from chapter3_bser.baselines.metrics import EpisodeRecorder, array

MASKED_COLUMNS = (6, 7, 8, 9, 10, 11, 15, 17, 19)
OBSERVATION_ID = 'public28_without_planner_v1'
ACTION_ID = 'direct_normalized_acceleration_v1'


def public_observation(raw, runtime):
    obs = array(raw)
    if obs.shape != (4, 28) or not np.isfinite(obs).all():
        raise ValueError('expected finite 4 x 28 observations')
    obs[:, MASKED_COLUMNS] = 0
    obs[:, 12:15] = 0  # Rebuild from committed local measurements, not intercept planning.
    finder = int(runtime.finder_idx)
    if runtime.task_found and finder >= 0:
        found = getattr(runtime, 'target_position_at_found', None)
        if found is not None:
            obs[finder, 12:15] = np.asarray(found) - obs[finder, :3]
    delivered = runtime.executor_delivered_target_state
    if runtime.executor_target_assigned and delivered is not None:
        obs[3, 12:15] = delivered.position - obs[3, :3]
    obs[:, 21] = 0
    if runtime.executor_target_assigned:
        obs[3, 21] = min(1., runtime._capture_hold_counter / max(1, runtime.target_capture_hold_steps))
    return obs


class PureMARLEnv:
    """One unbatched environment, explicit manifest reset, no auto-reset.

    Timeout remains a terminal finite-horizon task as in the source protocol;
    it is deliberately not converted into a bootstrap-enabled truncation.
    """
    n_agents, observation_dim, action_dim, state_dim = 4, 28, 3, 112

    def __init__(self, reference_config):
        self.config = copy.deepcopy(reference_config)
        self._rng = (random.Random(0).getstate(), np.random.RandomState(0).get_state(),
                     torch.Generator().manual_seed(0).get_state())
        self._done = True
        with self._rng_scope():
            settings = build_ch3_config(self.config['base_candidate'], self.config['profile'])
            settings.update({k: self.config[k] for k in (*PROTOCOL_FIELDS, 'collision_terminal_reward', 'waypoint_reward_timing')})
            settings.update(use_residual_prior=False, prior_strength_search=0., prior_strength_executor=0.,
                            residual_scale_search=1., residual_scale_executor=1.)
            self.core = MissionCoreEnv(**environment_kwargs_from_config(
                settings, device='cpu', max_steps=self.config['max_steps'], return_numpy=False))
            self.env = PRRACTrainingEnv(Phase1CV2TrainingEnv(self.core, reward_config=self.config['reward']),
                                      reward_objective_config=self.config, gamma=self.config['rl']['gamma'])

    @contextmanager
    def _rng_scope(self):
        outside = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        random.setstate(self._rng[0]); np.random.set_state(self._rng[1]); torch.set_rng_state(self._rng[2])
        try:
            yield
        finally:
            self._rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            random.setstate(outside[0]); np.random.set_state(outside[1]); torch.set_rng_state(outside[2])

    @property
    def unwrapped(self):
        return self.core.unwrapped

    def reset(self, scenario):
        if scenario['max_steps'] != self.config['max_steps'] or scenario['scenario_profile'] != self.config['profile']:
            raise ValueError('manifest horizon/profile differs from the common environment')
        self.scenario = copy.deepcopy(scenario)
        with self._rng_scope():
            seed = int(scenario['scenario_seed'])
            random.seed(seed); np.random.seed(np.random.SeedSequence(seed).generate_state(4)); torch.manual_seed(seed)
            raw = self.env.reset(scenario=scenario)
            # Parent reset uses planner_seed; innovations begin from scenario_seed.
            random.seed(seed); np.random.seed(np.random.SeedSequence(seed).generate_state(4)); torch.manual_seed(seed)
        self.observation = public_observation(raw, self.unwrapped)
        self._done = False
        self.recorder = EpisodeRecorder(scenario, self.config)
        self.recorder.start(self.unwrapped)
        self.source_returns = np.zeros(4, dtype=np.float64)
        return self.observation.copy(), self.observation.reshape(-1).copy()

    def step(self, actions):
        if self._done:
            raise RuntimeError('explicit reset required before stepping')
        proposed = np.asarray(actions, dtype=np.float32)
        if proposed.shape != (4, 3) or not np.isfinite(proposed).all():
            raise ValueError('actions must be finite 4 x 3')
        bounded = np.clip(proposed, -1., 1.)
        with self._rng_scope():
            raw, rewards, dones = self.env.step(torch.from_numpy(bounded.copy()))
        result = self.unwrapped.get_episode_result()
        self._done = bool(result['terminated'])
        if list(map(bool, dones)) != [self._done] * 4 or result['truncated']:
            raise RuntimeError('mission terminal semantics changed')
        self.observation = public_observation(raw, self.unwrapped)
        reward = array(rewards).reshape(4)
        self.recorder.observe(self.unwrapped, reward)
        self.source_returns += self.env.reward_accounting.last['source_reward_by_agent']
        # Environmental searcher freeze is part of the task, not a policy gate.
        # Report the actual physical command separately from the proposed action.
        limits = np.column_stack([array(self.unwrapped._a_xy_max)] * 2 + [array(self.unwrapped._a_z_max)])
        applied = array(self.unwrapped._agent_acc) / limits
        info = dict(result, proposed_actions=proposed.tolist(), bounded_actions=bounded.tolist(),
                    applied_actions=applied.tolist(), action_clipped_components=int((proposed != bounded).sum()),
                    observation_id=OBSERVATION_ID, action_id=ACTION_ID)
        return self.observation.copy(), self.observation.reshape(-1).copy(), reward, self._done, False, info

    def summary(self):
        return dict(self.recorder.finish(self.unwrapped), source_returns=self.source_returns.tolist(),
                    observation_id=OBSERVATION_ID, action_id=ACTION_ID)

    def close(self):
        self.env.close()
