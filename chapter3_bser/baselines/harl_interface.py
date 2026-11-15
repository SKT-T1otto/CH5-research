"""HARL environment and evaluation-policy interfaces; no HARL trainer rewrite."""
import copy
import numpy as np
from chapter3_bser.baselines.environment import PureMARLEnv


class HARLEnv:
    def __init__(self, reference_config, scenarios):
        from gym.spaces import Box
        if not scenarios:
            raise ValueError('explicit scenario manifest required')
        self.env = PureMARLEnv(reference_config)
        self.scenarios = copy.deepcopy(scenarios)
        self.cursor, self.n_agents = 0, 4
        self.observation_space = [Box(-np.inf, np.inf, (28,), dtype=np.float32) for _ in range(4)]
        self.share_observation_space = [Box(-np.inf, np.inf, (112,), dtype=np.float32) for _ in range(4)]
        self.action_space = [Box(-1., 1., (3,), dtype=np.float32) for _ in range(4)]

    def reset(self):
        scene = self.scenarios[self.cursor % len(self.scenarios)]
        self.cursor += 1
        obs, state = self.env.reset(scene)
        return obs, np.repeat(state[None], 4, axis=0), None

    def step(self, actions):
        obs, state, rewards, terminated, truncated, info = self.env.step(actions)
        if truncated:
            raise RuntimeError('finite-horizon mission must remain terminal')
        infos = [copy.deepcopy(info) for _ in range(4)]
        if terminated:
            for item in infos:
                item['episode_metrics'] = self.env.summary()
        return obs, np.repeat(state[None], 4, axis=0), rewards[:, None], np.full(4, terminated), infos, None

    def seed(self, seed):
        # Policy seed must not change the shared physical scenario sequence.
        self.policy_seed = int(seed)

    def close(self):
        self.env.close()


class HARLPolicy:
    """Wrap native HARL actors already constructed by the native runner.

    The evaluator never receives critic state. RNN state is reset per episode.
    """
    def __init__(self, actors, recurrent_n=1, hidden_size=128):
        if len(actors) != 4:
            raise ValueError('four native actors required')
        self.actors = actors
        self.recurrent_n, self.hidden_size = recurrent_n, hidden_size

    def reset(self):
        self.rnn = np.zeros((4, 1, self.recurrent_n, self.hidden_size), dtype=np.float32)

    def act(self, observations, *, deterministic=True):
        import torch
        actions = []
        with torch.no_grad():
            for index, actor in enumerate(self.actors):
                if hasattr(actor, 'act'):
                    action, rnn = actor.act(observations[index:index + 1], self.rnn[index],
                                            np.ones((1, 1), np.float32), deterministic=deterministic)
                    self.rnn[index] = rnn.detach().cpu().numpy()
                else:
                    action = actor.get_actions(observations[index:index + 1], add_noise=not deterministic)
                actions.append(action.detach().cpu().numpy().reshape(3))
        return np.stack(actions)
