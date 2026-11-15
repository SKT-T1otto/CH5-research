"""Optional TorchRL EnvBase and native BenchMARL MASAC policy adapters.

Import only in the separately installed BenchMARL environment. The native
SAC loss, replay sampling, alpha optimization and target updates stay upstream.
"""

_INTERFACES = None

def _load_native_interfaces():
    import copy
    import torch
    from tensordict import TensorDict
    from torchrl.data import Bounded, Categorical, Composite, Unbounded
    from torchrl.envs import EnvBase
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from chapter3_bser.baselines.environment import PureMARLEnv


    class BenchMARLEnv(EnvBase):
        def __init__(self, reference_config, scenarios, device='cpu'):
            super().__init__(device=device, batch_size=torch.Size([]))
            if not scenarios:
                raise ValueError('explicit scenario manifest required')
            self.mission = PureMARLEnv(reference_config)
            self.scenarios, self.cursor = copy.deepcopy(scenarios), 0
            self.observation_spec = Composite(
                agents=Composite(observation=Unbounded(shape=(4, 28), device=device), shape=(4,), device=device),
                state=Unbounded(shape=(112,), device=device), device=device)
            self.action_spec = Composite(
                agents=Composite(action=Bounded(-1., 1., shape=(4, 3), device=device), shape=(4,), device=device), device=device)
            self.reward_spec = Composite(
                agents=Composite(reward=Unbounded(shape=(4, 1), device=device), shape=(4,), device=device), device=device)
            self.done_spec = Composite(**{k: Categorical(2, shape=(1,), dtype=torch.bool, device=device)
                                         for k in ('done', 'terminated', 'truncated')}, device=device)

        def _packet(self, observations, state, terminated=False, reward=None):
            values = dict(agents=TensorDict(dict(observation=torch.as_tensor(observations, device=self.device)),
                                           batch_size=[4], device=self.device),
                          state=torch.as_tensor(state, device=self.device),
                          done=torch.tensor([terminated], device=self.device),
                          terminated=torch.tensor([terminated], device=self.device),
                          truncated=torch.tensor([False], device=self.device))
            td = TensorDict(values, batch_size=[], device=self.device)
            if reward is not None:
                td.set(('agents', 'reward'), torch.as_tensor(reward[:, None], device=self.device))
            return td

        def _reset(self, tensordict=None, **kwargs):
            scene = self.scenarios[self.cursor % len(self.scenarios)]
            self.cursor += 1
            obs, state = self.mission.reset(scene)
            return self._packet(obs, state)

        def _step(self, tensordict):
            action = tensordict.get(('agents', 'action')).detach().cpu().numpy()
            obs, state, reward, terminated, _, _ = self.mission.step(action)
            return self._packet(obs, state, terminated, reward)

        def _set_seed(self, seed):
            return seed  # Scenario/innovation seeds are read from the common manifest.

        def _close(self):
            self.mission.close()


    class BenchMARLPolicy:
        def __init__(self, native_policy, device='cpu'):
            self.policy, self.device = native_policy, torch.device(device)

        def reset(self):
            pass  # The prepared MASAC configuration is feed-forward.

        def act(self, observations, *, deterministic=True):
            td = TensorDict({'agents': TensorDict(
                {'observation': torch.as_tensor(observations, device=self.device)}, batch_size=[4])}, batch_size=[])
            exploration = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
            with torch.no_grad(), set_exploration_type(exploration):
                return self.policy(td).get(('agents', 'action')).detach().cpu().numpy()

    return {'BenchMARLEnv': BenchMARLEnv, 'BenchMARLPolicy': BenchMARLPolicy}


def __getattr__(name):
    global _INTERFACES
    if name not in ('BenchMARLEnv', 'BenchMARLPolicy'):
        raise AttributeError(name)
    if _INTERFACES is None:
        _INTERFACES = _load_native_interfaces()
    return _INTERFACES[name]
