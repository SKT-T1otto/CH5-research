"""Private-stream OU exploration on normalized residual actions."""
import copy
import math

import torch


class OUNoise:
    """One clock tick per training team decision, independent of control gates.

    Episode reset clears OU memory only. Disabled exploration consumes no random
    draws and leaves memory unchanged, while advancing the training-step clock.
    Evaluation uses a separate policy and never calls this object.
    """
    def __init__(self, seed, *, initial_sigma=.20, minimum_sigma=.02,
                 decay_per_environment_step=.00002, theta=.15, dt=1., mean=0.):
        if type(seed) is not int or seed < 0:
            raise ValueError('noise seed must be a nonnegative integer')
        values = (initial_sigma, minimum_sigma, decay_per_environment_step, theta, dt, mean)
        if (not all(math.isfinite(v) for v in values) or not 0 <= minimum_sigma <= initial_sigma
                or decay_per_environment_step <= 0 or theta <= 0 or dt <= 0):
            raise ValueError('invalid OU settings')
        self.settings = dict(initial_sigma=initial_sigma, minimum_sigma=minimum_sigma,
            decay_per_environment_step=decay_per_environment_step, theta=theta, dt=dt, mean=mean)
        self.seed, self.steps, self.draws = seed, 0, 0
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.memory = torch.zeros(4, 3, dtype=torch.float32)

    @property
    def sigma(self):
        p = self.settings
        return max(p['minimum_sigma'], p['initial_sigma']-p['decay_per_environment_step']*self.steps)

    def reset_episode(self):
        self.memory.fill_(self.settings['mean'])

    def sample(self, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('exploration switch must be boolean')
        sigma, p = self.sigma, self.settings
        if enabled:
            innovation = torch.randn((4, 3), generator=self.generator, dtype=torch.float32)
            self.memory = (self.memory + p['theta']*(p['mean']-self.memory)*p['dt']
                           + sigma*math.sqrt(p['dt'])*innovation)
            self.draws += 1
            result = self.memory.clone()
        else:
            result = torch.zeros_like(self.memory)
        self.steps += 1
        return result, sigma

    def state_dict(self):
        return dict(schema='auv.ch5.ou.v1', settings=copy.deepcopy(self.settings), seed=self.seed,
            steps=self.steps, draws=self.draws, memory=self.memory.clone(),
            rng=self.generator.get_state().clone())

    def load_state_dict(self, state):
        if (state.get('schema') != 'auv.ch5.ou.v1' or state.get('settings') != self.settings
                or state.get('seed') != self.seed or type(state.get('steps')) is not int
                or type(state.get('draws')) is not int or not 0 <= state['draws'] <= state['steps']):
            raise ValueError('noise state identity or clock differs')
        memory, rng = state.get('memory'), state.get('rng')
        if (not torch.is_tensor(memory) or memory.device.type != 'cpu' or memory.dtype != torch.float32
                or memory.shape != (4, 3) or not bool(torch.isfinite(memory).all())
                or not torch.is_tensor(rng) or rng.device.type != 'cpu' or rng.dtype != torch.uint8):
            raise ValueError('invalid noise memory or RNG state')
        probe = torch.Generator()
        probe.set_state(rng.clone())
        self.memory = memory.clone()
        self.generator.set_state(probe.get_state())
        self.steps, self.draws = state['steps'], state['draws']
