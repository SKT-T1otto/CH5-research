"""Local single-critic MADDPG network definitions verified against Chapter 4."""
import copy

import torch
from torch import nn
from core.algorithms.networks import MLPNetwork
from .config import validate


class SingleCriticNetworks(nn.Module):
    """Four decentralized actors and four centralized critics, with target copies."""

    def __init__(self, config, seed):
        super().__init__()
        c = validate(config)
        if type(seed) is not int or seed < 0:
            raise ValueError("initialization seed must be a nonnegative integer")
        a = c["algorithm"]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actors = nn.ModuleList(MLPNetwork(28, 3, hidden_dim=a["actor_hidden"][0],
                constrain_out=True, discrete_action=False, norm_in=True) for _ in range(4))
            self.critics = nn.ModuleList(MLPNetwork(124, 1, hidden_dim=a["critic_hidden"][0],
                constrain_out=False, discrete_action=False, norm_in=True) for _ in range(4))
        self.target_actors = copy.deepcopy(self.actors).requires_grad_(False)
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)
        self.eval()

    def actions(self, observations):
        if observations.ndim != 3 or observations.shape[0] < 1 or observations.shape[1:] != (4, 28):
            raise ValueError("observations must have shape [batch, 4, 28]")
        if not bool(torch.isfinite(observations).all()):
            raise ValueError("nonfinite observations")
        return torch.stack([actor(observations[:, i]) for i, actor in enumerate(self.actors)], dim=1)

    def values(self, observations, applied_actions):
        if observations.ndim != 3 or observations.shape[0] < 1 or observations.shape[1:] != (4, 28):
            raise ValueError("observations must have shape [batch, 4, 28]")
        if applied_actions.shape != (*observations.shape[:2], 3):
            raise ValueError("actions must have shape [batch, 4, 3]")
        joint = torch.cat((observations.flatten(1), applied_actions.flatten(1)), dim=1)
        if not bool(torch.isfinite(joint).all()) or bool((applied_actions.abs() > 1).any()):
            raise ValueError("invalid critic observations or normalized actions")
        return torch.cat([critic(joint) for critic in self.critics], dim=1)

    @torch.no_grad()
    def step(self, observations, *, explore=False):
        """One deterministic action per hull. Exploration belongs to the caller."""
        if explore:
            raise ValueError("add exploration explicitly before the runtime execution gate")
        if isinstance(observations, (list, tuple)):
            observations = torch.stack([torch.as_tensor(row) for row in observations])
        observations = torch.as_tensor(observations, dtype=torch.float32)
        if observations.shape != (4, 28):
            raise ValueError("expected a legal [4, 28] decision snapshot")
        return self.actions(observations.unsqueeze(0))[0]
