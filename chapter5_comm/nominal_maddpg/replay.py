"""Uniform replay over applied joint actions and team decision transitions."""
import numpy as np
import torch
from chapter3_bser.learning.replay import ExecutionReplay
from chapter3_bser.learning.replay_base import ExecutionSample


class NominalReplay(ExecutionReplay):
    def __init__(self, capacity, sampling_seed):
        super().__init__(capacity)
        self.generator = torch.Generator(device="cpu").manual_seed(sampling_seed)

    def push_transition(self, transition):
        rewards = np.asarray(transition["rewards"])
        if rewards.shape != (4,) or not np.all(rewards == rewards[0]):
            raise ValueError("expected four identical team rewards")
        packed = dict(transition)
        for key in ("observations", "next_observations"):
            packed[key] = np.asarray(packed[key], dtype=np.float32)
        super().push_transition(packed)

    def sample(self, n, *, device="cpu", norm_rews=False):
        if type(n) is not int or n <= 0 or not len(self) or norm_rews:
            raise ValueError("positive uniform sample of original rewards required")
        indices = torch.randperm(len(self), generator=self.generator)[:n]
        if not self.gate_initialized[indices].all():
            raise RuntimeError("sampled an unwritten execution gate")
        fields = tuple([buf[indices].to(device) for buf in buffers] for buffers in
            (self.obs_buffs, self.ac_buffs, self.rew_buffs, self.next_obs_buffs, self.done_buffs))
        return ExecutionSample((*fields, torch.ones(len(indices), device=device), indices.to(device),
            self.success_buffs[indices].to(device)), self.enabled[indices].to(device),
            self.next_enabled[indices].to(device))

