"""Single-critic MADDPG on gated Chapter 5 Replay samples."""
from dataclasses import dataclass
import math
import torch
from .config import validate
from .networks import SingleCriticNetworks


@dataclass(frozen=True)
class LearnerBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    enabled: torch.Tensor
    next_enabled: torch.Tensor

    @property
    def size(self):
        return self.observations.shape[0]


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite " + label)
    return value


def prepare_batch(sample):
    """Validate the real ExecutionSample contract before any optimizer mutation.

    The learner accepts CPU float32 Replay data, including both execution gates.
    The input tensors are detached and copied; rewards are never transformed.
    """
    if not isinstance(sample, tuple) or len(sample) != 8:
        raise ValueError("expected the eight-field execution Replay sample")
    packed = {}
    for name, values, tail in (
            ("observations", sample[0], (28,)), ("actions", sample[1], (3,)),
            ("rewards", sample[2], ()), ("next_observations", sample[3], (28,)),
            ("dones", sample[4], ())):
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError("expected four agents in " + name)
        for value in values:
            if (not torch.is_tensor(value) or value.device.type != "cpu"
                    or value.dtype != torch.float32 or value.ndim != 1 + len(tail)
                    or value.shape[1:] != tail or value.shape[0] < 1):
                raise ValueError("invalid CPU float32 Replay " + name)
            _finite(value, name)
        if len({v.shape[0] for v in values}) != 1:
            raise ValueError("inconsistent Replay rows in " + name)
        packed[name] = torch.stack(values, dim=1).detach().clone()
    size = packed["observations"].shape[0]
    if any(v.shape[0] != size for v in packed.values()):
        raise ValueError("inconsistent Replay batch sizes")
    for name, attribute in (("enabled", "enabled"), ("next_enabled", "next_enabled")):
        value = getattr(sample, attribute, None)
        if (not torch.is_tensor(value) or value.device.type != "cpu"
                or value.dtype != torch.bool or value.shape != (size, 4)):
            raise ValueError("invalid execution gate " + name)
        packed[name] = value.detach().clone()
    weights, indices, successes = sample[5:]
    for value, dtype, label in ((weights, torch.float32, "weights"),
                               (indices, torch.int64, "indices"),
                               (successes, torch.bool, "successes")):
        if (not torch.is_tensor(value) or value.device.type != "cpu"
                or value.dtype != dtype or value.shape != (size,)):
            raise ValueError("invalid Replay " + label)
    if not bool((weights == 1).all()) or bool((indices < 0).any()):
        raise ValueError("only uniform, unit-weight Replay samples are supported")
    dones = packed["dones"]
    if (not bool(((dones == 0) | (dones == 1)).all())
            or not torch.equal(dones, dones[:, :1].expand_as(dones))):
        raise ValueError("mission done must be binary and shared by all four agents")
    packed["dones"] = dones[:, 0].bool()
    if bool(packed["next_enabled"][packed["dones"]].any()):
        raise ValueError("terminal next-state execution gates must be zero")
    if bool((successes & ~packed["dones"]).any()):
        raise ValueError("mission success requires a terminal transition")
    rewards, actions = packed["rewards"], packed["actions"]
    if not torch.equal(rewards, rewards[:, :1].expand_as(rewards)):
        raise ValueError("expected the unchanged shared team reward")
    if bool((actions.abs() > 1).any()) or bool((actions[~packed["enabled"]] != 0).any()):
        raise ValueError("expected applied normalized residuals, zero behind closed gates")
    return LearnerBatch(**packed)


class SingleCriticLearner:
    """Registered MADDPG update equations; execution permission belongs to the caller.

    Standalone critic/actor methods support mechanism tests. ``rounds`` counts
    only complete calls to ``update``; per-agent counters count actual Adam steps.
    A numerical failure aborts the caller's run, never silently skips bad data.
    """

    def __init__(self, config, seed):
        self.config = validate(config)
        self.networks = SingleCriticNetworks(self.config, seed)
        learning = self.config["learning"]
        self.gamma, self.tau = learning["gamma"], learning["tau"]
        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=learning["actor_lr"],
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0) for a in self.networks.actors]
        self.critic_optimizers = [torch.optim.Adam(q.parameters(), lr=learning["critic_lr"],
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0) for q in self.networks.critics]
        self.rounds = self.target_updates = 0
        self.actor_steps, self.critic_steps, self.actor_skips = [0]*4, [0]*4, [0]*4

    @staticmethod
    def _agent(agent_i):
        if type(agent_i) is not int or not 0 <= agent_i < 4:
            raise ValueError("agent index must be an integer in [0, 4)")

    @staticmethod
    def _joint(observations, actions):
        return torch.cat((observations.flatten(1), actions.flatten(1)), dim=1)

    @staticmethod
    def _require_grad():
        if not torch.is_grad_enabled():
            raise RuntimeError("learner updates must run outside the no_grad environment step")

    @staticmethod
    def _gradient_norm(module):
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                _finite(p.grad, "gradient")
                total += float(p.grad.detach().double().square().sum())
        norm = math.sqrt(total)
        if not math.isfinite(norm):
            raise ValueError("nonfinite gradient norm")
        return norm

    @staticmethod
    def _check_optimizer(optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                _finite(p, "online parameter")
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    _finite(value, "Adam state")

    @torch.no_grad()
    def _targets(self, batch):
        actions = torch.zeros(batch.size, 4, 3, dtype=torch.float32)
        active = ~batch.dones
        # Do not evaluate terminal observations: 0 * NaN would not equal zero.
        for i, actor in enumerate(self.networks.target_actors):
            enabled = active & batch.next_enabled[:, i]
            if bool(enabled.any()):
                proposal = _finite(actor(batch.next_observations[enabled, i]), "target action")
                if proposal.shape != (int(enabled.sum()), 3) or bool((proposal.abs() > 1).any()):
                    raise ValueError("invalid normalized target action")
                actions[enabled, i] = proposal
        targets = batch.rewards.clone()
        if bool(active.any()):
            joint = self._joint(batch.next_observations[active], actions[active])
            for i, critic in enumerate(self.networks.target_critics):
                value = _finite(critic(joint), "nonterminal target Q")
                if value.shape != (int(active.sum()), 1):
                    raise ValueError("invalid target critic output")
                targets[active, i] += self.gamma * value[:, 0]
        _finite(targets, "TD target")
        return targets, actions

    def td_targets(self, sample):
        """Return detached [B,4] targets and gated [B,4,3] target actions."""
        return self._targets(prepare_batch(sample))

    def _critic_loss(self, batch, agent_i, targets):
        q = self.networks.critics[agent_i](self._joint(batch.observations, batch.actions))
        if q.shape != (batch.size, 1):
            raise ValueError("invalid online critic output")
        return _finite((q[:, 0] - targets[:, agent_i]).square().mean(), "critic MSE")

    def critic_loss(self, sample, agent_i):
        self._agent(agent_i)
        batch = prepare_batch(sample)
        targets, _ = self._targets(batch)
        return self._critic_loss(batch, agent_i, targets)

    def _step_critic(self, batch, agent_i, targets):
        optimizer = self.critic_optimizers[agent_i]
        self._check_optimizer(optimizer)
        optimizer.zero_grad(set_to_none=True)
        loss = self._critic_loss(batch, agent_i, targets)
        loss.backward()
        norm = self._gradient_norm(self.networks.critics[agent_i])
        optimizer.step()
        self._check_optimizer(optimizer)
        optimizer.zero_grad(set_to_none=True)
        self.critic_steps[agent_i] += 1
        return dict(loss=float(loss.detach()), gradient_norm=norm, steps=self.critic_steps[agent_i])

    def update_critic(self, sample, agent_i):
        self._require_grad()
        self._agent(agent_i)
        batch = prepare_batch(sample)
        targets, _ = self._targets(batch)
        return self._step_critic(batch, agent_i, targets)

    def _actor_loss(self, batch, agent_i):
        enabled = batch.enabled[:, agent_i]
        if not bool(enabled.any()):
            return None
        observations = batch.observations[enabled]
        actions = batch.actions[enabled].clone()
        actions[:, agent_i] = self.networks.actors[agent_i](observations[:, agent_i])
        critic = self.networks.critics[agent_i]
        flags = [p.requires_grad for p in critic.parameters()]
        try:
            # Q remains differentiable with respect to the selected actor action.
            critic.requires_grad_(False)
            q = critic(self._joint(observations, actions))
            if q.shape != (int(enabled.sum()), 1):
                raise ValueError("invalid actor critic output")
            return _finite(-q.mean(), "actor loss")
        finally:
            for p, flag in zip(critic.parameters(), flags):
                p.requires_grad_(flag)

    def actor_loss(self, sample, agent_i):
        self._agent(agent_i)
        return self._actor_loss(prepare_batch(sample), agent_i)

    def _step_actor(self, batch, agent_i):
        optimizer = self.actor_optimizers[agent_i]
        optimizer.zero_grad(set_to_none=True)
        rows = int(batch.enabled[:, agent_i].sum())
        if rows == 0:
            # No Adam.step: a zero gradient still advances momentum and the clock.
            self.actor_skips[agent_i] += 1
            return dict(skipped=True, rows=0, loss=None, gradient_norm=0.0,
                        steps=self.actor_steps[agent_i])
        self._check_optimizer(optimizer)
        loss = self._actor_loss(batch, agent_i)
        loss.backward()
        norm = self._gradient_norm(self.networks.actors[agent_i])
        optimizer.step()
        self._check_optimizer(optimizer)
        optimizer.zero_grad(set_to_none=True)
        self.actor_steps[agent_i] += 1
        return dict(skipped=False, rows=rows, loss=float(loss.detach()), gradient_norm=norm,
                    steps=self.actor_steps[agent_i])

    def update_actor(self, sample, agent_i):
        self._require_grad()
        self._agent(agent_i)
        return self._step_actor(prepare_batch(sample), agent_i)

    @torch.no_grad()
    def soft_update_targets(self):
        pairs = []
        for online, target in ((self.networks.actors, self.networks.target_actors),
                               (self.networks.critics, self.networks.target_critics)):
            sources, destinations = dict(online.named_parameters()), dict(target.named_parameters())
            if sources.keys() != destinations.keys():
                raise ValueError("target parameter names differ from online network")
            if dict(online.named_buffers()) or dict(target.named_buffers()):
                raise ValueError("registered LayerNorm networks have no running-statistic buffers")
            for name, value in sources.items():
                other = destinations[name]
                if (value.shape != other.shape or value.dtype != other.dtype
                        or other.requires_grad or other.grad is not None):
                    raise ValueError("target structure or gradient isolation differs")
                _finite(value, "online soft-update parameter")
                _finite(other, "target soft-update parameter")
                pairs.append((value, other))
        for value, other in pairs:
            other.mul_(1 - self.tau).add_(value, alpha=self.tau)
        self.target_updates += 1

    def counters(self):
        return dict(rounds=self.rounds, target_updates=self.target_updates,
                    actor_steps=self.actor_steps.copy(), critic_steps=self.critic_steps.copy(),
                    actor_skips=self.actor_skips.copy())

    def update(self, sample):
        """One shared minibatch; four critics, eligible actors, then soft targets.

        TD targets are computed before any online update in this round. No
        clipping, twin critics, reward relabeling or target-action noise is used.
        """
        self._require_grad()
        batch = prepare_batch(sample)
        targets, actions = self._targets(batch)
        critics = [self._step_critic(batch, i, targets) for i in range(4)]
        actors = [self._step_actor(batch, i) for i in range(4)]
        self.soft_update_targets()
        self.rounds += 1
        return dict(batch_size=batch.size, terminal_rows=int(batch.dones.sum()),
                    target_min=float(targets.min()), target_max=float(targets.max()),
                    closed_target_actions_zero=not bool(actions[~batch.next_enabled].any()),
                    critics=critics, actors=actors, counters=self.counters())
