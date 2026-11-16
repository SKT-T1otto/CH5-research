"""Bounded CH5 auxiliary, communication-RL and inherited MADDPG learners.

No simulator is imported or executed by this module. Auxiliary labels must be
provided by the factual trajectory dataset. Communication rewards are original
task rewards accumulated over physical-time intervals, never message scores.
The short-workflow checkpoint is for inference/new episodes. The separate
experiment_state module adds new, gated-replay episode-boundary snapshots;
neither schema restores an active simulator.
"""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from chapter5_comm.neural_modules import (
    BudgetedPacketScheduler, GeometryTemporalDecoder, RouteUtilityHead,
    TaskPacketEncoder, gaussian_prediction_nll, route_encounter_loss,
    temporal_composition_loss,
)


CHECKPOINT_SCHEMA = "ch5.communication.checkpoint.v1"
MAX_MECHANISM_UPDATES = 4


@dataclass(frozen=True)
class NeuralConfig:
    map_dim: int = 7
    dynamic_dim: int = 8
    query_dim: int = 9
    latent_dim: int = 32
    geometry_dim: int = 7
    state_dim: int = 6
    route_dim: int = 5
    context_dim: int = 7
    map_slots: int = 2
    dynamic_slots: int = 2
    num_heads: int = 4
    header_bytes: int = 16
    integration_steps: int = 4
    hidden_dim: int = 64
    query_conditioning: bool = True
    temporal_transport: bool = True
    geometry_conditioning: bool = True

    @classmethod
    def from_mapping(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("neural configuration must be a mapping")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("unknown neural configuration fields: " + str(sorted(unknown)))
        return cls(**dict(value))


class CH5Networks(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = NeuralConfig.from_mapping(config)
        c = self.config
        if any(type(x) is not bool for x in (c.query_conditioning, c.temporal_transport, c.geometry_conditioning)):
            raise ValueError("ablation switches must be boolean")
        self.encoder = TaskPacketEncoder(c.map_dim, c.dynamic_dim, c.query_dim,
                                         c.latent_dim, c.map_slots, c.dynamic_slots,
                                         c.num_heads, c.header_bytes)
        self.decoder = GeometryTemporalDecoder(c.latent_dim, c.geometry_dim,
                                               c.state_dim, c.integration_steps)
        self.decoder.transport_enabled = c.temporal_transport
        self.decoder.geometry_conditioning = c.geometry_conditioning
        self.route_head = RouteUtilityHead(c.latent_dim, c.route_dim, c.hidden_dim)
        self.scheduler = BudgetedPacketScheduler(c.latent_dim, c.context_dim, c.hidden_dim)


def build_neural_models(config=None, *, device="cpu"):
    return CH5Networks(config).to(device)


def freeze_auxiliary(networks, frozen=True):
    """Freeze codec/decoder/route head without freezing the communication head."""
    for module in (networks.encoder, networks.decoder, networks.route_head):
        module.requires_grad_(not frozen)
        module.train(not frozen)


def _cap(value, *, experiment=False):
    limit = 1000000 if experiment else MAX_MECHANISM_UPDATES
    if type(value) is not int or not 0 <= value <= limit:
        raise ValueError(f"learner update limit must be within 0..{limit}")
    return value


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("non-finite " + name)


def _tensor(value, *, device, boolean=False):
    return torch.as_tensor(value, device=device, dtype=torch.bool if boolean else torch.float32)


def _parameters_changed(parameters, before):
    return math.sqrt(sum(float((p.detach() - old).square().sum())
                         for p, old in zip(parameters, before)))


class AuxiliaryTrainer:
    """Actual optimization of route/prediction/time losses on supplied labels."""
    def __init__(self, networks, *, learning_rate=3e-4, route_weight=1.0,
                 prediction_weight=1.0, time_weight=0.1, max_updates=4, experiment=False):
        self.networks = networks
        self.experiment = bool(experiment)
        self.max_updates = _cap(max_updates, experiment=self.experiment)
        self.updates = 0
        self.weights = dict(route=float(route_weight), prediction=float(prediction_weight),
                            time=float(time_weight))
        if any(not math.isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError("loss weights must be finite and nonnegative")
        self.parameters = [p for m in (networks.encoder, networks.decoder, networks.route_head)
                           for p in m.parameters()]
        if not all(p.requires_grad for p in self.parameters):
            raise ValueError("auxiliary modules must be unfrozen before constructing their optimizer")
        self.optimizer = torch.optim.Adam(self.parameters, lr=float(learning_rate))
        self.history = []

    def step(self, batch):
        if self.updates >= self.max_updates:
            raise RuntimeError("bounded auxiliary update limit reached")
        n = self.networks
        device = next(n.parameters()).device
        booleans = {"query_available", "map_mask", "dynamic_mask", "received",
                    "route_valid", "state_valid"}
        required = ("map_tokens", "dynamic_tokens", "receiver_query", "query_available",
                    "map_mask", "dynamic_mask", "received", "ages", "local_geometry",
                    "route_features", "route_valid", "route_labels", "state_target", "state_valid")
        missing = set(required) - set(batch)
        if missing:
            raise ValueError("missing auxiliary batch fields: " + str(sorted(missing)))
        b = {key: _tensor(batch[key], device=device, boolean=key in booleans) for key in required}
        if b["state_valid"].ndim != 1:
            raise ValueError("state_valid must be a per-example [B] mask")
        if not bool(b["state_valid"].any() or b["route_valid"].any()):
            return {"skipped": True, "reason": "no_supervised_targets", "updates": self.updates}
        queries = b["query_available"] if n.config.query_conditioning else torch.zeros_like(b["query_available"])
        encoded = n.encoder(b["map_tokens"], b["dynamic_tokens"], b["receiver_query"],
                            queries, b["map_mask"], b["dynamic_mask"])
        received = b["received"] & encoded.valid
        decoded = n.decoder(encoded.tokens, encoded.stream_types, received,
                            b["ages"], b["local_geometry"])
        # A target label cannot supervise temporal reconstruction when the
        # actual received set contains no dynamic packet.
        state_valid = b["state_valid"] & decoded["has_dynamic"]
        if not bool(state_valid.any() or b["route_valid"].any()):
            return {"skipped": True, "reason": "no_supervised_targets", "updates": self.updates}
        zero = encoded.tokens.sum() * 0.0
        logits = n.route_head(decoded["map_context"], decoded["dynamic_context"], b["route_features"])
        route_loss = route_encounter_loss(logits, b["route_labels"], b["route_valid"]) if bool(b["route_valid"].any()) else zero
        prediction_loss = (gaussian_prediction_nll(decoded["state_mean"][state_valid],
                          decoded["state_logvar"][state_valid], b["state_target"][state_valid])
                           if bool(state_valid.any()) else zero)
        dynamics = received & (encoded.stream_types[None] == 1)
        if bool(dynamics.any()):
            geometry = decoded["map_context"][:, None].expand_as(encoded.tokens)
            elapsed = torch.where(dynamics, b["ages"], torch.zeros_like(b["ages"]))[dynamics]
            time_loss = temporal_composition_loss(n.decoder, encoded.tokens[dynamics],
                         geometry[dynamics], elapsed * 0.5, elapsed * 0.5)
        else:
            time_loss = zero
        loss = (self.weights["route"] * route_loss + self.weights["prediction"] * prediction_loss
                + self.weights["time"] * time_loss)
        _finite(loss, "auxiliary loss")
        before = [p.detach().clone() for p in self.parameters]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        row = dict(skipped=False, updates=self.updates, loss=float(loss.detach()),
                   route_loss=float(route_loss.detach()), prediction_loss=float(prediction_loss.detach()),
                   time_loss=float(time_loss.detach()), gradient_norm=float(norm),
                   parameter_delta_l2=_parameters_changed(self.parameters, before),
                   route_labels=int(b["route_valid"].sum()), state_labels=int(state_valid.sum()))
        self.history.append(row)
        return row

    def state_dict(self):
        return dict(updates=self.updates, max_updates=self.max_updates, experiment=self.experiment,
                    weights=self.weights, optimizer=self.optimizer.state_dict(), history=self.history)


def train_auxiliary(networks, batch, *, updates=1, **kwargs):
    trainer = AuxiliaryTrainer(networks, **kwargs)
    _cap(updates)
    for _ in range(updates):
        row = trainer.step(batch)
        if row["skipped"]:
            break
    return trainer


class CommunicationLearner:
    """On-policy single-step SMDP Actor-Critic with original task returns.

    Events must record the interval's discounted original task reward and next
    CTDE state. The latter may contain truth; scheduler inputs may not. A
    bootstrap is retained at short-test truncation (done=False). No CVaR claim
    or artificial communication reward is introduced.
    """
    def __init__(self, scheduler, critic_dim, *, gamma=.95, learning_rate=3e-4,
                 hidden_dim=64, value_weight=.5, entropy_weight=0., max_updates=4, experiment=False):
        self.scheduler = scheduler
        self.critic_dim = int(critic_dim)
        if self.critic_dim < 1 or not 0 < gamma <= 1:
            raise ValueError("critic dimension and physical-step discount are invalid")
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("critic hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self.gamma, self.value_weight, self.entropy_weight = float(gamma), float(value_weight), float(entropy_weight)
        if not math.isfinite(self.value_weight) or self.value_weight < 0 or not math.isfinite(self.entropy_weight) or self.entropy_weight < 0:
            raise ValueError("critic/entropy coefficients must be finite and nonnegative")
        self.experiment = bool(experiment)
        self.max_updates, self.updates, self.policy_version = _cap(max_updates, experiment=self.experiment), 0, 0
        device = next(scheduler.parameters()).device
        self.critic = nn.Sequential(nn.Linear(self.critic_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)).to(device)
        self.parameters = list(scheduler.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=float(learning_rate))
        self.history = []

    @property
    def device(self):
        return next(self.scheduler.parameters()).device

    def _distribution(self, event):
        tokens = _tensor(event["packet_tokens"], device=self.device)
        context = _tensor(event["context"], device=self.device)
        costs = _tensor(event["packet_bits"], device=self.device)
        budget = _tensor(event["budget_bits"], device=self.device)
        valid = _tensor(event["valid"], device=self.device, boolean=True)
        if tokens.ndim == 2:
            tokens = tokens[None]
        if context.ndim == 1:
            context = context[None]
        if budget.ndim == 0:
            budget = budget[None]
        if valid.ndim == 1:
            valid = valid[None]
        if tokens.shape[0] != 1:
            raise ValueError("one communication event must describe one sender decision")
        # Codec features are fixed during communication-policy optimization.
        return self.scheduler(tokens.detach(), context.detach(), costs, budget, valid)

    def _features(self, value):
        result = _tensor(value, device=self.device).reshape(-1)
        if result.shape != (self.critic_dim,):
            raise ValueError("CTDE feature size does not match the communication critic")
        _finite(result, "CTDE features")
        return result

    def choose(self, event, *, deterministic=False):
        with torch.no_grad():
            distribution = self._distribution(event)
            action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            return dict(action=int(action.item()), log_prob=float(distribution.log_prob(action).item()),
                        policy_version=self.policy_version)

    def value(self, features):
        with torch.no_grad():
            return float(self.critic(self._features(features)).item())

    def update(self, events):
        if self.updates >= self.max_updates:
            raise RuntimeError("bounded communication update limit reached")
        if not events:
            return dict(skipped=True, reason="no_communication_decisions", updates=self.updates)
        actor_terms, critic_terms, entropy_terms, targets = [], [], [], []
        for event in events:
            if event.get("policy_version") != self.policy_version:
                raise ValueError("stale communication trajectories are not on-policy")
            elapsed = event["elapsed"]
            step = event.get("episode_step", event.get("physical_step", 0))
            if type(elapsed) is not int or elapsed < 0 or type(step) is not int or step < 0:
                raise ValueError("SMDP elapsed/episode step must be nonnegative physical-step integers")
            reward = float(event["discounted_reward"])
            done = event["done"]
            if type(done) is not bool or not math.isfinite(reward):
                raise ValueError("event reward/terminal flag is invalid")
            distribution = self._distribution(event)
            action = torch.tensor([int(event["action"])], device=self.device)
            log_prob = distribution.log_prob(action).squeeze(0)
            _finite(log_prob, "selected communication log probability")
            old = event.get("log_prob")
            if old is not None and not math.isclose(float(log_prob.detach()), float(old), rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("communication policy changed since this decision was collected")
            value = self.critic(self._features(event["critic_features"])).squeeze(-1)
            with torch.no_grad():
                next_value = self.critic(self._features(event["next_critic_features"])).squeeze(-1)
                target = value.new_tensor(reward) + (0. if done else self.gamma ** elapsed) * next_value
            advantage = target - value
            actor_terms.append(-(self.gamma ** step) * log_prob * advantage.detach())
            critic_terms.append(advantage.square())
            entropy_terms.append(distribution.entropy().squeeze(0))
            targets.append(float(target))
        actor = torch.stack(actor_terms).mean()
        critic = torch.stack(critic_terms).mean()
        entropy = torch.stack(entropy_terms).mean()
        loss = actor + self.value_weight * critic - self.entropy_weight * entropy
        _finite(loss, "communication Actor-Critic loss")
        before = [p.detach().clone() for p in self.parameters]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.policy_version += 1
        row = dict(skipped=False, updates=self.updates, policy_version=self.policy_version,
                   events=len(events), loss=float(loss.detach()), actor_loss=float(actor.detach()),
                   critic_loss=float(critic.detach()), entropy=float(entropy.detach()),
                   gradient_norm=float(norm), parameter_delta_l2=_parameters_changed(self.parameters, before),
                   smdp_targets=targets, reward_source="original_task_interval_return")
        self.history.append(row)
        return row

    def state_dict(self):
        return dict(critic_dim=self.critic_dim, hidden_dim=self.hidden_dim, gamma=self.gamma, value_weight=self.value_weight, experiment=self.experiment,
                    entropy_weight=self.entropy_weight, max_updates=self.max_updates,
                    updates=self.updates, policy_version=self.policy_version,
                    critic=self.critic.state_dict(), optimizer=self.optimizer.state_dict(), history=self.history)


class TaskReplayUpdater:
    """Use the inherited gated 28D/3D replay and unmodified MADDPG update."""
    def __init__(self, model, *, capacity=1024, batch_size=8, minimum_batch_size=2, max_updates=4, experiment=False, device="cpu"):
        from chapter3_bser.learning.replay import ExecutionReplay
        if not 1 <= minimum_batch_size <= batch_size <= capacity:
            raise ValueError("invalid task replay batch/capacity")
        self.model = model
        self.replay = ExecutionReplay(capacity)
        self.batch_size, self.minimum_batch_size = batch_size, minimum_batch_size
        self.experiment, self.device = bool(experiment), str(device)
        self.max_updates, self.updates = _cap(max_updates, experiment=self.experiment), 0
        self.history = []

    def observe(self, transition):
        self.replay.push_transition(transition)

    def update(self):
        if self.updates >= self.max_updates:
            raise RuntimeError("bounded task MADDPG update limit reached")
        if len(self.replay) < self.minimum_batch_size:
            return dict(skipped=True, reason="not_enough_factual_transitions", updates=self.updates)
        count = min(self.batch_size, len(self.replay))
        self.model.prep_training(device=self.device)
        sample = self.replay.sample(count, device=self.device)
        losses = []
        for i in range(4):
            critic, actor, _ = self.model.update(sample, i)
            losses.append(dict(agent=i, critic_loss=critic, actor_loss=actor,
                               actor_updated=bool(self.model.last_actor_updated)))
        self.model.update_all_targets()
        self.model.prep_rollouts(device=self.device)
        self.updates += 1
        row = dict(skipped=False, updates=self.updates, batch_size=count, agents=losses,
                   reward_source="unmodified_transition_rewards", resume_supported=False)
        self.history.append(row)
        return row


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for part in ("core", "chapter3_bser", "chapter5_comm")
            for p in sorted((root / part).rglob("*.py"))}


def save_checkpoint(path, networks, *, auxiliary=None, communication=None, task_model=None,
                    metadata=None, source_identity=None):
    """Create a new inference checkpoint; never overwrite an existing file."""
    path = Path(path)
    if path.exists() or path.with_suffix(path.suffix + ".sha256.json").exists():
        raise FileExistsError(path)
    metadata = {} if metadata is None else dict(metadata)
    json.dumps(metadata, allow_nan=False)
    payload = dict(schema=CHECKPOINT_SCHEMA, neural_config=asdict(networks.config),
                   networks=networks.state_dict(), auxiliary=None if auxiliary is None else auxiliary.state_dict(),
                   communication=None if communication is None else communication.state_dict(),
                   task_model=None if task_model is None else task_model.training_state_dict(),
                   metadata=metadata, source_identity=_source_identity() if source_identity is None else source_identity,
                   torch_rng_state=torch.get_rng_state(), resume_supported=False,
                   restore_scope="inference_or_new_episode; no task replay or active environment state",
                   formal_performance_verified=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = path.with_suffix(path.suffix + ".sha256.json")
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(dict(file=path.name, sha256=sha, schema=CHECKPOINT_SCHEMA), stream, indent=2)
    return dict(path=str(path), sha256=sha, resume_supported=False)


def load_checkpoint(path, *, verify_source=True, expected_source_identity=None, device="cpu"):
    """Restore the newly defined structure; does not restore any environment."""
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if receipt.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("checkpoint hash differs from its receipt")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("resume_supported") is not False:
        raise ValueError("not a CH5 inference checkpoint")
    current = _source_identity() if expected_source_identity is None else expected_source_identity
    if verify_source and payload.get("source_identity") != current:
        raise ValueError("checkpoint code identity does not match this CH5 implementation")
    models = build_neural_models(payload["neural_config"], device=device)
    models.load_state_dict(payload["networks"], strict=True)
    for name, value in models.state_dict().items():
        if value.is_floating_point():
            _finite(value, "restored parameter " + name)
    models.eval()
    communication = None
    if payload["communication"] is not None:
        state = payload["communication"]
        communication = CommunicationLearner(models.scheduler, state["critic_dim"], gamma=state["gamma"],
                         value_weight=state["value_weight"], entropy_weight=state["entropy_weight"],
                         hidden_dim=state["hidden_dim"], max_updates=state["max_updates"], experiment=state.get("experiment", False))
        communication.critic.load_state_dict(state["critic"], strict=True)
        for name, value in communication.critic.state_dict().items():
            _finite(value, "restored communication critic " + name)
        communication.optimizer.load_state_dict(state["optimizer"])
        communication.updates, communication.policy_version = state["updates"], state["policy_version"]
        communication.history = state["history"]
        communication.critic.eval()
    return dict(networks=models, communication=communication, task_model_state=payload["task_model"],
                auxiliary_state=payload["auxiliary"], metadata=payload["metadata"], payload=payload)
