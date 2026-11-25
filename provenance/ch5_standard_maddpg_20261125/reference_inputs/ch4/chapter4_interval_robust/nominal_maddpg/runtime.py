"""Nominal closed-loop missions and real Replay transitions, with zero updates."""
import copy
import random

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import digest, state_digest
from chapter3_bser.learning.evaluate import preserved_rng
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.learning.scenarios import validate_manifest
from experiments.ch4_work1_recoverability import records
from experiments.ch4_work1_recoverability.plant_adapter import WaterParameters
from experiments.ch4_work1_recoverability.run import check_contract, completion_cost

from .config import read_json, ROOT, task_config, validate
from .learner import SingleCriticNetworks


class FrozenInitialPolicy:
    def __init__(self, config, seed):
        self.networks = SingleCriticNetworks(config, seed).requires_grad_(False)
        self.initial_sha256 = state_digest(self.networks.state_dict())
        self.policy_id = digest(dict(kind="frozen_initial_single_critic_maddpg", seed=seed,
                                     weights=self.initial_sha256))
        self.last_input_sha256 = None

    @torch.no_grad()
    def step(self, observations, explore=False):
        if explore:
            raise ValueError("frozen initialization validation uses zero exploration")
        array = np.asarray(observations, dtype=np.float32)
        if array.shape != (4, 28) or not np.isfinite(array).all():
            raise ValueError("actor receives four finite 28D observations")
        self.last_input_sha256 = records.stamp(array)
        actions = self.networks.actions(torch.from_numpy(array.copy()).unsqueeze(0))[0]
        return [actions[hull:hull+1].clone() for hull in range(4)]

    def verify_frozen(self):
        if (state_digest(self.networks.state_dict()) != self.initial_sha256
                or any(p.grad is not None for p in self.networks.parameters())):
            raise AssertionError("frozen initialization changed weights or accumulated gradients")
        return self.initial_sha256


class NominalRuntime:
    """Own task stepping and Replay insertion. The actor receives observations only."""

    def __init__(self, config, scene, policy, replay, episode_id=0):
        self.config = validate(config)
        validate_manifest(dict(scenarios=[scene]), self.config["task"],
                          evaluation=scene.get("scenario_role") == "eval")
        self.policy, self.replay = policy, replay
        self.closed = False
        protocol = read_json(ROOT/"experiments/ch4_work1_recoverability/protocol.json")
        self.world = records.Branch(task_config(self.config), scene, episode_id, protocol["task"]["approach_rule"])
        self.rt = self.world.rt
        try:
            plant = self.config["plant"]
            self.world.plant.set_parameters(WaterParameters(beta=plant["beta_m_per_s"],
                drag=plant["drag_multipliers"], thrust=plant["horizontal_thrust_multipliers"], law=plant["law"]))
            check_contract(self.rt, protocol)
        except BaseException:
            self.world.__exit__(None, None, None)
            self.closed = True
            raise
        self.world.policy = policy
        if isinstance(policy, FrozenInitialPolicy):
            self.world.prefix = digest([self.world.fingerprint, self.world.seed, policy.policy_id])
        self.phase_counts = records.phase_counts()
        self.discounted_return = self.undiscounted_return = 0.0
        self.transition_count = self.nonzero_residual_steps = 0
        self.closed_gate_boat_steps = self.suppressed_nonzero_boat_steps = 0
        self.max_lag_error = self.max_thrust_error = 0.0
        self.last_record = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self.closed:
            self.world.__exit__(None, None, None)
            self.closed = True

    @property
    def terminal(self):
        return self.rt.terminal

    @property
    def step(self):
        return self.rt.step

    @torch.no_grad()
    def advance(self):
        if self.closed or self.terminal:
            raise RuntimeError("cannot advance a closed or terminated nominal mission")
        before = self.world.decision()
        observations = np.asarray(self.rt.observations, dtype=np.float32).copy()
        for phase, hull in records.count_before(before, self.phase_counts).items():
            h = before["hulls"][hull]
            self.phase_counts[phase]["window"] = dict(t=before["t"], vehicle=hull,
                assignment=h["assignment_id"], tracking_waypoint=h["tracking_waypoint"])
        self.world.events = []
        with preserved_rng():
            random.setstate(self.world.rng[0])
            np.random.set_state(self.world.rng[1])
            torch.set_rng_state(self.world.rng[2])
            row = self.rt.advance(self.policy, explore=False)
        self.world.rng = records.step_stream(self.world.seed, before["t"]+1)
        self._check_transition(before, observations, row)
        outcome = self.rt.env.get_episode_result()
        row["success_flags"] = [bool(outcome["success"])]*4
        index = self.replay.next_idx
        self.replay.push_transition(row)
        self.replay.verify_slot(index, row)
        plant = self.world.plant
        evidence = [dict(id=f"{self.world.seed}/motion/{before['t']}/{i}", kind="command_response",
            source_vehicle=i, command_step=before["t"], sample_step=before["t"]+1,
            arrival_step=None, receiver_vehicle=None, teammate_visibility="offline_only") for i in range(4)]
        legacy = dict(before=before, residual=row["actions"].tolist(), command_mps2=plant.command.tolist(),
            lag_after=plant.lag.tolist(), thrust_mps2=plant.thrust_acc.tolist(),
            position_after_m=self.world.core._agent_pos.tolist(), velocity_after_mps=self.world.core._agent_vel.tolist(),
            team_reward=row["team_reward"], rewards=row["rewards"].tolist(), done=row["dones"],
            termination=outcome["termination_reason"], motion_evidence=evidence,
            planner_events=self.world.events.copy(), next_observations_sha256=records.stamp(row["next_observations"]))
        self.world.prefix = digest([self.world.prefix, digest(legacy)])
        self.last_record = dict(legacy=legacy, proposed_actions=row["proposed_actions"].tolist(),
            action_enabled=row["action_enabled"].tolist(), next_action_enabled=row["next_action_enabled"].tolist(),
            replay_index=index, source_rewards=row["source_reward_by_agent"],
            reward_transform_applied_count=row["reward_transform_applied_count"], outcome=copy.deepcopy(outcome))
        self.discounted_return += self.config["task"]["gamma"]**before["t"] * row["team_reward"]
        self.undiscounted_return += row["team_reward"]
        self.transition_count += 1
        self.nonzero_residual_steps += int(np.any(row["actions"]))
        return row

    def _check_transition(self, before, observations, row):
        if row["t"] != before["t"] or self.step != before["t"]+1:
            raise AssertionError("mission clock differs from the transition")
        if not np.array_equal(observations, np.asarray(row["observations"])):
            raise AssertionError("actor input differs from the stored pre-action observation")
        if isinstance(self.policy, FrozenInitialPolicy) and self.policy.last_input_sha256 != records.stamp(observations):
            raise AssertionError("frozen actor received a different input")
        gates = row["action_enabled"]
        expected = np.clip(row["proposed_actions"], -1, 1) * gates[:, None]
        if not np.array_equal(gates, [h["gate"] for h in before["hulls"]]) or not np.array_equal(row["actions"], expected):
            raise AssertionError("applied residual differs from the control gate or clipping")
        next_gates = np.zeros(4, dtype=bool) if self.terminal else self.rt._gates()
        if not np.array_equal(row["next_action_enabled"], next_gates):
            raise AssertionError("next-state execution gates differ")
        if not np.array_equal(np.asarray(row["next_observations"]), np.asarray(self.rt.observations)):
            raise AssertionError("Replay next observation missed closed-loop guidance refresh")
        outcome = self.rt.env.get_episode_result()
        if row["dones"] != [self.terminal]*4 or bool(outcome["terminated"]) != self.terminal:
            raise AssertionError("team terminal and Replay done flags differ")
        team = float(torch.as_tensor(row["source_reward_by_agent"], dtype=torch.float32).mean())
        if (team != row["team_reward"] or not np.array_equal(row["rewards"], np.full(4, team, dtype=np.float32))
                or row["reward_transform_applied_count"] != 1):
            raise AssertionError("original team reward was transformed more than once or relabeled")
        self.closed_gate_boat_steps += int((~gates).sum())
        self.suppressed_nonzero_boat_steps += int(((np.abs(row["proposed_actions"]).sum(axis=1)>0) & ~gates).sum())
        plant, core = self.world.plant, self.world.core
        command = torch.clamp(core._last_prior_acc + core._last_residual_acc, -plant.limits, plant.limits)
        if not torch.equal(command, plant.command):
            raise AssertionError("physical command was recomposed or scaled differently")
        lag = 0.15*np.asarray(before["lag_before"]) + 0.85*plant.sent.cpu().numpy()
        thrust = lag*plant.limits.cpu().numpy()
        thrust[:, :2] *= np.asarray(plant.parameters.thrust)[:, None]
        self.max_lag_error = max(self.max_lag_error, float(np.abs(lag-plant.lag.cpu().numpy()).max()))
        self.max_thrust_error = max(self.max_thrust_error, float(np.abs(thrust-plant.thrust_acc.cpu().numpy()).max()))
        if max(self.max_lag_error, self.max_thrust_error) > 2e-6:
            raise AssertionError("propulsion memory or one-time thrust multiplier differs")

    def finalize(self):
        if not self.terminal:
            raise RuntimeError("a partial collection prefix is not a completed task")
        outcome = self.rt.env.get_episode_result()
        accounting = self.rt.env.reward_accounting
        if (self.transition_count != self.step or self.discounted_return != accounting.team_discounted_return
                or self.undiscounted_return != accounting.team_undiscounted_return):
            raise AssertionError("stepwise return or transition count differs from native accounting")
        if isinstance(self.policy, FrozenInitialPolicy):
            self.policy.verify_frozen()
        return dict(steps=self.step, termination=outcome["termination_reason"], success=outcome["success"],
            penalized_completion_steps=completion_cost(outcome, self.step, self.config["task"]),
            team_discounted_return=self.discounted_return, team_undiscounted_return=self.undiscounted_return,
            transitions=self.transition_count, nonzero_residual_steps=self.nonzero_residual_steps,
            closed_gate_boat_steps=self.closed_gate_boat_steps,
            suppressed_nonzero_boat_steps=self.suppressed_nonzero_boat_steps,
            phase_counts=copy.deepcopy(self.phase_counts), max_lag_error=self.max_lag_error,
            max_thrust_error=self.max_thrust_error, trajectory_sha256=self.world.prefix)


def json_transition(row):
    result = {name:np.asarray(row[name]).tolist() for name in ("observations", "actions", "rewards",
              "next_observations", "dones", "action_enabled", "next_action_enabled", "success_flags")}
    result.update(t=row["t"], team_reward=row["team_reward"])
    return result
