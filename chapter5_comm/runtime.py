"""Causal Chapter 5 communication adapter for the inherited 28D task.

The inherited environment owns physics, sensing, target motion, task events,
capture, collisions and reward equations. Its shared map is retained ONLY for
the unchanged task/reward diagnostics. Deployment navigation uses four separate
instances of the inherited unknown-map planner, updated from own scans and
actually delivered scan frames. Training truth is exposed by a separate method.
"""
from collections import deque
from copy import deepcopy
import struct

import torch
import numpy as np

from core.config.active_environment import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.env.target_motion import TargetState, predict_target_state, solve_intercept_point, swept_relative_min_distance
from core.env.task_protocol import PROTOCOL_FIELDS, strict_terminal, validate_task_config
from core.env.uav_env import UAVEnv
from core.mapping.path_planner import OnlineUnknownMapTaskPlanner
from chapter3_bser.learning.base_runtime import seed_innovations
from chapter3_bser.rewards.training_env import Phase1CV2TrainingEnv
from chapter3_bser.protocols.team_environment import PRRACTrainingEnv
from chapter3_bser.experiments.coupled_search_v1.execution_forecast import nominal_step
from .d2_navigation import LocalD2Allocator
from .protocol import Kind, TDMAChannel


VEC6 = struct.Struct("<6f")
QUERY9 = struct.Struct("<9f")
GOAL3 = struct.Struct("<3f")
SCAN = struct.Struct("<29fI")  # own origin, 26 distances, 26-bit hit mask
ACK = struct.Struct("<I")


def _np(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()


def _observations(values):
    rows = [_np(value).astype(np.float32) for value in values]
    if np.asarray(rows).shape != (4, 28) or not np.isfinite(rows).all():
        raise ValueError("Chapter 5 requires finite inherited 4 x 28 inputs")
    return rows


class _LocalAgent:
    def __init__(self, mapper, position):
        self.mapper = mapper
        self.peers, self.queries = {}, {}
        self.target = None
        self.command = None
        self.last_command_step = -1
        self.scan = None
        self.scan_seen = set()
        self.goal = np.asarray(position, float).copy()
        self.waypoint = self.goal.copy()
        self.path = []
        self.path_index = 0
        self.last_plan = -100000
        self.last_goal = None
        self.last_allocation = -100000
        self.allocation = {}
        self.intercept_key = None
        self.soft_state = None
        self.soft_route = None
        self.soft_step = -1
        self.acks = deque(maxlen=32)


class _BootstrapEndpointGuard:
    """Keep an unresolved initial point without asserting map reachability.

    The inherited reset samples initial waypoints twice, including after scans
    quantize occupancy. A physically generated start may lie in a cell that
    the current uncertain map cannot connect. Only defer that diagnostic
    waypoint; do not clear occupancy, move the vehicle, resample the scenario,
    or alter collision/reward rules. CH5 local planning starts after reset.
    """
    def initial_search_targets(self, agent_positions):
        positions = self._as_points(agent_positions)
        if positions.shape != (self.n_search, 3) or not bool(torch.isfinite(positions).all()):
            raise ValueError("initial search positions must be a finite n_search x 3 tensor")
        if bool(((positions < 0.) | (positions > self.space_size)).any()):
            raise ValueError("initial search positions must remain inside the mission world")
        self.ch5_bootstrap_endpoint_deferrals = []
        targets, reserved = [], []
        for agent in range(self.n_search):
            status = self.endpoint_status(positions[agent], role="searcher")
            if status["reachable"]:
                target = self.sample_next_waypoint(agent, positions[agent], reserved_positions=reserved)
            else:
                target = positions[agent].clone()
                self.ch5_bootstrap_endpoint_deferrals.append(dict(agent=agent, reason=status["failure_reason"]))
            targets.append(target)
            reserved.append(target)
        return torch.stack(targets)


class _CH5BootstrapPlanner(_BootstrapEndpointGuard, OnlineUnknownMapTaskPlanner):
    """Original online-map kernel with guarded CH5 bootstrap only."""


class _CommunicatingUAVEnv(UAVEnv):
    """Only communication-dependent hooks differ from the inherited task."""
    def __init__(self, owner, **kwargs):
        self._comm_owner = owner
        super().__init__(**kwargs)

    def _active(self):
        return bool(self._comm_owner._ready)

    def _build_obstacle_planner(self):
        if self.obstacle_knowledge_mode != "online_unknown":
            return super()._build_obstacle_planner()
        kwargs = self._planner_common_kwargs()
        # Same inherited parameters; only the initial waypoint boundary differs.
        names = ("occupancy_free_logodds", "occupancy_occupied_logodds", "occupancy_logodds_clip",
            "occupancy_free_threshold", "occupancy_occupied_threshold", "occupancy_unknown_cost_weight",
            "occupancy_risk_cost_weight", "occupancy_replan_probability_delta",
            "target_negative_observation_strength", "target_negative_likelihood_floor",
            "target_revisit_half_life_steps", "target_recency_penalty_weight",
            "obstacle_information_gain_weight", "reservation_decay")
        kwargs.update({name: getattr(self, name) for name in names})
        return _CH5BootstrapPlanner(**kwargs)

    def _advance_fixed_handoff(self):
        if not self._active():
            return super()._advance_fixed_handoff()
        return False

    def _update_pse_executor_standby(self, *args, **kwargs):
        if not self._active():
            return super()._update_pse_executor_standby(*args, **kwargs)

    def _known_prediction_obstacles(self):
        if not self._active():
            return super()._known_prediction_obstacles()
        return self._comm_owner.locals[3].mapper.known_obstacle_aabbs()

    def _refresh_intercept(self, current_step):
        if not self._active():
            return super()._refresh_intercept(current_step)
        local = self._comm_owner.locals[3]
        if local.target is not None:
            if self._comm_owner.d2_allocator is not None:
                self._comm_owner._refresh_local_intercept(current_step)
                return
            estimate = self._comm_owner.predict_local_target(3)
            self.predicted_intercept_position = estimate.position.copy()
            self._agent_task_est[3].copy_(self._vec(estimate.position))

    def _update_nav_targets(self, force=False):
        if not self._active():
            return super()._update_nav_targets(force=force)
        self._comm_owner._update_navigation(force=force)

    def _publish_detection(self, finder_idx):
        super()._publish_detection(finder_idx)
        if self._active():
            state = self.target_state.copy()
            state.sample_step = int(self.step_count)
            self._comm_owner._accept_target(int(finder_idx), state, own_measurement=True)

    def _maybe_detect_swept(self, agent_start, target_start):
        super()._maybe_detect_swept(agent_start, target_start)
        if not self._active():
            return
        # A previously found target can still be measured independently. A
        # global Found event itself is never published into another local view.
        for agent in range(3):
            distance, _ = swept_relative_min_distance(agent_start[agent], _np(self._agent_pos[agent]),
                                                      target_start, self.target_state.position)
            if distance <= float(self._sensor_range[agent]) + self.detect_eps_bias:
                state = self.target_state.copy()
                state.sample_step = int(self.step_count)
                self._comm_owner._accept_target(agent, state, own_measurement=True)

    def _apply_agent_dynamics(self, actions):
        if not self._active():
            return super()._apply_agent_dynamics(actions)
        # Preserve the original physical integrator. Its communication-derived
        # stop mask must be local, while truth bookkeeping remains untouched for
        # the original reward/capture implementation immediately afterwards.
        truth_finished = self.agent_finished.clone()
        self.agent_finished[:3] = torch.tensor(
            [local.target is not None for local in self._comm_owner.locals[:3]], device=self.device)
        try:
            return super()._apply_agent_dynamics(actions)
        finally:
            self.agent_finished.copy_(truth_finished)

    def _sense_and_update_shared_map(self):
        changed = super()._sense_and_update_shared_map()
        if self._active():
            self._comm_owner._sense_local()
        return changed

    def _get_obs(self):
        if not self._active():
            return super()._get_obs()
        result = []
        # Deployment visibility adapter: retain the original feature scale and
        # 28D/task/reward contracts, but censor distances beyond own sensor range
        # even if an inherited distance backend returns ground-truth geometry.
        distances = torch.clamp(self._nearest_obstacle_distance(self._agent_pos),
                                max=float(self.obstacle_sensor_range))
        nearest = torch.clamp(distances / 10., 0., 1.)
        for i, local in enumerate(self._comm_owner.locals):
            pos, vel = self._agent_pos[i], self._agent_vel[i]
            relative = self._nav_targets[i] - pos
            norm = torch.norm(relative).clamp_min(self.eps)
            direction = relative / norm
            speed = torch.norm(vel).clamp_min(self.eps)
            known = local.target is not None
            estimate = self._comm_owner.predict_local_target(i) if known else None
            delta = self._vec(estimate.position) - pos if known else self._zero3
            finished = float(known) if i < 3 else float(self.agent_finished[i])
            # No global hold/waypoint event is communicated through the actor's
            # scalar features. Own reached progress is maintained by the local
            # path tracker; executor contact uses its own capture sensor only.
            progress = min(1., self._comm_owner.local_reached[i] / max(1., self._comm_owner.local_assigned[i]))
            hold = (min(1., float(self._capture_hold_counter) / self._hold_steps_for_agent(i))
                    if i == 3 and known else 0.)
            extras = self._vec([float(nearest[i]), progress, finished, hold])
            phase = self._vec([0., 1.] if known else [1., 0.])
            result.append(torch.cat((pos, vel, relative, direction, delta,
                torch.clamp(norm / 10., 0., 1.).reshape(1),
                torch.clamp(speed / (self._v_xy_max[i] + self.eps), 0., 1.).reshape(1),
                torch.tanh((torch.dot(vel, direction) / (self._v_xy_max[i] + self.eps)).reshape(1)),
                extras, self.role_onehots[i], phase)).to(self.dtype))
        return result


class _MissionAdapter(MissionCoreEnv):
    communication_mode = "ch5_all_channel_tdma_v1"

    def __init__(self, owner, **kwargs):
        self._owner = owner
        self._env = _CommunicatingUAVEnv(owner, **kwargs)
        self.return_numpy = bool(self._env.return_numpy)
        self._last_observations = None

    def reset(self, scenario=None):
        self._owner._ready = False
        self._env.reset(scenario=scenario)
        self._owner._initialize_locals(self._env)
        self._last_observations = self._env._obs_to_public(self._env._get_obs())
        return self._last_observations


class CH5Runtime:
    """Four causal knowledge stores around the existing third-chapter task.

    Ordering is explicit: begin_step(t) delivers packets due at t and sends at
    most one fair bootstrap frame. The caller decodes/drains received latents,
    installs soft context, sends at most one learned frame, then advance(actions)
    executes exactly one inherited physical transition. It never delivers t+1
    frames early. max_steps always remains the original configured deadline.
    """
    def __init__(self, config, scenario=None, *, seed=1, episode_id=0,
                 channel_config=None, horizon=8, candidate_count=4, map_token_count=32,
                 navigation="local_waypoints_v1"):
        validate_task_config(config)
        self.config, self.scenario = deepcopy(config), deepcopy(scenario)
        self.seed, self.episode_id = int(seed), int(episode_id)
        self.horizon, self.candidate_count = int(horizon), int(candidate_count)
        self.map_token_count = int(map_token_count)
        if navigation not in ("local_waypoints_v1", "causal_d2_v1"):
            raise ValueError("unknown Chapter 5 navigation")
        self.navigation = navigation
        self.d2_allocator = LocalD2Allocator() if navigation == "causal_d2_v1" else None
        if min(self.horizon, self.candidate_count, self.map_token_count) <= 0:
            raise ValueError("positive feature dimensions required")
        self._ready = False
        self.locals = []
        seed_innovations(self.seed)
        settings = build_ch3_config(str(config.get("base_candidate", "ch3_v3_full_reference")), str(config["profile"]))
        settings.update({key: config[key] for key in (*PROTOCOL_FIELDS, "collision_terminal_reward", "waypoint_reward_timing") if key in config})
        kwargs = environment_kwargs_from_config(settings, device="cpu", max_steps=int(config["max_steps"]), return_numpy=False)
        channel_settings = dict(channel_config or {})
        channel_settings.setdefault("seed", self.seed + 7919)
        channel_settings.setdefault("dt", float(kwargs.get("dt", .2)))
        self.channel = TDMAChannel(**channel_settings)
        self.base = _MissionAdapter(self, **kwargs)
        rewards = Phase1CV2TrainingEnv(self.base, reward_config=config["reward"])
        self.env = PRRACTrainingEnv(rewards, reward_objective_config=config, gamma=config["rl"]["gamma"])
        self.reset(scenario=scenario)

    @property
    def physical(self):
        return self.base.unwrapped

    @property
    def step(self):
        return int(self.physical.step_count)

    @property
    def dt(self):
        return float(self.physical.dt)

    @property
    def terminal(self):
        return bool(strict_terminal(self.env))

    @property
    def tdma_sender(self):
        return self.channel.owner(self.step)

    def reset(self, scenario=None):
        if scenario is not None:
            self.scenario = deepcopy(scenario)
        self.channel.reset()
        self._begun_step = -1
        self._neural_deliveries = []
        self._metadata = {}
        self._bootstrap_turn = np.zeros(4, dtype=int)
        self._receiver_turn = {(i, kind): 0 for i in range(4) for kind in Kind}
        self._learned_sent_step = -1
        self.local_reached = np.zeros(4, int)
        self.local_assigned = np.ones(4, int)
        self.observations = _observations(self.env.reset(scenario=self.scenario,
            episode_id=self.episode_id, episode_index=self.episode_id))
        return self.observations

    def _initialize_locals(self, env):
        self.locals = []
        self.bounds = _np(env.space_size).astype(float)
        for agent in range(4):
            mapper = env._build_obstacle_planner()
            mapper.reset(None, [])
            self.locals.append(_LocalAgent(mapper, _np(env._agent_pos[agent])))
        self._ready = True
        self._sense_local()
        self._update_navigation(force=True)

    def _sense_local(self):
        env = self.physical
        for i, local in enumerate(self.locals):
            position = _np(env._agent_pos[i]).astype(float)
            mapper = local.mapper
            mapper.set_mapping_step(self.step)
            executor_position = (position if i == 3 else
                                 local.peers[3][1] if 3 in local.peers else None)
            mapper.set_runtime_context(executor_pos=executor_position,
                executor_wait_point=None, use_belief=local.target is None,
                use_exec_cost=executor_position is not None, use_standby=i == 3,
                step=self.step)
            directions = _np(mapper.sensor_directions)
            readings = [env._raycast_ground_truth(position, ray) for ray in directions]
            distances = np.array([r[0] for r in readings], float)
            hits = np.array([r[1] for r in readings], bool)
            mapper.integrate_obstacle_scan(position[None], directions, distances[None], hits[None], current_step=self.step)
            mapper.predict_belief_motion()
            if i < 3 and local.target is None:
                mapper.update_belief_negative(env._agent_pos[i:i + 1], env._sensor_range[i:i + 1])
                if self.step % env.planner_step_update_interval == 0:
                    mapper.update_from_searcher_positions(env._agent_pos[i:i + 1],
                        apply_decay=False, suppress_only=env.planner_step_update_suppress_only,
                        sensor_ranges=env._sensor_range[i:i + 1])
            local.scan = (self.step, position, distances, hits)
            local.scan_seen.add((i, self.step))

    def predict_local_target(self, agent):
        local = self.locals[int(agent)]
        state = local.target
        if state is None:
            return None
        env = self.physical
        try:
            result = predict_target_state(state, max(0, self.step - state.sample_step), env.dt,
                self.bounds, local.mapper.known_obstacle_aabbs(), clearance=env.target_obstacle_clearance,
                max_reflections=env.target_max_reflections_per_step,
                max_prediction_steps=env.max_steps + env.failure_penalty_steps)
        except (ValueError, RuntimeError):
            # A prediction may fail in incomplete geometry. Holding the last
            # received sample is legal and requires no ground-truth fallback.
            result = state.copy()
        if local.soft_state is not None and self.step - local.soft_step <= self.horizon:
            result.position = np.clip(local.soft_state[:3], 0., self.bounds)
            result.velocity = local.soft_state[3:].copy()
        return result

    def _accept_target(self, receiver, state, *, own_measurement=False, delivery_step=None):
        local = self.locals[receiver]
        if local.target is not None and state.sample_step < local.target.sample_step:
            return
        local.target = state.copy()
        local.soft_state = None
        local.mapper.update_belief_detection(state.position)
        env = self.physical
        env._agent_task_known[receiver] = True
        env._agent_task_est[receiver].copy_(env._vec(state.position))
        if receiver == 3 and not own_measurement:
            first = not bool(env.executor_target_assigned)
            env.executor_target_assigned = True
            env.executor_delivered_target_state = state.copy()
            env.executor_received_target_step = int(self.step if delivery_step is None else delivery_step)
            env.current_target_arrived[3] = False
            if first:
                env.ch3_handoff_count += 1
                env.last_handoff_delay = float(env.executor_received_target_step - env.found_step)
                env.handoff_payload_sample_step = int(state.sample_step)
                env.handoff_payload_delivery_step = env.executor_received_target_step
                env.handoff_physical_age_at_delivery_steps = env.executor_received_target_step - state.sample_step
                env.predicted_target_position_at_delivery = self.predict_local_target(3).position.copy()
                # This metric stays training/diagnostic-only.
                env.target_prediction_error_at_delivery = float(np.linalg.norm(env.predicted_target_position_at_delivery - env.target_state.position))

    def _sample_goal(self, agent, position, mapper=None):
        mapper = self.locals[agent].mapper if mapper is None else mapper
        # Local planner stochasticity must not consume the physical flow RNG.
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed * 100003 + self.step * 13 + agent)
                return _np(mapper.sample_next_waypoint(agent, position)).astype(float)
        except RuntimeError:
            return np.asarray(position, float).copy()

    def _allocate_local_d2(self, agent):
        local, env = self.locals[agent], self.physical
        if self.step - local.last_allocation >= env.path_replan_interval:
            local.allocation = self.d2_allocator.allocate(
                mapper=local.mapper, step=self.step, receiver=agent,
                position=_np(env._agent_pos[agent]), velocity=_np(env._agent_vel[agent]),
                peers=local.peers, specs=env.agent_specs, standby_goal=local.goal,
            )
            local.last_allocation = self.step
        return local.allocation

    def _refresh_local_intercept(self, current_step):
        env, local = self.physical, self.locals[3]
        state = local.target
        if local.soft_state is not None and current_step - local.soft_step <= self.horizon:
            state = self.predict_local_target(3)
            state.sample_step = int(current_step)
        position = _np(env._agent_pos[3])
        key = (current_step, local.mapper.map_revision, position.tobytes(),
               state.sample_step, state.position.tobytes(), state.velocity.tobytes())
        if key == local.intercept_key:
            return
        local.intercept_key = key

        def travel_time(start, goal):
            result = local.mapper.grid_geodesic_cost(start, goal, role="executor")
            return float(result.get("travel_time", result["cost"])) if result["reachable"] else float("inf")

        options = dict(dt=env.dt, bounds=self.bounds, max_iterations=env.executor_intercept_iterations,
                       max_reflections=env.target_max_reflections_per_step,
                       max_prediction_steps=env.max_steps + env.failure_penalty_steps)
        try:
            result = solve_intercept_point(position, state, current_step, travel_time,
                obstacles=local.mapper.known_obstacle_aabbs(), clearance=env.target_obstacle_clearance, **options)
        except (RuntimeError, ValueError):
            env.target_prediction_map_fallback_count += 1
            result = solve_intercept_point(position, state, current_step, travel_time, **options)
        env.predicted_intercept_position = result["position"]
        if result["reachable"]:
            env._agent_task_est[3].copy_(env._vec(result["position"]))
        else:
            env.path_unreachable_count += 1

    def _d2_goal(self, agent, position):
        local, env = self.locals[agent], self.physical
        if agent == 3 and local.target is not None:
            self._refresh_local_intercept(self.step)
            goal = env.predicted_intercept_position
            return position if goal is None else goal
        if agent < 3 and local.target is not None:
            return position
        arrived = local.last_goal is None or np.linalg.norm(local.goal - position) <= env.search_arrive_eps
        if agent < 3 and local.command is not None:
            goal = local.command[1]
            if np.linalg.norm(goal - position) > env.search_arrive_eps:
                return goal
            local.command = None
        if agent == 3 or arrived or not local.path:
            goal = self._allocate_local_d2(agent).get(agent)
            if goal is not None:
                if agent < 3 and local.last_goal is not None and arrived and np.linalg.norm(goal - local.goal) > env.search_arrive_eps:
                    self.local_reached[agent] += 1
                    self.local_assigned[agent] += 1
                return goal
            return position
        return local.goal

    def _update_navigation(self, force=False):
        env = self.physical
        for i, local in enumerate(self.locals):
            position = _np(env._agent_pos[i])
            if self.d2_allocator is not None:
                local.goal = self._d2_goal(i, position)
            elif i < 3 and local.target is not None:
                local.goal = position.copy()
            elif i == 3 and local.target is not None:
                local.goal = self.predict_local_target(i).position.copy()
            elif local.command is not None:
                local.goal = local.command[1].copy()
            elif i < 3 and (local.last_goal is None or np.linalg.norm(local.goal - position) <= env.search_arrive_eps):
                if local.last_goal is not None:
                    self.local_reached[i] += 1
                    self.local_assigned[i] += 1
                local.goal = self._sample_goal(i, position)
            changed = local.last_goal is None or np.linalg.norm(local.goal - local.last_goal) > env.path_subgoal_radius
            periodic = self.step - local.last_plan >= env.path_replan_interval
            if force or changed or periodic:
                role = "executor" if i == 3 else "searcher"
                result = local.mapper.grid_astar_path(position, local.goal, role=role)
                local.path = [_np(p) for p in local.mapper.path_to_subgoals(result, local.goal)] if result["reachable"] else []
                local.path_index = 0
                local.last_plan = self.step
                local.last_goal = local.goal.copy()
            while local.path_index < len(local.path) - 1 and np.linalg.norm(position - local.path[local.path_index]) <= env.path_subgoal_radius:
                local.path_index += 1
            local.waypoint = local.path[local.path_index].copy() if local.path_index < len(local.path) else position.copy()
            if i < 3 and local.target is not None:
                local.waypoint = position.copy()
            env._nav_targets[i] = env._vec(local.waypoint)
        env._targets.copy_(env._nav_targets)

    def _query(self, agent):
        local = self.locals[agent]
        return np.r_[_np(self.physical._agent_pos[agent]) / self.bounds,
                     local.goal / self.bounds, float(local.target is not None),
                     (self.physical.max_steps - self.step) / self.physical.max_steps,
                     float(agent == 3)].astype(np.float32)

    def _send(self, kind, sender, receiver, payload, sample_step):
        # True distance is used by channel physics only, never by policy features.
        distance = float(torch.norm(self.physical._agent_pos[sender] - self.physical._agent_pos[receiver]))
        return self.channel.send(kind, sender, receiver, payload,
            sample_step=int(sample_step), step=self.step, distance=distance)

    def _bootstrap(self):
        sender = self.tdma_sender
        local = self.locals[sender]
        kinds = [Kind.STATE, Kind.SCAN, Kind.QUERY, Kind.TARGET, Kind.COMMAND, Kind.ACK]
        for offset in range(len(kinds)):
            index = (int(self._bootstrap_turn[sender]) + offset) % len(kinds)
            kind = kinds[index]
            peers = [i for i in range(4) if i != sender]
            if kind == Kind.TARGET:
                if local.target is None:
                    continue
                peers.sort(key=lambda i: (i != 3, i))
            if kind == Kind.COMMAND:
                peers = [i for i in sorted(local.peers) if i < 3 and i != sender]
                if sender != 0 or not peers or local.target is not None:
                    continue
            if kind == Kind.ACK and not local.acks:
                continue
            cursor = self._receiver_turn[sender, kind]
            receiver = peers[cursor % len(peers)]
            sample = self.step
            if kind == Kind.STATE:
                payload = VEC6.pack(*np.r_[_np(self.physical._agent_pos[sender]), _np(self.physical._agent_vel[sender])])
            elif kind == Kind.SCAN:
                sample, origin, distances, hits = local.scan
                mask = sum(int(hit) << j for j, hit in enumerate(hits))
                payload = SCAN.pack(*np.r_[origin, distances], mask)
            elif kind == Kind.QUERY:
                payload = QUERY9.pack(*self._query(sender))
            elif kind == Kind.TARGET:
                sample = local.target.sample_step
                payload = VEC6.pack(*np.r_[local.target.position, local.target.velocity])
            elif kind == Kind.COMMAND:
                if self.d2_allocator is None:
                    _, peer_position, _ = local.peers[receiver]
                    goal = self._sample_goal(receiver, peer_position, mapper=local.mapper)
                else:
                    goal = self._allocate_local_d2(sender).get(receiver)
                    if goal is None:
                        self._receiver_turn[sender, kind] += 1
                        continue
                    sample = local.last_allocation
                payload = GOAL3.pack(*goal)
            else:
                receiver, packet_id = local.acks[0]
                payload = ACK.pack(packet_id)
            result = self._send(kind, sender, receiver, payload, sample)
            self._bootstrap_turn[sender] = (index + 1) % len(kinds)
            if result["accepted"]:
                self._receiver_turn[sender, kind] += 1
                if kind == Kind.ACK:
                    local.acks.popleft()
            return result
        return None

    def _deliver(self, packet, due):
        local = self.locals[packet.receiver]
        kind, body = packet.kind, packet.payload
        if kind in (Kind.STATE, Kind.TARGET):
            values = np.array(VEC6.unpack(body), float)
            if kind == Kind.STATE:
                old = local.peers.get(packet.sender)
                if old is None or packet.sample_step >= old[0]:
                    local.peers[packet.sender] = (packet.sample_step, values[:3], values[3:])
            else:
                state = TargetState(values[:3], values[3:], packet.sample_step,
                                    self.physical.target_motion_mode)
                self._accept_target(packet.receiver, state, delivery_step=due)
        elif kind == Kind.SCAN:
            key = (packet.sender, packet.sample_step)
            if key not in local.scan_seen:
                values = SCAN.unpack(body)
                origin = np.array(values[:3], float)
                distances = np.array(values[3:29], float)
                hits = np.array([(values[29] >> j) & 1 for j in range(26)], bool)
                mapper = local.mapper
                # Age remains the source sample time, not falsely fresh receipt.
                mapper.integrate_obstacle_scan(origin[None], _np(mapper.sensor_directions),
                    distances[None], hits[None], current_step=packet.sample_step)
                mapper.set_mapping_step(self.step)
                local.scan_seen.add(key)
        elif kind == Kind.QUERY:
            old = local.queries.get(packet.sender)
            if old is None or packet.sample_step >= old[0]:
                local.queries[packet.sender] = (packet.sample_step, np.array(QUERY9.unpack(body), np.float32))
        elif kind == Kind.COMMAND:
            if packet.sender == 0 and packet.sample_step > local.last_command_step:
                local.command = (packet.sample_step, np.clip(GOAL3.unpack(body), 0., self.bounds))
                local.last_command_step = packet.sample_step
                local.last_plan = -100000
        elif kind == Kind.ACK:
            ACK.unpack(body)  # receipt is accounted, no free retransmission/ACK state
        elif kind in (Kind.MAP_LATENT, Kind.DYNAMIC_LATENT, Kind.TARGET_LATENT):
            codes = np.frombuffer(body, dtype=np.int8).copy()
            self._neural_deliveries.append(dict(sender=packet.sender, receiver=packet.receiver,
                source_step=packet.sample_step, delivery_step=due, stream_type=int(kind != Kind.MAP_LATENT),
                target_available=bool(kind == Kind.TARGET_LATENT),
                codes=codes, tokens=codes.copy(), packet_id=packet.sequence,
                metadata_id=self._metadata.get(packet.sequence)))
        if kind != Kind.ACK:
            local.acks.append((packet.sender, packet.sequence))

    def begin_step(self):
        if self._begun_step == self.step:
            return {"step": self.step, "tdma_sender": self.tdma_sender, "bootstrap": None}
        self.channel.refill(self.step)
        for packet, due in self.channel.receive(self.step):
            self._deliver(packet, due)
        self._update_navigation()
        self.observations = _observations(self.physical._get_obs())
        bootstrap = self._bootstrap()
        self._begun_step = self.step
        return {"step": self.step, "tdma_sender": self.tdma_sender, "bootstrap": bootstrap}

    def send_latent(self, sender, receiver, codes, stream_type, source_step=None, metadata_id=None,
                    target_available=False):
        self.begin_step()
        values = np.asarray(codes)
        if values.ndim != 1 or values.dtype != np.int8 or len(values) < 1:
            raise ValueError("latent codes must be a nonempty 1D int8 array")
        if int(stream_type) not in (0, 1):
            raise ValueError("stream_type must be geometry=0 or dynamic=1")
        if not isinstance(target_available, (bool, np.bool_)):
            raise ValueError("target_available must be a boolean evidence flag")
        if target_available and int(stream_type) != 1:
            raise ValueError("only a dynamic latent can carry target evidence")
        if target_available and self.locals[int(sender)].target is None:
            return {"accepted": False, "reason": "target_not_legally_known", "bits": (16 + len(values)) * 8}
        if self._learned_sent_step == self.step:
            return {"accepted": False, "reason": "one_learned_frame_per_slot", "bits": (16 + len(values)) * 8}
        kind = Kind.MAP_LATENT if int(stream_type) == 0 else Kind.DYNAMIC_LATENT
        if target_available:
            kind = Kind.TARGET_LATENT
        result = self._send(kind, int(sender), int(receiver), values.tobytes(), self.step if source_step is None else source_step)
        if result["accepted"]:
            self._learned_sent_step = self.step
            self._metadata[result["packet_id"]] = metadata_id
        return result

    def drain_neural_deliveries(self):
        result, self._neural_deliveries = self._neural_deliveries, []
        return result

    delivered_latents = drain_neural_deliveries

    def tokenize(self, agent, receiver):
        agent, receiver = int(agent), int(receiver)
        local, env = self.locals[agent], self.physical
        mapper = local.mapper
        centers = _np(mapper.flat_xyz_centers)
        counts = _np(mapper.occupancy_observation_count).reshape(-1).astype(float)
        observed = counts > 0
        last = _np(mapper.occupancy_last_observed_step).reshape(-1)
        probability = _np(mapper.occupancy_probability).reshape(-1)
        indices = np.flatnonzero(observed)
        if len(indices):
            distances = np.linalg.norm(centers[indices] - _np(env._agent_pos[agent]), axis=1)
            indices = indices[np.lexsort((indices, distances, -last[indices]))][:self.map_token_count]
        maps = np.zeros((self.map_token_count, 7), np.float32)
        mask = np.zeros(self.map_token_count, bool)
        for n, index in enumerate(indices):
            maps[n] = np.r_[centers[index] / self.bounds, probability[index], 1.,
                              counts[index] / (counts[index] + 1.), max(0, self.step - last[index]) / env.max_steps]
            mask[n] = True
        dynamics = np.zeros((5, 8), np.float32)
        dmask = np.zeros(5, bool)
        dynamics[agent] = np.r_[_np(env._agent_pos[agent]) / self.bounds, _np(env._agent_vel[agent]) / 2.5, 0., 1.]
        dmask[agent] = True
        for peer, (sample, pos, velocity) in local.peers.items():
            age = max(0, self.step - sample)
            dynamics[peer] = np.r_[pos / self.bounds, velocity / 2.5, 0., np.exp(-age / 30.)]
            dmask[peer] = True
        if local.target is not None:
            target = self.predict_local_target(agent)
            dynamics[4] = np.r_[target.position / self.bounds, target.velocity / 2.5, 1., np.exp(-max(0, self.step - local.target.sample_step) / 30.)]
            dmask[4] = True
        query = local.queries.get(receiver)
        return dict(map_tokens=maps, dynamic_tokens=dynamics, receiver_query=np.zeros(9, np.float32) if query is None else query[1].copy(),
                    query_available=bool(query is not None), map_mask=mask, dynamic_mask=dmask)

    def local_geometry(self, agent):
        inputs = self.tokenize(agent, (int(agent) + 1) % 4)
        mask = inputs["map_mask"]
        return inputs["map_tokens"][mask].mean(0) if mask.any() else np.zeros(7, np.float32)

    def route_features(self, agent, *, horizon=None, candidate_count=None, radius=None):
        agent, env = int(agent), self.physical
        local = self.locals[agent]
        horizon = self.horizon if horizon is None else int(horizon)
        count = self.candidate_count if candidate_count is None else int(candidate_count)
        if min(horizon, count) <= 0:
            raise ValueError("positive route horizon/count required")
        origin, velocity = _np(env._agent_pos[agent]), _np(env._agent_vel[agent])
        travel_radius = max(1., float(env._v_xy_max[agent]) * env.dt * horizon) if radius is None else float(radius)
        goals = [local.goal.copy()]
        for j in range(1, count):
            angle = (j - 1) * 2 * np.pi / max(1, count - 1) + agent * .7
            goals.append(np.clip(origin + np.array([np.cos(angle), np.sin(angle), 0.]) * travel_radius, 0., self.bounds))
        positions = np.repeat(origin[None, None], count * (horizon + 1), axis=0).reshape(count, horizon + 1, 3)
        valid = np.zeros(count, dtype=int)
        features = np.zeros((count, 5), np.float32)
        contract = self.base.get_public_control_contract()
        # Original nominal_step is search-prior specific; executor uses the
        # same equations with its own original public prior strength.
        contract["prior_strength_search"] = env.prior_strength_executor if agent == 3 else env.prior_strength_search
        spec = self.base.get_public_vehicle_specs()[agent]
        for c, goal in enumerate(goals):
            path = local.mapper.grid_astar_path(origin, goal, role="executor" if agent == 3 else "searcher")
            points = np.asarray(path.get("points", []), float).reshape(-1, 3)
            features[c, :3] = (goal - origin) / self.bounds
            features[c, 3] = min(10., float(path.get("cost", 10 * np.linalg.norm(self.bounds))) / np.linalg.norm(self.bounds)) if path["reachable"] else 10.
            features[c, 4] = float(path["reachable"])
            if not path["reachable"]:
                continue
            p, v, index = origin.copy(), velocity.copy(), min(1, len(points))
            for h in range(horizon):
                target = points[index] if index < len(points) else goal
                end, new_velocity = nominal_step(p, v, target, spec, contract)
                if not local.mapper.segment_is_free(p, end):
                    positions[c, h + 1:] = p
                    break
                p, v = end, new_velocity
                positions[c, h + 1] = p
                valid[c] = h + 1
                if index < len(points) and np.linalg.norm(p - points[index]) < env.path_subgoal_radius:
                    index += 1
        return dict(route_features=features, route_positions=positions, route_valid_steps=valid,
                    route_origin="receiver_local_map_forecast_v1", route_goals=np.asarray(goals))

    def install_decoded_state(self, receiver, decoded=None, *, state_mean=None, state_logvar=None,
                              route_logits=None, route_data=None, **metadata):
        receiver = int(receiver)
        local = self.locals[receiver]
        decoded = {} if decoded is None else decoded
        if state_mean is None:
            state_mean = decoded.get("state_mean")
        accepted = False
        # Neural hypotheses never create a discovery, handoff, known mask, or
        # authoritative occupancy observation. Legal target evidence is required.
        has_dynamic = bool(np.asarray(_np(decoded.get("has_dynamic", False))).all())
        if local.target is not None and state_mean is not None and has_dynamic:
            values = _np(state_mean).reshape(-1)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError("state_mean must be finite normalized target pos3/vel3")
            local.soft_state = np.r_[np.clip(values[:3], 0., 1.) * self.bounds,
                                     np.clip(values[3:], -1., 1.) * 2.5]
            local.soft_step = self.step
            accepted = True
        if route_logits is not None:
            routes = self.route_features(receiver) if route_data is None else route_data
            logits = _np(route_logits).reshape(-1)
            features = np.asarray(routes["route_features"])
            if len(logits) != len(features) or not np.isfinite(logits).all():
                raise ValueError("route logits must align with local candidates")
            allowed = (features[:, 4] > .5) & (np.asarray(routes["route_valid_steps"]) > 0)
            if allowed.any() and local.target is None and receiver < 3:
                best = int(np.argmax(np.where(allowed, logits, -np.inf)))
                # A goal is revalidated against current local geometry below.
                goals = routes.get("route_goals")
                if goals is not None:
                    goal = np.asarray(goals[best], float)
                else:
                    goal = np.asarray(routes["route_positions"])[best, int(routes["route_valid_steps"][best])]
                check = local.mapper.grid_astar_path(_np(self.physical._agent_pos[receiver]), goal, role="searcher")
                if check["reachable"]:
                    local.goal = goal.copy()
                    local.command = None
                    local.soft_route = best
                    local.last_plan = -100000
                    accepted = True
        self._update_navigation()
        self.observations = _observations(self.physical._get_obs())
        return {"installed_soft_context": accepted, "target_known": local.target is not None,
                "executor_assigned": bool(self.physical.executor_target_assigned)}

    def action_enabled(self):
        return np.array([local.target is None if i < 3 else not self.physical.mission_complete
                         for i, local in enumerate(self.locals)], dtype=bool)

    def advance(self, actions=None, *, model=None, explore=True):
        if self.terminal:
            raise RuntimeError("cannot advance a terminated task")
        self.begin_step()
        before = self.step
        observations = [row.copy() for row in self.observations]
        if actions is None:
            if model is None:
                raise ValueError("actions or the inherited MADDPG actor is required")
            actions = torch.cat(model.step(observations, explore=explore), dim=0)
        proposed = _np(actions).astype(np.float32)
        if proposed.shape != (4, 3) or not np.isfinite(proposed).all():
            raise ValueError("actions must be finite 4 x 3 residual accelerations")
        enabled = self.action_enabled()
        applied = np.clip(proposed, -1., 1.) * enabled[:, None]
        step_obs, rewards, dones = self.env.step(torch.tensor(applied, dtype=torch.float32))
        self.observations = _observations(step_obs)
        done_values = [bool(x) for x in dones]
        if self.step != before + 1 or self.terminal != all(done_values):
            raise RuntimeError("inherited physical clock/done contract violated")
        return dict(t=before, observations=observations, actions=applied.copy(), proposed_actions=proposed,
            rewards=_np(rewards), dones=done_values, next_observations=[row.copy() for row in self.observations],
            action_enabled=enabled, next_action_enabled=np.zeros(4, bool) if self.terminal else self.action_enabled(),
            **deepcopy(self.env.reward_accounting.last))

    def teacher_truth(self):
        """CTDE label data only; never consumed by deployment feature methods."""
        env = self.physical
        return dict(step=self.step, target_position=env.target_state.position.copy(),
            target_velocity=env.target_state.velocity.copy(), agent_positions=_np(env._agent_pos),
            agent_velocities=_np(env._agent_vel), task_found=bool(env.task_found),
            executor_known=bool(env.executor_target_assigned), bounds=self.bounds.copy())

    def summary(self):
        row = self.env.finalize_episode()
        scenario = self.scenario or {}
        row.update(scenario_id=scenario.get("scenario_id"), scenario_seed=scenario.get("scenario_seed"),
            actual_length=self.step, found_step=self.physical.found_step,
            handoff_event_step=self.physical.handoff_payload_delivery_step,
            last_target_received_step=self.physical.executor_received_target_step if self.physical.executor_target_assigned else None,
            capture_contact_steps=int(self.physical.capture_contact_step_count),
            remaining_task_steps=int(self.config["max_steps"]) - self.step,
            channel=self.channel.summary(), communication_protocol="ch5_all_channel_tdma_v1",
            navigation_controller=self.navigation,
            source_reward_adapter="Phase1CV2TrainingEnv+PRRACTrainingEnv", local_map_count=4,
            task_deadline_steps=int(self.physical.max_steps))
        row["bootstrap_endpoint_deferrals"] = deepcopy(getattr(self.physical.map_module,
            "ch5_bootstrap_endpoint_deferrals", []))
        return row

    def close(self):
        self.env.close()
