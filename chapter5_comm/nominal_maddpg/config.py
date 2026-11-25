"""Register the new learning identity without changing the inherited task."""
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path

from . import SCHEMA

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/chapter5/nominal_maddpg.json"
GROUPS = ("train", "validation", "baseline_admission", "link_fault_development")
BASE_SHA = "41e46d4d9ae35f7808d3bb82ae859f83117580013f8c2e69d93f4b83203dce48"
ENV_SHA = "d5881ea13b6f856e55f06051287761e9eca7c65739e03fb9232f6cadf5c1cb52"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON value: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject,
                      object_pairs_hook=unique)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def local_path(relative, root=ROOT):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("path must stay inside the Chapter 5 project")
    return path


def default_config():
    return dict(
        schema=SCHEMA, date="2026-11-25",
        task_reference=dict(path="configs/b2.json", sha256=BASE_SHA),
        environment_reference=dict(path="configs/environment.json", sha256=ENV_SHA),
        task=dict(profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400, dt=0.2,
                  agents=4, observation_dim=28, action_dim=3, critic_input_dim=124,
                  residual_scale_search=0.2, residual_scale_executor=0.15,
                  reward_objective="team_mean_v1", gamma=0.95,
                  task_protocol="collision_terminal_v1", failure_penalty_steps=100),
        algorithm=dict(name="MADDPG", revision="ch5.maddpg.single_critic.v1",
            actors=4, critics=4, critics_per_vehicle=1, decentralized_actor_input_dim=28,
            centralized_critic_input_dim=124, actor_hidden=[128, 128], critic_hidden=[256, 256],
            input_normalization="LayerNorm", actor_output="tanh", critic_loss="mse",
            target_critic_reduction="single", target_value_clip=None, target_action_noise=0.0,
            actor_other_actions="replay_applied_actions", actor_regularization=0.0,
            actor_sample_mask="execution_gate_open", replay_sampling="uniform",
            replay_reward_normalization=False, policy_sharing=False,
            actor_truth_parameters_visible=False, actor_privileged_state_visible=False),
        learning=dict(batch_size=128, replay_capacity=500000, warmup_steps=256,
            update_frequency=4, updates_per_train=2, gamma=0.95, tau=0.005,
            actor_lr=0.001, critic_lr=0.0005, optimizer="Adam", learner_device="cpu", cpu_threads=1,
            exploration=dict(type="OU", initial_sigma=0.2, minimum_sigma=0.02,
                             decay_per_environment_step=0.00002, evaluation_sigma=0.0),
            pilot_episodes=64, budget_observations=[200, 500, 1000], maximum_episodes_per_seed=1000,
            selection="minimum validation mean penalized completion steps, earlier checkpoint on ties"),
        runtime=dict(entry="chapter5_comm.runtime.CH5Runtime", navigation="causal_d2_v1",
            map_transport="own_history_v1", history_rule="native", history_tie_seed=11,
            early_brake=False, scan_first=False, command_override=None,
            horizon=8, candidate_count=4, map_token_count=32,
            plant="native_ch3", neural_communication=False, checkpoint=None),
        channel=dict(name="reliable_reference", capacity_bits=65536, refill_bits=16384,
            rate_bps=1000000, sound_speed=1500, dt=0.2, loss_probability=0.0,
            delay_steps=0, jitter_steps=0, burst_start_probability=0.0,
            burst_length=4, burst_loss_probability=0.8),
        training_seeds=[2729, 3733, 4751],
        seed_policy=dict(physical="scenario_seed", channel="scenario_seed + 7919",
            initialization="training_seed", sampling="independent stream per training_seed",
            exploration="independent stream per training_seed",
            evaluation="fixed physical and channel seeds; exploration disabled",
            shared_training_scene_pool=True),
        scene_groups={name:dict(start_seed=start, count=count,
            role="train" if name == "train" else "eval", purpose=purpose)
            for name, start, count, purpose in (
                ("train", 7100000, 1000, "shared training geometry; independent initialization and sampling"),
                ("validation", 7200000, 32, "development configuration and checkpoint selection"),
                ("baseline_admission", 7300000, 100, "one-time frozen task policy admission"),
                ("link_fault_development", 7400000, 24, "communication degradation after task admission"))},
        scene_manifest="configs/chapter5/nominal_maddpg_scenes.json",
        exclusion_manifest="provenance/ch5_standard_maddpg_20261125/history_exclusion.json",
        formal_communication_test=dict(status="NOT_REGISTERED", scenes=0),
        stage_status=dict(completed_steps=[1], training_enabled=False, rollouts_enabled=False,
            optimizer_updates_allowed=0, networks_implemented=False, runtime_validated=False,
            learner_update_validated=False, nominal_performance_verified=False))


def validate(config, root=ROOT):
    c = deepcopy(config)
    # This is an immutable first registration, not a permissive training preset.
    # Canonical serialization distinguishes booleans from integers and rejects NaN.
    if digest(c) != digest(default_config()):
        raise ValueError("registered MADDPG/task/channel/scene contract differs; register a new revision")
    for key in ("task_reference", "environment_reference"):
        ref = c[key]
        if file_sha(local_path(ref["path"], root)) != ref["sha256"]:
            raise ValueError("task dependency bytes differ: " + ref["path"])
    from chapter3_bser.learning.config import validate as validate_task
    validate_task(read_json(local_path(c["task_reference"]["path"], root)))
    return c


def load(path=DEFAULT_CONFIG):
    return validate(read_json(path))


def task_config(config):
    validate(config)
    return read_json(local_path(config["task_reference"]["path"]))


def _plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_default": str(value)}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def resolve(config):
    """Expand native constructor defaults. No mission reset, action or learner."""
    import random
    import numpy as np
    import torch
    from core.config.active_environment import build_ch3_config
    from core.env.mission_env import environment_kwargs_from_config
    from core.env.task_protocol import PROTOCOL_FIELDS
    from core.env.uav_env import UAVEnv
    from chapter5_comm.protocol import HEADER, TDMAChannel
    from chapter5_comm.history_evidence import SIZE, PREFIX, CELL

    native = task_config(config)
    settings = build_ch3_config(native["base_candidate"], native["profile"])
    settings.update({k: native[k] for k in (*PROTOCOL_FIELDS, "collision_terminal_reward",
                                          "waypoint_reward_timing") if k in native})
    kwargs = environment_kwargs_from_config(settings, device="cpu", max_steps=native["max_steps"],
                                           return_numpy=False)
    defaults = {}
    for constructor in (UAVEnv.__mro__[1].__init__, UAVEnv.__init__):
        for key, parameter in inspect.signature(constructor).parameters.items():
            if parameter.default is not inspect.Parameter.empty:
                defaults[key] = parameter.default
    defaults.update(kwargs)

    class ConstructorInventory(UAVEnv):
        def reset(self, scenario=None):
            # Native constructors call reset. Suppress it for parameter inspection.
            return None

    saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        env = ConstructorInventory(**kwargs)
        scalar_attributes = {k: _plain(v) for k, v in vars(env).items()
                             if isinstance(v, (str, bool, int, float)) or v is None}
        physics = dict(agent_specs=deepcopy(env.agent_specs), flow_gain=env.flow_gain,
            flow_gain_interpretation="additive horizontal acceleration field, not water speed",
            flow_z_gain=env.flow_z_gain, space_size=_plain(env.space_size), dtype=str(env.dtype),
            dynamics_source="core/env/uav_env.py:_apply_agent_dynamics",
            horizontal_update="v1=v+a*dt; v2=v1+(-drag_xy*v1+flow)*dt; speed_limit; p+=v2*dt",
            actuator_memory_coefficient=0.0, actuator_delay_steps=0,
            residual_scale_search=env.residual_scale_search,
            residual_scale_executor=env.residual_scale_executor)
    finally:
        random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])
    channel = {k: v for k, v in config["channel"].items() if k != "name"}
    TDMAChannel(seed=0, **channel)  # Constructor validation only; no sends/receives.
    return dict(schema="auv.ch5.nominal_maddpg.resolved.v1", native_task=native,
        environment_settings=settings, constructor_kwargs=kwargs,
        expanded_constructor_parameters=_plain(defaults), native_scalar_attributes=scalar_attributes,
        physics=physics, reward=dict(environment={k: scalar_attributes[k] for k in (
            "reward_scale", "time_penalty", "lambda_a", "lambda_da", "team_find_bonus",
            "finder_extra_bonus", "mission_complete_bonus", "early_find_bonus_gain",
            "early_success_bonus_gain", "collision_terminal_reward", "collision_penalty",
            "safe_dist", "sep_penalty_k", "coverage_reward_gain")},
            execution_wrapper=native["reward"], objective=native["reward_objective"],
            source_revision=native["source_reward_revision"], gamma=native["rl"]["gamma"]),
        runtime=config["runtime"], channel=channel,
        map_identity=dict(inherited_label=settings["map_sharing_mode"],
            effective="receiver_local_maps; own scans plus actually delivered source-replacing evidence",
            truth_access="simulator sensing/collision and offline dataset feasibility only",
            actor_access="28 local observation components only; no truth or team map"),
        wire=dict(header_bytes=HEADER.size, history_payload_bytes=SIZE,
            history_prefix_bytes=PREFIX.size, history_cell_bytes=CELL.size, history_max_cells=14,
            history_frame_bits=8*(HEADER.size+SIZE), command_frame_bits=8*(HEADER.size+12),
            accounting="actual serialized bytes including all metadata, queries and ACKs",
            earliest_delivery="at least one physical step; serialization and propagation remain active"),
        inherited_learning_fields_active=False, weights_loaded=[],
        inspection=dict(mode="native constructors with reset suppressed", mission_resets=0,
                        physics_steps=0, optimizer_updates=0))
