"""Freeze the task separately from the new single-critic learning identity."""
import copy
import hashlib
import json
from pathlib import Path

from chapter3_bser.learning.checkpoint import digest
from chapter3_bser.learning.config import validate as validate_native
from experiments.ch4_work1_recoverability.plant_adapter import RHO, WaterParameters

from . import SCHEMA


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/ch4/nominal_maddpg.json"
NATIVE_CONFIG_SHA256 = "41e46d4d9ae35f7808d3bb82ae859f83117580013f8c2e69d93f4b83203dce48"
TASK = dict(profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400, failure_penalty_steps=100,
            agents=4, role_order=["search_fast", "search_balanced", "search_precise", "executor"],
            observation_dim=28, action_dim=3, critic_input_dim=124, dt=0.2, gamma=0.95,
            residual_scale_search=0.20, residual_scale_executor=0.15,
            planner_protocol="d2_v1", reward_objective="team_mean_v1",
            task_protocol="collision_terminal_v1", collision_detection_revision="segment_closed_aabb_v1",
            terminal_reward_revision="team_failure_override_v1", collision_terminal_reward=-2.0,
            waypoint_reward_timing="arrival_before_reward_v1",
            repairs=dict(waypoint_arrival=True, inactive_actor=True, hold_residual=True))
PLANT = dict(law="linear", beta_m_per_s=0.10, drag_multipliers=[1.0]*4,
             horizontal_thrust_multipliers=[1.0]*4, lag_memory_coefficient=0.15,
             actuator_delay_steps=0, estimator_delay_steps=0, noise="N0",
             extra_vertical_bias_m_per_s2=0.0, water_field="beta * (sin(0.25*y), cos(0.22*x))")
ALGORITHM = dict(name="MADDPG", revision="ch4.maddpg.single_critic.v1",
                 actors=4, critics=4, critics_per_vehicle=1, decentralized_actor_input_dim=28,
                 centralized_critic_input_dim=124, actor_hidden=[128, 128], critic_hidden=[256, 256],
                 input_normalization="LayerNorm", actor_output="tanh", critic_loss="mse",
                 target_critic_reduction="single", target_value_clip=None, target_action_noise=0.0,
                 actor_other_actions="replay_applied_actions", actor_regularization=0.0,
                 actor_sample_mask="execution_gate_open", replay_sampling="uniform",
                 replay_reward_normalization=False, policy_sharing=False,
                 actor_truth_parameters_visible=False, actor_privileged_state_visible=False)
GROUPS = ("train", "validation", "nominal_admission", "mismatch_development", "mismatch_confirmation_reserve")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def local_path(relative, root=ROOT):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("artifact path must stay inside the Chapter 4 project")
    return path


def task_config(config, root=ROOT):
    reference = config["task_reference"]
    path = local_path(reference["path"], root)
    if reference != dict(path="configs/b2.json", sha256=NATIVE_CONFIG_SHA256) or file_sha(path) != NATIVE_CONFIG_SHA256:
        raise ValueError("frozen task reference differs")
    native = validate_native(read_json(path))
    for field in ("profile", "max_steps", "planner_protocol", "reward_objective", "task_protocol",
                  "collision_detection_revision", "terminal_reward_revision", "collision_terminal_reward",
                  "waypoint_reward_timing", "repairs", "observation_dim", "action_dim"):
        if native[field] != TASK[field]:
            raise ValueError("task reference violates contract: " + field)
    if native["rl"]["gamma"] != TASK["gamma"] or "map_experiment" in native:
        raise ValueError("task reference uses different discount or planner experiment")
    return native


def validate(config, root=ROOT):
    c = copy.deepcopy(config)
    required = {"schema", "date", "task_reference", "task", "plant", "algorithm", "learning",
                "training_seeds", "scene_groups", "scene_manifest", "nominal_references", "admission",
                "mismatch_gate", "stage_status", "module_boundaries"}
    if set(c) != required or c["schema"] != SCHEMA:
        raise ValueError("unexpected nominal MADDPG schema or fields")
    for name, expected in (("task", TASK), ("plant", PLANT), ("algorithm", ALGORITHM)):
        if digest(c[name]) != digest(expected):
            raise ValueError("frozen " + name + " contract differs")
    task_config(c, root)
    WaterParameters(beta=c["plant"]["beta_m_per_s"], drag=c["plant"]["drag_multipliers"],
                    thrust=c["plant"]["horizontal_thrust_multipliers"], law=c["plant"]["law"])
    if RHO != c["plant"]["lag_memory_coefficient"]:
        raise ValueError("plant adapter lag differs")
    expected_learning = dict(batch_size=128, replay_capacity=500000, warmup_steps=256,
        update_frequency=4, updates_per_train=2, gamma=0.95, tau=0.005,
        actor_lr=0.001, critic_lr=0.0005, optimizer="Adam", learner_device="cpu", cpu_threads=1,
        exploration=dict(type="OU", initial_sigma=0.2, minimum_sigma=0.02,
                         decay_per_environment_step=0.00002, evaluation_sigma=0.0),
        pilot_episodes=64, budget_observations=[200, 500, 1000], maximum_episodes_per_seed=1000,
        selection="minimum validation mean penalized completion steps, earlier checkpoint on ties")
    if digest(c["learning"]) != digest(expected_learning):
        raise ValueError("registered first-round learning settings differ")
    seeds = c["training_seeds"]
    if len(seeds) != 3 or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != 3:
        raise ValueError("three distinct nonnegative training seeds required")
    if set(c["scene_groups"]) != set(GROUPS):
        raise ValueError("five explicit scene purposes required")
    expected_counts = dict(zip(GROUPS, (1000, 32, 100, 24, 256)))
    used = set()
    for name in GROUPS:
        group = c["scene_groups"][name]
        if set(group) != {"start_seed", "count", "role", "purpose"}:
            raise ValueError("unrecognized scene group fields")
        first, count = group["start_seed"], group["count"]
        if type(first) is not int or type(count) is not int or first < 0 or count != expected_counts[name]:
            raise ValueError("scene seed or registered group count differs")
        block = set(range(first, first + count))
        if max(block) >= 2**31 or block & used:
            raise ValueError("overlapping or invalid scenario seed blocks")
        used.update(block)
        if group["role"] != ("train" if name == "train" else "eval") or not group["purpose"]:
            raise ValueError("scene role/purpose differs")
    if digest(c["stage_status"]) != digest(dict(completed_steps=[1, 2], training_enabled=False,
            rollouts_enabled=False, optimizer_updates_allowed=0, runtime_validated=False,
            learner_update_validated=False, nominal_performance_verified=False)):
        raise ValueError("this registration only admits steps 1 and 2")
    if c["nominal_references"] != dict(primary="relative_water_nominal", regression="native_ch3",
            native_performance_required=False):
        raise ValueError("nominal reference purposes changed")
    if c["admission"] != dict(scenes=100, mean_seed_success_at_least=0.60,
            each_seed_success_at_least=0.50, d2_paired_cost_improvement_ci_lower_above=0.0,
            each_seed_search_window_rate_at_least=0.80, each_seed_executor_window_rate_at_least=0.40,
            independent_unit="physical_scene", confidence_level=0.95, bootstrap_draws=10000,
            bootstrap_seed=6536000, seed_variability_reported_separately=True):
        raise ValueError("nominal admission criteria changed")
    if c["mismatch_gate"] != dict(minimum_mean_D_steps=5.0, ci_lower_above=0.0,
            confidence_level=0.95, bootstrap_draws=10000, bootstrap_seed=6636000,
            sample_size_rule="max(24, ceil((1.96 * development_scene_paired_sd / 2.5)^2))",
            confirmation_sample_size_locked=False, confirmation_reserve_is_candidate_only=True,
            overflow="register additional independent candidates before confirmation, or report insufficient precision",
            all_training_seeds_reported=True):
        raise ValueError("mismatch stopping rules changed")
    if c["scene_manifest"] != "configs/ch4/nominal_maddpg_scenes.json":
        raise ValueError("unexpected scene manifest path")
    boundaries = dict(task_runtime="chapter3_bser/learning/runtime.py", task_reference="configs/b2.json",
        nominal_plant="experiments/ch4_work1_recoverability/plant_adapter.py",
        network_construction="chapter4_interval_robust/nominal_maddpg/learner.py",
        scenario_registry="chapter4_interval_robust/nominal_maddpg/scenarios.py",
        source_registration="chapter4_interval_robust/nominal_maddpg/provenance.py",
        verification_entry="chapter4_interval_robust/nominal_maddpg/verify.py")
    if c["module_boundaries"] != boundaries:
        raise ValueError("module boundaries changed")
    for relative in boundaries.values():
        if not local_path(relative, root).is_file():
            raise ValueError("declared module is missing: " + relative)
    return c


def load(path=DEFAULT_CONFIG):
    return validate(read_json(path))
