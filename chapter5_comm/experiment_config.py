"""Versioned, explicit research budgets; does not execute experiments."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math

from .pipeline import PROJECT, read_config
from .training import NeuralConfig, _source_identity

SCHEMA = "ch5.linux.experiment.v1"
ARMS = ("full", "fixed_scheduler", "raw_only", "no_query", "no_transport",
        "no_geometry", "no_time_loss", "no_route_loss", "d2_zero", "d2_b2")
D2_ARMS = frozenset(("d2_zero", "d2_b2"))
RAW_ARMS = D2_ARMS | {"raw_only"}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_identity():
    # Training checkpoints bind to runtime source plus all execution scripts,
    # dependency pins and experiment configurations, independent of root path.
    identity = _source_identity()
    for folder in ("scripts", "configs/chapter5"):
        for p in sorted((PROJECT / folder).rglob("*")):
            if p.is_file() and (folder == "scripts" or p.suffix == ".json"):
                identity[p.relative_to(PROJECT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("cli.py", "requirements.txt", ".gitattributes", "configs/allocation.json"):
        identity[name] = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
    return identity


def read_experiment(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT / path
    c = json.loads(path.read_text(encoding="utf-8"))
    if c.get("config_name") != path.name:
        raise ValueError("config_name must identify the preset filename")
    if c.get("schema") != SCHEMA or c.get("kind") not in ("check", "baseline", "pilot", "formal"):
        raise ValueError("requires a versioned CH5 experiment configuration")
    _, base = read_config("configs/chapter5/short_pipeline_v1.json")
    if c["parent_config_sha256"] != hashlib.sha256((PROJECT / "configs/b2.json").read_bytes()).hexdigest():
        raise ValueError("parent task identity mismatch")
    if not c["seeds"] or len(c["seeds"]) != len(set(c["seeds"])) or any(type(s) is not int or not 0 <= s < 1000000 for s in c["seeds"]):
        raise ValueError("training seeds must be unique nonnegative integers below one million")
    if not c["arms"] or len(set(c["arms"])) != len(c["arms"]) or set(c["arms"]) - set(ARMS):
        raise ValueError("unknown or duplicate independently trained arm")
    if set(c["arms"]) & D2_ARMS and c.get("navigation") != "causal_d2_v1":
        raise ValueError("D2 control arms require causal D2 navigation")
    if c.get("navigation", "local_waypoints_v1") not in ("local_waypoints_v1", "causal_d2_v1"):
        raise ValueError("unknown experiment navigation")
    for key in ("collection_episodes", "auxiliary_updates", "auxiliary_batch_size", "task_episodes", "communication_episodes", "validation_episodes", "evaluation_episodes", "checkpoint_interval", "validation_interval", "replay_capacity", "task_batch_size", "task_warmup", "task_update_frequency", "task_updates_per_trigger", "label_horizon", "auxiliary_checkpoint_interval"):
        if type(c[key]) is not int or not 1 <= c[key] <= 1000000:
            raise ValueError("invalid positive budget: " + key)
    if type(c["rollout_limit_steps"]) is not int or not 1 <= c["rollout_limit_steps"] <= 400:
        raise ValueError("invalid rollout limit")
    if c["kind"] != "check" and c["rollout_limit_steps"] != 400:
        raise ValueError("baseline/pilot/formal must evaluate the unchanged full 400-step task")
    if c["task_batch_size"] > c["task_warmup"] or c["task_warmup"] > c["replay_capacity"]:
        raise ValueError("requires batch_size <= warmup <= replay capacity")
    max_task = c["task_episodes"] * c["rollout_limit_steps"] // c["task_update_frequency"] * c["task_updates_per_trigger"]
    if max_task > 1000000:
        raise ValueError("maximum task updates exceed the declared one-million update guard")
    if c["kind"] == "check" and (c["collection_episodes"] + c["task_episodes"] + c["communication_episodes"] + c["validation_episodes"] + c["evaluation_episodes"] > 8 or c["rollout_limit_steps"] > 32 or max_task > 4 or c["auxiliary_updates"] > 4):
        raise ValueError("local check must stay within eight short episodes and four updates")
    if len(c.get("collection_mechanism_seeds", [])) > c["collection_episodes"]:
        raise ValueError("too many mechanism collection cases")
    if any(type(s) is not int or s < 0 for s in c["collection_mechanism_seeds"]):
        raise ValueError("mechanism scenario seeds must be nonnegative integers")
    if any(type(c[k]) is not int or c[k] < 0 for k in ("minimum_state_labels", "minimum_positive_routes")):
        raise ValueError("label coverage minima cannot be negative")
    if type(c["save_episode_details"]) is not bool:
        raise ValueError("save_episode_details must be boolean")
    for key, expected in dict(map_dim=7, dynamic_dim=8, query_dim=9, geometry_dim=7,
                              state_dim=6, route_dim=5, context_dim=7, header_bytes=16).items():
        if c["neural"].get(key) != expected:
            raise ValueError("incompatible runtime feature/header contract: " + key)
    NeuralConfig.from_mapping(c["neural"])
    required = {"nominal", "loss_high", "delay_high", "burst_long", "budget_low", "combined_unseen", "reliable_reference"}
    if c["kind"] in ("pilot", "formal") and not required.issubset(c["evaluation_channels"]):
        raise ValueError("missing preregistered fault conditions")
    if c["kind"] == "baseline":
        budget = plan_summary(c)
        if (set(c["arms"]) != D2_ARMS or budget["independent_jobs"] != 2
                or budget["episodes_per_job_upper_bound"] > 8 or max_task > 4
                or set(c["evaluation_channels"]) != {"reliable_reference", "combined_unseen"}):
            raise ValueError("baseline requires one matched D2 pair, two declared channels, and a bounded run")
    from .protocol import TDMAChannel
    TDMAChannel(**c["channel"])
    for name, patch in c["evaluation_channels"].items():
        if not name.replace("_", "").isalnum():
            raise ValueError("unsafe channel label")
        TDMAChannel(**dict(c["channel"], **patch))
    return c, base


def arm_settings(c, arm):
    if arm not in c["arms"]:
        raise ValueError("arm is absent from the frozen plan")
    neural = deepcopy(c["neural"])
    neural.update(query_conditioning=arm != "no_query", temporal_transport=arm != "no_transport",
                  geometry_conditioning=arm != "no_geometry")
    losses = dict(route_weight=0. if arm == "no_route_loss" else 1.,
                  prediction_weight=1., time_weight=0. if arm in ("no_time_loss", "no_transport") else .1)
    mode = "neural_off" if arm in RAW_ARMS else "fixed" if arm == "fixed_scheduler" else "learned"
    return neural, losses, mode


def scene_seed(seed, split, index):
    # Fixed, disjoint numerical namespaces. Validation/test do not depend on
    # training seed, enabling common physical scenarios across trained seeds.
    origins = dict(collection=10**15 + seed * 1000000, task=2 * 10**15 + seed * 1000000,
                   communication=4 * 10**15 + seed * 1000000,
                   validation=6 * 10**15, test=8 * 10**15)
    return origins[split] + int(index)


def plan_summary(c):
    jobs = len(c["seeds"]) * len(c["arms"])
    validations = math.ceil(c["communication_episodes"] / c["validation_interval"])
    train = c["collection_episodes"] + c["task_episodes"] + c["communication_episodes"]
    evaluate = validations * c["validation_episodes"] + len(c["evaluation_channels"]) * c["evaluation_episodes"]
    return dict(kind=c["kind"], independent_jobs=jobs, seeds=c["seeds"], arms=c["arms"],
                navigation=c.get("navigation", "local_waypoints_v1"),
                task_horizon=400, rollout_limit=c["rollout_limit_steps"],
                maximum_physical_steps=jobs * (train + evaluate) * c["rollout_limit_steps"],
                episodes_per_job_upper_bound=train + evaluate,
                conditions=list(c["evaluation_channels"]),
                formal_performance_verified=False,
                execution="manual --execute required; baseline/pilot/formal execution requires Linux")
