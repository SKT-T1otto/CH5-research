"""Register ordinary task geometry; never run policies or select by outcomes."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.scenarios import validate_manifest
from .config import GROUPS, ROOT, digest, file_sha, local_path, read_json


def generate_scene(job):
    group, seed, index = job
    import torch
    from core.scenarios.ch3_generator_impl import build_scenario_manifests
    torch.set_num_threads(1)
    generated = build_scenario_manifests(count=1, generator_seed=seed,
        split="train" if group == "train" else "validation", profiles=["M20_MOVING_UNKNOWN_MULTI"])
    scene = deepcopy(generated["M20_MOVING_UNKNOWN_MULTI"]["scenarios"][0])
    role = "train" if group == "train" else "eval"
    scene.update(scenario_id=f"ch5_maddpg_{group}_{seed}", scenario_role=role,
                 scenario_split=role, pair_group_id=f"ch5_maddpg_physical_{seed}")
    return dict(index=index, physical_seed=seed, channel_seed=seed+7919,
                physical_fingerprint=scenario_fingerprint(scene), scene=scene)


def load_exclusions(config):
    manifest = read_json(local_path(config["exclusion_manifest"]))
    if manifest.get("schema") != "auv.ch5.maddpg.exclusions.v1":
        raise ValueError("invalid history exclusion schema")
    if set(manifest["cohorts"]) != {"historical_30", "development_24"}:
        raise ValueError("both previously inspected cohorts must be excluded")
    fps, seeds = set(), set()
    counts = {}
    for name, expected in (("historical_30", 30), ("development_24", 24)):
        cohort = manifest["cohorts"][name]
        path = local_path(cohort["archived_source"])
        if file_sha(path) != cohort["source_sha256"]:
            raise ValueError("historical source bytes differ: " + name)
        original = read_json(path)
        if len(original["scenes"]) != expected or len(cohort["entries"]) != expected:
            raise ValueError("historical cohort count differs")
        physical_seeds = (dict(original["seeds"]["typed_mapping"]) if name == "historical_30"
                          else dict(enumerate(original["physical_seeds"])))
        for i, entry in enumerate(cohort["entries"]):
            scene = original["scenes"][i]
            fp = scenario_fingerprint(scene)
            if (entry["index"] != i or entry["physical_fingerprint"] != fp
                    or entry["scenario_seed"] != scene["scenario_seed"]
                    or entry["scenario_id"] != scene["scenario_id"]
                    or entry["physical_seed"] != physical_seeds[i]):
                raise ValueError("historical physical fingerprint differs")
            fps.add(fp)
            seeds.add(scene["scenario_seed"])
            seeds.add(entry["physical_seed"])
        counts[name] = expected
    if len(fps) != 54:
        raise ValueError("expected 54 distinct previously inspected physical scenes")
    return dict(fingerprints=fps, seeds=seeds, counts=counts)


def validate_registry(config, registry, config_path):
    if (registry.get("schema") != "auv.ch5.nominal_maddpg.scenes.v1"
            or registry.get("config_sha256") != file_sha(config_path)
            or registry.get("exclusion_sha256") != file_sha(local_path(config["exclusion_manifest"]))
            or registry.get("task_rollouts") != 0 or registry.get("outcome_filtering") is not False
            or set(registry.get("groups", {})) != set(GROUPS)):
        raise ValueError("registry identity or scope differs")
    history = load_exclusions(config)
    used_fp, used_seed, used_ids = set(history["fingerprints"]), set(history["seeds"]), set()
    report = {}
    for name in GROUPS:
        spec, entries = config["scene_groups"][name], registry["groups"][name]
        if len(entries) != spec["count"]:
            raise ValueError("scene count differs: " + name)
        for index, entry in enumerate(entries):
            scene = entry["scene"]
            seed = spec["start_seed"] + index
            fp = scenario_fingerprint(scene)
            if (entry["index"] != index or entry["physical_seed"] != seed
                    or entry["channel_seed"] != seed+7919 or scene["scenario_seed"] != seed
                    or entry["physical_fingerprint"] != fp
                    or scene["scenario_id"] != f"ch5_maddpg_{name}_{seed}"):
                raise ValueError("scene identity or fingerprint differs")
            if (scene["planner_grid_size"] != [10, 10, 8]
                    or scene["target_position"] != scene["target_initial_position"]
                    or scene["obstacle_count"] != len(scene["obstacles"])
                    or not 2 <= scene["obstacle_count"] <= 4
                    or scene["obstacle_knowledge_mode"] != "online_unknown"
                    or scene["target_motion_mode"] != "constant_velocity_reflect_v1"
                    or scene["flow_phase_x"] != 0 or scene["flow_phase_y"] != 0):
                raise ValueError("ordinary native task geometry contract differs")
            if fp in used_fp or seed in used_seed or scene["scenario_id"] in used_ids:
                raise ValueError("duplicate historical or cross-group physical scene/seed/id")
            used_fp.add(fp); used_seed.add(seed); used_ids.add(scene["scenario_id"])
        validate_manifest(dict(scenarios=[e["scene"] for e in entries]), config["task"],
                          evaluation=name != "train")
        report[name] = dict(count=len(entries), fingerprints_sha256=digest([e["physical_fingerprint"] for e in entries]))
    return dict(groups=report, total_scenes=sum(v["count"] for v in report.values()),
                history_counts=history["counts"], history_overlap=0, cross_group_overlap=0)


def build_registry(config, config_path, workers=4):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("geometry generation permits 1 to 4 CPU workers")
    load_exclusions(config)  # Fail before expensive generation if evidence is absent.
    jobs = [(name, config["scene_groups"][name]["start_seed"]+i, i)
            for name in GROUPS for i in range(config["scene_groups"][name]["count"])]
    groups = {name: [] for name in GROUPS}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job, entry in zip(jobs, pool.map(generate_scene, jobs, chunksize=4)):
            groups[job[0]].append(entry)
            count = len(groups[job[0]])
            if count % 50 == 0 or count == config["scene_groups"][job[0]]["count"]:
                print(f"registered geometry: {job[0]} {count}/{config['scene_groups'][job[0]]['count']}", flush=True)
    registry = dict(schema="auv.ch5.nominal_maddpg.scenes.v1", config_sha256=file_sha(config_path),
        exclusion_sha256=file_sha(local_path(config["exclusion_manifest"])),
        generator="one native build_scenario_manifests(count=1) call per registered seed",
        generator_truth_use="existing feasibility checks and target-only trajectory; never policy input",
        task_rollouts=0, outcome_filtering=False, groups=groups)
    validate_registry(config, registry, config_path)
    return registry


def verify_regeneration(config, registry, workers=4):
    # Fixed indices chosen before any task outcomes; one fresh worker pool.
    entries = [(name, registry["groups"][name][i]) for name in GROUPS
               for i in sorted({0, len(registry["groups"][name])//2, len(registry["groups"][name])-1})]
    jobs = [(name, entry["physical_seed"], entry["index"]) for name, entry in entries]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (name, expected), actual in zip(entries, pool.map(generate_scene, jobs)):
            if digest(actual) != digest(expected):
                raise ValueError(f"independent regeneration differs: {name}/{expected['index']}")
    return dict(passed=True, count=len(entries), indices="first/middle/last per group",
                scene_ids=[e["scene"]["scenario_id"] for _, e in entries])
