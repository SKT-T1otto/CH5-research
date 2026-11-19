"""Paired D2 ZERO braking × scan-order evaluation on 30 fixed M20 scenes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.learning.scenarios import validate_manifest
from tools.executor_trace import digest

from .d2_diagnostics import PRESET, _initial_physics
from .experiment_config import read_experiment, scene_seed, source_identity
from .experiment_runner import _git_commit, environment
from .experiment_state import atomic_json, exclusive_job
from .pipeline import PROJECT, new_output, scenarios
from .runtime import CH5Runtime


SCHEMA = "ch5.d2.factorial.v1"
CASE_COUNT = 30
ARMS = {
    "baseline": (False, False),
    "brake_only": (True, False),
    "scan_only": (False, True),
    "both": (True, True),
}


def _scenes(manifest, base, seed, *, smoke):
    count = 1 if smoke and manifest is None else CASE_COUNT
    if manifest is None:
        data = {"scenarios": scenarios(count, scene_seed(seed, "test", 0), evaluation=True)}
        origin = {"kind": "generated_d2_short_test_namespace"}
    else:
        content = manifest.read_bytes()
        data = json.loads(content)
        if isinstance(data, list):
            data = {"scenarios": data}
        origin = {"kind": "supplied_manifest", "sha256": hashlib.sha256(content).hexdigest()}
    scenes = data.get("scenarios")
    if not isinstance(scenes, list):
        raise ValueError("a CH3 M20 scenario list is required")
    if len(scenes) != count:
        raise ValueError(f"expected exactly {count} unique M20 evaluation scenes")
    aliases = {"validation": "eval", "smoke_validation": "smoke_eval", "test": "eval"}
    normalized = [dict(s, **{k: aliases.get(s.get(k), s.get(k)) for k in ("scenario_role", "scenario_split")})
                  for s in scenes]
    validate_manifest(dict(data, scenarios=normalized), base, evaluation=True)
    return scenes, origin


def _evaluate(base, scene, seed, channel, *, steps, brake, scan):
    runtime = CH5Runtime(base, scene, seed=seed, channel_config=channel, navigation="causal_d2_v1",
                         early_brake=brake, scan_first=scan)
    try:
        initial = digest(_initial_physics(runtime.physical))
        policy = ZeroResidualPolicy()
        while runtime.step < steps and not runtime.terminal:
            runtime.advance(model=policy, explore=False)
        summary = runtime.summary()
        first_scan = [None] * 4
        for event in runtime.channel.events:
            receiver = event["receiver"]
            if event["kind"] == "SCAN" and first_scan[receiver] is None:
                first_scan[receiver] = event["delivery_step"]
        return dict(
            initial_physics_sha256=initial, steps=runtime.step, prefix_cutoff=not runtime.terminal,
            success=bool(summary["success"]) if runtime.terminal else None, found=bool(summary["found"]),
            collision=bool(summary["collision_episode"]), collision_phase=summary["first_collision_phase"],
            search_collision=summary["first_collision_phase"] == "Search",
            collision_step=summary["first_collision_step"], collision_records=summary["collision_records"],
            timed_out=summary["termination_reason"] == "timeout" if runtime.terminal else None,
            handoff_step=summary["handoff_event_step"], capture_contact_steps=summary["capture_contact_steps"],
            channel=summary["channel"], first_scan_receive_steps=first_scan, braking_events=runtime.braking_events,
        )
    finally:
        runtime.close()


def _report(rows, conditions, *, observation_limit, expected):
    counts, paired = {}, {}
    metrics = ("collision", "search_collision", "found", "success", "timed_out")
    for condition in conditions:
        groups = {arm: {x["case"]: x for x in rows if x["condition"] == condition and x["arm"] == arm}
                  for arm in ARMS}
        counts[condition] = {
            arm: dict(cases=len(group), prefix_cutoffs=sum(x["prefix_cutoff"] for x in group.values()),
                      sent_bits=sum(x["channel"]["sent_bits"] for x in group.values()),
                      brake_actions=sum(len(x["braking_events"]) for x in group.values()),
                      **{metric: sum(x[metric] is True for x in group.values()) for metric in metrics})
            for arm, group in groups.items()
        }
        paired[condition] = {}
        for left, right in (("baseline", "brake_only"), ("baseline", "scan_only"),
                            ("scan_only", "both"), ("brake_only", "both"), ("baseline", "both")):
            matched = sorted(groups[left].keys() & groups[right].keys())
            comparison = {}
            for metric in metrics:
                values = [(groups[left][i][metric], groups[right][i][metric]) for i in matched]
                comparison[metric] = dict(
                    matched=len(values),
                    observed=sum(a is not None and b is not None for a, b in values),
                    left_only=sum(a is True and b is False for a, b in values),
                    right_only=sum(a is False and b is True for a, b in values),
                )
            paired[condition][f"{left}_vs_{right}"] = comparison
    return dict(schema=SCHEMA, complete=len(rows) == expected, rows=len(rows), expected_rows=expected,
                observation_limit_steps=observation_limit, counts=counts, paired=paired)


def execute(root, manifest, *, smoke, resume):
    config, base = read_experiment(PRESET)
    scenes, origin = _scenes(manifest, base, config["seeds"][0], smoke=smoke)
    scenes = scenes[:1] if smoke else scenes
    steps = 25 if smoke else base["max_steps"]
    channels = {key: dict(config["channel"], **patch) for key, patch in config["evaluation_channels"].items()}
    source = source_identity()
    for name in ("configs/b2.json", "configs/environment.json", "configs/online.json",
                 "tools/executor_trace.py", "project_paths.py"):
        source[name] = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
    cases = [dict(case=i, scenario_id=scene["scenario_id"], scenario_seed=scene["scenario_seed"],
                  scenario_sha256=scenario_fingerprint(scene), physical_seed=scene["scenario_seed"] % (2**31 - 1))
             for i, scene in enumerate(scenes)]
    plan = dict(schema=SCHEMA, preset=PRESET, source=source, git_commit=_git_commit(),
                origin=origin, cases=cases, channels=channels, arms={k: list(v) for k, v in ARMS.items()},
                task_deadline_steps=base["max_steps"], observation_limit_steps=steps, training_updates=0)
    root = root.resolve()
    allowed = (PROJECT / "outputs/chapter5").resolve()
    if root == allowed or not root.is_relative_to(allowed):
        raise ValueError("run directory must be under outputs/chapter5")
    if not root.exists():
        root = new_output(root)
    elif not resume:
        raise FileExistsError(f"{root} already exists, use --resume for the same experiment")
    with exclusive_job(root):
        plan_path = root / "plan.json"
        if not plan_path.exists():
            atomic_json(plan_path, plan)
        elif json.loads(plan_path.read_bytes()) != plan:
            raise ValueError("resume requires identical source, channel settings and scene input")
        if not (root / "scenarios.json").exists():
            atomic_json(root / "scenarios.json", scenes)
        if json.loads((root / "scenarios.json").read_bytes()) != scenes:
            raise ValueError("stored scene inputs differ from the current experiment")
        if not (root / "environment.json").exists():
            atomic_json(root / "environment.json", environment("cpu"))
        rows = []
        for case, scene in zip(cases, scenes):
            initial = None
            for condition, channel in channels.items():
                for arm, (brake, scan) in ARMS.items():
                    path = root / "cells" / f"case_{case['case']:03d}" / condition / f"{arm}.json"
                    identity = dict(case=case["case"], scenario_sha256=case["scenario_sha256"], condition=condition, arm=arm)
                    if path.exists():
                        row = json.loads(path.read_bytes())
                        if any(row[key] != value for key, value in identity.items()):
                            raise ValueError(f"invalid stored cell {path}")
                    else:
                        row = dict(**identity, **_evaluate(base, scene, case["physical_seed"], channel,
                                                          steps=steps, brake=brake, scan=scan))
                    physical = row["initial_physics_sha256"]
                    if initial is not None and physical != initial:
                        raise ValueError(f"paired initial physical states disagree for case {case['case']}")
                    initial = physical
                    if not path.exists():
                        atomic_json(path, row)
                        print(json.dumps(dict(case=case["case"] + 1, arm=arm, condition=condition,
                                              steps=row["steps"], collision=row["collision"], found=row["found"]),
                                         ensure_ascii=False), flush=True)
                    rows.append(row)
        result = _report(rows, channels, observation_limit=steps, expected=len(scenes) * len(channels) * len(ARMS))
        if not (root / "report.json").exists():
            atomic_json(root / "report.json", result)
        elif json.loads((root / "report.json").read_bytes()) != result:
            raise ValueError("stored report differs from the completed cell results")
        return dict(report=str(root / "report.json"), **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    scenes = parser.add_mutually_exclusive_group()
    scenes.add_argument("--manifest", type=Path)
    scenes.add_argument("--generated", action="store_true", help="generate a separate reproducible suite")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--smoke", action="store_true", help="one paired case, at most 25 steps per cell")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    count, steps = (1, 25) if args.smoke else (CASE_COUNT, 400)
    budget = dict(cases=count, arms=list(ARMS), conditions=2, rollouts=count * len(ARMS) * 2,
                  maximum_physical_steps=count * len(ARMS) * 2 * steps, training_updates=0)
    if not args.execute:
        print(json.dumps(dict(executed=False, **budget), indent=2))
        return
    if args.root is None or (args.manifest is None and not args.generated):
        parser.error("--execute needs --root and either --manifest or --generated")
    if not args.smoke and platform.system() != "Linux":
        parser.error("the 30-scene evaluation runs on Linux; use --smoke for a local check")
    torch.set_num_threads(1)
    result = execute(args.root, args.manifest, smoke=args.smoke, resume=args.resume)
    print(json.dumps({k: v for k, v in result.items() if k not in ("counts", "paired")}, indent=2))
