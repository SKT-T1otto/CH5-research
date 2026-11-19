"""One paired 25-step diagnostic on the first D2-short M20 test scenario."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import torch

from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from tools.executor_trace import Blobs, MAP_FIELDS, digest, fields, plain, sha, tap_method, write

from .experiment_config import read_experiment, scene_seed, source_identity
from .experiment_runner import _git_commit, environment, factual_scene
from .pipeline import PROJECT, new_output
from .runtime import CH5Runtime


SCHEMA = "ch5.d2.collision_pair.v1"
PRESET = "configs/chapter5/d2_short_experiment_v1.json"
ROLLOUT_LIMIT = 25
FOCUS_AGENT = 1
SCAN_MAP_FIELDS = (*MAP_FIELDS, "sensor_directions")


def _initial_physics(raw):
    return dict(
        **fields(raw, ("_agent_pos", "_agent_vel", "_agent_acc", "space_size",
                       "flow_gain", "_flow_phase_x", "_flow_phase_y", "dt", "max_steps")),
        target=raw.target_state.to_payload(), obstacles=deepcopy(raw.ground_truth_obstacles),
    )


def _local_navigation(runtime, blobs):
    agents = []
    for i, local in enumerate(runtime.locals):
        sample, origin, distances, hits = local.scan
        agents.append(dict(
            agent_id=i, goal=local.goal, tracking_waypoint=local.waypoint, path=local.path,
            path_index=local.path_index, allocation=local.allocation, allocation_step=local.last_allocation,
            command=local.command, last_plan_step=local.last_plan,
            own_scan=dict(sample_step=sample, origin=origin, distances=distances, hits=hits),
            received_peers=local.peers, map_blob=blobs.put(fields(local.mapper, SCAN_MAP_FIELDS)),
        ))
    return agents


def _native_navigation(runtime, raw, blobs):
    return dict(
        guidance=runtime.guidance,
        allocation=runtime.controller.current_allocation,
        safety_status=runtime.bridge.last_safety_status,
        decision_map_step=runtime.state.step, decision_map_revision=runtime.state.map_revision,
        live_shared_map_blob=blobs.put(fields(raw.map_module, SCAN_MAP_FIELDS)),
        focus_path_tracker=runtime.bridge.path_tracker.snapshot(FOCUS_AGENT),
    )


def _record_arm(run, blobs, base, scene, seed, channel, *, native, expected_initial=None):
    arm = "ch3_native_d2_zero" if native else "ch5_d2_zero_reliable"
    directory = run / arm
    directory.mkdir()
    if native:
        runtime = RepairedMissionRuntime(base, deepcopy(scene), seed=seed, map_research=None)
        raw = runtime.env.unwrapped
    else:
        runtime = CH5Runtime(base, scene, seed=seed, channel_config=channel, navigation="causal_d2_v1")
        raw = runtime.physical
    policy = ZeroResidualPolicy()
    frames = []
    try:
        initial = _initial_physics(raw)
        write(directory / "initial_physics.json", initial)
        if expected_initial is not None and digest(initial) != expected_initial:
            raise ValueError("paired runtimes disagree on the initial physical state")
        physical, scans = [], []

        def observe_physics(when, args, kwargs, result):
            if when == "before":
                navigation = (_native_navigation(runtime, raw, blobs) if native
                              else _local_navigation(runtime, blobs))
                return plain(dict(
                    state=fields(raw, ("_agent_pos", "_agent_vel", "_nav_targets")),
                    phase=raw._phase_before_step, navigation=navigation, residual_input=args[0],
                ))
            before, _ = result
            physical.append(dict(
                before=before,
                after=fields(raw, ("_agent_pos", "_agent_vel", "_agent_acc", "_last_prior_acc", "_last_residual_acc")),
                collisions=plain(raw.last_collision_records),
            ))

        def observe_scan(when, args, kwargs, result):
            if when == "before":
                origins, directions, distances, hit_mask = args
                scans.append(plain(dict(step=kwargs["current_step"], origins=origins,
                                        directions=directions, distances=distances, hits=hit_mask)))

        with ExitStack() as stack:
            stream = stack.enter_context((directory / "steps.jsonl").open("x", encoding="utf-8"))
            stack.enter_context(tap_method(raw, "_apply_agent_dynamics", observe_physics))
            if native:
                stack.enter_context(tap_method(raw.map_module, "integrate_obstacle_scan", observe_scan))
            while runtime.step < ROLLOUT_LIMIT and not runtime.terminal:
                physical.clear()
                scans.clear()
                transition = (runtime.advance(policy, explore=False) if native
                              else runtime.advance(model=policy, explore=False))
                if len(physical) != 1 or int(transition["t"]) + 1 != runtime.step:
                    raise RuntimeError("expected exactly one recorded physical transition per diagnostic step")
                if np.any(transition["actions"]):
                    raise RuntimeError("D2 ZERO must apply zero learned residual")
                frame = dict(step_before=int(transition["t"]), step_after=runtime.step,
                             physical=physical[0], native_shared_scans=list(scans),
                             normalized_residual=plain(transition["actions"]),
                             action_enabled=plain(transition["action_enabled"]))
                stream.write(json.dumps(frame, allow_nan=False) + "\n")
                stream.flush()
                frames.append(frame)
        if not native:
            write(directory / "channel_events.json", runtime.channel.events)
        last = frames[-1]["physical"]
        summary = dict(arm=arm, steps=runtime.step, task_terminated=runtime.terminal,
                       prefix_cutoff=not runtime.terminal, collisions=last["collisions"],
                       collision_step=runtime.step if last["collisions"] else None,
                       final_phase=last["before"]["phase"], target_found=raw.task_found,
                       mission_complete=raw.mission_complete, initial_physics_sha256=digest(initial))
        write(directory / "summary.json", summary)
        return summary, frames
    finally:
        runtime.close()


def _first_difference(native, adapted, *, moment, key, agent=FOCUS_AGENT):
    for left, right in zip(native, adapted):
        before = moment == "before"
        a, b = (frame["physical"][moment] for frame in (left, right))
        av, bv = (row["state"][key][agent] if before else row[key][agent] for row in (a, b))
        if not np.allclose(av, bv, atol=1e-6, rtol=0):
            return dict(step=left["step_before" if before else "step_after"], native=av, adapted=bv)
    return None


def diagnose(root):
    config, base = read_experiment(PRESET)
    training_seed = config["seeds"][0]
    scene = factual_scene(training_seed, "test", 0)
    generator_seed = scene_seed(training_seed, "test", 0)
    physical_seed = generator_seed % (2 ** 31 - 1)
    channel = dict(config["channel"], **config["evaluation_channels"]["reliable_reference"])
    run = new_output(root)
    source = source_identity()
    for name in ("configs/b2.json", "configs/environment.json", "configs/online.json",
                 "tools/executor_trace.py", "project_paths.py"):
        source[name] = sha(PROJECT / name)
    plan = dict(schema=SCHEMA, git_commit=_git_commit(), source=source, preset=PRESET,
                scene_generator_seed=generator_seed, scenario_sha256=scenario_fingerprint(scene),
                physical_seed=physical_seed, scenario_id=scene["scenario_id"],
                task_deadline_steps=base["max_steps"], observation_limit_steps=ROLLOUT_LIMIT,
                focus_agent=FOCUS_AGENT, ch5_channel=channel, environment=environment("cpu"),
                arms=["ch3_native_d2_zero", "ch5_d2_zero_reliable"], training_updates=0)
    write(run / "plan.json", plan)
    write(run / "scenario.json", scene)
    blobs = Blobs(run / "maps")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        native, native_frames = _record_arm(run, blobs, base, scene, physical_seed, channel, native=True)
        adapted, adapted_frames = _record_arm(run, blobs, base, scene, physical_seed, channel,
                                              native=False, expected_initial=native["initial_physics_sha256"])
        result = dict(schema=SCHEMA, complete=True, arms=[native, adapted], focus_agent=FOCUS_AGENT,
                      first_tracking_target_difference=_first_difference(native_frames, adapted_frames, moment="before", key="_nav_targets"),
                      first_applied_acceleration_difference=_first_difference(native_frames, adapted_frames, moment="after", key="_agent_acc"),
                      first_position_difference=_first_difference(native_frames, adapted_frames, moment="after", key="_agent_pos"),
                      shared_observed_transitions=min(len(native_frames), len(adapted_frames)))
        write(run / "result.json", result)
        return dict(report=str(run / "result.json"), **result)
    except Exception as error:
        write(run / "failure.json", dict(complete=False, error=repr(error)))
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and args.root is None:
        parser.error("--execute requires a fresh --root under outputs/chapter5")
    result = (diagnose(args.root) if args.execute else
              dict(executed=False, preset=PRESET, test_case_index=0, observation_limit_steps=ROLLOUT_LIMIT,
                   maximum_physical_steps=2 * ROLLOUT_LIMIT, training_updates=0))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
