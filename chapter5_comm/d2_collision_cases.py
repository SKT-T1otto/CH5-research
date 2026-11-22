"""Replay two reliable-link D2 collisions against the archived Linux R123 decisions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from chapter3_bser.learning.base_runtime import _make_base_env, seed_innovations
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.path_option.path_context import public_route
from core.env.task_protocol import closed_segment_aabb_first_hit
from tools.executor_trace import digest, plain, sha, tap_method, write

from .d2_collision_inputs import BATCH_PLAN, CASES, R123_PLAN, load_cases, locate_runs
from .d2_diagnostics import PRESET, _initial_physics
from .experiment_config import read_experiment
from .experiment_runner import _git_commit
from .pipeline import PROJECT, new_output
from .runtime import CH5Runtime


def obstacle_bounds(obstacle):
    center = np.asarray(obstacle["center"])
    half_size = np.asarray(obstacle["size"]) / 2
    return center - half_size, center + half_size


def route_digest(points):
    return hashlib.sha256(np.asarray(points, dtype="<f4").reshape(-1, 3).tobytes()).hexdigest()[:16]


def map_sample(mapper, anchor, history):
    grid = mapper.occupancy_probability.detach().cpu().numpy().copy()
    cell = mapper._cell_from_point_np(anchor)
    history.append(grid)
    state = "unknown"
    if mapper.known_free_mask[cell]:
        state = "free"
    elif mapper.known_occupied_mask[cell]:
        state = "occupied"
    return dict(revision=int(mapper.map_revision), impact_voxel=cell, impact_voxel_state=state,
                impact_voxel_probability=float(grid[cell]))


def navigation(step, mapper, anchor, box, grids, positions, velocity, agent, goal, points, index, waypoint):
    p = np.array(positions[agent], dtype=float)
    waypoint = np.array(waypoint, dtype=float)
    route = np.asarray(points, dtype=float).reshape(-1, 3)
    remaining = np.vstack((p, route[index:]))
    crosses = [closed_segment_aabb_first_hit(a, b, *box) is not None for a, b in zip(remaining, remaining[1:])]
    lo, hi = box
    return plain(dict(step=step, team_positions=positions, position=p, velocity=velocity,
                      speed_m_s=float(np.linalg.norm(velocity)), goal=goal, route=route, route_index=index,
                      waypoint=waypoint, map=map_sample(mapper, anchor, grids),
                      waypoint_segment_map_free=bool(mapper.segment_is_free(p, waypoint)),
                      waypoint_segment_hits_box=closed_segment_aabb_first_hit(p, waypoint, *box) is not None,
                      remaining_route_hits_box=any(crosses) if crosses else None,
                      obstacle_distance_m=float(np.linalg.norm(np.maximum(np.maximum(lo-p, p-hi), 0.0)))))


def r123_reference(base, case, anchor, box):
    # Reintegrate only sensor evidence at the four *logged* poses. No policy,
    # simulated motion, clock advances through the dynamics, or CH5 map feedback.
    seed_innovations(case["seed3"])
    env = _make_base_env(base)
    frames, grids, routes = [], [], {}
    try:
        env.reset(scenario=case["scene"])
        raw = env.unwrapped
        initial = digest(_initial_physics(raw))
        mapper = raw.map_module
        grid_spec = dict(shape=list(mapper.grid_size), bounds=raw.space_size.tolist(),
                         spacing=[mapper.cell_dx, mapper.cell_dy, mapper.cell_dz], z_range=list(mapper.z_range),
                         free_threshold=mapper.occupancy_free_threshold,
                         occupied_threshold=mapper.occupancy_occupied_threshold)
        agent = case["agent"]
        for step in range(case["steps"]):
            team = case["trace"][4*step:4*step+4]
            row = team[agent]
            positions = np.array([r["position"] for r in team], dtype=np.float32)
            if step == 0:
                np.testing.assert_array_equal(raw._agent_pos.numpy(), positions)
            else:
                raw._agent_pos.copy_(torch.from_numpy(positions))
                raw._agent_vel.copy_(torch.tensor([r["velocity"] for r in team]))
                raw.step_count = step
                raw._sense_and_update_shared_map()
            if int(mapper.map_revision) != row["map_revision"]:
                raise ValueError(f"R123 occupancy revision mismatch at action {step}")
            key = row["route_id"]
            if key not in routes:
                query = public_route(env, agent, row["endpoint"])
                if route_digest(query.path_points) != key:
                    raise ValueError(f"R123 full route does not match the archived route ID {key} at action {step}")
                routes[key] = query.path_points
            points = routes[key]
            index = row["route_index"]
            waypoint = points[index] if len(points) else row["position"]
            frame = navigation(step, mapper, anchor, box, grids, positions, row["velocity"], agent,
                               row["endpoint"], points, index, waypoint)
            frame.update(route_id=key, next_position=row["next_position"], event_target=row["event_target"],
                         applied_normalized_acceleration=row["executed"], action=row["decision"]["action"],
                         collisions=[])
            frames.append(frame)
        return frames, grids, initial, grid_spec
    finally:
        env.close()


def sonar_on_box(local, box):
    _, origin, distances, hits = local.scan
    directions = local.mapper.sensor_directions.detach().cpu().numpy().astype(float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    endpoints = origin + distances[:, None] * directions
    lo, hi = box
    on_face = np.min(np.minimum(abs(endpoints-lo), abs(endpoints-hi)), axis=1) <= 1e-5
    inside = np.all((endpoints >= lo-1e-5) & (endpoints <= hi+1e-5), axis=1)
    observed = distances[hits & on_face & inside]
    return float(np.min(observed)) if len(observed) else None


def replay_ch5(base, case, seed, channel, anchor, box):
    runtime = CH5Runtime(base, case["scene"], seed=seed, channel_config=channel, navigation="causal_d2_v1")
    frames, grids, shared = [], [], []
    try:
        raw = runtime.physical
        initial = digest(_initial_physics(raw))
        agent = case["agent"]
        local = runtime.locals[agent]
        latest_query = None

        def observe_query(when, args, kwargs, result):
            nonlocal latest_query
            if when == "after":
                _, answer = result
                latest_query = dict(step=runtime.step, reachable=bool(answer["reachable"]),
                                    reason=answer["failure_reason"], start=plain(args[0]), goal=plain(args[1]))

        def observe(when, args, kwargs, result):
            if when == "before":
                frame = navigation(runtime.step, local.mapper, anchor, box, grids, raw._agent_pos.numpy(),
                                   raw._agent_vel[agent].numpy(), agent, local.goal, local.path,
                                   local.path_index, raw._nav_targets[agent].numpy())
                frame.update(last_replan_step=int(local.last_plan),
                             latest_observed_astar_query=latest_query,
                             own_allocated_goal=plain(local.allocation.get(agent)),
                             allocation_step=int(local.last_allocation), command=plain(local.command),
                             peer_state_samples={str(i): int(row[0]) for i, row in local.peers.items()},
                             flow_acceleration_xy=raw._flow_at(raw._agent_pos)[agent, :2].tolist(),
                             own_sonar_min_range_to_box_m=sonar_on_box(local, box),
                             latest_scan_samples=[max((t for a, t in local.scan_seen if a == peer), default=None)
                                                  for peer in range(4)],
                             shared_at_same_poses=map_sample(raw.map_module, anchor, shared))
                return frame
            before, _ = result
            before.update(next_position=raw._agent_pos[agent].tolist(), prior_acceleration=raw._last_prior_acc[agent].tolist(),
                          applied_acceleration=raw._agent_acc[agent].tolist(),
                          collisions=plain(raw.last_collision_records))
            frames.append(before)

        policy = ZeroResidualPolicy()
        with tap_method(local.mapper, "grid_astar_path", observe_query), tap_method(raw, "_apply_agent_dynamics", observe):
            while runtime.step < case["steps"] and not runtime.terminal:
                transition = runtime.advance(model=policy, explore=False)
                if np.any(transition["actions"]) or len(frames) != runtime.step:
                    raise RuntimeError("expected one observed transition with zero residual per step")
        summary = runtime.summary()
        return frames, grids, shared, dict(seed=seed, initial_physics_sha256=initial,
            steps=runtime.step, prefix_cutoff=not runtime.terminal, found=bool(summary["found"]),
            collision_step=summary["first_collision_step"], collisions=summary["collision_records"],
            channel=summary["channel"], events=runtime.channel.events)
    finally:
        runtime.close()


def historical_checks(actual, old):
    records = actual["collisions"]
    wanted = old["collision_records"]
    same_ids = [(r["agent_id"], r["obstacle_id"]) for r in records] == [
        (r["agent_id"], r["obstacle_id"]) for r in wanted]
    return dict(initial=actual["initial_physics_sha256"] == old["initial_physics_sha256"],
                termination=(not actual["prefix_cutoff"] and actual["steps"] == old["steps"]
                             and actual["collision_step"] == old["collision_step"] and actual["found"] == old["found"]),
                agents_and_obstacles=same_ids,
                collision_geometry=same_ids and all(
                    np.allclose(a[k], b[k], rtol=0, atol=1e-5) for a, b in zip(records, wanted)
                    for k in ("position_before", "candidate_position", "first_hit_point", "hit_fraction")),
                channel=actual["channel"] == old["channel"])


def first_at(rows, predicate):
    return next((r["step"] for r in rows if predicate(r)), None)


def compare(reference, replay, summary, agent):
    first = {}
    for key in ("goal", "waypoint", "position", "next_position"):
        first[key] = next((a["step"] for a, b in zip(reference, replay)
                           if not np.allclose(a[key], b[key], rtol=0, atol=1e-5)), None)
    last = replay[-1]
    deliveries = [e["delivery_step"] for e in summary["events"] if e["kind"] == "SCAN"
                  and e["receiver"] == agent and e["delivery_step"] is not None]
    final_empty = 0
    for frame in reversed(replay):
        if frame["route"]:
            break
        final_empty += 1
    return dict(first_different_action_index=first, first_empty_path=first_at(replay, lambda r: not r["route"]),
                consecutive_empty_path_actions_at_end=final_empty,
                actions_tracking_through_box_on_locally_clear_segment=sum(
                    r["waypoint_segment_map_free"] and r["waypoint_segment_hits_box"] for r in replay),
                actions_with_different_impact_voxel_class=sum(
                    r["map"]["impact_voxel_state"] != r["shared_at_same_poses"]["impact_voxel_state"] for r in replay),
                first_planned_segment_hits_box=first_at(replay, lambda r: r["waypoint_segment_hits_box"]),
                first_own_sonar_on_box=first_at(replay, lambda r: r["own_sonar_min_range_to_box_m"] is not None),
                first_peer_scan_delivery=min(deliveries, default=None),
                last_pre_action=last, r123_same_action=reference[last["step"]])


def diagnose_case(directory, base, case, channel):
    directory.mkdir()
    old = case["cell"]
    hit = old["collision_records"][0]
    anchor = hit["first_hit_point"]
    box = obstacle_bounds(case["scene"]["obstacles"][hit["obstacle_id"]])
    ref, shared, initial, grid = r123_reference(base, case, anchor, box)
    traces = {"r123_shared": ref}
    arrays = {"r123_shared": shared}
    result = dict(case=case["scene"]["scenario_id"], focus_agent=case["agent"], historical_collision=hit,
                  action_window=case["steps"], r123_seed=case["seed3"], r123_initial_physics_sha256=initial,
                  r123_arms_equal_in_window=True, r123_verified_route_ids=sorted({r["route_id"] for r in ref}),
                  r123_first_planned_segment_hits_box=first_at(ref, lambda r: r["waypoint_segment_hits_box"]))
    for name, seed in (("ch5_batch_seed", case["seed5"]), ("ch5_r123_seed", case["seed3"])):
        print(f"[{case['index']:04d}] R123 prefix verified, replaying {name} with seed {seed}", flush=True)
        frames, grids, shadow, summary = replay_ch5(base, case, seed, channel, anchor, box)
        comparison = compare(ref, frames, summary, case["agent"])
        write(directory / f"{name}_channel.json", summary.pop("events"))
        traces[name] = frames
        arrays[name] = grids
        arrays[name+"_shared_at_same_poses"] = shadow
        if name == "ch5_batch_seed":
            result["historical_checks"] = historical_checks(summary, old)
        else:
            result["matched_initial_physics"] = initial == summary["initial_physics_sha256"]
        result[name] = dict(**summary, **comparison)
    result["ch5_two_seed_traces_equal"] = traces["ch5_batch_seed"] == traces["ch5_r123_seed"]
    result["verified"] = all(result["historical_checks"].values()) and result["matched_initial_physics"]
    write(directory / "grid.json", dict(**grid, impact_point=anchor, obstacle_id=hit["obstacle_id"],
                                       obstacle_bounds=box, map_timing="pre-action", scenario=case["scene"]))
    np.savez_compressed(directory / "occupancy.npz", **{k: np.stack(v) for k, v in arrays.items()})
    for name, frames in traces.items():
        with (directory / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            for frame in frames:
                stream.write(json.dumps(frame, allow_nan=False) + "\n")
    from .d2_collision_plot import plot
    plot(directory / "comparison.png", grid, box, anchor, case["agent"], traces, arrays)
    write(directory / "comparison.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, default=PROJECT.parent)
    parser.add_argument("--ch3-run", type=Path)
    parser.add_argument("--batch-run", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    budget = dict(cases=["0003", "0009"], physical_step_upper_bound=2*sum(v[1] for v in CASES.values()),
                  r123_new_physical_steps=0, training_updates=0)
    if not args.execute:
        print(json.dumps(dict(executed=False, **budget), indent=2))
        return
    if args.root is None:
        parser.error("--execute requires a new --root under outputs/chapter5")
    ch3, batch = locate_runs(args.search_root, args.ch3_run, args.batch_run)
    cases, plan5, base_hash = load_cases(ch3, batch)
    _, base = read_experiment(PRESET)
    frozen = {k: v for k, v in plan5["source"].items()
              if k.startswith(("core/", "chapter3_bser/", "chapter5_comm/", "configs/"))}
    if sha(PROJECT / "configs/b2.json") != base_hash or any(sha(PROJECT / k) != v for k, v in frozen.items()):
        raise ValueError("historical runtime, mapping or task configuration differs from the frozen run")
    run = new_output(args.root)
    write(run / "plan.json", dict(**budget, ch3=str(ch3), batch=str(batch), git_commit=_git_commit(),
                                  ch3_plan_sha256=R123_PLAN, batch_plan_sha256=BATCH_PLAN,
                                  channel=plan5["channels"]["reliable_reference"]))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        results = {}
        for case in cases:
            label = f"{case['index']:04d}"
            print(f"[{label}] comparing the R123 archive and two D2 reliable-link replays", flush=True)
            results[label] = diagnose_case(run / label, base, case, plan5["channels"]["reliable_reference"])
            print(f"[{label}] archived collision checks: {results[label]['historical_checks']}", flush=True)
            replay = results[label]["ch5_batch_seed"]
            print(f"[{label}] collision step={replay['collision_step']}, "
                  f"final consecutive actions with empty route={replay['consecutive_empty_path_actions_at_end']}, "
                  f"first tracking segment through obstacle={replay['first_planned_segment_hits_box']}", flush=True)
        verified = all(r["verified"] for r in results.values())
        report = dict(complete=True, verified=verified, **budget, results=results)
        write(run / "report.json", report)
        print(json.dumps(dict(report=str(run / "report.json"), complete=True, verified=verified)))
        if not verified:
            raise RuntimeError("replay differs from the archived physics or collision; see report.json")
    except Exception as exc:
        write(run / "failure.json", dict(error=str(exc)))
        raise
    finally:
        torch.set_num_threads(threads)


if __name__ == "__main__":
    main()
