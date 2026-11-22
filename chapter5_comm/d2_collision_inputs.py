"""Read-only inputs from the two completed Linux runs, pinned to their receipts."""
import json
import os
from pathlib import Path

from chapter3_bser.learning.config import scenario_fingerprint
from tools.executor_trace import sha


R123_RUN = "r123_expansion_20261120_123244_3078331"
BATCH_RUN = "d2_batch_r123_20261121_140145_3923224"
R123_PLAN = "9efebcdb73c0b326b9d62653c89da558e8f817cbb7e190f34fc0b1f09536b168"
BATCH_PLAN = "017c6544f15bbd2508c2a96fe13d83138b286e1ed2eb69a5d27410c2e1653816"
CASES = {
    3: (3, 66, "e492c121ea41b988b92f42778fd6fba61b20be20af410329366373a48a263d4b"),
    9: (2, 37, "4177f72287fbc14e15163c292fd455315c8265fea3e8d20c2c77a05dbb4341ce"),
}
RECEIPTS = {
    (3, "tracking_both"): "c02efdcb57c7a58a9cc7e7627c98a57bcfc9646c7730a3e51b762c3dad9d2470",
    (3, "task_sync"): "b4931ba0d818d316a57ffba42cde210773ab2a51f51fa2c5989aedcdd3540476",
    (9, "tracking_both"): "e8f93d40f20d825e26a322214f2121dd6c5458c8b3d10650426a2615a4686e3b",
    (9, "task_sync"): "3aeaeea0fdbd259a92acbcc8d6234f9a9b440da3c21a0afbb149160ab0ab26b8",
}
TRACE_FIELDS = ("step", "agent", "position", "velocity", "next_position", "map_revision",
                "route_id", "route_index", "endpoint", "event_target", "executed", "decision", "collision")


def checked_json(path, expected):
    if sha(path) != expected:
        raise ValueError(f"input SHA-256 mismatch: {path}")
    return json.loads(path.read_bytes())


def locate_runs(search, ch3, batch):
    found = {R123_RUN: ch3, BATCH_RUN: batch}
    needed = {name for name, path in found.items() if path is None}
    candidates = {name: [] for name in needed}
    if needed:
        for parent, dirs, _ in os.walk(search):
            for name in needed.intersection(dirs):
                candidates[name].append(Path(parent) / name)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in needed
                             and d not in ("__pycache__", "cells", "evaluation", "checkpoints", "node_modules"))
    for name, paths in candidates.items():
        if len(paths) > 1:
            # A copied run in a backup can have the same name. Prefer a complete
            # copy of the pinned inputs. load_cases verifies their full contents.
            if name == R123_RUN:
                required = ["plan/scenes.json", *(f"evaluation/{i:04d}_{arm}/episode.json" for i, arm in RECEIPTS),
                            *(f"evaluation/{i:04d}_{arm}/completed.json" for i, arm in RECEIPTS)]
                plan, checksum = "plan/plan.json", R123_PLAN
            else:
                required = ["scenarios.json", *(f"cells/case_{i:03d}/reliable_reference/baseline.json" for i in CASES)]
                plan, checksum = "plan.json", BATCH_PLAN
            paths = sorted((d for d in paths if all((d / f).is_file() for f in [plan, *required])
                            and sha(d / plan) == checksum), key=lambda d: (len(d.parts), str(d)))
        if not paths:
            option = "--ch3-run" if name == R123_RUN else "--batch-run"
            raise ValueError(f"could not locate the original {name} under {search}. "
                             f"Pass {option} with the extracted run directory.")
        found[name] = paths[0]
    return found[R123_RUN].resolve(), found[BATCH_RUN].resolve()


def load_cases(ch3, batch):
    plan3 = checked_json(ch3 / "plan/plan.json", R123_PLAN)
    scenes3 = checked_json(ch3 / "plan/scenes.json", plan3["inputs"]["scenes.json"])["scenarios"]
    plan5 = checked_json(batch / "plan.json", BATCH_PLAN)
    scenes5 = checked_json(batch / "scenarios.json",
                           "681c5afe276ad329c20cbb6b2f5c37aa1aad5b8afbdb504096a6712398324eb9")
    cases = []
    for i, (agent, steps, cell_hash) in CASES.items():
        scene = scenes3[i]
        cell = checked_json(batch / f"cells/case_{i:03d}/reliable_reference/baseline.json", cell_hash)
        if (scene != scenes5[i] or scene["scenario_id"] != f"r123_expand_20261119_{i:04d}"
                or scenario_fingerprint(scene) != cell["scenario_sha256"]
                or cell["collision_step"] != steps or cell["collision_records"][0]["agent_id"] != agent):
            raise ValueError(f"invalid historical pair for case {i}")
        traces = {}
        seed3 = plan3["evaluation_seed"] + i
        for arm in ("tracking_both", "task_sync"):
            folder = ch3 / f"evaluation/{i:04d}_{arm}"
            receipt = checked_json(folder / "completed.json", RECEIPTS[i, arm])
            episode = checked_json(folder / "episode.json", receipt["artifacts"]["episode.json"])
            if (receipt["seed"] != seed3 or episode["scenario_id"] != scene["scenario_id"]
                    or episode["collision_episode"] or episode["actual_length"] != 400):
                raise ValueError(f"unexpected R123 episode: {folder}")
            trace = episode["path_option"]["trace"][:4 * steps]
            if [(row["step"], row["agent"]) for row in trace] != [
                    (t, a) for t in range(steps) for a in range(4)]:
                raise ValueError(f"incomplete action-indexed R123 prefix: {folder}")
            traces[arm] = [{key: row[key] for key in TRACE_FIELDS} for row in trace]
        if traces["tracking_both"] != traces["task_sync"]:
            raise ValueError(f"R123 arms differ within the collision window of case {i}")
        cases.append(dict(index=i, agent=agent, scene=scene, cell=cell, steps=steps, seed3=seed3,
                          seed5=plan5["cases"][i]["physical_seed"], trace=traces["tracking_both"]))
    return cases, plan5, plan3["inputs"]["config.json"]
