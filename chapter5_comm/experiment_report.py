"""Hierarchical reporting: independent training seeds are the replication unit."""
from pathlib import Path
import csv
import json
import math
import uuid

import numpy as np

from .pipeline import write_json
from .experiment_runner import load_plan, job_path


def seed_interval(values, seed=104729):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, ci95=None, independent_seeds=0)
    if len(values) < 2:
        return dict(mean=float(values.mean()), ci95=None, independent_seeds=len(values))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (5000, len(values)), replace=True).mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                independent_seeds=len(values), interval="percentile bootstrap of training-seed means; small-n exploratory")


def report(root):
    root, plan = load_plan(root)
    c = plan["config"]
    rows, missing, by_key = [], [], {}
    for seed in c["seeds"]:
        for arm in c["arms"]:
            for condition in c["evaluation_channels"]:
                path = job_path(root, seed, arm) / "evaluation" / condition / "result.json"
                if not path.exists():
                    missing.append(dict(seed=seed, arm=arm, condition=condition))
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                if data["config_sha"] != plan["config_sha"] or (data["seed"], data["arm"], data["condition"]) != (seed, arm, condition):
                    raise ValueError("evaluation identity mismatch")
                episodes = data["rows"]
                if len(episodes) != c["evaluation_episodes"]:
                    raise ValueError("completed marker has an incomplete episode count")
                returns = [x["observed_return"] for x in episodes]
                complete = all(x["completed_task_episode"] for x in episodes)
                returns_sorted = sorted(returns)
                tail = max(1, math.ceil(.1 * len(returns_sorted)))
                state_count = sum(x.get("state_estimate_count", 0) for x in episodes)
                value = dict(mean_return=float(np.mean(returns)),
                    success_rate=float(np.mean([bool(x.get("success", False)) for x in episodes])) if complete else None,
                    found_rate=float(np.mean([bool(x.get("found")) for x in episodes])),
                    collision_rate=float(np.mean([bool(x.get("collision_episode", x.get("collision_records"))) for x in episodes])),
                    handoff_rate=float(np.mean([x.get("handoff_event_step") is not None for x in episodes])),
                    capture_contact_rate=float(np.mean([x.get("capture_contact_steps", 0) > 0 for x in episodes])),
                    return_cvar10=float(np.mean(returns_sorted[:tail])), tail_episodes=tail,
                    mean_sent_bits=float(np.mean([x["channel"]["sent_bits"] for x in episodes])),
                    mean_wall_seconds=float(np.mean([x["wall_seconds"] for x in episodes])),
                    mean_state_position_error=(sum(x["mean_state_position_error"] * x["state_estimate_count"] for x in episodes if x.get("state_estimate_count", 0)) / state_count) if state_count else None,
                    mean_normalized_state_nll=(sum(x["mean_normalized_state_nll"] * x["state_estimate_count"] for x in episodes if x.get("state_estimate_count", 0)) / state_count) if state_count else None)
                by_key[seed, arm, condition] = value
                for x in episodes:
                    rows.append(dict(seed=seed, arm=arm, condition=condition,
                        scenario=x["scenario_generator_seed"], length=x["actual_length"],
                        return_value=x["observed_return"], success=x.get("success"),
                        found=x.get("found"), termination_reason=x.get("termination_reason"),
                        collision=bool(x.get("collision_episode", x.get("collision_records"))),
                        handoff_step=x.get("handoff_event_step"), capture_contact_steps=x.get("capture_contact_steps"),
                        task_policy=x.get("task_policy"), task_update_count=x.get("task_update_count"),
                        budget_truncated=x["budget_truncated"], sent_bits=x["channel"]["sent_bits"],
                        delivered_bits=x["channel"]["delivered_bits"], wall_seconds=x["wall_seconds"],
                        state_estimates=x.get("state_estimate_count", 0), position_error=x.get("mean_state_position_error"),
                        normalized_state_nll=x.get("mean_normalized_state_nll")))
    aggregate, comparisons = [], []
    for arm in c["arms"]:
        for condition in c["evaluation_channels"]:
            values = [by_key[s, arm, condition] for s in c["seeds"] if (s, arm, condition) in by_key]
            metrics = {}
            for key in ("mean_return", "success_rate", "found_rate", "collision_rate", "handoff_rate",
                        "capture_contact_rate", "return_cvar10", "mean_sent_bits", "mean_wall_seconds",
                        "mean_state_position_error", "mean_normalized_state_nll"):
                metrics[key] = seed_interval([x[key] for x in values if x[key] is not None])
            aggregate.append(dict(arm=arm, condition=condition, metrics=metrics,
                expected_training_seeds=len(c["seeds"]), complete=len(values) == len(c["seeds"])))
    reference = "d2_b2" if "d2_zero" in c["arms"] else "full"
    for arm in c["arms"]:
        if arm == reference:
            continue
        for condition in c["evaluation_channels"]:
            seeds = [s for s in c["seeds"] if (s, arm, condition) in by_key and (s, reference, condition) in by_key]
            delta = [by_key[s, reference, condition]["mean_return"] - by_key[s, arm, condition]["mean_return"] for s in seeds]
            comparisons.append(dict(reference=reference, comparator=arm, condition=condition,
                paired_training_seeds=seeds, return_difference=seed_interval(delta),
                pairing="matching training-seed labels and physical scenarios; NOT identical packet-fault draws"))
    destination = root / "reports" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    if rows:
        with (destination / "episodes.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(schema="ch5.experiment.report.v1", complete=not missing, missing=missing,
        configured_training_seeds=c["seeds"], aggregate=aggregate, comparisons=comparisons,
        formal_performance_verified=False,
        interpretation="Completion is not a superiority claim; reliable_reference is not budget-matched; CVaR is descriptive, not risk training.")
    write_json(destination / "report.json", result)
    lines = ["# CH5 experiment report", "", f"Complete: {not missing}; missing arm/seed/condition cells: {len(missing)}.", "",
             "Training seeds, not pooled episodes, are the replication unit. Confidence intervals with few seeds are exploratory.", "",
             "| Arm | Condition | Return mean | Seed-bootstrap 95% interval | Completed seeds |", "|---|---|---:|---|---:|"]
    for row in aggregate:
        metric = row["metrics"]["mean_return"]
        lines.append(f"| {row['arm']} | {row['condition']} | {metric['mean']} | {metric['ci95']} | {metric['independent_seeds']} |")
    lines += ["", "No automatic novelty, convergence or significant-improvement acceptance is inferred.",
              "External CoDe/VA-MARO/VIL2C baselines are not implemented reproductions in this suite."]
    with (destination / "report.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return dict(report=str(destination), complete=not missing, missing_cells=len(missing))
