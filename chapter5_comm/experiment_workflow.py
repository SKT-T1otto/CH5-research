"""Explicit manual Linux experiment launcher, separate from bounded ch5 smoke."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

from .pipeline import PROJECT, write_json
from .experiment_config import read_experiment, plan_summary
from .experiment_runner import prepare, train, evaluate, load_plan
from .experiment_report import report


def suite(root, stage, device, resume=False):
    root, plan = load_plan(root)
    destination = root / "suite_attempts" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    outcomes = []
    for seed in plan["config"]["seeds"]:
        for arm in plan["config"]["arms"]:
            for action in (("train", "evaluate") if stage == "all" else (stage,)):
                command = [sys.executable, "-B", "cli.py", "ch5-experiment", action,
                    "--root", str(root), "--seed", str(seed), "--arm", arm, "--device", device, "--execute"]
                if resume and action == "train":
                    command.append("--resume")
                with (destination / f"{seed}_{arm}_{action}.log").open("x", encoding="utf-8") as f:
                    status = subprocess.run(command, cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT).returncode
                row = dict(seed=seed, arm=arm, stage=action, exit_code=status)
                outcomes.append(row)
                print(json.dumps(row), flush=True)
                if status:
                    break
    write_json(destination / "status.json", dict(jobs=outcomes, passed=all(x["exit_code"] == 0 for x in outcomes)))
    summary = report(root)
    return dict(attempt=str(destination), passed=all(x["exit_code"] == 0 for x in outcomes), **summary)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "prepare", "train", "evaluate", "suite", "report"))
    p.add_argument("--config", default="configs/chapter5/pilot_experiment_v1.json")
    p.add_argument("--root")
    p.add_argument("--seed", type=int)
    p.add_argument("--arm")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    p.add_argument("--stop-after", choices=("collection", "auxiliary", "task"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    if a.command == "plan" or not a.execute:
        c = load_plan(a.root)[1]["config"] if a.root else read_experiment(a.config)[0]
        result = dict(command=a.command, executed=False, **plan_summary(c))
    else:
        if not a.root:
            p.error("execution requires --root under outputs/chapter5")
        if a.command == "prepare":
            result = prepare(a.config, a.root)
        elif a.command in ("train", "evaluate"):
            if a.seed is None or a.arm is None:
                p.error("one job requires --seed and --arm from the frozen plan")
            if a.command == "train":
                result = train(a.root, a.seed, a.arm, device=a.device, resume=a.resume, stop_after=a.stop_after)
            else:
                result = evaluate(a.root, a.seed, a.arm, device=a.device)
        elif a.command == "suite":
            result = suite(a.root, a.stage, a.device, a.resume)
        else:
            result = report(a.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result.get("passed") is False:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
