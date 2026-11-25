"""Prepare, execute, resume and report the registered three-seed Linux pilot."""
import argparse
import json
import math
import platform
import uuid
from pathlib import Path

import torch
from chapter3_bser.learning.checkpoint import state_digest
from chapter5_comm.experiment_state import atomic_json, exclusive_job
from chapter5_comm.nominal_maddpg import checkpoint as cp
from chapter5_comm.nominal_maddpg.config import ROOT, digest, file_sha, read_json
from tools.ch5_nominal_pilot_train import (Pilot, PROTOCOL, SOURCE, preflight, require, rounds_due,
                                          seed_process)

DEFAULT_ROOT = ROOT / "outputs/chapter5/nominal_maddpg_pilot64"


def publish(path, value):
    if path.exists():
        require(digest(read_json(path)) == digest(value), "immutable record differs: " + str(path))
    else:
        atomic_json(path,value)


def score(rows):
    require(bool(rows), "empty validation set")
    n = len(rows)
    wins = sum(row["outcome"] == "success" for row in rows)
    collisions = sum(row["outcome"] == "obstacle_collision" for row in rows)
    timeouts = sum(row["outcome"] == "timeout" for row in rows)
    require(wins+collisions+timeouts == n and all(math.isfinite(row["team_return"]) for row in rows),
            "invalid validation outcome or return")
    return dict(scenes=n, success_count=wins, collision_count=collisions, timeout_count=timeouts,
        success_rate=wins/n, collision_rate=collisions/n, mean_team_return=math.fsum(r["team_return"] for r in rows)/n,
        env_steps=sum(r["steps"] for r in rows))


def selection(rows):
    trained = [row for row in rows if row["episode"] > 0]
    return max(trained, key=lambda r:(r["success_count"],-r["collision_count"],r["mean_team_return"],-r["episode"]))


def evaluation_folder(root, episode):
    return root / "validation" / f"{episode:04d}"


def committed(root):
    folder = root/"boundaries"
    if not folder.exists():
        return []
    ids = sorted(int(p.name) for p in folder.iterdir() if p.name.isascii() and p.name.isdecimal())
    require(ids == list(range(len(ids))) and len(ids) <= 65, "missing or unexpected checkpoint boundary")
    return ids


def scene_row(path, tag, entry):
    data = read_json(path)
    require(data["checkpoint"] == tag and data["sha256"] == digest(data["row"]), "invalid evaluation receipt")
    row = data["row"]
    require(row["fingerprint"] == entry["physical_fingerprint"] and row["physical_seed"] == entry["physical_seed"]
            and 1 <= row["steps"] <= 400, "validation scene identity or length differs")
    return row


def evaluate(pilot, boundary):
    marker = read_json(boundary/"committed.json")
    episode = len(pilot.episodes)
    folder = evaluation_folder(pilot.root, episode)
    tag = marker["sha256"]
    before = state_digest(pilot.snapshot())
    require(before == marker["state_sha256"], "evaluation does not start at the registered boundary")
    saved = cp.global_rng_state()
    rows = []
    try:
        for i, entry in enumerate(pilot.groups["validation"]):
            path = folder/f"scene_{i:02d}.json"
            if not path.exists():
                seed_process(entry["physical_seed"])
                row = pilot.validation_scene(i)
                atomic_json(path, dict(checkpoint=tag, row=row, sha256=digest(row)))
            rows.append(scene_row(path,tag,entry))
    finally:
        cp.restore_global_rng(saved)
    require(state_digest(pilot.snapshot()) == before, "validation changed training state")
    metrics = dict(episode=episode, checkpoint=tag, state_sha256=before, scene_rows_sha256=digest(rows),
                   eval_updates=0, restore_verified=True, **score(rows))
    publish(folder/"verified.json",metrics)
    return metrics


def run_seed(root, seed, startup, run_id):
    config, protocol, groups, _ = startup
    folder = root/f"seed_{seed}"
    with exclusive_job(folder):
        seed_process(seed)
        pilot = Pilot(folder,seed,config,protocol,groups,run_id)
        ids = committed(folder)
        if ids:
            # A canonical but damaged checkpoint is an error; never roll it back silently.
            boundary = folder/"boundaries"/f"{ids[-1]:04d}"
            pilot.restore(boundary)
        else:
            boundary = cp.save_boundary(pilot)
        while True:
            episode = len(pilot.episodes)
            if episode in protocol["validation_checkpoints"]:
                # Exercise the disk restore in the normal training path at every validation boundary.
                pilot.restore(boundary)
                metrics = evaluate(pilot,boundary)
                print(f"seed={seed} validation after {episode}: {metrics['success_count']}/32 successes",flush=True)
            if episode == protocol["episodes_per_seed"]:
                break
            pilot.train_one()
            boundary = cp.save_boundary(pilot)
            print(f"seed={seed} episode={episode+1} steps={pilot.team_steps} updates={pilot.learner.rounds}",flush=True)
        publish(folder/"complete.json",dict(identity=pilot.identity, episodes=pilot.episodes,
                train_steps=pilot.team_steps, joint_updates=pilot.learner.rounds, final_payload=file_sha(boundary/"state.pt")))


def check_run(root):
    row = read_json(root/"run.json")
    require(row["source_sha256"] == file_sha(SOURCE) and row["protocol_sha256"] == file_sha(PROTOCOL),
            "pilot source or protocol differs from run")
    return row["id"]


def seed_report(root, seed, startup, run_id):
    config, protocol, groups, _ = startup
    folder = root/f"seed_{seed}"
    sealed = (folder/"complete.json").is_file()
    ids = committed(folder)
    metrics = []
    statuses = []
    for n in protocol["validation_checkpoints"]:
        output = evaluation_folder(folder,n)
        if n not in ids or not (output/"verified.json").is_file():
            statuses.append(dict(episode=n,status="pending"))
            continue
        require(n in ids, "registered evaluation has no boundary")
        boundary = folder/"boundaries"/f"{n:04d}"
        marker = read_json(boundary/"committed.json")
        require(marker["identity"]["run"] == run_id and marker["identity"]["seed"] == seed and
                marker["identity"]["source"] == file_sha(SOURCE) and marker["sha256"] == file_sha(boundary/"state.pt"),
                "evaluation boundary integrity differs")
        rows = [scene_row(output/f"scene_{i:02d}.json",marker["sha256"],entry)
                for i,entry in enumerate(groups["validation"])]
        verified = read_json(output/"verified.json")
        require(verified == dict(episode=n, checkpoint=marker["sha256"], state_sha256=marker["state_sha256"],
                scene_rows_sha256=digest(rows), eval_updates=0,restore_verified=True,**score(rows)),
                "validation aggregation or restore receipt differs")
        metrics.append(verified)
        statuses.append(dict(status="verified",**verified))
    result = dict(seed=seed, completed=False, last_boundary=ids[-1] if ids else None,
                  validation_checkpoints=statuses)
    if not sealed:
        return result
    require(ids == list(range(65)) and len(metrics) == 5, "seed sealed without all training and evaluations")
    complete = read_json(folder/"complete.json")
    identity = complete["identity"]
    require(identity == marker["identity"], "seed completion identity differs")
    state = cp.read_boundary(folder,boundary,identity)
    steps = state["team_steps"]
    require(complete["episodes"] == state["episodes"] and complete["final_payload"] == marker["sha256"]
            and complete["train_steps"] == steps == sum(r["steps"] for r in state["episodes"])
            and complete["joint_updates"] == state["learner"]["counters"]["rounds"] == rounds_due(steps,config["learning"]),
            "training ledger and final state disagree")
    updates = {r["episode"]:sum(ep["updates"] for ep in state["episodes"][:r["episode"]]) for r in metrics}
    eligible = [r for r in metrics if updates[r["episode"]] > 0]
    best = selection(eligible) if eligible else None
    result.update(completed=True, train_steps=steps, joint_updates=complete["joint_updates"],
        training_episodes=state["episodes"], candidate_episode=best["episode"] if best else None,
        candidate_payload=best["checkpoint"] if best else None, data_budget_restore_passed=bool(eligible),
        candidate_joint_updates=updates[best["episode"]] if best else 0,
        initialized_successes=metrics[0]["success_count"], candidate_successes=best["success_count"] if best else None,
        strict_gain=bool(best and best["success_count"] > metrics[0]["success_count"]))
    return result


def gate(seeds):
    if not all(s["completed"] for s in seeds) or len(seeds) != 3:
        return "PENDING"
    return "PASS" if all(s["data_budget_restore_passed"] for s in seeds) and sum(s["strict_gain"] for s in seeds) >= 2 else "STOP"


def report(root, startup, run_id):
    _, protocol, _, _ = startup
    seeds = [seed_report(root,s,startup,run_id) for s in protocol["training_seeds"]]
    decision = gate(seeds)
    final = dict(schema=protocol["schema"], run_id=run_id, decision=decision, seeds=seeds,
                 confirmed_training_steps=sum(s.get("train_steps",0) for s in seeds),
                 verified_validation_steps=sum(r.get("env_steps",0) for s in seeds for r in s["validation_checkpoints"]),
                 training_budget=76800, validation_budget=192000)
    if decision != "PENDING":
        with exclusive_job(root):
            publish(root/"report.json",final)
    for s in seeds:
        print(f"seed={s['seed']} last_boundary={s['last_boundary']} candidate={s.get('candidate_episode')}")
        for r in s["validation_checkpoints"]:
            print(f"  {r['episode']:2d}  {r['status']}  success={r.get('success_count')}  "
                  f"collisions={r.get('collision_count')}  mean_return={r.get('mean_team_return')}")
    print(f"Step 6: {decision}. Output: {root}",flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("check","run","report"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--seed",type=int,choices=(2729,3733,4751))
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    startup = preflight()
    config, protocol, _, registry = startup
    if args.command == "check":
        print(json.dumps(dict(registry=registry, training_seeds=config["training_seeds"],
              validation_checkpoints=protocol["validation_checkpoints"], training_step_limit=76800,
              validation_episode_count=480, device="cpu", training_episodes_started=0),indent=2))
        return
    root = args.root.resolve()
    if args.command == "report":
        report(root,startup,check_run(root))
        return
    if not args.execute or platform.system() != "Linux":
        parser.error("run requires Linux and --execute")
    with exclusive_job(root):
        if not (root/"run.json").exists():
            atomic_json(root/"run.json", dict(id=uuid.uuid4().hex, source_sha256=file_sha(SOURCE),
                                              protocol_sha256=file_sha(PROTOCOL)))
        run_id = check_run(root)
    for seed in ([args.seed] if args.seed else config["training_seeds"]):
        run_seed(root,seed,startup,run_id)
    report(root,startup,run_id)


if __name__ == "__main__":
    main()
