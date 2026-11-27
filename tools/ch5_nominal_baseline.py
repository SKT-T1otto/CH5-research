"""Fresh standard MADDPG training, bounded stages, fixed validation and one-time admission."""
import argparse
import copy
from contextlib import ExitStack
import json
import platform
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter5_comm.experiment_state import atomic_json, exclusive_job
from chapter5_comm.nominal_maddpg import checkpoint as cp
from chapter5_comm.nominal_maddpg.config import ROOT, digest, file_sha, read_json
from chapter5_comm.nominal_maddpg.exploration import OUNoise
from chapter5_comm.nominal_maddpg.replay import NominalReplay
from chapter5_comm.nominal_maddpg.runtime import NominalRuntime
from tools.ch5_nominal_pilot import publish, scene_row, score, selection
from tools.ch5_nominal_pilot_train import Pilot, preflight, episode_result, seed_process

REGISTER = ROOT / "provenance/ch5_nominal_baseline"
PROTOCOL = REGISTER / "protocol.json"
SOURCE = REGISTER / "source.json"
OUTPUT = ROOT / "outputs/chapter5/nominal_baseline_v2"


def require(value, message):
    if not value:
        raise ValueError(message)


def checked_setup():
    setup = preflight()
    source = read_json(SOURCE)
    for name, expected in source["files"].items():
        require(file_sha(ROOT/name) == expected, "baseline source differs: " + name)
    return setup, read_json(PROTOCOL)


def ids(root):
    folder = root/"boundaries"
    values = sorted(int(p.name) for p in folder.iterdir() if p.name.isascii() and p.name.isdecimal()) if folder.exists() else []
    require(values == list(range(len(values))) and len(values) <= 1001, "missing training boundary")
    return values


class Baseline(Pilot):
    def __init__(self, root, seed, setup, spec, job):
        config, old_protocol, groups, _ = setup
        protocol = copy.deepcopy(old_protocol)
        protocol.update(episodes_per_seed=1000, max_train_team_steps_per_seed=400000)
        super().__init__(root,seed,config,protocol,groups,job)
        self.identity.update(baseline_source=file_sha(SOURCE), baseline_protocol=file_sha(PROTOCOL))
        self.learning = copy.deepcopy(self.learning)
        self.learning["actor_lr"] = spec["actor_lr"]
        for optimizer in self.learner.actor_optimizers:
            for group in optimizer.param_groups:
                group["lr"] = spec["actor_lr"]
        noise = dict(protocol["exploration"])
        noise["decay_per_environment_step"] = spec["exploration_decay_per_step"]
        self.noise = OUNoise(seed+protocol["exploration_seed_offset"], **noise)


@torch.no_grad()
def validate_scene(model, entry, index, config):
    seed_process(entry["physical_seed"])
    rt = NominalRuntime(config,entry["scene"],entry["physical_seed"],NominalReplay(1,0),episode_id=index)
    reward = 0.
    opened = np.zeros(4,dtype=np.int64)
    saturated = np.zeros(4,dtype=np.int64)
    try:
        while not rt.core.terminal:
            tr = rt.advance(model.learner.networks)
            reward += float(tr["rewards"][0])
            gate = tr["action_enabled"]
            opened += gate
            saturated += (np.abs(tr["proposed_actions"]) >= .999).any(axis=1) & gate
        return episode_result(rt,entry,reward), dict(open=opened.tolist(),saturated=saturated.tolist())
    finally:
        rt.close()


def evaluate(model, group):
    n = len(model.episodes)
    boundary = model.root/"boundaries"/f"{n:04d}"
    marker = read_json(boundary/"committed.json")
    before = state_digest(model.snapshot())
    require(before == marker["state_sha256"], "evaluation boundary differs")
    tag = marker["sha256"]
    folder = model.root/group/f"{n:04d}"
    rows, actions = [], []
    rng = cp.global_rng_state()
    try:
        for i,entry in enumerate(model.groups[group]):
            path = folder/f"scene_{i:03d}.json"
            if not path.exists():
                row,action = validate_scene(model,entry,i,model.config)
                atomic_json(path, dict(checkpoint=tag,row=row,sha256=digest(row),actions=action,
                                       actions_sha256=digest(action)))
            packet = read_json(path)
            row = scene_row(path,tag,entry)
            action = packet["actions"]
            require(packet["actions_sha256"] == digest(action) and len(action["open"]) == 4
                    and len(action["saturated"]) == 4 and
                    all(0 <= a <= b <= row["steps"] for a,b in zip(action["saturated"],action["open"])),
                    "invalid evaluation action receipt")
            rows.append(row)
            actions.append(action)
            print(f"seed={model.identity['seed']} {group} ep={n} {i+1}/{len(model.groups[group])} "
                  f"{row['outcome']}",flush=True)
    finally:
        cp.restore_global_rng(rng)
    require(state_digest(model.snapshot()) == before, "evaluation changed learner, replay or RNG")
    opened = np.sum([a["open"] for a in actions],axis=0)
    saturated = np.sum([a["saturated"] for a in actions],axis=0)
    metrics = dict(episode=n, checkpoint=tag,state_sha256=before,rows_sha256=digest(rows),
        action_rows_sha256=digest(actions), optimizer_updates=0, restored=True,
        saturation_open=[float(a/b) if b else None for a,b in zip(saturated,opened)], **score(rows))
    publish(folder/"verified.json",metrics)
    return metrics


def stable(metrics, spec):
    if len(metrics) < spec["stable_validation_points"]:
        return dict(stable=False,strong=False,weak=False,reason="insufficient_validation_points")
    recent = metrics[-spec["stable_validation_points"]:]
    success = [r["success_count"] for r in recent]
    returns = [r["mean_team_return"] for r in recent]
    scale = max(spec["stable_return_scale_floor"],abs(sum(returns)/len(returns)))
    spread = max(returns)-min(returns)
    plateau = max(success)-min(success) <= spec["stable_success_range_count"] and spread <= spec["stable_return_range_relative"]*scale
    safe = max(r["collision_count"] for r in recent) <= spec["maximum_validation_collisions"]
    return dict(stable=plateau, strong=plateau and safe and min(success)>=spec["strong_validation_successes"],
        weak=plateau and safe and min(success)>=spec["weak_validation_successes"],
        success_range=[min(success),max(success)], return_range=[min(returns),max(returns)],
        return_spread_limit=spec["stable_return_range_relative"]*scale,
        last_episodes=[r["episode"] for r in recent], maximum_collisions=max(r["collision_count"] for r in recent))


def job_identity(root):
    job = read_json(root/"run.json")
    require(job["source"] == file_sha(SOURCE) and job["protocol"] == file_sha(PROTOCOL), "run source identity differs")
    return job["id"]


def training(root, seed, setup, spec, job, until):
    with exclusive_job(root):
        progress = ids(root)
        require(not (root/"freeze.json").exists(), "policy frozen; use a new experiment for further development")
        seed_process(seed)
        model = Baseline(root,seed,setup,spec,job)
        if progress:
            model.restore(root/"boundaries"/f"{progress[-1]:04d}")
        else:
            cp.save_boundary(model)
        while len(model.episodes) <= until:
            n = len(model.episodes)
            if n % spec["validation_interval"] == 0:
                model.restore(root/"boundaries"/f"{n:04d}")
                evaluate(model,"validation")
            if n == until:
                break
            model.train_one()
            cp.save_boundary(model)
            row = model.episodes[-1]
            print(f"seed={seed} train={len(model.episodes)}/{until} steps={model.team_steps} "
                  f"return={row['team_return']:.6f} outcome={row['outcome']} "
                  f"updates={model.learner.rounds} sigma={model.noise.sigma:.4f}",flush=True)


def report(root,setup,spec):
    seeds = []
    for seed in spec["training_seeds"]:
        d = root/f"seed_{seed}"
        progress = ids(d)
        rows = []
        for n in progress:
            path = d/"validation"/f"{n:04d}"/"verified.json"
            if n%spec["validation_interval"] == 0 and path.exists():
                metrics = read_json(path)
                marker = read_json(d/"boundaries"/f"{n:04d}"/"committed.json")
                require(metrics["episode"] == n and metrics["checkpoint"] == marker["sha256"]
                        and metrics["optimizer_updates"] == 0, "validation summary identity differs")
                packets = [read_json(path.parent/f"scene_{i:03d}.json") for i in range(32)]
                for i,p in enumerate(packets):
                    scene_row(path.parent/f"scene_{i:03d}.json",marker["sha256"],setup[2]["validation"][i])
                    require(digest(p["actions"])==p["actions_sha256"],"action receipt differs")
                require(metrics["rows_sha256"]==digest([p["row"] for p in packets]) and
                        metrics["action_rows_sha256"]==digest([p["actions"] for p in packets]) and
                        all(metrics[k]==v for k,v in score([p["row"] for p in packets]).items()),
                        "validation aggregate differs")
                rows.append(metrics)
        expected = list(range(0,(progress[-1] if progress else 0)+1,spec["validation_interval"]))
        all_points = [r["episode"] for r in rows] == expected
        status = stable(rows,spec)
        status["strong"] = status["strong"] and all_points and bool(progress) and rows[-1]["episode"]==progress[-1]
        status["weak"] = status["weak"] and all_points and bool(progress) and rows[-1]["episode"]==progress[-1]
        train_rows = []
        if progress:
            boundary = d/"boundaries"/f"{progress[-1]:04d}"
            marker = read_json(boundary/"committed.json")
            state = cp.read_boundary(d,boundary,marker["identity"])
            require(marker["identity"]["run"]==job_identity(root) and marker["identity"]["seed"]==seed
                    and marker["identity"]["baseline_source"]==file_sha(SOURCE), "checkpoint run differs")
            train_rows = state["episodes"]
        seeds.append(dict(seed=seed, last_boundary=progress[-1] if progress else None, training=train_rows,
                          validation=rows,stability=status))
    passed = all(s["stability"]["weak"] for s in seeds) and sum(s["stability"]["strong"] for s in seeds)>=spec["required_strong_seeds"]
    data = dict(seeds=seeds,development_ready_to_freeze=passed,formal_admission="PENDING",
                source=file_sha(SOURCE), protocol=file_sha(PROTOCOL))
    return data


def curves(root,data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(12,8))
    for s in data["seeds"]:
        train, val = s["training"],s["validation"]
        if train:
            values = [r["team_return"] for r in train]
            mean = [sum(values[max(0,i-19):i+1])/len(values[max(0,i-19):i+1]) for i in range(len(values))]
            axes[0,0].plot(range(1,len(mean)+1),mean,label=str(s["seed"]))
        if val:
            x=[r["episode"] for r in val]
            axes[0,1].plot(x,[r["success_rate"] for r in val],marker=".",label=str(s["seed"]))
            axes[1,0].plot(x,[r["mean_team_return"] for r in val],marker=".",label=str(s["seed"]))
            axes[1,1].plot(x,[r["collision_rate"] for r in val],marker=".",label=str(s["seed"]))
    for a,title in zip(axes.flat,("Training return, trailing 20 episodes","Validation success rate",
                                "Validation task return","Validation collision rate")):
        a.set_title(title);a.set_xlabel("Completed training episodes");a.grid(alpha=.3)
        if a.lines:a.legend()
    fig.tight_layout()
    name=root/"reports"/("curves_"+uuid.uuid4().hex[:8]+".png")
    name.parent.mkdir(exist_ok=True)
    fig.savefig(name,dpi=160);plt.close(fig)
    return str(name)


def freeze(root,setup,spec,data):
    require(data["development_ready_to_freeze"], "development stability and capability gate not met")
    for s in data["seeds"]:
        d=root/f"seed_{s['seed']}"
        best=selection(s["validation"][-spec["stable_validation_points"]:])
        publish(d/"freeze.json",dict(episode=best["episode"],checkpoint=best["checkpoint"],
                                     source=file_sha(SOURCE),protocol=file_sha(PROTOCOL)))
    print("All policies frozen before opening the 100 admission scenes.",flush=True)


def admission(root,setup,spec,job):
    for seed in spec["training_seeds"]:
        require((root/f"seed_{seed}"/"freeze.json").is_file(),"all three policies must be frozen first")
    results=[]
    for seed in spec["training_seeds"]:
        d=root/f"seed_{seed}"
        with exclusive_job(d):
            selected=read_json(d/"freeze.json")
            require(selected["source"]==file_sha(SOURCE) and selected["protocol"]==file_sha(PROTOCOL),"freeze identity differs")
            model=Baseline(d,seed,setup,spec,job)
            model.restore(d/"boundaries"/f"{selected['episode']:04d}")
            require(file_sha(d/"boundaries"/f"{selected['episode']:04d}"/"state.pt")==selected["checkpoint"],"frozen bytes differ")
            result=evaluate(model,"baseline_admission")
            results.append(dict(seed=seed,passed=result["success_count"]>=spec["admission_min_successes"]
                         and result["collision_count"]<=spec["admission_max_collisions"],metrics=result))
    result=dict(passed=all(x["passed"] for x in results),seeds=results)
    publish(root/"admission.json",result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("check","train","report","freeze","admit"))
    parser.add_argument("--root",type=Path,default=OUTPUT)
    parser.add_argument("--until",type=int,choices=(200,500,1000),default=200)
    parser.add_argument("--seed",type=int,choices=(2729,3733,4751))
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    setup,spec=checked_setup()
    if args.command=="check":
        print(json.dumps(dict(protocol=spec,mode="fresh_initialization",source=file_sha(SOURCE),
                              task_rollouts=0,training_updates=0),indent=2))
        return
    root=args.root.resolve()
    if args.command in ("train","freeze","admit"):
        require(args.execute and platform.system()=="Linux","Linux and --execute required")
    with exclusive_job(root):
        if args.command=="train" and not (root/"run.json").exists():
            atomic_json(root/"run.json",dict(id=uuid.uuid4().hex,source=file_sha(SOURCE),protocol=file_sha(PROTOCOL)))
        job=job_identity(root)
    if args.command=="train":
        for seed in ([args.seed] if args.seed else spec["training_seeds"]):
            training(root/f"seed_{seed}",seed,setup,spec,job,args.until)
    if args.command=="admit":
        admission(root,setup,spec,job)
        return
    if args.command=="freeze":
        with exclusive_job(root),ExitStack() as locks:
            for seed in spec["training_seeds"]:
                locks.enter_context(exclusive_job(root/f"seed_{seed}"))
            data=report(root,setup,spec)
            freeze(root,setup,spec,data)
    else:
        data=report(root,setup,spec)
    filename=root/"reports"/("report_"+uuid.uuid4().hex[:8]+".json")
    atomic_json(filename,data)
    chart=curves(root,data)
    for s in data["seeds"]:
        print(f"seed={s['seed']} completed={s['last_boundary']} {s['stability']}")
    print(f"Development ready to freeze: {data['development_ready_to_freeze']}; report={filename}; curves={chart}")


if __name__=="__main__":
    main()
