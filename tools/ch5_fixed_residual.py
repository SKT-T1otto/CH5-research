"""Paired frozen-Actor versus per-hull constant residual diagnostic for pilot 2729."""
import argparse
from collections import Counter
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from chapter5_comm.experiment_state import atomic_json, exclusive_job
from chapter5_comm.nominal_maddpg.checkpoint import read_boundary
from chapter5_comm.nominal_maddpg.config import ROOT, digest, file_sha, read_json
from chapter5_comm.nominal_maddpg.networks import SingleCriticNetworks
from chapter5_comm.nominal_maddpg.replay import NominalReplay
from chapter5_comm.nominal_maddpg.runtime import NominalRuntime
from tools.ch5_nominal_pilot import DEFAULT_ROOT, check_run, publish, scene_row, score
from tools.ch5_nominal_pilot_train import PROTOCOL, SOURCE, episode_result, preflight, require, seed_process

SEED = 2729
ANCHOR = 4
RETURN_ATOL = 1e-7
OUTPUT = ROOT / "outputs/chapter5/audits/fixed_residual_2729_ep32"
# Identify the two published pilot evaluations by checkpoint and ordered scene receipts.
REFERENCES = {
    0: ("5ba68eb27c0b608eec5f3daba962d50f903eb714c57d12819abc33d708924c1f",
        "fa62c649680c8124f525a2b9c3f83759b732ee055158777d6b059d269c9ac45b"),
    32: ("b8d961f0d48bbf2c20708dd81c98ae9ee2e648febb37dbdb80916a806d221a17",
         "50a655f449da590b48137960019958a5554af301a77e92b36c55c3f952469763"),
}


class FixedResidual:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        require(self.values.shape == (4, 3) and np.isfinite(self.values).all()
                and (np.abs(self.values) <= 1).all(), "invalid normalized per-hull residuals")

    def step(self, observations, *, explore=False):
        require(not explore, "the diagnostic uses deterministic actions")
        return self.values.copy()


@torch.no_grad()
def calibrate(networks, replay, steps):
    base = replay["base"]
    require(steps > 0 and steps == base["filled_i"] == base["next_idx"], "training replay cursor differs")
    gates = replay["enabled"]
    require(gates.shape == (steps, 4) and gates.dtype == torch.bool
            and replay["gate_initialized"].shape == (steps,) and bool(replay["gate_initialized"].all()),
            "missing replay execution gates")
    obs = torch.stack(base["obs_buffs"], dim=1)
    require(obs.shape == (steps, 4, 28), "invalid local observation storage")
    actions = torch.cat([networks.actions(chunk) for chunk in obs.split(512)]).double()
    require(bool(torch.isfinite(actions).all()) and not bool((actions.abs() > 1).any()), "invalid frozen actions")
    counts = gates.sum(0)
    require(bool((counts > 0).all()), "each hull needs an executed training decision")
    mean = (actions * gates.unsqueeze(-1)).sum(0) / counts[:, None]
    rms = ((actions - mean).square() * gates.unsqueeze(-1)).sum(0).div(counts[:, None]).sqrt()
    return dict(training_team_steps=steps, open_steps=counts.tolist(), per_hull_xyz=mean.float().tolist(),
                rms_deviation_by_axis=rms.tolist())


def archived_rows(seed_root, entries, episode):
    checkpoint, rows_hash = REFERENCES[episode]
    folder = seed_root/"validation"/f"{episode:04d}"
    rows = [scene_row(folder/f"scene_{i:02d}.json", checkpoint, item) for i,item in enumerate(entries)]
    receipt = read_json(folder/"verified.json")
    require(digest(rows) == rows_hash == receipt["scene_rows_sha256"] and receipt["episode"] == episode
            and receipt["checkpoint"] == checkpoint and receipt["restore_verified"] and receipt["eval_updates"] == 0
            and all(receipt[k] == v for k,v in score(rows).items()), "archived validation set differs")
    return rows


def prepare(pilot_root):
    config, _, groups, _ = preflight()
    run_id = check_run(pilot_root)
    seed_root = pilot_root/f"seed_{SEED}"
    path = seed_root/"boundaries/0032"
    marker = read_json(path/"committed.json")
    identity = marker["identity"]
    require(marker["sha256"] == REFERENCES[32][0] and identity["run"] == run_id and identity["seed"] == SEED
            and identity["source"] == file_sha(SOURCE) and identity["protocol"] == file_sha(PROTOCOL),
            "original 2729/0032 pilot checkpoint required")
    state = read_boundary(seed_root, path, identity)
    require(state["cursor"] == len(state["episodes"]) == 32 and
            sum(r["steps"] for r in state["episodes"]) == state["team_steps"], "training cutoff differs")
    for idx,row in enumerate(state["episodes"]):
        require(row["fingerprint"] == groups["train"][int(state["order"][idx])]["physical_fingerprint"],
                "non-training source in calibration")
    networks = SingleCriticNetworks(config, SEED).eval().requires_grad_(False)
    networks.load_state_dict(state["learner"]["networks"], strict=True)
    calibration = calibrate(networks, state["replay"], state["team_steps"])
    entries = groups["validation"]
    reference = {n:archived_rows(seed_root,entries,n) for n in REFERENCES}
    manifest = dict(schema="ch5.fixed_residual.2729.32.v1", pilot_run=run_id, checkpoint=marker["sha256"],
                    source_sha256=file_sha(Path(__file__)), pilot_source=identity["source"],
                    python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                    platform=platform.platform(), cpu_threads=torch.get_num_threads(),
                    reference_rows_sha256={str(k):v[1] for k,v in REFERENCES.items()},
                    calibration=calibration, anchor_seed=entries[ANCHOR]["physical_seed"])
    return config, entries, reference, networks, manifest


@torch.no_grad()
def rollout(config, entry, index, policy):
    seed_process(entry["physical_seed"])
    rt = NominalRuntime(config, entry["scene"], entry["physical_seed"], NominalReplay(1,0), episode_id=index)
    reward = 0.
    try:
        while not rt.core.terminal:
            tr = rt.advance(policy)
            reward += float(tr["rewards"][0])
            if isinstance(policy, FixedResidual):
                expected = policy.values * tr["action_enabled"][:, None]
                require(np.array_equal(expected, tr["actions"]),
                        f"applied fixed residual differs on scene {index}, step {tr['t']}")
        return episode_result(rt, entry, reward)
    finally:
        rt.close()


def compare_anchor(expected, observed):
    require(expected.keys() == observed.keys(), "Actor reproduction fields differ")
    for field, value in expected.items():
        if field == "team_return" and abs(observed[field] - value) <= RETURN_ATOL:
            continue
        if observed[field] != value:
            raise ValueError(f"Actor reproduction diverged on {expected['physical_seed']}: {field}; "
                             f"archived={value!r}, current={observed[field]!r}")


def paired_report(baseline, actor, fixed):
    require(len(baseline) == len(actor) == len(fixed) > 0, "unequal paired sets")
    pairs = []
    gain_seeds, retained = [], []
    actor_only, fixed_only = [], []
    flips = Counter()
    for b,a,f in zip(baseline, actor, fixed):
        seed = b["physical_seed"]
        require(seed == a["physical_seed"] == f["physical_seed"] and
                b["fingerprint"] == a["fingerprint"] == f["fingerprint"], "unpaired geometry")
        if b["outcome"] != "success" and a["outcome"] == "success":
            gain_seeds.append(seed)
            if f["outcome"] == "success":
                retained.append(seed)
        if a["outcome"] == "success" and f["outcome"] != "success":
            actor_only.append(seed)
        if f["outcome"] == "success" and a["outcome"] != "success":
            fixed_only.append(seed)
        flips[a["outcome"],f["outcome"]] += 1
        pairs.append(dict(physical_seed=seed, initialized=b, actor=a, fixed=f))
    require(len({r["physical_seed"] for r in pairs}) == len(pairs), "duplicate paired scene")
    n = len(retained)
    interpretation = ("majority_reproduced" if n*2 > len(gain_seeds) else
                      "exactly_half_reproduced" if n*2 == len(gain_seeds) else "minority_reproduced")
    if not gain_seeds:
        interpretation = "no_gains_to_compare"
    return dict(initialized=score(baseline), actor=score(actor), fixed=score(fixed),
                actor_new_successes=gain_seeds, fixed_retained_new_successes=retained,
                retained_count=n, new_success_count=len(gain_seeds), interpretation=interpretation,
                actor_only_successes=actor_only, fixed_only_successes=fixed_only,
                actor_to_fixed_counts=[dict(actor=a,fixed=f,count=count) for (a,f),count in sorted(flips.items())],
                scenes=pairs)


def run_pairs(out, config, entries, reference, networks, manifest):
    fixed = FixedResidual(manifest["calibration"]["per_hull_xyz"])
    tag = digest(manifest)
    with exclusive_job(out):
        publish(out/"run.json", manifest)
        anchor_path = out/"actor_anchor.json"
        expected = reference[32][ANCHOR]
        if not anchor_path.exists():
            print(f"Checking archived Actor on {expected['physical_seed']} ...", flush=True)
            observed = rollout(config, entries[ANCHOR], ANCHOR, networks)
            compare_anchor(expected, observed)
            atomic_json(anchor_path,dict(checkpoint=tag, row=observed, sha256=digest(observed)))
        observed_anchor = scene_row(anchor_path,tag,entries[ANCHOR])
        compare_anchor(expected,observed_anchor)
        rows = []
        for i,entry in enumerate(entries):
            path = out/"fixed"/f"scene_{i:02d}.json"
            if not path.exists():
                row = rollout(config, entry, i, fixed)
                atomic_json(path,dict(checkpoint=tag, row=row, sha256=digest(row)))
            row = scene_row(path,tag,entry)
            rows.append(row)
            print(f"{i+1:02d}/{len(entries)}  seed={row['physical_seed']}  "
                  f"Actor={reference[32][i]['outcome']}  fixed={row['outcome']}", flush=True)
        report = paired_report(reference[0],reference[32],rows)
        report.update(run_sha256=tag, calibration=manifest["calibration"], actor_anchor_passed=True,
                      optimizer_updates=0, anchor_env_steps=expected["steps"],
                      anchor_return_abs_difference=abs(observed_anchor["team_return"]-expected["team_return"]),
                      anchor_return_abs_tolerance=RETURN_ATOL)
        publish(out/"report.json",report)
        print(f"Fixed successes: {report['fixed']['success_count']}/{len(rows)}; "
              f"retained new successes: {report['retained_count']}/{report['new_success_count']}; "
              f"{report['interpretation']}. Full pairs: {out/'report.json'}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check","run"))
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, out = args.pilot_root.resolve(), args.output.resolve()
    require(root != out and root not in out.parents and out not in root.parents,
            "diagnostic output must be separate from pilot artifacts")
    config, entries, reference, networks, manifest = prepare(root)
    print(f"Calibration: {manifest['calibration']}",flush=True)
    if args.command == "run":
        run_pairs(out,config,entries,reference,networks,manifest)
    else:
        print(f"Verified {len(entries)} matched archived scenes at both checkpoints. No rollouts started.")


if __name__ == "__main__":
    main()
