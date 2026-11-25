"""Training and collection for the registered Chapter 5 pilot."""
import math
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter5_comm.experiment_state import load_replay, replay_state
from chapter5_comm.nominal_maddpg import checkpoint as cp
from chapter5_comm.nominal_maddpg.config import DEFAULT_CONFIG, ROOT, digest, file_sha, load, local_path, read_json
from chapter5_comm.nominal_maddpg.exploration import OUNoise
from chapter5_comm.nominal_maddpg.learner import SingleCriticLearner
from chapter5_comm.nominal_maddpg.provenance import MANIFEST, DIRECTORY
from chapter5_comm.nominal_maddpg.replay import NominalReplay
from chapter5_comm.nominal_maddpg.runtime import NominalRuntime
from chapter5_comm.nominal_maddpg.scenarios import validate_registry

REGISTRATION = ROOT / "provenance/ch5_nominal_pilot"
PROTOCOL = REGISTRATION / "protocol.json"
SOURCE = REGISTRATION / "source.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight():
    config, protocol = load(), read_json(PROTOCOL)
    source, base = read_json(SOURCE), read_json(MANIFEST)
    require(file_sha(MANIFEST) == source["parent_sha256"], "source registration parent differs")
    # The legacy registration launcher is not invoked by the pilot.
    for name, expected in base["files"].items():
        if name != "chapter5_comm/nominal_maddpg/__main__.py":
            require(file_sha(local_path(name)) == expected["sha256"], "registered file differs: " + name)
    for name, checksum in source["files"].items():
        require(file_sha(local_path(name)) == checksum, "pilot file differs: " + name)
    archive = read_json(DIRECTORY/"ch4_reference.json")["files"]["configs/ch4/nominal_maddpg.json"]
    ch4 = read_json(local_path(archive["archived_path"]))["learning"]["exploration"]
    require(ch4 == config["learning"]["exploration"] and ch4["evaluation_sigma"] == 0, "OU settings differ")
    require(all(protocol["exploration"][k] == ch4[k] for k in ("initial_sigma","minimum_sigma",
            "decay_per_environment_step")), "OU schedule differs")
    require(protocol["training_seeds"] == config["training_seeds"], "seed registration differs")
    registry = read_json(local_path(config["scene_manifest"]))
    checked = validate_registry(config, registry, DEFAULT_CONFIG)
    return config, protocol, registry["groups"], checked


def rounds_due(steps, learning):
    threshold = max(learning["warmup_steps"], learning["batch_size"])
    freq = learning["update_frequency"]
    triggers = max(0, steps//freq - (threshold-1)//freq)
    return triggers * learning["updates_per_train"]


def episode_result(runtime, entry, observed_return):
    summary = runtime.core.summary()
    outcome, steps = summary["termination_reason"], summary["actual_length"]
    value = summary["team_undiscounted_return"]
    require(outcome in ("success","obstacle_collision","timeout") and summary["episode_terminated"]
            and summary["success"] == (outcome == "success")
            and summary["collision_episode"] == (outcome == "obstacle_collision"),
            "unexpected original task termination")
    require(math.isfinite(value) and value == observed_return, "original team return disagrees with replay rewards")
    return dict(physical_seed=entry["physical_seed"], fingerprint=entry["physical_fingerprint"],
                steps=steps, outcome=outcome, team_return=value, found_step=summary["found_step"],
                handoff_step=summary["handoff_event_step"], capture_steps=summary["capture_contact_steps"],
                tx_bits=summary["channel"]["sent_bits"], rx_bits=summary["channel"]["delivered_bits"])


class Pilot:
    def __init__(self, root, seed, config, protocol, groups, run_id):
        self.root, self.config, self.protocol, self.groups = Path(root), config, protocol, groups
        require(seed in protocol["training_seeds"] and torch.get_num_threads() == 1, "unregistered seed or CPU threads")
        self.identity = dict(schema=protocol["schema"], run=run_id, seed=seed,
            source=file_sha(SOURCE), protocol=file_sha(PROTOCOL), python=sys.version, torch=str(torch.__version__),
            numpy=np.__version__, platform=platform.system(), machine=platform.machine(), threads=1, device="cpu")
        self.learning = config["learning"]
        self.order_rng = torch.Generator().manual_seed(seed+protocol["order_seed_offset"])
        self.order = torch.randperm(len(groups["train"]), generator=self.order_rng)
        self.replay = NominalReplay(self.learning["replay_capacity"], seed+protocol["sampling_seed_offset"])
        self.noise = OUNoise(seed+protocol["exploration_seed_offset"], **protocol["exploration"])
        self.learner = SingleCriticLearner(config, seed)
        self.episodes = []
        self.team_steps = 0
        self.boundary = True

    def step(self, observations, *, explore=False):
        base = self.learner.networks.step(observations)
        noise, _ = self.noise.sample()
        return base + noise

    def train_one(self):
        index = len(self.episodes)
        require(self.boundary and index < self.protocol["episodes_per_seed"], "pilot episode budget exhausted")
        self.boundary = False
        self.noise.reset_episode()
        entry = self.groups["train"][int(self.order[index])]
        rt = NominalRuntime(self.config, entry["scene"], entry["physical_seed"], self.replay, episode_id=index)
        observed_return = critic_mse = 0.
        open_steps = np.zeros(4, dtype=int)
        saturated = np.zeros(4, dtype=int)
        action_abs = np.zeros(4)
        initial_round = self.learner.rounds
        try:
            while not rt.core.terminal:
                tr = rt.advance(self)
                self.team_steps += 1
                observed_return += float(tr["rewards"][0])
                open_steps += tr["action_enabled"]
                saturated += ((abs(tr["actions"]) >= 0.999).any(axis=1) & tr["action_enabled"])
                action_abs += abs(tr["actions"]).mean(axis=1)
                require(len(self.replay) == self.team_steps, "replay has missing or extra environment rows")
                for _ in range(rounds_due(self.team_steps,self.learning)-self.learner.rounds):
                    batch = self.replay.sample(self.learning["batch_size"])
                    require(batch[6].numel() == self.learning["batch_size"], "short replay batch")
                    update = self.learner.update(batch)
                    critic_mse += sum(v["loss"] for v in update["critics"])/4
            row = episode_result(rt,entry,observed_return)
        finally:
            rt.close()
        updates = self.learner.rounds - initial_round
        row.update(episode=index+1, updates=updates, sigma_next=self.noise.sigma,
                   open_gate_steps=open_steps.tolist(), saturated_steps=saturated.tolist(),
                   sum_mean_abs_applied_residual=action_abs.tolist(), mean_critic_mse=critic_mse/updates if updates else None)
        self.episodes.append(row)
        self.boundary = True

    def snapshot(self):
        require(self.boundary, "checkpoint requires completed episode")
        state = dict(episodes=list(self.episodes), team_steps=self.team_steps, cursor=len(self.episodes),
            order=self.order.clone(), order_rng=self.order_rng.get_state(),
            learner=cp.learner_state(self.learner), replay=replay_state(self.replay),
            sampling_rng=self.replay.generator.get_state(), exploration=self.noise.state_dict(),
            global_rng=cp.global_rng_state())
        self.validate(state)
        return state

    def validate(self, state):
        rows, steps = state["episodes"], state["team_steps"]
        require(state["cursor"] == len(rows) <= self.protocol["episodes_per_seed"]
                and torch.equal(state["order"], self.order), "training scene order or cursor differs")
        total = 0
        for i, row in enumerate(rows):
            entry = self.groups["train"][int(self.order[i])]
            require(row["episode"] == i+1 and entry["physical_fingerprint"] == row["fingerprint"]
                    and entry["physical_seed"] == row["physical_seed"] and 1 <= row["steps"] <= 400,
                    "training scene or length differs")
            prev = total
            total += row["steps"]
            require(row["updates"] == rounds_due(total,self.learning)-rounds_due(prev,self.learning),
                    "updates differ from registered budget")
        replay = state["replay"]["base"]
        require(total == steps == replay["filled_i"] == replay["next_idx"] == state["exploration"]["steps"]
                == state["exploration"]["draws"] and steps <= self.protocol["max_train_team_steps_per_seed"],
                "training / exploration / replay counters disagree")
        counts = state["learner"]["counters"]
        rounds = rounds_due(steps,self.learning)
        require(counts["rounds"] == counts["target_updates"] == rounds and counts["critic_steps"] == [rounds]*4
                and all(a+s==rounds for a,s in zip(counts["actor_steps"],counts["actor_skips"])),
                "optimizer counters disagree")

    def restore(self, boundary):
        state = cp.read_boundary(self.root, boundary, self.identity)
        self.validate(state)
        cp.restore_learner(self.learner, state["learner"])
        load_replay(self.replay, state["replay"])
        self.replay.generator.set_state(state["sampling_rng"])
        self.order_rng.set_state(state["order_rng"])
        self.noise.load_state_dict(state["exploration"])
        self.team_steps, self.episodes = state["team_steps"], state["episodes"]
        cp.restore_global_rng(state["global_rng"])
        self.boundary = True
        require(state_digest(self.snapshot()) == state_digest(state), "boundary did not restore exactly")

    def validation_scene(self, index):
        entry = self.groups["validation"][index]
        rt = NominalRuntime(self.config,entry["scene"],entry["physical_seed"],NominalReplay(1,0),episode_id=index)
        observed = 0.
        try:
            with torch.no_grad():
                while not rt.core.terminal:
                    observed += float(rt.advance(self.learner.networks)["rewards"][0])
            return episode_result(rt,entry,observed)
        finally:
            rt.close()


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
