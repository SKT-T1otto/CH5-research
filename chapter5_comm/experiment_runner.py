"""Manual staged experiments. Original task/reward; full episodes in research presets."""
from copy import deepcopy
from pathlib import Path
import json
import platform
import time
import uuid
import sys
import os
from importlib import metadata

import torch
import numpy as np

from .pipeline import (PROJECT, auxiliary_batch, new_output, run_episode, scenarios,
                       task_model, write_json, evaluation_summary)
from .training import (AuxiliaryTrainer, CommunicationLearner, TaskReplayUpdater,
                       build_neural_models, freeze_auxiliary)
from .experiment_config import (canonical, read_experiment, source_identity, arm_settings,
                                scene_seed, plan_summary)
from .experiment_state import (append_json, exclusive_job, latest_receipt, load_state,
                               save_state, load_replay, restore_rng, atomic_json)


def prepare(config_path, root):
    c, base = read_experiment(config_path)
    root = new_output(root)
    write_json(root / "plan.json", dict(config=c, base=base, config_sha=canonical(c),
        source=source_identity(), budget=plan_summary(c), created_on=platform.platform(),
        git_commit=_git_commit(), formal_performance_verified=False))
    write_json(root / "environment.json", environment())
    return dict(prepared=str(root), **plan_summary(c))


def environment(device=None):
    return dict(platform=platform.platform(), system=platform.system(), python=sys.version,
        numpy=str(np.__version__), torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
        cuda_available=torch.cuda.is_available(), requested_device=device,
        cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        distributions={d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]},
        threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_plan(root):
    root = Path(root).resolve()
    if not root.is_relative_to(PROJECT / "outputs/chapter5"):
        raise ValueError("run must belong to outputs/chapter5 in this checkout")
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan["config_sha"] != canonical(plan["config"]) or plan["source"] != source_identity():
        raise ValueError("frozen plan/source mismatch; start a new run after changing code/config")
    # Revalidate the frozen config, including nested values, without writing it.
    original = PROJECT / "configs/chapter5" / plan["config"]["config_name"]
    c, base = read_experiment(original)
    if c != plan["config"] or base != plan["base"]:
        raise ValueError("preset differs from the frozen plan")
    return root, plan


def device_name(request):
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if request not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda or auto")
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no silent CPU fallback")
    return request


def execution_guard(c):
    if c["kind"] != "check" and platform.system() != "Linux":
        raise ValueError("pilot/formal execution is reserved for manual Linux runs; use check locally")


def identity_for(plan, seed, arm):
    if seed not in plan["config"]["seeds"] or arm not in plan["config"]["arms"]:
        raise ValueError("seed/arm absent from frozen experiment plan")
    return dict(config_sha=plan["config_sha"], seed=seed, arm=arm)


def job_path(root, seed, arm):
    return root / "jobs" / f"seed_{seed}" / arm


def build_objects(c, base, seed, arm, device):
    neural, losses, mode = arm_settings(c, arm)
    model = task_model(base, seed, device=device)
    networks = build_neural_models(neural, device=device)
    auxiliary = AuxiliaryTrainer(networks, max_updates=c["auxiliary_updates"], experiment=True, **losses)
    communication = CommunicationLearner(networks.scheduler, 44, gamma=base["rl"]["gamma"],
        max_updates=c["communication_episodes"], experiment=True)
    maximum = c["task_episodes"] * c["rollout_limit_steps"] // c["task_update_frequency"] * c["task_updates_per_trigger"]
    task = TaskReplayUpdater(model, capacity=c["replay_capacity"], batch_size=c["task_batch_size"],
        minimum_batch_size=c["task_batch_size"], max_updates=maximum, experiment=True, device=device)
    return networks, model, auxiliary, communication, task, mode


def restore_objects(payload, objects, device):
    n, model, auxiliary, comm, task, _ = objects
    n.load_state_dict(payload["networks"], strict=True)
    model.load_training_state_dict(payload["model"])
    model.prep_rollouts(device=device)
    for agent, state in zip(model.agents, payload["exploration"]):
        agent.noise.state.copy_(state["state"].to(device))
        agent.noise.sigma = state["sigma"]
        agent.current_sigma = state["current_sigma"]
    auxiliary.optimizer.load_state_dict(payload["auxiliary"]["optimizer"])
    auxiliary.updates = payload["auxiliary"]["updates"]
    auxiliary.history = payload["auxiliary"]["history"]
    comm.critic.load_state_dict(payload["communication"]["critic"])
    comm.optimizer.load_state_dict(payload["communication"]["optimizer"])
    comm.updates = payload["communication"]["updates"]
    comm.policy_version = payload["communication"]["policy_version"]
    comm.history = payload["communication"]["history"]
    task.updates = payload["task"]["updates"]
    task.history = payload["task"]["history"]
    load_replay(task.replay, payload["task"]["replay"])
    # Optimizer restoration maps momentum tensors to their parameter device.
    for optimizer in (auxiliary.optimizer, comm.optimizer):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
    restore_rng(payload["rng"])


class OnlineTaskUpdates:
    def __init__(self, task, c, log, clock=0):
        self.task, self.c, self.log, self.clock = task, c, log, clock

    def observe(self, transition):
        self.task.observe(transition)
        self.clock += 1
        if len(self.task.replay) < self.c["task_warmup"] or self.clock % self.c["task_update_frequency"]:
            return
        for _ in range(self.c["task_updates_per_trigger"]):
            if self.task.updates >= self.task.max_updates:
                return
            row = self.task.update()
            append_json(self.log, dict(stage="task_update", clock=self.clock, **row))
            self.task.history[:] = self.task.history[-100:]


def episode_config(c, channel=None):
    return dict(rollout_limit_steps=c["rollout_limit_steps"], label_horizon=c["label_horizon"],
                neural_cache_steps=32, channel=deepcopy(c["channel"] if channel is None else channel))


def factual_scene(seed, split, index):
    # Native generator is left untouched; split namespaces are explicit even
    # though the inherited generator only names train/validation splits.
    value = scene_seed(seed, split, index)
    return scenarios(1, value, evaluation=split in ("validation", "test"))[0]


def episode(c, base, objects, *, seed, split, index, directory, mode, labels=False,
            updater=None, channel=None, scene=None, stochastic=False):
    n, model, _, learner, _, _ = objects
    scene = factual_scene(seed, split, index) if scene is None else scene
    case_seed = scene_seed(seed, split, index) % (2 ** 31 - 1)
    started = time.perf_counter()
    result = run_episode(episode_config(c, channel), base, scene, n, model, seed=case_seed,
        mode=mode, episode_id=f"{split}_{index}", output=directory, learner=learner,
        task_updater=updater, record_labels=labels, explore=stochastic,
        task_explore=stochastic and split in ("collection", "task"), save_details=c["save_episode_details"])
    result["summary"].update(wall_seconds=time.perf_counter() - started, scenario_generator_seed=scene.get("scenario_seed"))
    write_json(directory / "experiment_episode.json", result["summary"])
    return result


def train(root, seed, arm, *, device="cpu", resume=False, stop_after=None):
    root, plan = load_plan(root)
    c, base = plan["config"], plan["base"]
    execution_guard(c)
    device = device_name(device)
    identity = identity_for(plan, seed, arm)
    job = job_path(root, seed, arm)
    with exclusive_job(job):
        receipt = latest_receipt(job)
        if receipt and not resume:
            raise FileExistsError("job has checkpoints; use --resume or a fresh experiment root")
        if receipt and receipt["progress"]["complete"]:
            load_state(job, receipt["path"], identity)
            return dict(status="already_trained", seed=seed, arm=arm)
        attempt = job / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        log = attempt / "events.jsonl"
        write_json(attempt / "environment.json", environment(device))
        objects = build_objects(c, base, seed, arm, device)
        n, model, auxiliary, learner, task, final_mode = objects
        progress = dict(collection=0, auxiliary=0, task=0, communication=0, serial=0,
            collection_paths=[], best_score=None, best_serial=None, complete=False)
        clock = 0
        if receipt:
            payload = load_state(job, receipt["path"], identity)
            restore_objects(payload, objects, device)
            progress, clock = payload["progress"], payload["task"]["clock"]
        updater = OnlineTaskUpdates(task, c, log, clock)

        def checkpoint(score=None):
            progress["serial"] += 1
            if score is not None and (progress["best_score"] is None or score > progress["best_score"]):
                progress.update(best_score=score, best_serial=progress["serial"])
            return save_state(job, identity=identity, progress=deepcopy(progress), networks=n,
                model=model, auxiliary=auxiliary, communication=learner, task=task, task_clock=updater.clock)

        try:
            for i in range(progress["collection"], c["collection_episodes"]):
                scene = None
                if i < len(c["collection_mechanism_seeds"]):
                    scene = scenarios(1, c["collection_mechanism_seeds"][i])[0]
                result = episode(c, base, objects, seed=seed, split="collection", index=i,
                    directory=attempt / f"collection_{i}", mode="neural_off" if arm == "raw_only" else "fixed",
                    labels=arm != "raw_only", scene=scene, stochastic=True)
                append_json(log, dict(stage="collection", **result["summary"]))
                if arm != "raw_only":
                    progress["collection_paths"].append((attempt / f"collection_{i}" / "labels.jsonl").relative_to(job).as_posix())
                progress["collection"] = i + 1
                checkpoint()
            if stop_after == "collection":
                return dict(status="stopped_at_collection_boundary", **progress)
            examples = []
            if arm != "raw_only":
                for relative in progress["collection_paths"]:
                    examples.extend(json.loads(line) for line in (job / relative).read_text(encoding="utf-8").splitlines())
                positive = sum(sum(y * valid for y, valid in zip(x["route_labels"], x["route_valid"])) for x in examples)
                states = sum(bool(x["dynamic_target_valid"] and x["inputs"]["state_observable"]) for x in examples)
                write_json(attempt / "label_coverage.json", dict(examples=len(examples), positive_routes=positive, observable_states=states))
                if not examples or positive < c["minimum_positive_routes"] or states < c["minimum_state_labels"]:
                    raise ValueError("insufficient factual label coverage; retain run and revise a NEW collection plan")
                for i in range(progress["auxiliary"], c["auxiliary_updates"]):
                    freeze_auxiliary(n, False)
                    indices = np.random.choice(len(examples), min(c["auxiliary_batch_size"], len(examples)), replace=False)
                    row = auxiliary.step(auxiliary_batch([examples[j] for j in indices]))
                    if row["skipped"]:
                        raise ValueError("sampled batch has no supervision; no silent progress increment")
                    append_json(log, dict(stage="auxiliary", **row))
                    auxiliary.history[:] = auxiliary.history[-100:]
                    progress["auxiliary"] = i + 1
                    if (i + 1) % c["auxiliary_checkpoint_interval"] == 0 or i + 1 == c["auxiliary_updates"]:
                        checkpoint()
            if stop_after == "auxiliary":
                return dict(status="stopped_at_auxiliary_boundary", **progress)
            freeze_auxiliary(n, True)
            examples.clear()
            for i in range(progress["task"], c["task_episodes"]):
                result = episode(c, base, objects, seed=seed, split="task", index=i,
                    directory=attempt / f"task_{i}", mode="neural_off" if arm == "raw_only" else "fixed",
                    updater=updater, stochastic=True)
                append_json(log, dict(stage="task", **result["summary"]))
                progress["task"] = i + 1
                if (i + 1) % c["checkpoint_interval"] == 0 or i + 1 == c["task_episodes"]:
                    checkpoint()
            if stop_after == "task":
                return dict(status="stopped_at_task_boundary", **progress)
            # Task actor remains deterministic and is not updated in this stage.
            for i in range(progress["communication"], c["communication_episodes"]):
                result = episode(c, base, objects, seed=seed, split="communication", index=i,
                    directory=attempt / f"communication_{i}", mode=final_mode, stochastic=True)
                if final_mode == "learned":
                    if not result["events"]:
                        raise ValueError("no complete SMDP decision intervals")
                    append_json(log, dict(stage="communication_update", **learner.update(result["events"])))
                    learner.history[:] = learner.history[-100:]
                append_json(log, dict(stage="communication", **result["summary"]))
                progress["communication"] = i + 1
                final = i + 1 == c["communication_episodes"]
                score = None
                if (i + 1) % c["validation_interval"] == 0 or final:
                    from .experiment_state import rng_state
                    saved_rng = rng_state()
                    rows = []
                    for v in range(c["validation_episodes"]):
                        value = episode(c, base, objects, seed=seed, split="validation", index=v,
                            directory=attempt / f"validation_{i}_{v}", mode=final_mode)["summary"]
                        rows.append(value)
                    restore_rng(saved_rng)
                    score = float(np.mean([x["observed_return"] for x in rows]))
                    append_json(log, dict(stage="validation", communication_episode=i + 1, mean_return=score, episodes=rows))
                progress["complete"] = final
                if score is not None or (i + 1) % c["checkpoint_interval"] == 0:
                    checkpoint(score)
                print(json.dumps(dict(seed=seed, arm=arm, stage="communication", episode=i + 1,
                    return_value=result["summary"]["observed_return"], validation=score)), flush=True)
            return dict(status="trained", seed=seed, arm=arm, **progress)
        except BaseException as error:
            write_json(attempt / "failure.json", dict(error=type(error).__name__, message=str(error),
                last_committed=latest_receipt(job), attempted_progress=progress))
            raise


def evaluate(root, seed, arm, *, device="cpu"):
    root, plan = load_plan(root)
    c, base = plan["config"], plan["base"]
    execution_guard(c)
    identity = identity_for(plan, seed, arm)
    job = job_path(root, seed, arm)
    device = device_name(device)
    with exclusive_job(job):
        latest = latest_receipt(job)
        if not latest or not latest["progress"]["complete"]:
            raise ValueError("training incomplete; test does not select an interim checkpoint")
        best = latest["progress"]["best_serial"]
        choices = [json.loads(p.read_text()) for p in (job / "checkpoints").glob("*.receipt.json")]
        selected = next(x for x in choices if x["progress"]["serial"] == best)
        payload = load_state(job, selected["path"], identity)
        objects = build_objects(c, base, seed, arm, device)
        restore_objects(payload, objects, device)
        freeze_auxiliary(objects[0], True)
        for condition, patch in c["evaluation_channels"].items():
            done = job / "evaluation" / condition / "result.json"
            if done.exists():
                old = json.loads(done.read_text())
                if old["checkpoint_sha256"] != selected["sha256"]:
                    raise ValueError("evaluation checkpoint identity changed")
                continue
            attempt = done.parent / uuid.uuid4().hex
            rows = []
            for i in range(c["evaluation_episodes"]):
                row = episode(c, base, objects, seed=seed, split="test", index=i,
                    directory=attempt / f"episode_{i}", mode=objects[-1],
                    channel=dict(c["channel"], **patch))["summary"]
                rows.append(row)
                append_json(attempt / "episodes.jsonl", row)
            # Unique completed-condition marker. Partial attempts remain intact.
            atomic_json(done, dict(schema="ch5.experiment.evaluation.v1", seed=seed, arm=arm,
                condition=condition, config_sha=plan["config_sha"], checkpoint_sha256=selected["sha256"],
                episode_source=(attempt / "episodes.jsonl").relative_to(job).as_posix(),
                summary=evaluation_summary(rows), rows=rows,
                information_regime="costed all-channel TDMA", fault_pairing="same scenes/seeds; packet RNG is decision-dependent",
                role="information-rich reference" if condition == "reliable_reference" else "restricted-channel test"))
        return dict(status="evaluated", seed=seed, arm=arm, conditions=list(c["evaluation_channels"]))
