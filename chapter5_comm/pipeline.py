"""Bounded factual collection -> labels -> learners -> checkpoint -> evaluation.

The 400-step parent task is never shortened. A short rollout is an external
budget truncation with a nonterminal bootstrap, not a task failure or terminal.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import random

import torch
import numpy as np

from chapter5_comm.data import TrajectoryRecorder, write_labels
from chapter5_comm.training import (
    AuxiliaryTrainer, CommunicationLearner, TaskReplayUpdater,
    build_neural_models, freeze_auxiliary, load_checkpoint, save_checkpoint,
)

PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = "ch5.integrated.short.v1"


def plain(value):
    if torch.is_tensor(value):
        return plain(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError("requires a CH5 integrated short-test configuration")
    for name in ("collection_episodes", "communication_episodes", "evaluation_episodes"):
        if type(config[name]) is not int or not 1 <= config[name] <= 2:
            raise ValueError("this entry point permits one or two episodes per stage")
    if type(config["rollout_limit_steps"]) is not int or not 1 <= config["rollout_limit_steps"] <= 32:
        raise ValueError("external short rollout limit must be within 1..32")
    modes = config["evaluation_modes"]
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"learned", "fixed", "neural_off"}:
        raise ValueError("invalid evaluation modes")
    episodes = config["collection_episodes"] + config["communication_episodes"] + len(modes) * config["evaluation_episodes"]
    if episodes > 8:
        raise ValueError("combined short-test budget exceeds eight episodes")
    for name in ("auxiliary_updates", "task_updates"):
        if type(config[name]) is not int or not 0 <= config[name] <= 4:
            raise ValueError("short learner update limit must be within 0..4")
    seeds = config.get("collection_scenario_seeds")
    if seeds is not None and (len(seeds) != config["collection_episodes"] or
                              any(type(x) is not int or x < 0 for x in seeds)):
        raise ValueError("collection seeds must explicitly identify each factual collection case")
    base_path = (PROJECT / config["parent_config"]).resolve()
    if not base_path.is_relative_to(PROJECT / "configs"):
        raise ValueError("parent configuration must belong to this project")
    data = base_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != config["parent_config_sha256"]:
        raise ValueError("frozen third-chapter configuration changed")
    base = json.loads(data)
    if base["max_steps"] != 400 or base["rl"]["gamma"] != .95:
        raise ValueError("the inherited 400-step/.95 task contract changed")
    from chapter3_bser.learning.config import validate
    validate(base)
    return config, base


def new_output(path):
    path = Path(path).resolve()
    allowed = (PROJECT / "outputs/chapter5").resolve()
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError("new outputs must be a child of outputs/chapter5")
    if path.exists():
        raise FileExistsError(path)
    for parent in path.parents:
        if parent == allowed.parent:
            break
        if parent.exists() and parent.is_symlink():
            raise ValueError("output ancestry cannot contain links")
    path.mkdir(parents=True)
    return path


def task_model(base, seed):
    from chapter3_bser.learning.model import build_model
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = build_model(base, device="cpu")
    model.prep_rollouts(device="cpu")
    return model


def scenarios(count, seed, *, evaluation=False):
    from core.scenarios.ch3_generator_impl import build_scenario_manifests
    split = "smoke_validation" if evaluation else "smoke_train"
    result = build_scenario_manifests(count=count, generator_seed=seed, split=split,
                                     profiles=["M20_MOVING_UNKNOWN_MULTI"])
    # The parent's actual generated task instances and horizon are preserved.
    return deepcopy(result["M20_MOVING_UNKNOWN_MULTI"]["scenarios"])


def collection_scenarios(config):
    seeds = config.get("collection_scenario_seeds")
    return ([scenarios(1, seed)[0] for seed in seeds] if seeds is not None else
            scenarios(config["collection_episodes"], config["train_scenario_seed"]))


def _tensor(value, boolean=False):
    return torch.as_tensor(value, dtype=torch.bool if boolean else torch.float32)


def _encode(networks, inputs):
    return networks.encoder(
        _tensor(inputs["map_tokens"])[None], _tensor(inputs["dynamic_tokens"])[None],
        _tensor(inputs["receiver_query"])[None], _tensor([inputs["query_available"]], True),
        _tensor(inputs["map_mask"], True)[None], _tensor(inputs["dynamic_mask"], True)[None],
    )


def delivered_tokens(cache):
    """Use only the actual on-wire code, never a sender's float activation."""
    values = np.stack([x["codes"] for x in cache])
    if values.dtype != np.int8:
        raise ValueError("received packet codes must retain their serialized int8 dtype")
    return _tensor(values)[None] / 127.


def _critic_features(runtime, sender):
    """Privileged training baseline only; never passed into a deployed head."""
    truth = runtime.teacher_truth()
    position = np.asarray(truth["agent_positions"], dtype=np.float32).reshape(-1) / np.tile([20., 20., 8.], 4)
    velocity = np.asarray(truth["agent_velocities"], dtype=np.float32).reshape(-1) / 2.5
    target = np.r_[np.asarray(truth["target_position"]) / [20., 20., 8.], np.asarray(truth["target_velocity"]) / 2.5]
    phase = [runtime.step / 400., float(truth["task_found"]), float(truth["executor_known"])]
    budgets = [runtime.channel.remaining_bits(i) / runtime.channel.capacity_bits for i in range(4)]
    queue = list(runtime.channel.queue)
    ages = [max(0, runtime.step - int(item[-1].sample_step)) for item in queue] if queue and hasattr(queue[0][-1], "sample_step") else []
    transport = [len(queue) / 64., float(np.mean(ages)) / 400. if ages else 0., float(np.max(ages)) / 400. if ages else 0.]
    # This is a finite CTDE baseline summary, not a claim of exact Markov state.
    return np.r_[position, velocity, target, phase, budgets, transport, np.eye(4)[sender]].astype(np.float32)


def _context(runtime, sender, inputs):
    return np.asarray([
        runtime.step / 400., runtime.channel.remaining_bits(sender) / runtime.channel.capacity_bits,
        float(sender == 3), float(inputs["query_available"]),
        float(inputs["receiver_query"][6]) if inputs["query_available"] else 0.,
        float(np.mean(inputs["map_mask"])), float(np.mean(inputs["dynamic_mask"])),
    ], dtype=np.float32)


def run_episode(config, base, scene, networks, model, *, seed, mode, episode_id,
                output, learner=None, task_updater=None, record_labels=False):
    from chapter5_comm.runtime import CH5Runtime
    if mode not in {"learned", "fixed", "neural_off"}:
        raise ValueError(mode)
    runtime = CH5Runtime(base, scene, seed=seed, channel_config=config["channel"])
    recorder = TrajectoryRecorder(episode_id, source="real_CH5Runtime_factual_target_v1")
    snapshots, caches, pending, events, transitions, network_trace = {}, {i: [] for i in range(4)}, {}, [], [], []
    gamma, horizon = base["rl"]["gamma"], config["label_horizon"]
    last_observations = None
    model.reset_noise()

    def frame():
        truth = runtime.teacher_truth()
        recorder.add_frame(runtime.step, truth["target_position"], truth["target_velocity"])

    def finish(sender):
        event = pending.pop(sender)
        event.update(next_critic_features=_critic_features(runtime, sender), done=bool(runtime.terminal))
        if event["elapsed"]:
            events.append(event)

    try:
        frame()
        for _ in range(config["rollout_limit_steps"]):
            if runtime.terminal:
                break
            runtime.begin_step()
            for delivered in runtime.drain_neural_deliveries():
                receiver = int(delivered["receiver"])
                snapshots_key = str(delivered["metadata_id"])
                snapshot = snapshots.get(snapshots_key)
                if snapshot is None:
                    raise RuntimeError("a received neural packet has no factual send record")
                cache = caches[receiver]
                cache[:] = [x for x in cache if not (x["sender"] == delivered["sender"] and x["stream_type"] == delivered["stream_type"])]
                cache.append(delivered)
                cache[:] = [x for x in cache if runtime.step - x["source_step"] <= config["neural_cache_steps"]]
                with torch.no_grad():
                    # Reconstruct exactly the encoder's on-wire int8 quantizer.
                    tokens = delivered_tokens(cache)
                    types = torch.tensor([x["stream_type"] for x in cache], dtype=torch.long)
                    ages = _tensor([[max(0, runtime.step-x["source_step"]) * runtime.dt for x in cache]])
                    geometry = runtime.local_geometry(receiver)
                    decoded = networks.decoder(tokens, types, torch.ones(1, len(cache), dtype=torch.bool), ages, _tensor(geometry)[None])
                    routes = runtime.route_features(receiver, horizon=horizon)
                    logits = networks.route_head(decoded["map_context"], decoded["dynamic_context"], _tensor(routes["route_features"])[None])
                    if mode != "neural_off":
                        runtime.install_decoded_state(receiver,
                            decoded={"has_dynamic": bool(decoded["has_dynamic"][0]) and
                                     any(x.get("target_available", False) for x in cache)},
                            state_mean=decoded["state_mean"][0].numpy(),
                            state_logvar=decoded["state_logvar"][0].numpy(),
                            route_logits=logits[0].numpy(), route_data=routes)
                network_trace.append(dict(step=runtime.step, receiver=receiver, packet_id=delivered["packet_id"], source_step=delivered["source_step"], has_dynamic=bool(decoded["has_dynamic"][0])))
                if record_labels:
                    inputs = deepcopy(snapshot["inputs"])
                    selected = snapshot["selected"]
                    received = np.zeros(networks.config.map_slots + networks.config.dynamic_slots, dtype=bool)
                    received[selected] = True
                    inputs.update(received=received, ages=np.full(len(received), (runtime.step-delivered["source_step"])*runtime.dt, np.float32),
                                  local_geometry=np.asarray(geometry, np.float32),
                                  state_observable=bool(inputs["dynamic_mask"][-1] and delivered["stream_type"] == 1))
                    recorder.add_decision(runtime.step, plain(inputs), routes["route_positions"], routes["route_features"],
                        receiver_id=receiver, sender_id=int(delivered["sender"]), route_valid_steps=routes["route_valid_steps"],
                        decision_id=f"{episode_id}:{delivered['packet_id']}", route_origin=routes["route_origin"],
                        detect_radius=float(runtime.physical.target_capture_radius if receiver == 3 else
                                            runtime.physical._sensor_range[receiver] + runtime.physical.detect_eps_bias))
            sender = runtime.tdma_sender
            if sender in pending:
                finish(sender)
            receiver = (sender + 1 + (runtime.step // 4) % 3) % 4
            inputs = runtime.tokenize(sender, receiver)
            with torch.no_grad():
                encoded = _encode(networks, inputs)
            event = dict(packet_tokens=encoded.tokens[0].numpy().copy(), context=_context(runtime, sender, inputs),
                         packet_bits=encoded.packet_bits.numpy().copy(), budget_bits=runtime.channel.remaining_bits(sender),
                         valid=encoded.valid[0].numpy().copy(), critic_features=_critic_features(runtime, sender),
                         episode_step=runtime.step, elapsed=0, discounted_reward=0.)
            if mode == "learned":
                if learner is None:
                    raise ValueError("learned mode requires the communication learner")
                choice = learner.choose(event, deterministic=not record_labels)
            else:
                feasible = np.flatnonzero(event["valid"] & (event["packet_bits"] <= event["budget_bits"]))
                action = int(feasible[(runtime.step // 4) % len(feasible)]) if len(feasible) and mode != "neural_off" else len(event["valid"])
                choice = dict(action=action, log_prob=None, policy_version=-1)
            event.update(choice)
            pending[sender] = event
            selected = int(choice["action"])
            if selected < encoded.tokens.shape[1]:
                key = f"{episode_id}:{runtime.step}:{sender}:{receiver}:{selected}"
                sent = runtime.send_latent(sender, receiver, encoded.codes[0, selected].numpy(),
                    int(encoded.stream_types[selected]), source_step=runtime.step, metadata_id=key,
                    target_available=bool(inputs["dynamic_mask"][-1] and int(encoded.stream_types[selected]) == 1))
                if sent["accepted"]:
                    if sent["bits"] != int(encoded.packet_bits[selected]):
                        raise RuntimeError("declared neural frame cost differs from actual serialized bytes")
                    snapshots[key] = dict(inputs=plain(inputs), selected=selected)
            observations = [np.asarray(x, np.float32).copy() for x in runtime.observations]
            if transitions:
                previous = transitions[-1]
                previous.update(next_observations=deepcopy(observations),
                    next_action_enabled=runtime.action_enabled(), replay_eligible=True,
                    next_observation_phase="after_channel_and_decoder_before_next_action")
                if task_updater is not None:
                    task_updater.observe(previous)
            with torch.no_grad():
                actions = torch.cat(model.step(observations, explore=record_labels), dim=0).detach().cpu().numpy()
            transition = runtime.advance(actions)
            transition.update(replay_eligible=False, next_observation_phase="post_physics_pending_channel")
            transitions.append(transition)
            rewards = np.asarray(transition["rewards"], dtype=np.float64)
            if not np.allclose(rewards, rewards[0]):
                raise RuntimeError("the inherited team_mean reward contract is not uniform")
            for item in pending.values():
                item["discounted_reward"] += gamma ** item["elapsed"] * float(rewards[0])
                item["elapsed"] += 1
            last_observations = transition["next_observations"]
            frame()
        if runtime.terminal:
            for sender in list(pending):
                finish(sender)
            transitions[-1].update(replay_eligible=True, next_observation_phase="task_terminal")
            if task_updater is not None:
                task_updater.observe(transitions[-1])
        summary = runtime.summary()
        summary.update(episode_id=episode_id, method=mode, original_task_horizon=400,
                       budget_truncated=not runtime.terminal, external_rollout_limit=config["rollout_limit_steps"],
                       observed_return=float(sum(float(np.mean(x["rewards"])) for x in transitions)),
                       completed_task_episode=bool(runtime.terminal), neural_delivery_applications=len(network_trace),
                       censored_communication_intervals=len(pending),
                       replay_eligible_transitions=sum(x["replay_eligible"] for x in transitions))
        examples = recorder.labels(horizon, detect_radius=1., dynamic_horizon=0) if record_labels else []
        output.mkdir(parents=True, exist_ok=False)
        recorder.write_jsonl(output / "factual_trajectory.jsonl")
        write_labels(output / "labels.jsonl", examples)
        write_json(output / "summary.json", summary)
        write_json(output / "communication_events.json", events)
        write_json(output / "network_deliveries.json", network_trace)
        write_json(output / "task_transitions.json", transitions)
        return dict(summary=summary, examples=examples, events=events, last_observations=last_observations)
    except Exception as error:
        # Preserve factual partial runs instead of silently discarding failures.
        output.mkdir(parents=True, exist_ok=True)
        partial = output / "interrupted_factual_trajectory.jsonl"
        if not partial.exists():
            recorder.write_jsonl(partial)
        write_json(output / "interrupted_episode.json", dict(
            status="interrupted", actual_length=runtime.step, transitions=transitions,
            network_deliveries=network_trace, error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        runtime.close()


def auxiliary_batch(examples):
    if not examples:
        raise ValueError("no neural packet was actually delivered; no real auxiliary dataset exists")
    keys = ["map_tokens", "dynamic_tokens", "receiver_query", "query_available", "map_mask", "dynamic_mask", "received", "ages", "local_geometry"]
    result = {k: np.stack([np.asarray(x["inputs"][k]) for x in examples]) for k in keys}
    for key in ("route_features", "route_valid", "route_labels"):
        result[key] = np.stack([np.asarray(x[key]) for x in examples])
    scales = np.asarray([20., 20., 8., 2.5, 2.5, 2.5], np.float32)
    result["state_target"] = np.stack([x["dynamic_target"] for x in examples]) / scales
    result["state_valid"] = np.asarray([x["dynamic_target_valid"] and x["inputs"]["state_observable"] for x in examples], dtype=bool)
    return result


def evaluation_summary(rows):
    complete = [x for x in rows if x["completed_task_episode"]]
    return dict(episodes=len(rows), task_horizon=400, budget_truncated=sum(x["budget_truncated"] for x in rows),
                full_task_success_rate=None if len(complete) != len(rows) else float(np.mean([x.get("success", x.get("mission_success", False)) for x in rows])),
                mean_observed_short_return=float(np.mean([x["observed_return"] for x in rows])),
                neural_deliveries=sum(x["neural_delivery_applications"] for x in rows),
                interpretation="short plumbing/mechanism evidence; externally truncated episodes do not estimate 400-step task success")


def smoke(config, base, output):
    model = task_model(base, config["seed"])
    networks = build_neural_models(config["neural"])
    auxiliary = AuxiliaryTrainer(networks, max_updates=config["auxiliary_updates"])
    updater = TaskReplayUpdater(model, batch_size=8, max_updates=config["task_updates"])
    train_scenes = collection_scenarios(config)
    write_json(output / "config.json", config)
    write_json(output / "parent_task_config.json", base)
    write_json(output / "train_scenarios.json", train_scenes)
    examples, collection_rows = [], []
    for i, scene in enumerate(train_scenes):
        result = run_episode(config, base, scene, networks, model, seed=config["seed"]+i,
            mode="fixed", episode_id=f"collect_{i}", output=output/f"collect/episode_{i}", task_updater=updater, record_labels=True)
        examples.extend(result["examples"])
        collection_rows.append(result["summary"])
    batch = auxiliary_batch(examples)
    for _ in range(config["auxiliary_updates"]):
        auxiliary.step(batch)
    for _ in range(config["task_updates"]):
        updater.update()
    freeze_auxiliary(networks, True)
    # Fixed dimensional CTDE summary: 12+12+6+3+4+3+4 = 44.
    learner = CommunicationLearner(networks.scheduler, 44, gamma=base["rl"]["gamma"], max_updates=4)
    comm_rows = []
    comm_scenes = scenarios(config["communication_episodes"], config["train_scenario_seed"]+1)
    for i, scene in enumerate(comm_scenes):
        result = run_episode(config, base, scene, networks, model, seed=config["seed"]+100+i,
            mode="learned", episode_id=f"comm_{i}", output=output/f"communication_train/episode_{i}", learner=learner, record_labels=True)
        learner.update(result["events"])
        comm_rows.append(result["summary"])
    checkpoint = save_checkpoint(output/"checkpoint.pt", networks, auxiliary=auxiliary, communication=learner,
        task_model=model, metadata=dict(config=config, base_config=base, status="short_test_not_formal_performance"))
    restored = load_checkpoint(output/"checkpoint.pt")
    exact = all(torch.equal(v, restored["networks"].state_dict()[k]) for k, v in networks.state_dict().items())
    critic_exact = all(torch.equal(v, restored["communication"].critic.state_dict()[k])
                       for k, v in learner.critic.state_dict().items())
    if not exact or not critic_exact:
        raise RuntimeError("neural/communication critic checkpoint roundtrip differs")
    restored_model = task_model(base, config["seed"])
    restored_model.load_training_state_dict(restored["task_model_state"])
    restored_model.prep_rollouts(device="cpu")
    with torch.no_grad():
        check_observations = result["last_observations"]
        original_actions = torch.cat(model.step(check_observations, explore=False))
        restored_actions = torch.cat(restored_model.step(check_observations, explore=False))
    task_actions_exact = torch.equal(original_actions, restored_actions)
    if not task_actions_exact:
        raise RuntimeError("inherited task actor checkpoint roundtrip differs")
    eval_scenes = scenarios(config["evaluation_episodes"], config["eval_scenario_seed"], evaluation=True)
    write_json(output/"evaluation_scenarios.json", eval_scenes)
    results = {}
    for mode in config["evaluation_modes"]:
        rows = []
        for i, scene in enumerate(eval_scenes):
            result = run_episode(config, base, scene, restored["networks"], restored_model,
                seed=config["seed"]+200+i, mode=mode, episode_id=f"eval_{mode}_{i}",
                output=output/f"evaluate/{mode}/episode_{i}", learner=restored["communication"], record_labels=False)
            rows.append(result["summary"])
        results[mode] = evaluation_summary(rows)
    report = dict(schema=SCHEMA, status="completed_short_test", formal_performance_verified=False,
        original_task_horizon=400, reward_unchanged=True, source_parent_config_sha256=config["parent_config_sha256"],
        external_rollout_limit=config["rollout_limit_steps"], actual_episodes=len(collection_rows)+len(comm_rows)+sum(x["episodes"] for x in results.values()),
        factual_auxiliary_examples=len(examples), route_supervised_labels=int(np.asarray(batch["route_valid"]).sum()),
        positive_route_labels=int((np.asarray(batch["route_labels"])*np.asarray(batch["route_valid"])).sum()),
        observable_state_labels=int(np.asarray(batch["state_valid"]).sum()), auxiliary_updates=auxiliary.history,
        task_updates=updater.history, communication_updates=learner.history, checkpoint=checkpoint,
        neural_checkpoint_roundtrip_exact=exact, communication_critic_roundtrip_exact=critic_exact,
        task_actor_roundtrip_exact=task_actions_exact, evaluation=results)
    write_json(output/"report.json", report)
    return report
