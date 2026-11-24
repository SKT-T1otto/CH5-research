"""Teacher labels from recorded task rollouts, never from random future tensors.

The caller records factual target position/velocity at each physical step and
receiver-local candidate execution forecasts at the decision step.  This file
does not run the simulator, infer a missing future, generate a truth-map route,
or change task rewards.  The geometric label uses the third-chapter swept
distance kernel; it is not a counterfactual full-task success label.

``inputs`` are deployment-side features supplied by the legal runtime.  CTDE
target facts are stored separately and are consumed only as training labels.
Numerical validation cannot prove that the caller constructed legal features;
that information-flow contract is enforced by the runtime and its tests.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from chapter3_bser.experiments.coupled_search_v1.numerics import swept_distances


ROLLOUT_SCHEMA = "ch5_factual_rollout_v1"
LABEL_SCHEMA = "ch5_factual_geometric_labels_v1"
ROUTE_ORIGIN = "receiver_local_map_forecast_v1"
_TEACHER_INPUT_KEYS = frozenset({
    "ctde_target", "ctde_target_trajectory", "target_truth", "future_target",
    "future_targets", "future_target_valid", "dynamic_target", "route_labels",
})


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _vector3(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite physical 3-vector")
    return result.copy()


def _plain(value):
    """Owned JSON-compatible values; no implicit tensor/device or RNG work."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("record mappings require string keys")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("record values must be finite; use an explicit validity mask")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported record value: {type(value).__name__}")


def _legal_feature_container(inputs):
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be a mapping of runtime-supplied legal features")

    def inspect(value):
        if isinstance(value, dict):
            blocked = set(value) & _TEACHER_INPUT_KEYS
            if blocked:
                raise ValueError("teacher-only fields cannot enter inputs: " + ", ".join(sorted(blocked)))
            for item in value.values():
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)

    inspect(inputs)
    return _plain(inputs)


def build_route_encounter_labels(route_positions, target_positions, horizon,
                                 detect_radius, route_valid_steps=None,
                                 target_valid=None):
    """Compare saved local route forecasts against factual target motion.

    ``route_positions`` is [routes, forecast_steps+1, 3], including the
    receiver's position at decision time. ``target_positions`` is a [T,3]
    factual trajectory on the same physical clock, also including that step.
    Inputs may be shorter than ``horizon+1``; the absent tail is censored.

    A positive encounter already observed within a valid prefix is labeled
    positive even when its tail is censored. A negative is valid only after
    the entire requested horizon is observed and the candidate forecast is
    valid for that horizon. No held position or extrapolated future is added.
    """
    horizon = _integer(horizon, "horizon", 1)
    radius = float(detect_radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("detect_radius must be finite and nonnegative")
    routes = np.asarray(route_positions, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    if routes.ndim != 3 or routes.shape[-1] != 3 or routes.shape[0] < 1 or routes.shape[1] < 1:
        raise ValueError("route_positions must have shape [C,T,3] with nonempty routes/time")
    if targets.ndim != 2 or targets.shape[-1] != 3 or targets.shape[0] < 1:
        raise ValueError("target_positions must have shape [T,3]")
    count, times, _ = routes.shape
    if route_valid_steps is None:
        valid_steps = np.full(count, times - 1, dtype=np.int64)
    else:
        raw = np.asarray(route_valid_steps)
        if raw.shape != (count,) or raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw > times - 1):
            raise ValueError("route_valid_steps must be [C] integers within the saved forecast")
        valid_steps = raw.astype(np.int64, copy=True)
    if target_valid is None:
        known = np.ones(len(targets), dtype=bool)
    else:
        known = np.asarray(target_valid)
        if known.shape != (len(targets),) or known.dtype != np.bool_:
            raise ValueError("target_valid must be a bool mask matching target_positions")
    if not np.isfinite(targets[known]).all():
        raise ValueError("known target states must be finite")
    for index, end in enumerate(valid_steps):
        if not np.isfinite(routes[index, :end + 1]).all():
            raise ValueError("valid route forecast positions must be finite")

    prefix = 0
    for k in range(1, min(horizon, len(targets) - 1) + 1):
        if not (known[k - 1] and known[k]):
            break
        prefix = k
    observed = np.minimum(valid_steps, min(horizon, prefix))
    labels = np.zeros(count, dtype=np.float32)
    first_hit = np.zeros(count, dtype=np.int64)
    for index, end in enumerate(observed):
        for k in range(1, int(end) + 1):
            # Same interval/swept geometry as third-chapter discovery.signature.
            distance = swept_distances(routes[index, k - 1], routes[index, k],
                                       targets[k - 1], targets[k])
            if float(distance) <= radius:
                labels[index] = 1.0
                first_hit[index] = k
                break
    censored = observed < horizon
    valid = (labels == 1.0) | ~censored
    return {
        "route_labels": labels,
        "route_valid": valid,
        "route_first_hit_steps": first_hit,
        "route_observed_intervals": observed,
        "route_censored": censored,
    }


def _indexed_frames(records):
    indexed, decisions = {}, []
    for source in records:
        record = deepcopy(source)
        if not isinstance(record, dict):
            raise ValueError("each rollout frame must be a mapping")
        if record.get("schema", ROLLOUT_SCHEMA) != ROLLOUT_SCHEMA:
            raise ValueError("unsupported rollout schema")
        episode = str(record.get("episode_id", "episode_0"))
        step = _integer(record.get("step"), "step")
        key = episode, step
        if key in indexed:
            raise ValueError("duplicate factual frame: " + repr(key))
        truth = record.get("ctde_target")
        if not isinstance(truth, dict):
            raise ValueError("each frame requires separate ctde_target position/velocity facts")
        if _integer(truth.get("sample_step", step), "target sample_step") != step:
            raise ValueError("target sample_step must equal the recorded physical frame step")
        target = np.concatenate((_vector3(truth.get("position"), "target position"),
                                 _vector3(truth.get("velocity"), "target velocity")))
        indexed[key] = target
        frame_decisions = record.get("decisions", [])
        if not isinstance(frame_decisions, list):
            raise ValueError("frame decisions must be a list")
        for decision in frame_decisions:
            if not isinstance(decision, dict):
                raise ValueError("each decision must be a mapping")
            if _integer(decision.get("step", step), "decision step") != step:
                raise ValueError("decision and candidate forecast must start on their frame step")
            decisions.append((episode, step, decision))
    return indexed, decisions


def labels_from_rollout(records, horizon, detect_radius, dynamic_horizon=0):
    """Make teacher-only examples from physical-step factual rollout frames.

    Dynamic targets retain physical world units. Normalization belongs in the
    training adapter and must use recorded training-only/configured scales.
    ``dynamic_horizon=0`` supervises the decoder at its current receiving
    step; a positive value selects an actually recorded future state.
    """
    horizon = _integer(horizon, "horizon", 1)
    dynamic_horizon = _integer(dynamic_horizon, "dynamic_horizon")
    indexed, decisions = _indexed_frames(records)
    examples, ids = [], set()
    for episode, step, decision in sorted(decisions, key=lambda item: (item[0], item[1])):
        origin = decision.get("route_origin")
        if origin != ROUTE_ORIGIN:
            raise ValueError("candidate routes must be saved receiver-local map forecasts")
        decision_id = str(decision.get("decision_id", f"{episode}:{step}:{len(examples)}"))
        if (episode, decision_id) in ids:
            raise ValueError("duplicate decision_id within an episode")
        ids.add((episode, decision_id))
        inputs = _legal_feature_container(decision.get("inputs"))
        routes = np.asarray(decision.get("route_positions"), dtype=np.float64)
        if routes.ndim != 3 or routes.shape[-1] != 3:
            raise ValueError("decision route_positions must have shape [C,T,3]")
        features = np.asarray(decision.get("route_features"), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != routes.shape[0] or not np.isfinite(features).all():
            raise ValueError("route_features must be finite [C,F] aligned to saved routes")
        future = np.zeros((horizon + 1, 6), dtype=np.float64)
        future_valid = np.zeros(horizon + 1, dtype=bool)
        for k in range(horizon + 1):
            fact = indexed.get((episode, step + k))
            if fact is not None:
                future[k] = fact
                future_valid[k] = True
        radius = float(decision.get("detect_radius", detect_radius))
        labels = build_route_encounter_labels(
            routes, future[:, :3], horizon, radius,
            decision.get("route_valid_steps"), future_valid,
        )
        dynamic = indexed.get((episode, step + dynamic_horizon))
        example = {
            "schema": LABEL_SCHEMA,
            "episode_id": episode,
            "decision_id": decision_id,
            "step": step,
            "sender_id": decision.get("sender_id"),
            "receiver_id": decision.get("receiver_id"),
            "inputs": inputs,
            "route_features": features.tolist(),
            **{key: value.tolist() for key, value in labels.items()},
            "dynamic_target": (np.zeros(6) if dynamic is None else dynamic).tolist(),
            "dynamic_target_valid": dynamic is not None,
            "dynamic_target_step": step + dynamic_horizon,
            "future_targets": future.tolist(),
            "future_target_valid": future_valid.tolist(),
            "label_metadata": {
                "teacher_only": True,
                "target_source": "recorded_physical_rollout_no_extrapolation",
                "route_origin": origin,
                "horizon_physical_steps": horizon,
                "detect_radius_world_units": radius,
                "encounter_definition": "third_chapter_swept_interval_distance",
                "not_full_task_counterfactual_return": True,
                "invalid_target_fill": "zero_storage_only; validity mask must be applied",
            },
        }
        examples.append(example)
    return examples


class TrajectoryRecorder:
    """Keep actual rollout facts and local decision inputs in separate fields.

    Facts must be sampled after reset/step on the same physical clock as the
    decision route's initial position. Missing records stay missing. Recording
    functions have no simulator access and cannot manufacture factual states.
    """

    def __init__(self, episode_id="episode_0", source="chapter5_runtime"):
        self.episode_id = str(episode_id)
        self.source = str(source)
        self.records = []
        self._lookup = {}
        self._last_step = {}

    def add_frame(self, step, target_position, target_velocity, episode_id=None):
        episode = self.episode_id if episode_id is None else str(episode_id)
        step = _integer(step, "step")
        if step <= self._last_step.get(episode, -1):
            raise ValueError("frames must be appended with strictly increasing per-episode steps")
        position = _vector3(target_position, "target_position")
        velocity = _vector3(target_velocity, "target_velocity")
        frame = {"schema": ROLLOUT_SCHEMA, "episode_id": episode, "step": step,
                 "recording_source": self.source,
                 "ctde_target": {"position": position.tolist(), "velocity": velocity.tolist(), "sample_step": step},
                 "decisions": []}
        self.records.append(frame)
        self._lookup[episode, step] = frame
        self._last_step[episode] = step
        return deepcopy(frame)

    def add_decision(self, step, inputs, route_positions, route_features,
                     receiver_id=None, sender_id=None, route_valid_steps=None,
                     decision_id=None, route_origin=ROUTE_ORIGIN, episode_id=None,
                     detect_radius=None):
        episode = self.episode_id if episode_id is None else str(episode_id)
        step = _integer(step, "step")
        frame = self._lookup.get((episode, step))
        if frame is None:
            raise ValueError("record the factual decision-step frame before its decision")
        if route_origin != ROUTE_ORIGIN:
            raise ValueError("route_origin must identify a receiver-local map forecast")
        routes = np.asarray(route_positions, dtype=np.float64)
        features = np.asarray(route_features, dtype=np.float64)
        if routes.ndim != 3 or routes.shape[-1] != 3 or routes.shape[0] < 1 or routes.shape[1] < 1:
            raise ValueError("route_positions must be nonempty [C,T,3]")
        if features.ndim != 2 or features.shape[0] != routes.shape[0]:
            raise ValueError("route_features must align to route_positions")
        decision = {"decision_id": str(decision_id if decision_id is not None else f"{episode}:{step}:{len(frame['decisions'])}"),
                    "step": step, "sender_id": sender_id, "receiver_id": receiver_id,
                    "inputs": _legal_feature_container(inputs),
                    "route_positions": _plain(routes), "route_features": _plain(features),
                    "route_origin": route_origin}
        if route_valid_steps is not None:
            values = np.asarray(route_valid_steps)
            if values.shape != (len(routes),) or values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= routes.shape[1]):
                raise ValueError("route_valid_steps must be integers within each saved route")
            decision["route_valid_steps"] = values.tolist()
        if detect_radius is not None:
            radius = float(detect_radius)
            if not math.isfinite(radius) or radius < 0:
                raise ValueError("decision detect_radius must be finite and nonnegative")
            decision["detect_radius"] = radius
        frame["decisions"].append(decision)
        return deepcopy(decision)

    def labels(self, horizon, detect_radius, dynamic_horizon=0):
        return labels_from_rollout(self.records, horizon, detect_radius, dynamic_horizon)

    def to_jsonl(self):
        return "".join(json.dumps(_plain(row), ensure_ascii=False, allow_nan=False) + "\n" for row in self.records)

    def write_jsonl(self, path):
        """Create a new recording; never overwrite a historical run file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(self.to_jsonl())
        return destination


def read_rollout(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    _indexed_frames(rows)
    return rows


def write_labels(path, examples):
    """Create a teacher-example JSONL without overwriting an existing run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            if example.get("schema") != LABEL_SCHEMA:
                raise ValueError("unsupported teacher label schema")
            stream.write(json.dumps(_plain(example), ensure_ascii=False, allow_nan=False) + "\n")
    return destination


COMMAND_TRIPLET_SCHEMA = "ch5_native_command_choice_triplet_v1"
COMMAND_ARMS = ("CMD", "PAD", "REPLAN_ONLY")


def command_triplet_record(opportunity, sender_features, arms, route_encounter_labels, diagnostics_ref):
    """Bind a legal pre-send example to three observed closed-loop suffixes and separate teacher labels."""
    from .protocol import COMMAND_PADDING, Kind, Packet
    from .runtime import GOAL3

    kinds = dict(zip(COMMAND_ARMS, (Kind.COMMAND, Kind.COMMAND_PAD, Kind.REPLAN_ONLY)))
    if set(arms) != set(kinds) or set(route_encounter_labels) != set(kinds):
        raise ValueError("a triplet requires all three named arms and their separate geometric labels")
    legal = {"sender", "receiver", "send_step", "sample_step", "candidate_goal", "previous_command",
             "available_bits", "query_sample_step", "peer_sample_step", "tokens"}
    token_fields = {"map_tokens", "map_mask", "dynamic_tokens", "dynamic_mask", "receiver_query", "query_available"}
    if set(sender_features) != legal or set(sender_features["tokens"]) != token_fields:
        raise ValueError("sender inputs must match the pre-send feature allowlist")
    inputs = _legal_feature_container(sender_features)
    src, dst, t = (inputs[k] for k in ("sender", "receiver", "send_step"))
    if (opportunity["receiver"], opportunity["send_step"]) != (dst, t):
        raise ValueError("sender features refer to a different native opportunity")
    prev = inputs["previous_command"]
    if (src != 0 or dst not in (1, 2) or set(prev) != {"goal", "sample_step", "send_step"}
            or not 0 <= prev["sample_step"] <= prev["send_step"] < t or not 0 <= inputs["sample_step"] <= t):
        raise ValueError("invalid sender chronology or D2 peer")
    if any(s is not None and not 0 <= s <= t for s in (inputs["query_sample_step"], inputs["peer_sample_step"])):
        raise ValueError("sender evidence cannot come from the future")
    body = GOAL3.pack(*_vector3(inputs["candidate_goal"], "candidate_goal"))
    if body == GOAL3.pack(*_vector3(prev["goal"], "previous_goal")):
        raise ValueError("expected a changed native endpoint")
    if not diagnostics_ref or len(opportunity["prefix_sha256"]) != 64:
        raise ValueError("missing diagnostic provenance")
    due_steps = set()
    for name, kind in kinds.items():
        arm = arms[name]
        packet = Packet.from_bytes(bytes.fromhex(arm["sent_hex"]))
        expected = Packet(kind, src, dst, opportunity["packet_id"], inputs["sample_step"], t,
                          body if name == "CMD" else COMMAND_PADDING)
        if packet != expected or packet.bits != 224 or arm["prefix_sha256"] != opportunity["prefix_sha256"]:
            raise ValueError("mismatched cut or actual command frame")
        end = arm["outcome"]
        if sum(int(end[k]) for k in ("success", "collision", "timeout")) != 1 or end["terminal_step"] <= t:
            raise ValueError("one original terminal outcome is required")
        due_steps.add(arm["scheduled_delivery"])
        arrival, execution = arm["arrival"], arm["next_executed"]
        if arrival is None:
            if execution is not None or arm["received_hex"] is not None or route_encounter_labels[name] is not None:
                raise ValueError("an undelivered command has no receipt or arrival label")
        elif (arm["received_hex"] != arm["sent_hex"] or not t < arrival < end["terminal_step"]
              or execution is None or execution["action_step"] != arrival or route_encounter_labels[name] is None):
            raise ValueError("received bytes and first executed waypoint must align to actual delivery")
    if len(due_steps) != 1 or len({a["arrival"] for a in arms.values()}) != 1:
        raise ValueError("intervention delivery differs before any treatment could act")
    return _plain(dict(schema=COMMAND_TRIPLET_SCHEMA, opportunity=opportunity, sender_features=inputs,
                       arms=arms, route_encounter_labels=route_encounter_labels, offline_diagnostics_ref=diagnostics_ref))
