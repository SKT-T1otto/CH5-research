"""New episode-boundary research checkpoints; no historical replay loading."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import random
import uuid

import torch
import numpy as np

from core.replay.ch3_buffer import CH3ReplayBuffer
from .pipeline import plain, write_json
from .experiment_config import canonical, source_identity

STATE_SCHEMA = "ch5.experiment.episode_boundary.v1"


def atomic_json(path, value):
    """Publish a NEW completion record atomically; interrupted pending files stay."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pending.open("x", encoding="utf-8") as f:
        json.dump(plain(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.rename(pending, path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def append_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def journal(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8")
    rows = []
    for line in data.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Only an interrupted final append is recoverable, never a
            # corrupted interior record. Subsequent writes use a new journal.
            if line != data.splitlines()[-1] or data.endswith("\n"):
                raise
    return rows


@contextmanager
def exclusive_job(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def replay_state(replay):
    used = replay.filled_i
    return dict(base=CH3ReplayBuffer.state_dict(replay),
                enabled=replay.enabled[:used].clone(), next_enabled=replay.next_enabled[:used].clone(),
                gate_initialized=replay.gate_initialized[:used].clone())


def load_replay(replay, state):
    used = state["base"]["filled_i"]
    for key in ("enabled", "next_enabled", "gate_initialized"):
        x = state[key]
        shape = (used,) if key == "gate_initialized" else (used, 4)
        if x.shape != shape or x.dtype != torch.bool or (key == "gate_initialized" and not bool(x.all())):
            raise ValueError("invalid new gated replay snapshot: " + key)
    CH3ReplayBuffer.load_state_dict(replay, state["base"])
    for key in ("enabled", "next_enabled", "gate_initialized"):
        getattr(replay, key)[:used].copy_(state[key])


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device topology changed; exact resume is unsupported")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_state(directory, *, identity, progress, networks, model, auxiliary, communication, task, task_clock):
    directory = Path(directory)
    path = directory / "checkpoints" / (uuid.uuid4().hex + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Snapshot only between physical episodes; no active simulator is serialized.
    payload = dict(schema=STATE_SCHEMA, identity=identity, progress=progress,
        source=source_identity(), networks=networks.state_dict(),
        model=None if task is None else model.training_state_dict(),
        exploration=[dict(state=agent.noise.state.detach().cpu().clone(), sigma=agent.noise.sigma,
                          current_sigma=agent.current_sigma) for agent in model.agents] if task is not None else [],
        auxiliary=auxiliary.state_dict(), communication=communication.state_dict(),
        task=None if task is None else dict(updates=task.updates, history=task.history,
                                           replay=replay_state(task.replay), clock=task_clock),
        rng=rng_state(), resume_scope="last_committed_episode_or_auxiliary_update_boundary")
    with path.open("xb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dict(path=path.relative_to(directory).as_posix(), sha256=sha,
                   progress=progress, schema=STATE_SCHEMA)
    atomic_json(path.with_suffix(".receipt.json"), receipt)
    # Immutable per-checkpoint receipts are the commit record. A killed append
    # cannot conceal a newer fully committed checkpoint.
    return receipt


def load_state(directory, relative, identity):
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory / "checkpoints"):
        raise ValueError("checkpoint must belong to this new experiment job")
    receipt = json.loads(path.with_suffix(".receipt.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("checkpoint digest mismatch")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload["schema"] != STATE_SCHEMA or payload["identity"] != identity:
        raise ValueError("checkpoint belongs to a different experiment/arm/seed")
    if payload["source"] != source_identity():
        raise ValueError("source, scripts, dependency pins or configs changed; use a fresh run")
    return payload


def latest_receipt(directory):
    receipts = [json.loads(p.read_text(encoding="utf-8")) for p in (Path(directory) / "checkpoints").glob("*.receipt.json")]
    return max(receipts, key=lambda x: x["progress"]["serial"]) if receipts else None
