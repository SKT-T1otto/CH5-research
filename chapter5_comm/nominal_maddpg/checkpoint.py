"""CPU episode-boundary snapshots with a durable commit marker and strict identity."""
import copy
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter5_comm.experiment_state import atomic_json
from .config import file_sha, read_json

SCHEMA = 'auv.ch5.nominal_maddpg.boundary.v1'


def global_rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
                torch=torch.get_rng_state().clone())


def restore_global_rng(state):
    # Probe every stream before changing the process-global state.
    p = random.Random(); p.setstate(state['python'])
    n = state['numpy']; numpy_state=(n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4])
    probe = np.random.RandomState(0); probe.set_state(numpy_state)
    t = torch.Generator(); t.set_state(state['torch'])
    random.setstate(p.getstate()); np.random.set_state(probe.get_state()); torch.set_rng_state(t.get_state())


def learner_state(learner):
    return dict(networks=copy.deepcopy(learner.networks.state_dict()),
        actors=[copy.deepcopy(o.state_dict()) for o in learner.actor_optimizers],
        critics=[copy.deepcopy(o.state_dict()) for o in learner.critic_optimizers], counters=learner.counters())


def restore_learner(learner, state):
    expected=learner.networks.state_dict(); loaded=state['networks']
    if expected.keys()!=loaded.keys(): raise ValueError('checkpoint network keys differ')
    for key,value in loaded.items():
        if (not torch.is_tensor(value) or value.shape!=expected[key].shape or value.dtype!=expected[key].dtype
                or value.device.type!='cpu' or not bool(torch.isfinite(value).all())):
            raise ValueError('invalid checkpoint network tensor: '+key)
    counts=state['counters']; rounds=counts.get('rounds')
    if type(rounds) is not int or rounds<0 or counts.get('target_updates')!=rounds:
        raise ValueError('invalid team update counters')
    for key in ('actor_steps','actor_skips','critic_steps'):
        if not isinstance(counts.get(key),list) or len(counts[key])!=4 or any(type(n) is not int or n<0 for n in counts[key]):
            raise ValueError('invalid per-agent update counters')
    if counts['critic_steps']!=[rounds]*4 or any(a+s!=rounds for a,s in zip(counts['actor_steps'],counts['actor_skips'])):
        raise ValueError('round/critic/actor/skip counters disagree')
    for kind,optimizers,counters in (('actors',learner.actor_optimizers,counts['actor_steps']),
                                    ('critics',learner.critic_optimizers,counts['critic_steps'])):
        if len(state[kind])!=4: raise ValueError('missing optimizer state')
        for opt,stored,steps in zip(optimizers,state[kind],counters):
            if stored['param_groups']!=opt.state_dict()['param_groups']:
                raise ValueError('optimizer configuration or parameter ownership differs')
            ids=stored['param_groups'][0]['params']; params=opt.param_groups[0]['params']
            if set(stored['state'])!=(set(ids) if steps else set()):
                raise ValueError('optimizer state does not match actual steps')
            for index,param in zip(ids,params):
                if not steps: continue
                item=stored['state'][index]
                if set(item)!={'step','exp_avg','exp_avg_sq'} or float(item['step'])!=steps:
                    raise ValueError('Adam clock differs from per-agent steps')
                for key in ('exp_avg','exp_avg_sq'):
                    v=item[key]
                    if not torch.is_tensor(v) or v.dtype!=param.dtype or v.shape!=param.shape or not bool(torch.isfinite(v).all()):
                        raise ValueError('invalid Adam moments')
    learner.networks.load_state_dict(loaded,strict=True)
    for kind,optimizers in (('actors',learner.actor_optimizers),('critics',learner.critic_optimizers)):
        for opt,stored in zip(optimizers,state[kind]): opt.load_state_dict(copy.deepcopy(stored))
    for key,value in counts.items(): setattr(learner,key,copy.deepcopy(value))
    learner.networks.zero_grad(set_to_none=True)


def save_boundary(session):
    """Stage in a private directory; publish the committed boundary in one rename."""
    state = session.snapshot()
    path = session.root / "boundaries" / f"{len(state['episodes']):04d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent))
    pending = stage / "state.pt.pending"
    with pending.open("xb") as stream:
        torch.save(dict(schema=SCHEMA, identity=session.identity, state=state), stream)
        stream.flush()
        os.fsync(stream.fileno())
    artifact = stage / "state.pt"
    pending.rename(artifact)
    marker = dict(schema=SCHEMA, identity=session.identity, sha256=file_sha(artifact),
                  state_sha256=state_digest(state), episodes=len(state['episodes']),
                  team_steps=state['team_steps'])
    atomic_json(stage / "committed.json", marker)
    stage.rename(path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return path


def read_boundary(root, path, expected_identity):
    path = Path(path).resolve()
    if path.parent != (Path(root)/"boundaries").resolve():
        raise ValueError("checkpoint must belong to this run")
    if not (len(path.name) == 4 and path.name.isascii() and path.name.isdecimal()):
        raise ValueError("incomplete or unregistered boundary name")
    marker_path = path / "committed.json"
    if not marker_path.is_file():
        raise ValueError("incomplete checkpoint: missing commit marker")
    marker = read_json(marker_path)
    if marker.get("schema") != SCHEMA or marker.get("identity") != expected_identity:
        raise ValueError("checkpoint run, seed, source, or configuration differs")
    artifact = path / "state.pt"
    if not artifact.is_file() or file_sha(artifact) != marker["sha256"]:
        raise ValueError("checkpoint byte hash differs")
    payload = torch.load(artifact, weights_only=True, map_location="cpu")
    state = payload["state"]
    if (payload["identity"] != expected_identity or payload["schema"] != SCHEMA
            or state_digest(state) != marker["state_sha256"]
            or marker["episodes"] != len(state["episodes"]) or marker["team_steps"] != state["team_steps"]
            or path.name != f"{marker['episodes']:04d}"):
        raise ValueError("checkpoint state or boundary counters differ")
    return state


