"""Passive observations and exact in-memory forks for the D2 command experiment."""
from copy import deepcopy
import hashlib
import random

import numpy as np
import torch

from tools.executor_trace import digest, plain, rng_hash, tap_method


def fingerprint(rt):
    """Same behavioral state projection as the frozen Windows native A/A receipt."""
    raw, ch = rt.physical, rt.channel
    agents = []
    for local in rt.locals:
        state = {k: v for k, v in vars(local).items() if k not in ('mapper', 'own_evidence')}
        for name in ('mapper', 'own_evidence'):
            state[name] = {k: v for k, v in vars(getattr(local, name)).items()
                           if torch.is_tensor(v) or isinstance(v, (bool, int, float, str))}
        agents.append(state)
    channel = {k: v for k, v in vars(ch).items() if k not in ('queue', 'rng') and not callable(v)}
    channel['queue'] = [(due, tie, body.hex(), event) for due, tie, body, event in ch.queue]
    channel['rng_state'] = ch.rng.bit_generator.state
    return digest(dict(step=rt.step, begun=rt._begun_step, target=raw.target_state,
        physics={k: v for k, v in vars(raw).items() if torch.is_tensor(v) or isinstance(v, (bool, int, float, str))},
        observations=rt.observations, agents=agents, bootstrap_turn=rt._bootstrap_turn,
        receiver_turn=rt._receiver_turn, previously_sent=rt._evidence_sent,
        evidence_events=rt.evidence_events, channel=channel, external_rng=rng_hash()))


def fork(rt):
    # These forwarding wrappers need initialized shells in the deepcopy memo.
    wrappers = (rt.env, rt.env.env)
    memo = {id(w): object.__new__(type(w)) for w in wrappers}
    clone = deepcopy(rt, memo)
    for wrapper in wrappers:
        vars(memo[id(wrapper)]).update(deepcopy(vars(wrapper), memo))
    assert clone.base._owner is clone and clone.physical._comm_owner is clone
    assert not np.shares_memory(rt.channel.tokens, clone.channel.tokens)
    assert clone.channel.rng is not rt.channel.rng and clone.channel.queue is not rt.channel.queue
    assert clone.physical._agent_pos.data_ptr() != rt.physical._agent_pos.data_ptr()
    for left, right in zip(rt.locals, clone.locals):
        for name in ('mapper', 'own_evidence'):
            assert getattr(left, name).occupancy_logodds.data_ptr() != getattr(right, name).occupancy_logodds.data_ptr()
    return clone


def save_rng():
    return random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def navigation(rt):
    agents = []
    for i, local in enumerate(rt.locals):
        cell = local.mapper._cell_from_point_np(local.waypoint)
        agents.append(dict(agent=i, position=rt.physical._agent_pos[i], velocity=rt.physical._agent_vel[i],
            goal=local.goal, route=local.path, route_index=local.path_index, waypoint=local.waypoint,
            waypoint_cell=cell, waypoint_probability=float(local.mapper.occupancy_probability[cell]),
            last_plan=local.last_plan))
    return plain(agents)


def map_digest(rt):
    def arrays(mapper):
        result = {}
        for name, v in vars(mapper).items():
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                a = np.asarray(v)
                result[name] = dict(dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest())
        return result
    return digest([dict(agent=i, fused=arrays(local.mapper), firsthand=arrays(local.own_evidence),
                        remote_evidence=digest(local.remote_evidence)) for i, local in enumerate(rt.locals)])


def receiver_state(rt, receiver):
    local = rt.locals[receiver]
    return plain(dict(command=local.command, command_sample=local.last_command_step, goal=local.goal,
                      last_plan=local.last_plan, path=local.path, waypoint=local.waypoint, acks=local.acks,
                      last_replan_request=rt._last_replan_request.get(receiver)))


class BeforeTransmission(Exception):
    """Reached the original send call with its goal and receiver already allocated."""


class CommandTrace:
    def __init__(self, rt, send_step, *, prefix=False):
        self.rt, self.send_step, self.prefix = rt, send_step, prefix
        self.events = []
        self.pre_send = None

    def add(self, step, event, data, agent=None):
        self.events.append(dict(step=step, event=event, agent=agent, data=plain(data)))

    def send(self, when, args, kwargs, result):
        rt = self.rt
        kind, sender, receiver, payload, sample = args
        if when == 'before':
            if rt.step == self.send_step:
                self.pre_send = dict(state=fingerprint(rt), sender=sender, receiver=receiver, sample=sample,
                    kind=kind.name, payload=payload.hex(), tokens=rt.channel.remaining_bits(sender),
                    sequence=rt.channel.sequence)
                if self.prefix:
                    raise BeforeTransmission
            return
        accepted = result[1]
        due, _, raw, event = next(q for q in rt.channel.queue if q[3]['packet_id'] == accepted['packet_id'])
        self.add(rt.step, 'tx', dict(wire_hex=raw.hex(), accepted=accepted, due=due, metadata=event), sender)

    def receive(self, when, args, kwargs, result):
        step = args[0]
        if when == 'before':
            return {event['packet_id']: raw for due, _, raw, event in self.rt.channel.queue if due <= step}
        wires, packets = result
        for packet, due in packets:
            raw = wires.pop(packet.sequence)
            assert packet.to_bytes() == raw, (step, 'wire roundtrip', packet.sequence)
            self.add(step, 'rx', dict(wire_hex=raw.hex(), packet=packet.sequence, kind=packet.kind.name, due=due), packet.receiver)
        assert not wires, (step, 'due packets left over')

    def deliver(self, when, args, kwargs, result):
        packet, due = args
        if packet.send_step == self.send_step and packet.sender == 0 and packet.receiver == 1:
            self.add(self.rt.step, 'delivery_' + when,
                     dict(packet=packet.sequence, due=due, **receiver_state(self.rt, packet.receiver)), packet.receiver)

    def astar(self, agent):
        def observe(when, args, kwargs, result):
            if when == 'before':
                return plain(dict(start=args[0], goal=args[1], role=kwargs.get('role', args[2] if len(args) > 2 else 'searcher')))
            query, answer = result
            self.add(self.rt.step, 'astar', dict(**query, answer=answer), agent)
        return observe

    def dynamics(self, when, args, kwargs, result):
        rt, raw = self.rt, self.rt.physical
        if when == 'before':
            if not self.prefix:
                self.add(rt.step, 'local_maps', map_digest(rt))
            return dict(step=rt.step, agents=navigation(rt))
        frame = result[0]
        frame.update(next_positions=raw._agent_pos.tolist(), collisions=plain(raw.last_collision_records))
        self.add(frame['step'], 'trajectory', frame)
        self.add(frame['step'], 'control', dict(applied_residual_action=args[0], prior_acc=raw._last_prior_acc,
            residual_acc=raw._last_residual_acc, acceleration=raw._agent_acc, post_velocity=raw._agent_vel,
            post_position=raw._agent_pos))

    def attach(self, stack):
        rt = self.rt
        for owner, name, observer in ((rt, '_send', self.send), (rt.channel, 'receive', self.receive),
                (rt, '_deliver', self.deliver), (rt.physical, '_apply_agent_dynamics', self.dynamics)):
            stack.enter_context(tap_method(owner, name, observer))
        for i, local in enumerate(rt.locals):
            stack.enter_context(tap_method(local.mapper, 'grid_astar_path', self.astar(i)))

    def advance(self, policy):
        step = self.rt.step
        self.add(step, 'transition', self.rt.advance(model=policy, explore=False))


def matching(rows, event, *, step=None, agent=None):
    return [r for r in rows if r['event'] == event and (step is None or r['step'] == step)
            and (agent is None or r['agent'] == agent)]


def reference_signature(events):
    comparable = [r for r in events if r['event'] not in ('delivery_before', 'delivery_after')]
    names = sorted({row['event'] for row in comparable})
    return dict(events=len(comparable), sha256=digest(comparable),
                by_event={name: digest(matching(comparable, name)) for name in names})
