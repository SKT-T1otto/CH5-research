"""Byte-exact CH5 frames and a bounded causal TDMA acoustic channel.

No physics or task rewards are defined here. All frame types use the same
serialization, token bucket, loss and delay process, including ACK/control.
"""
from dataclasses import dataclass
from enum import IntEnum
import heapq
import math
import struct

import numpy as np


class Kind(IntEnum):
    STATE = 1
    SCAN = 2
    TARGET = 3
    QUERY = 4
    COMMAND = 5
    ACK = 6
    MAP_LATENT = 7
    DYNAMIC_LATENT = 8
    TARGET_LATENT = 9


HEADER = struct.Struct("<BBBBIHHI")
assert HEADER.size == 16


@dataclass(frozen=True)
class Packet:
    kind: Kind
    sender: int
    receiver: int
    sequence: int
    sample_step: int
    send_step: int
    payload: bytes

    def to_bytes(self):
        if self.sender not in range(4) or self.receiver not in range(4) or self.sender == self.receiver:
            raise ValueError("frame endpoints must be distinct existing agents")
        if not 0 <= self.sequence < 2 ** 32:
            raise ValueError("sequence must fit uint32")
        if not 0 <= self.sample_step <= self.send_step < 2 ** 16:
            raise ValueError("sample/send steps must be causal uint16 values")
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must already be serialized bytes")
        return HEADER.pack(1, int(self.kind), self.sender, self.receiver,
                           self.sequence, self.sample_step, self.send_step, len(self.payload)) + self.payload

    @classmethod
    def from_bytes(cls, frame):
        if len(frame) < HEADER.size:
            raise ValueError("truncated header")
        version, kind, sender, receiver, sequence, sample, sent, length = HEADER.unpack_from(frame)
        if version != 1 or len(frame) != HEADER.size + length:
            raise ValueError("unsupported frame version or payload length")
        packet = cls(Kind(kind), sender, receiver, sequence, sample, sent, bytes(frame[HEADER.size:]))
        packet.to_bytes()
        return packet

    @property
    def bits(self):
        return 8 * len(self.to_bytes())


class TDMAChannel:
    """One known sender per physical step, with independently seeded faults.

    A single bootstrap frame plus learned data can share the same sender's
    token bucket. The simulator may use true distances for propagation, never
    exposing those distances to the sender's policy.
    """
    def __init__(self, *, seed=1, capacity_bits=8192, refill_bits=2048,
                 loss_probability=0.1, delay_steps=1, jitter_steps=2,
                 rate_bps=9600., dt=.2, sound_speed=1500.,
                 burst_start_probability=0., burst_length=4, burst_loss_probability=.9):
        if capacity_bits <= 0 or refill_bits < 0 or rate_bps <= 0 or dt <= 0 or sound_speed <= 0:
            raise ValueError("invalid channel capacity/time/rate")
        if any(not 0 <= x <= 1 for x in (loss_probability, burst_start_probability, burst_loss_probability)):
            raise ValueError("fault probabilities must lie in [0,1]")
        if delay_steps < 0 or jitter_steps < 0 or burst_length < 1:
            raise ValueError("invalid delay/burst duration")
        self.seed = int(seed)
        self.capacity_bits, self.refill_bits = int(capacity_bits), int(refill_bits)
        self.loss_probability = float(loss_probability)
        self.delay_steps, self.jitter_steps = int(delay_steps), int(jitter_steps)
        self.rate_bps, self.dt, self.sound_speed = float(rate_bps), float(dt), float(sound_speed)
        self.burst_start_probability = float(burst_start_probability)
        self.burst_length, self.burst_loss_probability = int(burst_length), float(burst_loss_probability)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.tokens = np.full(4, self.capacity_bits, dtype=np.int64)
        self.last_refill_step = 0
        self.queue, self.events = [], []
        self.sequence = 0
        self.sent_bits = self.delivered_bits = self.dropped_frames = self.rejected_frames = 0
        self.sent_by_kind, self.delivered_by_kind = {}, {}
        self.burst_until = np.zeros((4, 4), dtype=np.int64)
        self.tx_free_time = 0.
        self._tie = 0

    @staticmethod
    def owner(step):
        return int(step) % 4

    def refill(self, step):
        step = int(step)
        if step < self.last_refill_step:
            raise ValueError("channel clock cannot go backwards")
        delta = step - self.last_refill_step
        self.tokens = np.minimum(self.capacity_bits, self.tokens + delta * self.refill_bits)
        self.last_refill_step = step

    def remaining_bits(self, sender):
        return int(self.tokens[int(sender)])

    def send(self, kind, sender, receiver, payload, *, sample_step, step, distance=0., sequence=None):
        self.refill(step)
        sender, receiver, step = int(sender), int(receiver), int(step)
        if sender != self.owner(step):
            self.rejected_frames += 1
            return {"accepted": False, "reason": "not_tdma_owner", "bits": 0}
        seq = self.sequence if sequence is None else int(sequence)
        packet = Packet(Kind(kind), sender, receiver, seq, int(sample_step), step, bytes(payload))
        frame = packet.to_bytes()
        bits = 8 * len(frame)
        if bits > self.tokens[sender]:
            self.rejected_frames += 1
            return {"accepted": False, "reason": "budget", "bits": bits}
        self.sequence = max(self.sequence + 1, seq + 1)
        self.tokens[sender] -= bits
        self.sent_bits += bits
        name = packet.kind.name
        self.sent_by_kind[name] = self.sent_by_kind.get(name, 0) + bits
        if step >= self.burst_until[sender, receiver] and self.rng.random() < self.burst_start_probability:
            self.burst_until[sender, receiver] = step + self.burst_length
        loss = self.burst_loss_probability if step < self.burst_until[sender, receiver] else self.loss_probability
        dropped = bool(self.rng.random() < loss)
        serialization_seconds = bits / self.rate_bps
        propagation_seconds = max(0., float(distance)) / self.sound_speed
        jitter = int(self.rng.integers(self.jitter_steps + 1))
        # All senders share one acoustic serialization queue, with sub-step
        # precision. Independently rounding each frame would fabricate backlog;
        # independent per-sender queues would fabricate parallel bandwidth.
        start_time = max(step * self.dt, self.tx_free_time)
        self.tx_free_time = start_time + serialization_seconds
        physical_arrival = math.ceil((self.tx_free_time + propagation_seconds) / self.dt - 1e-12)
        due = max(step + 1, physical_arrival + self.delay_steps + jitter)
        event = {"packet_id": seq, "kind": name, "sender": sender, "receiver": receiver,
                 "sample_step": int(sample_step), "send_step": step, "delivery_step": None,
                 "scheduled_delivery_step": due, "bits": bits, "dropped": dropped}
        self.events.append(event)
        if dropped:
            self.dropped_frames += 1
        else:
            self._tie += 1
            heapq.heappush(self.queue, (due, self._tie, frame, event))
        return {"accepted": True, "reason": "sent", "bits": bits, "packet_id": seq,
                "dropped": dropped, "scheduled_delivery_step": due}

    def receive(self, step):
        self.refill(step)
        result = []
        while self.queue and self.queue[0][0] <= int(step):
            due, _, frame, event = heapq.heappop(self.queue)
            packet = Packet.from_bytes(frame)
            event["delivery_step"] = int(due)
            self.delivered_bits += len(frame) * 8
            name = packet.kind.name
            self.delivered_by_kind[name] = self.delivered_by_kind.get(name, 0) + len(frame) * 8
            result.append((packet, int(due)))
        return result

    def summary(self):
        return {"sent_bits": self.sent_bits, "delivered_bits": self.delivered_bits,
                "dropped_frames": self.dropped_frames, "rejected_frames": self.rejected_frames,
                "in_flight_frames": len(self.queue), "sent_bits_by_kind": dict(self.sent_by_kind),
                "delivered_bits_by_kind": dict(self.delivered_by_kind),
                "remaining_bits": self.tokens.tolist(), "header_bytes": HEADER.size}
