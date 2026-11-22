"""Deterministic own-scan sufficient-statistic transport (no neural inference).

120 payload bytes, exactly like SCAN: 8-byte header and fourteen 8-byte cells.
Grid identity is public mission configuration. Source snapshots replace earlier
snapshots; they are never re-added. No forwarding of received map evidence.
"""
import struct
import numpy as np
import torch

PREFIX = struct.Struct('<4sBBH')
CELL = struct.Struct('<HHhH')  # flat index, last observation, logodds/1024, count
SIZE = 120


def encode(mapper, previous=()):
    odds = mapper.occupancy_logodds.detach().cpu().numpy().reshape(-1)
    last = mapper.occupancy_last_observed_step.cpu().numpy().reshape(-1)
    counts = mapper.occupancy_observation_count.cpu().numpy().reshape(-1)
    # Refresh previously sent cells too, including later free evidence. This
    # prevents stale occupied snapshots persisting merely because rank changed.
    indices = np.flatnonzero(counts > 0)
    ranked = sorted(indices, key=lambda i: (-float(odds[i]), -int(counts[i]), -int(last[i]), int(i)))
    corrections = [i for i, q in dict(previous).items() if round(float(odds[i])*1024) != q]
    selected = list(dict.fromkeys([*corrections, *ranked]))[:14]
    entries = [(int(i), int(last[i]), int(np.clip(round(float(odds[i])*1024), -32768, 32767)),
                min(int(counts[i]), 65535)) for i in selected]
    body = PREFIX.pack(b'OEV1', len(entries), 0, len(odds))
    body += b''.join(CELL.pack(*row) for row in entries)
    return body.ljust(SIZE, b'\0'), entries


def decode(body, cells, sample_step):
    if len(body) != SIZE:
        raise ValueError('evidence must have SCAN payload length')
    magic, count, reserved, size = PREFIX.unpack_from(body)
    if magic != b'OEV1' or count > 14 or reserved or size != cells:
        raise ValueError('invalid evidence grid/header')
    rows = [CELL.unpack_from(body, PREFIX.size+i*CELL.size) for i in range(count)]
    if len({r[0] for r in rows}) != len(rows) or any(i >= cells or t > sample_step or n == 0 for i,t,q,n in rows):
        raise ValueError('invalid or noncausal evidence cell')
    return rows


def fuse(local):
    mapper = local.mapper
    total = local.own_evidence.occupancy_logodds.clone().reshape(-1)
    for source in local.remote_evidence.values():
        for index, (stamp, quantized, count) in source.items():
            total[index] += quantized / 1024.
    mapper.occupancy_logodds.copy_(total.reshape_as(mapper.occupancy_logodds).clamp(
        -mapper.occupancy_logodds_clip, mapper.occupancy_logodds_clip))
    mapper._refresh_online_masks()
