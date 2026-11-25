"""Deterministic two-cell map-frame baselines. All arguments are sender-local."""
from copy import copy
import hashlib
import struct

import numpy as np

from . import history_evidence as hist

RULES = ('native', 'no_query', 'straight', 'route', 'native_tie', 'route_tie')


def route_geometry(mapper, query, receiver, use_astar):
    if query is None:
        return None, 'missing_query'
    bounds = mapper.space_size.detach().cpu().numpy()
    values = np.asarray(query, dtype=float)
    start, goal = values[:3] * bounds, values[3:6] * bounds
    if np.linalg.norm(start-goal) <= 1e-5:
        return None, 'stationary_query'
    path = [start, goal]
    if use_astar:
        # A* writes caches and diagnostic counters. Keep them private to this lookup.
        planner = copy(mapper)
        planner._geodesic_cache = {}
        res = planner.grid_astar_path(start, goal, role='executor' if receiver == 3 else 'searcher')
        if not res['reachable']:
            return None, 'unreachable_query'
        path = res['points']
    scale = np.asarray([mapper.cell_dx, mapper.cell_dy, mapper.cell_dz])
    points = mapper.flat_xyz_centers.detach().cpu().numpy() / scale
    path = np.asarray(path) / scale
    distance = np.linalg.norm(points-path[0], axis=1)
    for left, right in zip(path,path[1:]):
        delta = right-left
        u = np.clip((points-left) @ delta / max(delta @ delta, 1.e-20), 0., 1.)
        distance = np.minimum(distance, np.linalg.norm(points-left-u[:,None]*delta,axis=1))
    return np.round(distance,6), 'available'


def choose(rule, native, candidates, odds, previous, distance=None, *, context=(0,0,0), seed=11):
    """Keep native corrections and edit only the last two unprotected entries."""
    if rule not in RULES:
        raise ValueError('unknown map selection rule')
    if rule == 'native' or (rule in ('straight','route','route_tie') and distance is None):
        return list(native)
    lookup = {r[0]:r for r in candidates}
    corrected = {i for i,_,q,_ in native if i in previous and previous[i] != q}
    base = [r[0] for r in native]
    tail = [i for i in base if i not in corrected][-2:]
    if not tail:
        return list(native)
    fixed = set(base)-set(tail)
    def rank(i):
        _, stamp, value, count = lookup[i]
        if rule == 'native_tie':
            key = (-float(odds[i]),-count,-stamp)
        else:
            key = (int(previous.get(i)==value),)
            if distance is not None and rule!='no_query':
                key += (distance[i],)
            key += (-stamp,-abs(value))
        if rule.endswith('_tie'):
            tie = hashlib.blake2b(struct.pack('<5I',seed,*context,i),digest_size=8).digest()
            return (*key,tie,i)
        return (*key,i not in tail,i)
    winners = sorted((i for i in lookup if i not in fixed),key=rank)[:len(tail)]
    retain = fixed|set(winners)
    return [r for r in native if r[0] in retain]+[lookup[i] for i in winners if i not in base]


def encode(own, sender_map, previous, native_body, native, rule, query, sender, receiver, step, seed):
    odds = own.occupancy_logodds.detach().cpu().numpy().reshape(-1)
    count = own.occupancy_observation_count.detach().cpu().numpy().reshape(-1)
    stamp = own.occupancy_last_observed_step.detach().cpu().numpy().reshape(-1)
    distance = None
    if rule in ('route','route_tie','straight'):
        distance,_ = route_geometry(sender_map,query,receiver,rule!='straight')
        if distance is None:
            return native_body,native
    rows = [(int(i),int(stamp[i]),int(np.clip(round(float(odds[i])*1024),-32768,32767)),min(65535,int(count[i])))
            for i in np.flatnonzero(count>0)]
    selected = choose(rule,native,rows,odds,previous,distance,context=(sender,receiver,step),seed=seed)
    body = native_body[:hist.PREFIX.size]+b''.join(hist.CELL.pack(*r) for r in selected)
    return body.ljust(hist.SIZE,b'\0'),selected
