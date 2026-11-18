"""Chapter 3 D2 allocation on a single receiver's causal planning view."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from chapter3_bser.candidate_generator import generate_search_candidates, generate_standby_candidates
from chapter3_bser.config import load_bser_phase1a1_config
from chapter3_bser.d2.solver import fixed_greedy, solve
from chapter3_bser.objective import build_objective_context
from core.mapping.planning_state import extract_planning_state
from core.mapping.travel_cost_service import TravelCostService


class LocalD2Allocator:
    def __init__(self):
        self.config = load_bser_phase1a1_config()

    def allocate(self, *, mapper, step, receiver, position, velocity, peers, specs, standby_goal=None):
        reports = {i: (row[1], row[2]) for i, row in peers.items()}
        reports[receiver] = (position, velocity)
        ids = sorted(reports)
        if ids == [3]:
            return {}
        # The inherited view indexes agents by position in the tuple. Compact
        # the set of known reports and translate IDs at the packet boundary.
        agents = SimpleNamespace(
            role_order=[specs[i]["name"] for i in ids],
            positions=[reports[i][0] for i in ids],
            velocities=[reports[i][1] for i in ids],
            navigation_targets=[standby_goal if i == receiver == 3 else reports[i][0] for i in ids],
        )
        view = SimpleNamespace(
            get_task_state=lambda: SimpleNamespace(step=step, target_found=False, mission_complete=False),
            get_agent_state=lambda: agents,
            get_scenario_identity=lambda: {"obstacle_knowledge_mode": "online_unknown"},
            unwrapped=SimpleNamespace(map_module=mapper, agent_specs=[specs[i] for i in ids]),
        )
        state = replace(extract_planning_state(view),
                        searcher_ids=tuple(j for j, i in enumerate(ids) if i < 3),
                        executor_id=ids.index(3) if 3 in reports else 0)
        generation = self.config["candidate_generation"]
        service = TravelCostService(state)
        candidates, _, _ = generate_search_candidates(
            state, service, k_search=int(generation["k_search_exact"]),
            minimum_separation=float(generation["minimum_separation"]),
            maximum_travel_time=float(generation["maximum_physical_travel_time"]),
        )
        standbys = ()
        if receiver == 3:
            standbys, _, _ = generate_standby_candidates(state, service, k_standby=int(generation["k_standby_exact"]))
        context = build_objective_context(state, candidates, standbys, self.config)
        result = solve("D2", candidates, standbys, context) if standbys else fixed_greedy(candidates, None, context, pure=True)
        goals = {ids[c.agent_id]: np.asarray(c.waypoint) for c in result.selected}
        if result.standby is not None:
            goals[3] = np.asarray(result.standby.waypoint)
        return goals
