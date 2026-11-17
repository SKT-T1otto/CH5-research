"""Reset regressions: native pilot case, no resampling and no physical steps."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import numpy as np

from core.mapping.path_planner import OnlineUnknownMapTaskPlanner
from chapter5_comm.runtime import CH5Runtime, _BootstrapEndpointGuard, _CH5BootstrapPlanner
from chapter5_comm.pipeline import PROJECT
from chapter5_comm.experiment_config import scene_seed


class _BoundaryPlanner(_BootstrapEndpointGuard):
    n_search = 3
    space_size = torch.tensor([20., 20., 8.])

    def __init__(self):
        self.called = []
        self.occupied = {1}

    def _as_points(self, values):
        return torch.as_tensor(values, dtype=torch.float32)

    def endpoint_status(self, point, role="searcher"):
        valid = int(point[0]) not in self.occupied
        return dict(reachable=valid, failure_reason=None if valid else "point_invalid")

    def sample_next_waypoint(self, agent, position, reserved_positions):
        if not self.endpoint_status(position)["reachable"]:
            raise RuntimeError("current_pos is not a legal reachable planner point")
        self.called.append(agent)
        return position + torch.tensor([.25, 0., 0.])


class ResetBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_deferred_start_is_not_sampled_moved_or_declared_free(self):
        p = _BoundaryPlanner()
        positions = torch.tensor([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]])
        before = positions.clone()
        targets = p.initial_search_targets(positions)
        self.assertEqual(p.called, [1, 2])
        self.assertTrue(torch.equal(targets[0], positions[0]))
        self.assertTrue(torch.equal(before, positions))
        self.assertFalse(p.endpoint_status(positions[0])["reachable"])
        self.assertEqual(p.ch5_bootstrap_endpoint_deferrals, [dict(agent=0, reason="point_invalid")])

    def test_invalid_coordinates_are_not_silently_accepted(self):
        for value in (float("nan"), -1., 21.):
            positions = torch.ones(3, 3)
            positions[0, 0] = value
            with self.assertRaises(ValueError):
                _BoundaryPlanner().initial_search_targets(positions)

    def test_reachable_initial_targets_match_inherited_planner_exactly(self):
        kwargs = dict(space_size=(20, 20, 8), grid_size=(10, 10, 8), z_range=(.8, 7.2), device="cpu")
        positions = torch.tensor([[2., 3., 2.], [5., 6., 3.], [12., 14., 5.]])
        original, guarded = OnlineUnknownMapTaskPlanner(**kwargs), _CH5BootstrapPlanner(**kwargs)
        original.reset(positions, [])
        guarded.reset(positions, [])
        torch.manual_seed(71)
        expected = original.initial_search_targets(positions)
        torch.manual_seed(71)
        actual = guarded.initial_search_targets(positions)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(original.claim_count, guarded.claim_count))
        self.assertTrue(torch.equal(original.occupancy_logodds, guarded.occupancy_logodds))
        self.assertEqual(guarded.ch5_bootstrap_endpoint_deferrals, [])

    def test_original_native_pilot_scene_fails_then_guarded_reset_preserves_scene(self):
        scene = json.loads((PROJECT / "checks/fixtures/ch5_pilot_collection_1.json").read_text(encoding="utf-8"))
        base = json.loads((PROJECT / "configs/b2.json").read_text(encoding="utf-8"))
        seed = scene_seed(4317, "collection", 1) % (2 ** 31 - 1)
        with patch.object(_CH5BootstrapPlanner, "initial_search_targets", OnlineUnknownMapTaskPlanner.initial_search_targets):
            with self.assertRaisesRegex(RuntimeError, "current_pos is not a legal reachable planner point"):
                CH5Runtime(base, scene, seed=seed)
        runtime = CH5Runtime(base, scene, seed=seed)
        try:
            self.assertEqual(runtime.step, 0)
            self.assertEqual(runtime.physical.max_steps, 400)
            np.testing.assert_allclose(runtime.teacher_truth()["agent_positions"], scene["initial_agent_positions"], atol=1e-6)
            np.testing.assert_allclose(runtime.teacher_truth()["target_position"], scene["target_initial_position"], atol=1e-6)
            np.testing.assert_allclose(runtime.teacher_truth()["target_velocity"], scene["target_initial_velocity"], atol=1e-6)
            self.assertEqual(len(runtime.physical.ground_truth_obstacles), len(scene["obstacles"]))
            for actual, expected in zip(runtime.physical.ground_truth_obstacles, scene["obstacles"]):
                for key in ("center", "size"):
                    # The unchanged parent environment stores geometry in its
                    # float32 physical tensors; compare the same representation.
                    np.testing.assert_array_equal(actual[key], np.asarray(expected[key], dtype=np.asarray(actual[key]).dtype))
            self.assertTrue(runtime.physical.map_module.ch5_bootstrap_endpoint_deferrals)
            self.assertTrue(all(local.mapper is not runtime.physical.map_module for local in runtime.locals))
            self.assertFalse(any(local.target is not None for local in runtime.locals))
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
