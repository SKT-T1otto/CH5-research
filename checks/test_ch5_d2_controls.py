from unittest import TestCase
from unittest.mock import patch

import numpy as np

from chapter5_comm.experiment_config import arm_settings, plan_summary, read_experiment
from chapter5_comm.experiment_runner import build_objects
from chapter5_comm.pipeline import scenarios
from chapter5_comm.runtime import CH5Runtime, TargetState


class D2ControlsTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.base = read_experiment("configs/chapter5/d2_check_experiment_v1.json")

    def test_two_arms_share_navigation_and_zero_has_no_task_learner(self):
        short, _ = read_experiment("configs/chapter5/d2_short_experiment_v1.json")
        self.assertEqual(plan_summary(short)["rollout_limit"], 400)
        self.assertEqual(short["navigation"], "causal_d2_v1")
        self.assertEqual(arm_settings(short, "d2_zero")[-1], "neural_off")
        self.assertEqual(arm_settings(short, "d2_b2")[-1], "neural_off")
        _, model, _, _, task, _ = build_objects(self.config, self.base, 4317, "d2_zero", "cpu")
        self.assertIsNone(task)
        self.assertTrue(all(not action.any() for action in model.step(None)))

    def test_missing_peers_and_intercept_use_only_receiver_views(self):
        runtime = CH5Runtime(self.base, scenarios(1, 699)[0], navigation="causal_d2_v1")
        self.addCleanup(runtime.close)
        self.assertTrue(all(not local.peers for local in runtime.locals))
        self.assertLessEqual(set(runtime.locals[2].allocation), {2})
        local = runtime.locals[3]
        state = TargetState(np.array([10., 11., 3.]), np.array([.1, 0., 0.]), 0,
                            runtime.physical.target_motion_mode)
        local.target = state
        predicted = np.array([12., 11., 3.])
        with patch.object(local.mapper, "grid_geodesic_cost", return_value={"reachable": True, "cost": 3.}) as geo:
            with patch("chapter5_comm.runtime.solve_intercept_point") as solver:
                def intercept(position, payload, step, travel_time, **options):
                    np.testing.assert_array_equal(payload.position, state.position)
                    self.assertEqual(travel_time(position, predicted), 3.)
                    return {"reachable": True, "position": predicted}
                solver.side_effect = intercept
                np.testing.assert_array_equal(runtime._d2_goal(3, runtime.physical._agent_pos[3]), predicted)
                self.assertEqual(geo.call_args.kwargs["role"], "executor")
                runtime._d2_goal(3, runtime.physical._agent_pos[3])
                self.assertEqual(solver.call_count, 1)
