from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from chapter5_comm.d2_batch import _report
from chapter5_comm.experiment_config import arm_settings, plan_summary, read_experiment
from chapter5_comm.experiment_runner import build_objects
from chapter5_comm.pipeline import scenarios
from chapter5_comm.runtime import CH5Runtime, TargetState
from chapter5_comm.search_braking import stopping_risk


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

    def test_handoff_summary_keeps_first_and_last_receive_steps_separate(self):
        runtime = CH5Runtime(self.base, scenarios(1, 699)[0])
        self.addCleanup(runtime.close)
        runtime.physical.found_step = 2
        for sent, received in [(2, 6), (8, 11)]:
            state = TargetState(np.array([10., 11., 3.]), np.array([.1, 0., 0.]), sent,
                                runtime.physical.target_motion_mode)
            runtime._accept_target(3, state, delivery_step=received)
        with patch.object(runtime.env, "finalize_episode", return_value={}):
            self.assertEqual(runtime.summary()["handoff_event_step"], 6)
            self.assertEqual(runtime.summary()["last_target_received_step"], 11)
        self.assertEqual(runtime.physical.last_handoff_delay, 4)

    def test_stopping_risk_only_uses_hits_ahead_of_boat(self):
        mapper = type("Mapper", (), {
            "planner_obstacle_clearance": .4,
            "sensor_directions": torch.tensor([[1., 0., 0.], [-1., 0., 0.]]),
            "segment_is_free": staticmethod(lambda start, end: True),
        })()
        pos = np.ones(3)
        options = dict(dt=.2, sensor_range=4.5, bounds=(20, 20, 8), prior_gain=(1.1, 1.),
                       prior_strength=1., acceleration_limit=(1., .7))
        scan = (0, pos, np.array([.5, .5]), np.array([True, False]))
        self.assertEqual(stopping_risk(pos, np.array([1., 0., 0.]), mapper, scan, **options), "sonar")
        self.assertIsNone(stopping_risk(pos, np.array([-1., 0., 0.]), mapper, scan, **options))

    def test_factorial_report_excludes_censored_success_from_paired_counts(self):
        baseline = dict(case=0, condition="fault", arm="baseline", prefix_cutoff=False, collision=True,
                        search_collision=True, found=False, success=False, timed_out=False,
                        channel={"sent_bits": 960}, braking_events=[])
        brake = dict(baseline, arm="brake_only", prefix_cutoff=True, collision=False, search_collision=False,
                     success=None, timed_out=None)
        report = _report([baseline, brake], ["fault"], observation_limit=25, expected=8)
        pair = report["paired"]["fault"]["baseline_vs_brake_only"]
        self.assertEqual(pair["collision"]["left_only"], 1)
        self.assertEqual(pair["success"]["observed"], 0)
        self.assertFalse(report["complete"])
