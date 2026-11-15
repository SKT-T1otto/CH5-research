"""Deterministic geometry/schema checks; these are not task training data."""
import json
import unittest

import numpy as np

from chapter5_comm.data import (
    ROUTE_ORIGIN,
    TrajectoryRecorder,
    build_route_encounter_labels,
    labels_from_rollout,
)


class FactualLabelContracts(unittest.TestCase):
    def test_swept_crossing_is_detected_when_endpoints_are_outside_radius(self):
        routes = np.array([[[-1., 0, 0], [1., 0, 0]], [[-1., 1, 0], [1., 1, 0]]])
        result = build_route_encounter_labels(routes, np.zeros((2, 3)), 1, .1)
        np.testing.assert_array_equal(result["route_labels"], [1., 0.])
        np.testing.assert_array_equal(result["route_valid"], [True, True])
        np.testing.assert_array_equal(result["route_first_hit_steps"], [1, 0])

    def test_censored_negative_is_not_filled_but_known_positive_survives(self):
        routes = np.array([[[-1., 0, 0], [1., 0, 0], [2., 0, 0], [3., 0, 0]],
                           [[-1., 1, 0], [1., 1, 0], [2., 1, 0], [3., 1, 0]]])
        result = build_route_encounter_labels(routes, np.zeros((2, 3)), 3, .1)
        np.testing.assert_array_equal(result["route_labels"], [1., 0.])
        np.testing.assert_array_equal(result["route_valid"], [True, False])
        np.testing.assert_array_equal(result["route_censored"], [True, True])
        np.testing.assert_array_equal(result["route_observed_intervals"], [1, 1])

    def test_gap_and_truncated_route_do_not_produce_false_negatives(self):
        route = np.full((2, 4, 3), 2.)
        result = build_route_encounter_labels(
            route, np.zeros((4, 3)), 3, .1,
            route_valid_steps=np.array([3, 1]),
            target_valid=np.array([True, True, False, True]),
        )
        self.assertFalse(result["route_valid"].any())
        np.testing.assert_array_equal(result["route_observed_intervals"], [1, 1])

    @staticmethod
    def recording():
        recorder = TrajectoryRecorder("real_episode")
        recorder.add_frame(0, [0., 0, 0], [1., 0, 0])
        recorder.add_decision(
            0, {"map_tokens": [[.2, .3]], "receiver_query": [0., 1.]},
            [[[-1., 0, 0], [1., 0, 0], [2., 0, 0]]], [[1., 2., 3., 4., 5.]],
            receiver_id=3, sender_id=0,
        )
        recorder.add_frame(1, [0., 0, 0], [-1., 0, 0])
        return recorder

    def test_dynamic_target_uses_recorded_velocity_not_constant_motion_guess(self):
        recorder = self.recording()
        example = recorder.labels(2, .1, dynamic_horizon=1)[0]
        self.assertTrue(example["dynamic_target_valid"])
        self.assertEqual(example["dynamic_target"], [0., 0, 0, -1., 0, 0])
        self.assertEqual(example["future_target_valid"], [True, True, False])
        self.assertEqual(example["future_targets"][2], [0.] * 6)
        self.assertTrue(example["label_metadata"]["teacher_only"])

    def test_future_truth_changes_labels_without_changing_deployment_inputs(self):
        recorder = self.recording()
        first = recorder.labels(1, .1)[0]
        rows = json.loads("[" + ",".join(recorder.to_jsonl().splitlines()) + "]")
        rows[1]["ctde_target"]["position"] = [0., 4., 0]
        second = labels_from_rollout(rows, 1, .1)[0]
        self.assertEqual(first["inputs"], second["inputs"])
        self.assertEqual(first["route_features"], second["route_features"])
        self.assertNotEqual(first["route_labels"], second["route_labels"])

    def test_missing_future_dynamic_target_has_explicit_invalid_mask(self):
        example = self.recording().labels(2, .1, dynamic_horizon=2)[0]
        self.assertFalse(example["dynamic_target_valid"])
        self.assertEqual(example["dynamic_target"], [0.] * 6)

    def test_episode_boundaries_cannot_supply_future_states(self):
        recorder = self.recording()
        recorder.add_frame(2, [99., 99, 99], [0., 0, 0], episode_id="other_episode")
        example = recorder.labels(2, .1, dynamic_horizon=2)[0]
        self.assertFalse(example["dynamic_target_valid"])

    def test_receiver_specific_radius_is_recorded_in_each_teacher_label(self):
        recorder = TrajectoryRecorder()
        recorder.add_frame(0, [0, 0, 0], [0, 0, 0])
        routes = [[[-1., .25, 0], [1., .25, 0]]]
        for receiver, radius in ((0, .1), (1, .5)):
            recorder.add_decision(0, {}, routes, [[0.]], receiver_id=receiver, detect_radius=radius)
        recorder.add_frame(1, [0, 0, 0], [0, 0, 0])
        examples = recorder.labels(1, .01)
        self.assertEqual([x["route_labels"] for x in examples], [[0.], [1.]])
        self.assertEqual([x["label_metadata"]["detect_radius_world_units"] for x in examples], [.1, .5])

    def test_recorder_copies_inputs_and_json_roundtrip_retains_schema(self):
        recorder = self.recording()
        rows = [json.loads(line) for line in recorder.to_jsonl().splitlines()]
        source = rows[0]["decisions"][0]["inputs"]
        labels = labels_from_rollout(rows, 2, .1)
        labels[0]["inputs"]["map_tokens"][0][0] = 999.
        self.assertEqual(source["map_tokens"][0][0], .2)
        self.assertEqual(rows[0]["decisions"][0]["route_origin"], ROUTE_ORIGIN)

    def test_duplicate_frames_wrong_clock_and_teacher_features_are_rejected(self):
        recorder = self.recording()
        with self.assertRaises(ValueError):
            recorder.add_frame(1, [0, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            recorder.add_decision(0, {"ctde_target": [0, 0, 0]}, [[[0, 0, 0]]], [[0.]])
        rows = [json.loads(line) for line in recorder.to_jsonl().splitlines()]
        with self.assertRaises(ValueError):
            labels_from_rollout(rows + rows[:1], 2, .1)
        rows[0]["ctde_target"]["sample_step"] = 1
        with self.assertRaises(ValueError):
            labels_from_rollout(rows, 2, .1)
        rows[0]["ctde_target"]["sample_step"] = 0
        rows[0]["decisions"][0].pop("route_origin")
        with self.assertRaises(ValueError):
            labels_from_rollout(rows, 2, .1)


if __name__ == "__main__":
    unittest.main()
