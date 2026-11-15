"""Short-workflow guards; these tests do not run physical episodes."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import torch
import numpy as np

from chapter5_comm.pipeline import (
    PROJECT, auxiliary_batch, delivered_tokens, evaluation_summary, read_config,
)


class WorkflowContracts(unittest.TestCase):
    def test_original_task_identity_and_bounded_plan(self):
        config, parent = read_config("configs/chapter5/short_pipeline_v1.json")
        self.assertEqual(parent["max_steps"], 400)
        self.assertEqual(config["rollout_limit_steps"], 32)
        self.assertEqual(parent["rl"]["gamma"], .95)
        for patch in ({"rollout_limit_steps": 400}, {"task_updates": 5},
                      {"evaluation_modes": ["learned", "fixed", "neural_off"]}):
            invalid = dict(config, **patch)
            with TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_config(path)

    def test_received_int8_is_dequantized_exactly(self):
        codes = np.array([-127, 0, 64, 127], dtype=np.int8)
        actual = delivered_tokens([dict(codes=codes, tokens=np.full(4, 99.))])
        self.assertTrue(torch.equal(actual[0, 0], torch.tensor(codes.astype(float), dtype=torch.float32) / 127))
        with self.assertRaises(ValueError):
            delivered_tokens([dict(codes=codes.astype(float))])

    def test_state_units_and_target_knowledge_mask(self):
        keys = ["map_tokens", "dynamic_tokens", "receiver_query", "query_available",
                "map_mask", "dynamic_mask", "received", "ages", "local_geometry"]
        row = dict(inputs={k: [0] for k in keys}, route_features=[[0] * 5],
                   route_valid=[True], route_labels=[False], dynamic_target=[20, 10, 4, 2.5, 0, -2.5],
                   dynamic_target_valid=True)
        row["inputs"]["state_observable"] = True
        unknown = deepcopy(row)
        unknown["inputs"]["state_observable"] = False
        batch = auxiliary_batch([row, unknown])
        np.testing.assert_allclose(batch["state_target"][0], [1, .5, .5, 1, 0, -1])
        np.testing.assert_array_equal(batch["state_valid"], [True, False])

    def test_external_truncation_is_not_full_task_success_rate(self):
        summary = evaluation_summary([dict(completed_task_episode=False, budget_truncated=True,
            success=False, observed_return=1., neural_delivery_applications=2)])
        self.assertIsNone(summary["full_task_success_rate"])
        self.assertEqual(summary["budget_truncated"], 1)


if __name__ == "__main__":
    unittest.main()
