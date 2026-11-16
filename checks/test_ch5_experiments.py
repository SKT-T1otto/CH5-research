"""Experiment contracts and episode-boundary restoration; no physical rollout."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
from types import SimpleNamespace

import torch
import numpy as np

from chapter5_comm.experiment_config import (read_experiment, plan_summary, arm_settings, scene_seed)
from chapter5_comm.experiment_runner import (execution_guard, build_objects, restore_objects, device_name)
from chapter5_comm.experiment_state import (replay_state, load_replay, rng_state, restore_rng,
    save_state, load_state, latest_receipt, exclusive_job, atomic_json)
from chapter5_comm.experiment_report import seed_interval
from chapter5_comm.experiment_workflow import main
from chapter5_comm.experiment_report import report
from chapter5_comm.experiment_runner import job_path
from chapter5_comm.experiment_workflow import suite
from chapter5_comm.pipeline import _encode


class ExperimentContracts(unittest.TestCase):
    def setUp(self):
        self.c, self.base = read_experiment("configs/chapter5/check_experiment_v1.json")
        torch.set_num_threads(1)

    def test_frozen_presets_and_dry_run_never_start_training(self):
        self.assertEqual(plan_summary(self.c)["maximum_physical_steps"], 120)
        for name in ("pilot", "formal"):
            c, _ = read_experiment(f"configs/chapter5/{name}_experiment_v1.json")
            self.assertEqual(c["rollout_limit_steps"], 400)
            self.assertEqual(len(c["evaluation_channels"]), 7)
        with patch("chapter5_comm.experiment_workflow.train", side_effect=AssertionError("unexpected execution")):
            with redirect_stdout(StringIO()):
                result = main(["train", "--config", "configs/chapter5/check_experiment_v1.json"])
        self.assertFalse(result["executed"])
        with patch("platform.system", return_value="Windows"):
            c = dict(self.c, kind="formal")
            with self.assertRaises(ValueError):
                execution_guard(c)

    def test_scene_namespaces_are_disjoint_and_test_is_not_training_seed_selected(self):
        values = {scene_seed(4317, s, i) for s in ("collection", "task", "communication", "validation", "test") for i in range(20)}
        self.assertEqual(len(values), 100)
        self.assertEqual(scene_seed(4317, "test", 0), scene_seed(5329, "test", 0))
        self.assertNotEqual(scene_seed(4317, "task", 0), scene_seed(5329, "task", 0))

    def test_transport_ablations_change_the_equations(self):
        c = deepcopy(self.c)
        c["arms"] += ["no_geometry", "no_transport"]
        z, age = torch.randn(3, 32), torch.ones(3)
        for arm in ("no_geometry", "no_transport"):
            objects = build_objects(c, self.base, 7, arm, "cpu")
            decoder = objects[0].decoder
            a = decoder.transport(z, torch.zeros_like(z), age)
            b = decoder.transport(z, torch.ones_like(z), age)
            self.assertTrue(torch.equal(a, b))
            if arm == "no_transport":
                self.assertTrue(torch.equal(a, z))

    def test_no_query_uses_the_same_unknown_query_during_deployment(self):
        c = deepcopy(self.c)
        c["arms"].append("no_query")
        n = build_objects(c, self.base, 7, "no_query", "cpu")[0]
        x = dict(map_tokens=np.ones((2, 7)), dynamic_tokens=np.ones((2, 8)),
            receiver_query=np.ones(9), query_available=True, map_mask=np.ones(2, bool), dynamic_mask=np.ones(2, bool))
        a = _encode(n, x).codes
        x["receiver_query"][:] = -987
        self.assertTrue(torch.equal(a, _encode(n, x).codes))

    def test_new_checkpoint_restores_replay_optimizer_rng_and_next_update(self):
        c = deepcopy(self.c)
        c["task_episodes"] = 2
        objects = build_objects(c, self.base, 7, "full", "cpu")
        n, model, auxiliary, communication, task, _ = objects
        transition = dict(observations=np.zeros((4, 28)), next_observations=np.ones((4, 28)) * .01,
            actions=np.zeros((4, 3)), rewards=np.ones(4) * .1, dones=np.zeros(4),
            action_enabled=np.ones(4, bool), next_action_enabled=np.ones(4, bool))
        for _ in range(10):
            task.observe(transition)
        task.update()
        model.agents[0].noise.sigma = .123
        identity = dict(config_sha="test", seed=7, arm="full")
        progress = dict(serial=1, complete=False)
        with TemporaryDirectory() as d:
            receipt = save_state(d, identity=identity, progress=progress, networks=n, model=model,
                auxiliary=auxiliary, communication=communication, task=task, task_clock=10)
            payload = load_state(d, receipt["path"], identity)
            self.assertEqual(latest_receipt(d)["sha256"], receipt["sha256"])
            expected = task.update()
            target = [x.detach().clone() for x in model.agents[0].policy.parameters()]
            restored = build_objects(c, self.base, 7, "full", "cpu")
            restore_objects(payload, restored, "cpu")
            actual = restored[4].update()
            self.assertEqual(actual["agents"], expected["agents"])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(target, restored[1].agents[0].policy.parameters())))
            self.assertEqual(restored[1].agents[0].noise.sigma, .123)
            with self.assertRaises(ValueError):
                load_state(d, receipt["path"], dict(identity, arm="raw_only"))
            with patch("chapter5_comm.experiment_state.source_identity", return_value={}):
                with self.assertRaises(ValueError):
                    load_state(d, receipt["path"], identity)

    def test_gate_ring_is_validated_and_sampling_rng_is_restored(self):
        objects = build_objects(self.c, self.base, 7, "full", "cpu")
        replay = objects[4].replay
        state = replay_state(replay)
        state["gate_initialized"] = torch.ones(1)
        with self.assertRaises(ValueError):
            load_replay(replay, state)
        s = rng_state()
        a, b = torch.randn(3), np.random.randn(3)
        restore_rng(s)
        self.assertTrue(torch.equal(a, torch.randn(3)))
        np.testing.assert_array_equal(b, np.random.randn(3))

    def test_seed_intervals_do_not_treat_episodes_as_training_replicates(self):
        self.assertIsNone(seed_interval([1.])["ci95"])
        self.assertEqual(seed_interval([1., 2., 3.])["independent_seeds"], 3)

    def test_device_failure_is_explicit(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(ValueError):
                device_name("cuda")
            self.assertEqual(device_name("auto"), "cpu")

    def test_interrupted_completion_marker_is_not_visible_and_existing_records_are_not_overwritten(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "result.json"
            with patch("chapter5_comm.experiment_state.os.rename", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    atomic_json(path, dict(complete=True))
            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(d).glob("*.pending"))), 1)
            atomic_json(path, dict(complete=True))
            with self.assertRaises(FileExistsError):
                atomic_json(path, dict(complete=False))
            self.assertTrue(json.loads(path.read_text())["complete"])

    def test_report_retains_missing_seeds_and_censored_success_rate(self):
        c = deepcopy(self.c)
        c.update(seeds=[1, 2], arms=["full", "fixed_scheduler"])
        plan = dict(config=c, config_sha="unit")
        row = dict(observed_return=1., completed_task_episode=False, budget_truncated=True,
            channel=dict(sent_bits=384, delivered_bits=384), wall_seconds=.1, actual_length=24,
            scenario_generator_seed=3, success=False, found=False, termination_reason="running")
        with TemporaryDirectory() as d:
            root = Path(d)
            path = job_path(root, 1, "full") / "evaluation/nominal/result.json"
            atomic_json(path, dict(config_sha="unit", seed=1, arm="full", condition="nominal", rows=[row]))
            with patch("chapter5_comm.experiment_report.load_plan", return_value=(root, plan)):
                result = report(root)
            data = json.loads((Path(result["report"]) / "report.json").read_text())
            self.assertFalse(data["complete"])
            self.assertEqual(len(data["missing"]), 3)
            self.assertIsNone(data["aggregate"][0]["metrics"]["success_rate"]["mean"])
            self.assertEqual(data["aggregate"][0]["metrics"]["mean_return"]["independent_seeds"], 1)

    def test_suite_retains_failed_job_and_continues_other_arm(self):
        c = deepcopy(self.c)
        c.update(seeds=[1], arms=["full", "fixed_scheduler"])
        with TemporaryDirectory() as d:
            root = Path(d)
            with patch("chapter5_comm.experiment_workflow.load_plan", return_value=(root, dict(config=c))):
                with patch("chapter5_comm.experiment_workflow.subprocess.run", side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0), SimpleNamespace(returncode=0)]):
                    with patch("chapter5_comm.experiment_workflow.report", return_value=dict(complete=False)):
                        with redirect_stdout(StringIO()):
                            result = suite(root, "all", "cpu", True)
            self.assertFalse(result["passed"])
            records = json.loads((Path(result["attempt"]) / "status.json").read_text())["jobs"]
            self.assertEqual(len(records), 3)
            self.assertEqual(records[-1]["stage"], "evaluate")


if __name__ == "__main__":
    unittest.main()
