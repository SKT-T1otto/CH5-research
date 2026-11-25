"""Fail-closed identity tests; no task rollout, learner or checkpoint restore."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from chapter5_comm.nominal_maddpg.config import (
    DEFAULT_CONFIG, ROOT, default_config, digest, file_sha, load, local_path,
    read_json, resolve, validate, write_new,
)
from chapter5_comm.nominal_maddpg.scenarios import load_exclusions, validate_registry
from chapter5_comm.nominal_maddpg.provenance import compare_ch4


class ConfigTests(unittest.TestCase):
    def test_exact_registered_config(self):
        self.assertEqual(load(), default_config())

    def test_algorithm_mutations_rejected(self):
        mutations = dict(critics_per_vehicle=2, critic_loss="huber", target_critic_reduction="min",
                         target_value_clip=10, target_action_noise=0.2, actor_other_actions="current_actors")
        for field, value in mutations.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                config = default_config(); config["algorithm"][field] = value; validate(config)

    def test_task_information_and_runtime_mutations_rejected(self):
        for section, key, value in (("task", "max_steps", 32), ("task", "residual_scale_search", .45),
                ("runtime", "plant", "relative_water_nominal"), ("runtime", "navigation", "local_waypoints_v1"),
                ("runtime", "map_transport", "raw_scan"), ("runtime", "history_rule", "route"),
                ("runtime", "checkpoint", "historical.pt"), ("channel", "loss_probability", .1),
                ("stage_status", "training_enabled", True), ("stage_status", "optimizer_updates_allowed", 1)):
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                config = default_config(); config[section][key] = value; validate(config)

    def test_seed_and_purpose_mutations_rejected(self):
        c = default_config(); c["training_seeds"][1] = c["training_seeds"][0]
        with self.assertRaises(ValueError): validate(c)
        c = default_config(); c["scene_groups"]["validation"]["start_seed"] = 7100000
        with self.assertRaises(ValueError): validate(c)
        c = default_config(); c["scene_groups"]["baseline_admission"]["purpose"] = "tune"
        with self.assertRaises(ValueError): validate(c)

    def test_json_and_path_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
                path.write_text(raw)
                with self.assertRaises(ValueError): read_json(path)
            with self.assertRaises(FileExistsError): write_new(path, {})
        with self.assertRaises(ValueError): local_path("../outside.json")
        with self.assertRaises(ValueError): local_path(".")

    def test_changed_dependency_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "configs").mkdir()
            for name in ("b2.json", "environment.json"):
                (root / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
            # A whitespace-only edit must change the identity even when JSON values agree.
            path = root / "configs/environment.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                validate(default_config(), root)

    def test_resolved_defaults_and_rng_preserved(self):
        import random
        import numpy as np
        import torch
        before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        r = resolve(load())
        self.assertEqual(random.getstate(), before[0])
        after = np.random.get_state()
        self.assertEqual(after[0], before[1][0]); self.assertTrue(np.array_equal(after[1], before[1][1]))
        self.assertEqual(after[2:], before[1][2:]); self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(r["physics"]["residual_scale_search"], .2)
        self.assertEqual(r["physics"]["residual_scale_executor"], .15)
        self.assertEqual(r["physics"]["flow_gain"], .18)
        self.assertEqual(len(r["physics"]["agent_specs"]), 4)
        self.assertEqual(r["expanded_constructor_parameters"]["dt"], .2)
        self.assertEqual(r["constructor_kwargs"]["max_steps"], 400)
        self.assertEqual(r["wire"]["history_frame_bits"], 1088)
        self.assertEqual(r["wire"]["command_frame_bits"], 224)
        self.assertEqual(r["reward"]["environment"]["collision_terminal_reward"], -2.)
        self.assertEqual(r["inspection"]["physics_steps"], 0)

    def test_ch4_common_algorithm_and_separate_plant(self):
        c = load(); comparison = compare_ch4(c, resolve(c))
        self.assertEqual(comparison["algorithm_common_fields"]["critics_per_vehicle"], 1)
        self.assertFalse(comparison["plant_migrated"])
        self.assertEqual(comparison["plant_migration_status"], "REVIEW_REQUIRED")
        self.assertGreaterEqual(len(comparison["environment_differences"]), 4)

    def test_historical_exclusions(self):
        h = load_exclusions(load())
        self.assertEqual(h["counts"], dict(historical_30=30, development_24=24))
        self.assertEqual(len(h["fingerprints"]), 54)


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load()
        cls.registry = read_json(local_path(cls.config["scene_manifest"]))

    def test_all_scenes_disjoint(self):
        r = validate_registry(self.config, self.registry, DEFAULT_CONFIG)
        self.assertEqual(r["total_scenes"], 1156)
        self.assertEqual(r["cross_group_overlap"], 0)
        self.assertEqual(r["history_overlap"], 0)

    def test_geometry_tamper_rejected(self):
        r = deepcopy(self.registry)
        r["groups"]["validation"][0]["scene"]["target_initial_position"][0] += .1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_registry(self.config, r, DEFAULT_CONFIG)

    def test_metadata_renaming_cannot_hide_physical_duplicate(self):
        from chapter3_bser.learning.config import scenario_fingerprint
        r = deepcopy(self.registry)
        source = r["groups"]["train"][0]["scene"]
        dest = r["groups"]["validation"][0]
        old = dest["scene"]
        dest["scene"] = dict(source, **{key: old[key] for key in
            ("scenario_id", "scenario_role", "scenario_split", "scenario_seed", "pair_group_id")})
        dest["physical_fingerprint"] = scenario_fingerprint(dest["scene"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_registry(self.config, r, DEFAULT_CONFIG)

    def test_historical_geometry_cannot_be_relabelled(self):
        from chapter3_bser.learning.config import scenario_fingerprint
        r = deepcopy(self.registry)
        history = read_json(local_path(self.config["exclusion_manifest"]))
        old = read_json(local_path(history["cohorts"]["historical_30"]["archived_source"]))["scenes"][0]
        dest = r["groups"]["train"][0]
        dest["scene"] = dict(old, **{key: dest["scene"][key] for key in
            ("scenario_id", "scenario_role", "scenario_split", "scenario_seed", "pair_group_id")})
        dest["physical_fingerprint"] = scenario_fingerprint(dest["scene"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_registry(self.config, r, DEFAULT_CONFIG)

    def test_missing_scene_and_changed_horizon_rejected(self):
        r = deepcopy(self.registry); r["groups"]["validation"].pop()
        with self.assertRaisesRegex(ValueError, "count"):
            validate_registry(self.config, r, DEFAULT_CONFIG)
        r = deepcopy(self.registry); r["groups"]["validation"][0]["scene"]["max_steps"] = 32
        with self.assertRaises(ValueError): validate_registry(self.config, r, DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
