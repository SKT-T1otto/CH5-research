import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from chapter5_comm.d2_collision_cases import historical_checks, route_digest
from chapter5_comm.d2_collision_inputs import BATCH_RUN, R123_RUN, locate_runs


class CollisionEvidenceTest(TestCase):
    def test_route_identity_uses_ordered_float32_vertices(self):
        path = [[1., 2., 3.], [4., 5., 6.]]
        self.assertEqual(route_digest(path), "24ae2dfe8df57c1b")
        self.assertNotEqual(route_digest(path[::-1]), route_digest(path))

    def test_reproduction_rejects_a_different_contact_on_the_same_step(self):
        hit = dict(agent_id=3, obstacle_id=0, first_hit_point=[1, 2, 3], position_before=[0, 2, 3],
                   candidate_position=[2, 2, 3], hit_fraction=.5)
        actual = dict(steps=66, prefix_cutoff=False, collision_step=66, found=False,
                      initial_physics_sha256="initial", collisions=[hit], channel={"dropped_frames": 0})
        old = dict(actual, collision_records=copy.deepcopy(actual["collisions"]))
        self.assertTrue(all(historical_checks(actual, old).values()))
        old["collision_records"][0]["first_hit_point"][1] += .1
        self.assertFalse(historical_checks(actual, old)["collision_geometry"])

    def test_directory_discovery_reports_duplicate_archives_and_accepts_exact_paths(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ch3 = root / "chapter3" / R123_RUN
            batch = root / "chapter5" / BATCH_RUN
            ch3.mkdir(parents=True)
            batch.mkdir(parents=True)
            self.assertEqual(locate_runs(root, None, None), (ch3.resolve(), batch.resolve()))
            (root / "backups" / R123_RUN).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "--ch3-run"):
                locate_runs(root, None, batch)
            self.assertEqual(locate_runs(root, ch3, None), (ch3.resolve(), batch.resolve()))
