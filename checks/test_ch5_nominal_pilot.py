"""Pilot orchestration checks; training and validation scene pools stay unopened for collection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter5_comm.nominal_maddpg import checkpoint as cp
from chapter5_comm.nominal_maddpg.config import load, read_json
from tools.ch5_nominal_pilot import committed, evaluate, gate, report, run_seed, score, selection
from tools.ch5_nominal_pilot_train import Pilot, PROTOCOL, SOURCE, file_sha, rounds_due, seed_process


class PilotTests(unittest.TestCase):
    def test_schedule_and_budget(self):
        learning = load()["learning"]
        self.assertEqual([rounds_due(t,learning) for t in (127,252,255,256,259,260,25600)],
                         [0,0,0,2,2,4,12674])
        self.assertEqual(3*64*400,76800)
        self.assertEqual(3*5*32*400,192000)

    def test_selection_all_tiebreaks_and_strict_gate(self):
        def row(n,s,c,r):
            return dict(episode=n,success_count=s,collision_count=c,mean_team_return=r)
        cases = [row(0,32,0,1000),row(16,3,15,100),row(32,4,2,2),row(48,4,2,2),row(64,4,3,100)]
        self.assertEqual(selection(cases)["episode"],32)
        cases[3]["mean_team_return"] = 2.1
        self.assertEqual(selection(cases)["episode"],48)
        for wins, result in (([True,True,False],"PASS"),([True,False,False],"STOP")):
            self.assertEqual(gate([dict(completed=True,data_budget_restore_passed=True,strict_gain=w) for w in wins]),result)
        seeds = [dict(completed=True,data_budget_restore_passed=True,strict_gain=True) for _ in range(3)]
        seeds[2]["data_budget_restore_passed"] = False
        self.assertEqual(gate(seeds),"STOP")
        self.assertEqual(gate([dict(completed=False)]*3),"PENDING")

    def test_score_uses_untransformed_mean_and_all_outcomes(self):
        rows = [dict(steps=2,outcome=o,team_return=r) for o,r in
                (("success",1.),("obstacle_collision",-2.),("timeout",4.))]
        result = score(rows)
        self.assertEqual(result["mean_team_return"],1.)
        self.assertEqual([result[k] for k in ("success_count","collision_count","timeout_count")],[1,1,1])
        rows[0]["outcome"] = "running"
        with self.assertRaises(ValueError):
            score(rows)

    def test_interrupted_evaluation_restore_and_corrupt_partial(self):
        torch.set_num_threads(1)
        cfg, p = load(), read_json(PROTOCOL)
        groups = {"train":[{}], "validation":[dict(physical_seed=i,physical_fingerprint=str(i)) for i in range(3)]}
        with tempfile.TemporaryDirectory() as d:
            seed_process(2729)
            obj = Pilot(Path(d),2729,cfg,p,groups,"unit")
            boundary = cp.save_boundary(obj)
            initial = state_digest(obj.snapshot())
            def run(i):
                torch.rand(2)
                np.random.random()
                return dict(physical_seed=i,fingerprint=str(i),steps=1,outcome="success",team_return=0.25)
            with patch.object(obj,"validation_scene",side_effect=[run(0),RuntimeError("interrupted")]):
                # Generate fixture data first, then compare to the saved boundary.
                obj.restore(boundary)
                with self.assertRaises(RuntimeError):
                    evaluate(obj,boundary)
            self.assertEqual(state_digest(obj.snapshot()),initial)
            with patch.object(obj,"validation_scene",side_effect=run) as remaining:
                row = evaluate(obj,boundary)
                self.assertEqual([c.args[0] for c in remaining.call_args_list],[1,2])
            self.assertEqual(row["eval_updates"],0)
            self.assertEqual(row["success_count"],3)
            self.assertEqual(state_digest(obj.snapshot()),initial)
            self.assertEqual(committed(obj.root),[0])
            with patch.object(obj,"validation_scene",side_effect=AssertionError("repeated validation")):
                self.assertEqual(evaluate(obj,boundary),row)
            (obj.root/"validation/0000/scene_01.json").write_text("{}",encoding="utf-8")
            with self.assertRaises((KeyError,ValueError)):
                evaluate(obj,boundary)

    def test_reject_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"boundaries/0001").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,"missing"):
                committed(Path(d))

    def test_training_resume_crosses_real_optimizer_triggers_on_fixture(self):
        class FixtureRuntime:
            def __init__(self, config, scene, seed, replay, *, episode_id):
                self.core = self
                self.replay = replay
                self.steps = 0
                self.length = 258 if episode_id == 0 else 2

            @property
            def terminal(self):
                return self.steps == self.length

            def advance(self, policy):
                obs = np.full((4,28), self.steps/258., dtype=np.float32)
                act = policy.step(obs).clamp(-1,1).numpy()
                self.steps += 1
                tr = dict(observations=obs, next_observations=obs+1/258., actions=act,
                          rewards=np.ones(4,dtype=np.float32), dones=[self.terminal]*4,
                          success_flags=[False]*4, action_enabled=np.ones(4,dtype=bool),
                          next_action_enabled=np.full(4,not self.terminal))
                self.replay.push_transition(tr)
                return tr

            def summary(self):
                return dict(termination_reason="obstacle_collision", episode_terminated=True, success=False,
                    collision_episode=True, actual_length=self.steps, team_undiscounted_return=float(self.steps),
                    found_step=None, handoff_event_step=None, capture_contact_steps=0,
                    channel=dict(sent_bits=0,delivered_bits=0))

            def close(self):
                pass

        torch.set_num_threads(1)
        groups = {"train":[dict(scene={},physical_seed=i,physical_fingerprint=str(i)) for i in range(2)]}
        with tempfile.TemporaryDirectory() as d, patch("tools.ch5_nominal_pilot_train.NominalRuntime",FixtureRuntime):
            args = (Path(d),2729,load(),read_json(PROTOCOL),groups,"fixture")
            train = Pilot(*args)
            train.train_one()
            self.assertEqual(train.learner.rounds,2)
            boundary = cp.save_boundary(train)
            train.train_one()
            self.assertEqual((len(train.replay),train.learner.rounds),(260,4))
            expected = state_digest(train.snapshot())
            del train
            resumed = Pilot(*args)
            resumed.restore(boundary)
            resumed.train_one()
            self.assertEqual(state_digest(resumed.snapshot()),expected)

    def test_three_seed_report_and_repeat_on_metadata_fixtures(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from types import SimpleNamespace

        class FixturePilot:
            def __init__(self, root, seed, config, protocol, groups, run_id):
                self.root, self.groups, self.seed = root, groups, seed
                self.identity = dict(seed=seed, run=run_id, source=file_sha(SOURCE))
                self.team_steps, self.episodes = 0, []
                self.learner = SimpleNamespace(rounds=0)

            def train_one(self):
                self.team_steps += 4
                rounds = rounds_due(self.team_steps,cfg["learning"])
                self.episodes.append(dict(steps=4,updates=rounds-self.learner.rounds,team_return=-2.))
                self.learner.rounds = rounds

            def snapshot(self):
                return dict(team_steps=self.team_steps, episodes=self.episodes,
                            learner=dict(counters=dict(rounds=self.learner.rounds)))

            def restore(self, boundary):
                state = cp.read_boundary(self.root,boundary,self.identity)
                self.episodes, self.team_steps = state["episodes"], state["team_steps"]
                self.learner.rounds = state["learner"]["counters"]["rounds"]

            def validation_scene(self, i):
                improved = self.learner.rounds > 0 and self.seed != 4751 and i == 0
                return dict(steps=4, physical_seed=i, fingerprint=str(i),
                            outcome="success" if improved else "timeout",team_return=1.)

        cfg, protocol = load(), read_json(PROTOCOL)
        groups = dict(validation=[dict(physical_seed=i,physical_fingerprint=str(i)) for i in range(32)])
        startup = cfg,protocol,groups,{}
        with tempfile.TemporaryDirectory() as d, patch("tools.ch5_nominal_pilot.Pilot",FixturePilot), redirect_stdout(StringIO()):
            root = Path(d)
            for seed in protocol["training_seeds"]:
                self.assertEqual(report(root,startup,"fixture")["decision"],"PENDING")
                run_seed(root,seed,startup,"fixture")
            outcome = report(root,startup,"fixture")
            self.assertEqual(outcome["decision"],"PASS")
            self.assertEqual([r["candidate_episode"] for r in outcome["seeds"]],[64,64,64])
            self.assertEqual((outcome["confirmed_training_steps"],outcome["verified_validation_steps"]),(768,1920))
            with patch.object(FixturePilot,"train_one",side_effect=AssertionError("repeated training")):
                run_seed(root,2729,startup,"fixture")
            self.assertEqual(report(root,startup,"fixture"),outcome)


if __name__ == "__main__":
    unittest.main()
