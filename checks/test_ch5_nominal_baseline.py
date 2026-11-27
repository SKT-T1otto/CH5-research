"""Regression checks for the fresh baseline's gates, budget and saved optimizer settings."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter5_comm.nominal_maddpg import checkpoint as cp
from tools.ch5_nominal_baseline import (Baseline, PROTOCOL, ids, read_json, stable)
from tools.ch5_nominal_pilot_train import preflight


class BaselineTests(unittest.TestCase):
    def test_plateau_requires_task_capability_and_safety(self):
        spec=read_json(PROTOCOL)
        def rows(s=29,c=1):
            return [dict(episode=n,success_count=s,collision_count=c,mean_team_return=2.) for n in (120,140,160,180,200)]
        self.assertTrue(stable(rows(),spec)["strong"])
        self.assertTrue(stable(rows(26),spec)["weak"])
        self.assertFalse(stable(rows(26),spec)["strong"])
        self.assertTrue(stable(rows(6),spec)["stable"])
        self.assertFalse(stable(rows(6),spec)["weak"])
        self.assertFalse(stable(rows(c=2),spec)["strong"])
        self.assertFalse(stable(rows()[:4],spec)["stable"])

    def test_success_and_return_instability(self):
        spec=read_json(PROTOCOL)
        rows=[dict(episode=n,success_count=29,collision_count=0,mean_team_return=2.) for n in range(5)]
        rows[0]["success_count"]=26
        self.assertFalse(stable(rows,spec)["stable"])
        rows[0]["success_count"]=29
        rows[0]["mean_team_return"]=1.5
        self.assertFalse(stable(rows,spec)["stable"])

    def test_cumulative_boundaries_and_gap_rejection(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            for i in range(201):
                (root/"boundaries"/f"{i:04d}").mkdir(parents=True)
            self.assertEqual(ids(root),list(range(201)))
            (root/"boundaries/0202").mkdir()
            with self.assertRaisesRegex(ValueError,"missing"):
                ids(root)

    def test_fresh_initialization_and_boundary_state_restore(self):
        torch.set_num_threads(1)
        setup=preflight()
        spec=read_json(PROTOCOL)
        with TemporaryDirectory() as d:
            args=(Path(d),2729,setup,spec,"test")
            model=Baseline(*args)
            self.assertEqual((model.team_steps,len(model.replay),model.learner.rounds),(0,0,0))
            self.assertEqual(model.noise.settings["decay_per_environment_step"],.000002)
            self.assertTrue(all(opt.param_groups[0]["lr"]==.0003 for opt in model.learner.actor_optimizers))
            path=cp.save_boundary(model)
            expected=state_digest(model.snapshot())
            model.noise.sample()
            model.restore(path)
            self.assertEqual(state_digest(model.snapshot()),expected)
            self.assertEqual(model.protocol["episodes_per_seed"],1000)
            self.assertEqual(model.protocol["max_train_team_steps_per_seed"],400000)

    def test_stage_freeze_before_admission(self):
        from tools.ch5_nominal_baseline import freeze
        with TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"gate"):
                freeze(Path(d),None,None,dict(development_ready_to_freeze=False))
            self.assertFalse((Path(d)/"seed_2729/freeze.json").exists())

    def test_training_then_restored_suffix_matches_with_new_learning_rate(self):
        import numpy as np
        from unittest.mock import patch

        class FixtureRuntime:
            def __init__(self, config, scene, seed, replay, *, episode_id):
                self.core=self
                self.replay=replay
                self.steps=0
                self.length=258 if episode_id==0 else 2

            @property
            def terminal(self):
                return self.steps==self.length

            def advance(self, policy):
                obs=np.full((4,28),self.steps/258.,dtype=np.float32)
                action=policy.step(obs).clamp(-1,1).numpy()
                self.steps+=1
                tr=dict(observations=obs,next_observations=obs+1/258.,actions=action,
                        rewards=np.ones(4,dtype=np.float32),dones=[self.terminal]*4,
                        success_flags=[False]*4,action_enabled=np.ones(4,dtype=bool),
                        next_action_enabled=np.full(4,not self.terminal))
                self.replay.push_transition(tr)
                return tr

            def summary(self):
                return dict(termination_reason="obstacle_collision",episode_terminated=True,success=False,
                            collision_episode=True,actual_length=self.steps,team_undiscounted_return=float(self.steps),
                            found_step=None,handoff_event_step=None,capture_contact_steps=0,
                            channel=dict(sent_bits=0,delivered_bits=0))

            def close(self):
                pass

        torch.set_num_threads(1)
        setup=preflight()
        spec=read_json(PROTOCOL)
        with TemporaryDirectory() as d,patch("tools.ch5_nominal_pilot_train.NominalRuntime",FixtureRuntime):
            args=(Path(d),2729,setup,spec,"restore-test")
            model=Baseline(*args)
            model.train_one()
            self.assertEqual(model.learner.rounds,2)
            boundary=cp.save_boundary(model)
            model.train_one()
            self.assertEqual(model.learner.rounds,4)
            expected=state_digest(model.snapshot())
            del model
            resumed=Baseline(*args)
            resumed.restore(boundary)
            resumed.train_one()
            self.assertEqual(state_digest(resumed.snapshot()),expected)


if __name__=="__main__":
    unittest.main()
