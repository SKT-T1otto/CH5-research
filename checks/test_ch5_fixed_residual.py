"""Unit and orchestration tests for signed-mean action substitution, without training."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter5_comm.nominal_maddpg.config import read_json
from tools.ch5_fixed_residual import (ANCHOR, FixedResidual, calibrate, compare_anchor,
                                     paired_report, run_pairs)


def row(i, outcome):
    return dict(physical_seed=100+i, fingerprint=str(i), steps=3, outcome=outcome,
                team_return=0.5, found_step=None, handoff_step=None, capture_steps=0,
                tx_bits=32, rx_bits=0)


class FixedResidualTests(unittest.TestCase):
    def test_signed_mean_uses_only_current_open_observations(self):
        class Model:
            def actions(self, obs):
                return obs[:,:,:3]

        observations = torch.zeros(3,4,28)
        observations[0,:,:3] = torch.tensor([-1.,0.,1.])
        observations[1,:,:3] = torch.tensor([1.,1.,0.])
        observations[2,:,:3] = torch.tensor([0.,-1.,0.])
        gates = torch.tensor([[1,0,0,1],[1,1,0,1],[0,0,1,1]],dtype=torch.bool)
        replay = dict(base=dict(filled_i=3,next_idx=3,obs_buffs=list(observations.unbind(1))),
                      enabled=gates,gate_initialized=torch.ones(3,dtype=torch.bool))
        output = calibrate(Model(),replay,3)
        self.assertEqual(output["open_steps"],[2,1,1,3])
        np.testing.assert_allclose(output["per_hull_xyz"],[[0,.5,.5],[1,1,0],[0,-1,0],[0,0,1/3]])
        replay["enabled"][:,2] = False
        with self.assertRaisesRegex(ValueError,"each hull"):
            calibrate(Model(),replay,3)
        replay["gate_initialized"][0] = False
        with self.assertRaisesRegex(ValueError,"gates"):
            calibrate(Model(),replay,3)

    def test_fixed_policy_independent_of_observation_and_mutable_caller(self):
        values = np.arange(12,dtype=np.float32).reshape(4,3)/12
        policy = FixedResidual(values.tolist())
        proposal = policy.step(np.ones((4,28)))
        proposal[0] = -1
        np.testing.assert_array_equal(policy.step(np.zeros((4,28))),values)
        with self.assertRaises(ValueError):
            policy.step(None,explore=True)

    def test_matched_gains_separate_substitution_from_different_wins(self):
        success, timeout, collision = "success","timeout","obstacle_collision"
        before = [timeout,timeout,timeout,timeout,success,timeout,collision]
        actor = [success,success,success,success,success,timeout,collision]
        fixed = [success,success,success,timeout,collision,success,success]
        reference = [row(i,o) for i,o in enumerate(before)]
        trained = [row(i,o) for i,o in enumerate(actor)]
        for keep,meaning in ((3,"majority_reproduced"),(2,"exactly_half_reproduced"),(1,"minority_reproduced")):
            outcomes = fixed.copy()
            outcomes[keep:4] = [timeout]*(4-keep)
            result = paired_report(reference,trained,[row(i,o) for i,o in enumerate(outcomes)])
            self.assertEqual((result["interpretation"],result["retained_count"],result["new_success_count"]),
                             (meaning,keep,4))
            self.assertEqual(result["fixed_only_successes"],[105,106])
            self.assertEqual(result["fixed"]["collision_count"],1)
            self.assertIn(104,result["actor_only_successes"])
        with self.assertRaisesRegex(ValueError,"unpaired"):
            paired_report(reference,trained,list(reversed(trained)))

    def test_anchor_identifies_changed_record_even_if_both_succeeded(self):
        expected = row(4,"success")
        with self.assertRaisesRegex(ValueError,"104: tx_bits; archived=32, current=64"):
            compare_anchor(expected,dict(expected,tx_bits=64))
        compare_anchor(expected,dict(expected,team_return=0.5+1e-9))
        with self.assertRaisesRegex(ValueError,"team_return"):
            compare_anchor(expected,dict(expected,team_return=0.5+2e-7))
        with self.assertRaisesRegex(ValueError,"fields"):
            compare_anchor(expected,{k:v for k,v in expected.items() if k!="found_step"})

    def test_resume_verifies_finished_pairs_and_anchor_before_fixed_run(self):
        entries = [dict(physical_seed=100+i,physical_fingerprint=str(i)) for i in range(6)]
        trained = [row(i,"success") for i in range(6)]
        ref = {0:[row(i,"timeout") for i in range(6)],32:trained}
        manifest = dict(calibration=dict(per_hull_xyz=[[0,0,0]]*4))
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            out = Path(folder)
            with patch("tools.ch5_fixed_residual.rollout",return_value=dict(trained[ANCHOR],steps=2)) as run:
                with self.assertRaisesRegex(ValueError,"steps"):
                    run_pairs(out,None,entries,ref,None,manifest)
                self.assertEqual(run.call_count,1)
                self.assertFalse((out/"fixed").exists())
            with patch("tools.ch5_fixed_residual.rollout",side_effect=[trained[ANCHOR],trained[0],OSError("stop")]):
                with self.assertRaises(OSError):
                    run_pairs(out,None,entries,ref,None,manifest)
            with patch("tools.ch5_fixed_residual.rollout",side_effect=lambda c,e,i,p:trained[i]) as run:
                run_pairs(out,None,entries,ref,None,manifest)
                self.assertEqual([call.args[2] for call in run.call_args_list],[1,2,3,4,5])
            with patch("tools.ch5_fixed_residual.rollout",side_effect=AssertionError("unexpected repeat")):
                run_pairs(out,None,entries,ref,None,manifest)
            self.assertEqual(read_json(out/"report.json")["retained_count"],6)
            p = out/"fixed/scene_01.json"
            p.write_text(p.read_text(encoding="utf8").replace('"steps": 3','"steps": 2'),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"receipt"):
                run_pairs(out,None,entries,ref,None,manifest)


if __name__ == "__main__":
    unittest.main()
