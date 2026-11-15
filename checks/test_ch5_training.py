"""Finite learner-math tests; no environment rollouts or historical restore."""
import unittest

import torch

from chapter5_comm.training import (
    AuxiliaryTrainer, CommunicationLearner, NeuralConfig, build_neural_models,
    freeze_auxiliary,
)


class CH5TrainingMath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        self.models = build_neural_models(dict(latent_dim=16, hidden_dim=16))

    def batch(self):
        return dict(map_tokens=torch.randn(2, 4, 7), dynamic_tokens=torch.randn(2, 3, 8),
                    receiver_query=torch.randn(2, 9), query_available=torch.tensor([True, False]),
                    map_mask=torch.ones(2, 4, dtype=torch.bool),
                    dynamic_mask=torch.ones(2, 3, dtype=torch.bool),
                    received=torch.tensor([[True, False, True, False], [True, True, False, True]]),
                    ages=torch.ones(2, 4), local_geometry=torch.randn(2, 7),
                    route_features=torch.randn(2, 3, 5), route_valid=torch.ones(2, 3, dtype=torch.bool),
                    route_labels=torch.tensor([[1., 0., 1.], [0., 1., 0.]]),
                    state_target=torch.randn(2, 6), state_valid=torch.tensor([True, False]))

    def event(self, learner):
        event = dict(packet_tokens=torch.randn(4, 16), context=torch.randn(7),
                     packet_bits=torch.full((4,), 256), budget_bits=512,
                     valid=torch.ones(4, dtype=torch.bool), critic_features=torch.zeros(3),
                     next_critic_features=torch.ones(3), discounted_reward=1.,
                     elapsed=3, episode_step=2, done=False)
        event.update(learner.choose(event))
        return event

    def test_shape_contract_and_frozen_auxiliary(self):
        self.assertEqual(self.models.config.dynamic_dim, 8)
        self.assertEqual(self.models.config.geometry_dim, 7)
        with self.assertRaises(ValueError):
            NeuralConfig.from_mapping(dict(unrecognized=True))
        freeze_auxiliary(self.models)
        self.assertFalse(any(p.requires_grad for p in self.models.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.models.scheduler.parameters()))

    def test_auxiliary_update_changes_codec_with_partial_censoring(self):
        trainer = AuxiliaryTrainer(self.models, max_updates=1)
        row = trainer.step(self.batch())
        self.assertFalse(row["skipped"])
        self.assertGreater(row["parameter_delta_l2"], 0.)
        self.assertEqual(row["state_labels"], 1)
        with self.assertRaises(RuntimeError):
            trainer.step(self.batch())

    def test_no_supervision_skips_without_time_identity_training(self):
        batch = self.batch()
        batch["state_valid"].zero_()
        batch["route_valid"].zero_()
        batch["state_target"].fill_(float("nan"))
        batch["route_labels"].fill_(float("nan"))
        trainer = AuxiliaryTrainer(self.models)
        before = [p.detach().clone() for p in trainer.parameters]
        row = trainer.step(batch)
        self.assertTrue(row["skipped"])
        self.assertEqual(trainer.updates, 0)
        self.assertTrue(all(torch.equal(p, old) for p, old in zip(trainer.parameters, before)))

    def test_map_only_packets_cannot_supervise_dynamic_reconstruction(self):
        batch = self.batch()
        batch["received"][:, 2:] = False
        batch["state_valid"].fill_(True)
        batch["state_target"].fill_(float("nan"))
        trainer = AuxiliaryTrainer(self.models)
        row = trainer.step(batch)
        self.assertEqual(row["state_labels"], 0)
        self.assertEqual(row["prediction_loss"], 0.)
        batch["route_valid"].zero_()
        row = trainer.step(batch)
        self.assertTrue(row["skipped"])
        self.assertEqual(trainer.updates, 1)

    def test_smdp_discount_uses_physical_interval_and_real_terminal(self):
        learner = CommunicationLearner(self.models.scheduler, 3, max_updates=1)
        with torch.no_grad():
            for p in learner.critic.parameters():
                p.zero_()
            learner.critic[-1].bias.fill_(2.)
        continuing = self.event(learner)
        terminal = self.event(learner)
        terminal["done"] = True
        result = learner.update([continuing, terminal])
        self.assertAlmostEqual(result["smdp_targets"][0], 1. + .95 ** 3 * 2., places=5)
        self.assertAlmostEqual(result["smdp_targets"][1], 1., places=5)
        self.assertGreater(result["parameter_delta_l2"], 0.)
        self.assertEqual(result["reward_source"], "original_task_interval_return")

    def test_stale_actor_trajectories_are_not_reused_as_on_policy(self):
        learner = CommunicationLearner(self.models.scheduler, 3, max_updates=2)
        event = self.event(learner)
        learner.update([event])
        with self.assertRaises(ValueError):
            learner.update([event])

    def test_long_update_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            AuxiliaryTrainer(self.models, max_updates=5)
        with self.assertRaises(ValueError):
            CommunicationLearner(self.models.scheduler, 3, max_updates=5)


if __name__ == "__main__":
    unittest.main()
