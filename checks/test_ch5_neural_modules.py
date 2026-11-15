"""Neural communication contract checks; no environment or optimizer steps."""
import unittest

import torch

from chapter5_comm.neural_modules import (
    BudgetedPacketScheduler,
    GeometryTemporalDecoder,
    RouteUtilityHead,
    TaskPacketEncoder,
    gaussian_prediction_nll,
    route_distillation_loss,
    route_encounter_loss,
    temporal_composition_loss,
)


class NeuralCommunicationContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)
        self.batch, self.width = 2, 16
        self.encoder = TaskPacketEncoder(7, 6, 9, latent_dim=self.width)
        self.decoder = GeometryTemporalDecoder(self.width, geometry_dim=8)
        self.map_tokens = torch.randn(2, 4, 7)
        self.dynamic_tokens = torch.randn(2, 3, 6)
        self.query = torch.randn(2, 9)
        self.map_mask = torch.ones(2, 4, dtype=torch.bool)
        self.dynamic_mask = torch.ones(2, 3, dtype=torch.bool)
        self.geometry = torch.randn(2, 8)

    def encode(self, query=None, available=None, map_mask=None, dynamic_mask=None):
        return self.encoder(
            self.map_tokens, self.dynamic_tokens,
            self.query if query is None else query,
            torch.ones(2, dtype=torch.bool) if available is None else available,
            self.map_mask if map_mask is None else map_mask,
            self.dynamic_mask if dynamic_mask is None else dynamic_mask,
        )

    def test_unknown_query_does_not_read_hidden_receiver_information(self):
        missing = torch.zeros(2, dtype=torch.bool)
        left = self.encode(torch.full_like(self.query, float("nan")), missing)
        right = self.encode(torch.randn_like(self.query) * 1000, missing)
        self.assertTrue(torch.isfinite(left.tokens).all())
        self.assertTrue(torch.equal(left.tokens, right.tokens))

    def test_masked_source_tokens_do_not_enter_packets(self):
        self.map_tokens.fill_(float("nan"))
        self.dynamic_tokens.fill_(float("nan"))
        encoded = self.encode(
            map_mask=torch.zeros_like(self.map_mask),
            dynamic_mask=torch.zeros_like(self.dynamic_mask),
        )
        self.assertFalse(encoded.valid.any())
        self.assertTrue(torch.isfinite(encoded.tokens).all())
        self.assertEqual(float(encoded.tokens.detach().abs().sum()), 0.0)

    def test_erased_payload_and_age_cannot_change_decoder_output(self):
        encoded = self.encode()
        received = torch.tensor([[True, False, True, False], [False] * 4])
        ages = torch.tensor([[4., 8., 1., 3.], [1., 2., 3., 4.]])
        expected = self.decoder(encoded.tokens, encoded.stream_types, received, ages, self.geometry)
        payload = torch.where(received[..., None], encoded.tokens, torch.full_like(encoded.tokens, float("nan")))
        hidden_ages = torch.where(received, ages, torch.full_like(ages, float("nan")))
        actual = self.decoder(payload, encoded.stream_types, received, hidden_ages, self.geometry)
        for key in ("map_context", "dynamic_context", "state_mean", "state_logvar"):
            self.assertTrue(torch.isfinite(actual[key]).all(), key)
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
        self.assertFalse(bool(actual["has_dynamic"][1]))
        self.assertFalse(bool(actual["has_map"][1]))

    def test_static_map_is_not_motion_extrapolated_by_packet_age(self):
        encoded = self.encode()
        received = (encoded.stream_types == 0)[None].expand(2, -1)
        young = self.decoder(encoded.tokens, encoded.stream_types, received, torch.zeros(2, 4), self.geometry)
        old = self.decoder(encoded.tokens, encoded.stream_types, received, torch.full((2, 4), 1000.), self.geometry)
        self.assertTrue(torch.equal(young["map_context"], old["map_context"]))

    def test_zero_age_transport_is_identity_and_geometry_affects_dynamics(self):
        z, g = torch.randn(2, 16), torch.randn(2, 16)
        self.assertTrue(torch.equal(self.decoder.transport(z, g, torch.zeros(2)), z))
        left = self.decoder.transport(z, g, torch.ones(2))
        right = self.decoder.transport(z, g + 2., torch.ones(2))
        self.assertGreater(float((left - right).detach().abs().max()), 1e-7)

    def test_quantized_forward_and_payload_cost_are_explicit(self):
        encoded = self.encode()
        self.assertEqual(encoded.codes.dtype, torch.int8)
        self.assertTrue(torch.equal(encoded.tokens, encoded.codes.to(encoded.tokens.dtype) / 127.))
        self.assertTrue(torch.equal(encoded.packet_bits, torch.full((4,), (16 + 16) * 8, dtype=torch.long)))

    def test_auxiliary_task_and_prediction_losses_reach_both_modules(self):
        encoded = self.encode()
        received = torch.ones(2, 4, dtype=torch.bool)
        decoded = self.decoder(encoded.tokens, encoded.stream_types, received, torch.ones(2, 4), self.geometry)
        route_head = RouteUtilityHead(16, 5)
        logits = route_head(decoded["map_context"], decoded["dynamic_context"], torch.randn(2, 3, 5))
        valid = torch.ones(2, 3, dtype=torch.bool)
        encounter = route_encounter_loss(logits, torch.tensor([[1., 0., 1.], [0., 1., 0.]]), valid)
        prediction = gaussian_prediction_nll(decoded["state_mean"], decoded["state_logvar"], torch.randn(2, 6))
        consistency = temporal_composition_loss(self.decoder, torch.randn(2, 16), decoded["map_context"], torch.full((2,), .3), torch.full((2,), .4))
        loss = encounter + prediction + consistency
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (self.encoder, self.decoder, route_head):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        for parameter in (self.encoder.map_input.weight, self.encoder.dynamic_input.weight,
                          self.decoder.vector_field[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_teacher_is_detached_and_empty_route_set_is_rejected(self):
        student = torch.randn(2, 3, requires_grad=True)
        teacher = torch.randn(2, 3, requires_grad=True)
        route_distillation_loss(student, teacher, torch.tensor([[True, False, True], [False, True, True]])).backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        with self.assertRaises(ValueError):
            route_encounter_loss(student, torch.zeros_like(student), torch.zeros_like(student, dtype=torch.bool))

    def test_budget_mask_and_noop_remain_valid_without_packets(self):
        encoded = self.encode()
        scheduler = BudgetedPacketScheduler(16, 7)
        context = torch.randn(2, 7)
        distribution = scheduler(encoded.tokens, context, encoded.packet_bits, torch.zeros(2), encoded.valid)
        self.assertTrue(torch.equal(distribution.probs[:, -1], torch.ones(2)))
        self.assertEqual(float(distribution.probs[:, :-1].detach().sum()), 0.)
        valid = encoded.valid.clone()
        valid[:, 1:] = False
        distribution = scheduler(encoded.tokens, context, encoded.packet_bits, torch.full((2,), 256.), valid)
        self.assertEqual(float(distribution.probs[:, 1:-1].detach().sum()), 0.)
        loss = -distribution.log_prob(torch.zeros(2, dtype=torch.long)).mean()
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in scheduler.parameters() if p.grad is not None), 0.)


if __name__ == "__main__":
    unittest.main()
