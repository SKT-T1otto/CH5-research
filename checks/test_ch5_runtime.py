"""Protocol and local-information checks without launching task rollouts."""
from types import SimpleNamespace
import unittest

import torch
import numpy as np

from chapter5_comm.protocol import HEADER, Kind, Packet, TDMAChannel
from chapter5_comm.runtime import CH5Runtime, VEC6, _LocalAgent, _CommunicatingUAVEnv


class ProtocolTests(unittest.TestCase):
    def test_exact_int8_frame_bytes(self):
        codes = np.arange(32, dtype=np.int8)
        packet = Packet(Kind.MAP_LATENT, 0, 1, 17, 0, 0, codes.tobytes())
        self.assertEqual(HEADER.size, 16)
        self.assertEqual(packet.bits, 384)
        decoded = Packet.from_bytes(packet.to_bytes())
        np.testing.assert_array_equal(np.frombuffer(decoded.payload, np.int8), codes)
        self.assertEqual(decoded, packet)

    def test_frame_rejects_truncation_or_future_sample(self):
        packet = Packet(Kind.STATE, 0, 1, 0, 0, 0, b"abc")
        with self.assertRaises(ValueError):
            Packet.from_bytes(packet.to_bytes()[:-1])
        with self.assertRaises(ValueError):
            Packet(Kind.STATE, 0, 1, 0, 2, 1, b"").to_bytes()

    def test_all_kinds_spend_actual_serialized_bits(self):
        channel = TDMAChannel(seed=2, capacity_bits=16384, refill_bits=0,
                              loss_probability=1., jitter_steps=0)
        expected = 0
        for kind in Kind:
            result = channel.send(kind, 0, 1, b"1234", sample_step=0, step=0)
            self.assertTrue(result["accepted"])
            self.assertTrue(result["dropped"])
            expected += 160
        self.assertEqual(channel.sent_bits, expected)
        self.assertEqual(channel.remaining_bits(0), 16384 - expected)
        self.assertEqual(channel.receive(100), [])
        self.assertEqual(set(channel.summary()["sent_bits_by_kind"]), {kind.name for kind in Kind})

    def test_tdma_and_budget_rejections_do_not_consume_bits(self):
        channel = TDMAChannel(capacity_bits=200, refill_bits=0, loss_probability=0.)
        rejected = channel.send(Kind.STATE, 1, 0, b"", sample_step=0, step=0)
        self.assertEqual(rejected["reason"], "not_tdma_owner")
        accepted = channel.send(Kind.ACK, 0, 1, b"1234", sample_step=0, step=0)
        self.assertTrue(accepted["accepted"])
        rejected = channel.send(Kind.STATE, 0, 1, b"", sample_step=0, step=0)
        self.assertEqual(rejected["reason"], "budget")
        self.assertEqual(channel.sent_bits, 160)

    def test_delivery_causal_and_shared_queue_serialized(self):
        channel = TDMAChannel(seed=2, capacity_bits=10000, refill_bits=0,
            rate_bps=160., dt=1., loss_probability=0., delay_steps=0, jitter_steps=0)
        first = channel.send(Kind.ACK, 0, 1, b"1234", sample_step=0, step=0)
        second = channel.send(Kind.ACK, 1, 2, b"1234", sample_step=1, step=1)
        self.assertEqual(first["scheduled_delivery_step"], 1)
        self.assertEqual(second["scheduled_delivery_step"], 2)
        self.assertEqual(channel.receive(1)[0][0].sequence, first["packet_id"])
        self.assertEqual(channel.receive(2)[0][0].sequence, second["packet_id"])
        with self.assertRaises(ValueError):
            channel.refill(1)

    def test_fault_rng_independent_and_reproducible(self):
        def events():
            channel = TDMAChannel(seed=7, loss_probability=.4, jitter_steps=3,
                burst_start_probability=.2, burst_length=3)
            for step in range(16):
                sender = step % 4
                channel.send(Kind.QUERY, sender, (sender + 1) % 4, b"abc", sample_step=step, step=step)
            return channel.events
        first = events()
        np.random.seed(111)
        np.random.random(100)
        self.assertEqual(first, events())


class _Mapper:
    flat_xyz_centers = torch.tensor([[1., 1., 1.], [8., 8., 2.]])
    occupancy_observation_count = torch.tensor([1, 0])
    occupancy_last_observed_step = torch.tensor([0, -1])
    occupancy_probability = torch.tensor([.3, .5])
    belief_map = torch.tensor([.6, .4])

    def grid_astar_path(self, start, goal, role="searcher"):
        return {"reachable": True, "points": [np.asarray(start), np.asarray(goal)],
                "cost": float(np.linalg.norm(np.asarray(goal) - start))}

    def segment_is_free(self, start, end):
        return True


def fake_runtime():
    runtime = CH5Runtime.__new__(CH5Runtime)
    env = SimpleNamespace(step_count=0, max_steps=400, executor_target_assigned=False,
        _agent_pos=torch.tensor([[1., 1., 1.], [2., 2., 1.], [3., 3., 1.], [4., 4., 1.]]),
        _agent_vel=torch.zeros(4, 3), _get_obs=lambda: [torch.zeros(28) for _ in range(4)])
    runtime.base = SimpleNamespace(unwrapped=env)
    runtime.bounds = np.array([20., 20., 5.])
    runtime.locals = [_LocalAgent(_Mapper(), env._agent_pos[i]) for i in range(4)]
    runtime.map_token_count = 2
    runtime.horizon = 8
    runtime._neural_deliveries, runtime._metadata = [], {}
    runtime._update_navigation = lambda **kwargs: None
    return runtime


class InformationFlowTests(unittest.TestCase):
    def test_actor_obstacle_feature_censors_geometry_outside_sensor_range(self):
        runtime = fake_runtime()
        env = runtime.physical
        runtime._ready = True
        runtime.local_reached, runtime.local_assigned = np.zeros(4), np.ones(4)
        env._active = lambda: True
        env._comm_owner = runtime
        env._nav_targets = env._agent_pos + 1.
        env._v_xy_max = torch.ones(4)
        env._zero3 = torch.zeros(3)
        env.eps, env.dtype = 1e-8, torch.float32
        env.role_onehots, env.agent_finished = torch.eye(4), torch.zeros(4, dtype=torch.bool)
        env._capture_hold_counter, env.obstacle_sensor_range = 0, 4.5
        env._hold_steps_for_agent = lambda agent: 5
        env._vec = lambda values: torch.tensor(values, dtype=torch.float32)
        env._nearest_obstacle_distance = lambda points: torch.full((4,), 5.)
        near_outside = np.asarray(_CommunicatingUAVEnv._get_obs(env))
        env._nearest_obstacle_distance = lambda points: torch.full((4,), 999.)
        far_outside = np.asarray(_CommunicatingUAVEnv._get_obs(env))
        np.testing.assert_array_equal(near_outside, far_outside)
        np.testing.assert_allclose(far_outside[:, 18], .45)
        env._nearest_obstacle_distance = lambda points: torch.full((4,), 2.)
        in_range = np.asarray(_CommunicatingUAVEnv._get_obs(env))
        np.testing.assert_allclose(in_range[:, 18], .2)

    def test_undelivered_peer_and_target_are_masked(self):
        runtime = fake_runtime()
        before = runtime.tokenize(0, 1)
        self.assertEqual(before["map_tokens"].shape, (2, 7))
        self.assertEqual(before["dynamic_tokens"].shape, (5, 8))
        self.assertEqual(before["receiver_query"].shape, (9,))
        self.assertEqual(before["dynamic_mask"].tolist(), [True, False, False, False, False])
        self.assertFalse(before["query_available"])
        runtime.physical._agent_pos[1:] = 1000.
        runtime.physical._agent_vel[1:] = 1000.
        runtime.physical.target_state = SimpleNamespace(position=np.ones(3) * 999)
        after = runtime.tokenize(0, 1)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])

    def test_map_tokens_exclude_dynamic_target_belief(self):
        runtime = fake_runtime()
        before = runtime.tokenize(0, 1)
        self.assertEqual(float(before["map_tokens"][0, 5]), .5)
        runtime.locals[0].mapper.belief_map = torch.tensor([0., 1.])
        after = runtime.tokenize(0, 1)
        np.testing.assert_array_equal(before["map_tokens"], after["map_tokens"])

    def test_state_only_becomes_available_after_delivered_frame(self):
        runtime = fake_runtime()
        packet = Packet(Kind.STATE, 1, 0, 2, 0, 1, VEC6.pack(2., 3., 1., .2, .3, 0.))
        runtime._deliver(packet, 3)
        inputs = runtime.tokenize(0, 1)
        self.assertTrue(inputs["dynamic_mask"][1])
        np.testing.assert_allclose(inputs["dynamic_tokens"][1, :3], np.array([2., 3., 1.]) / runtime.bounds)
        self.assertEqual(runtime.locals[0].acks.popleft(), (1, 2))
        self.assertFalse(inputs["dynamic_mask"][4])

    def test_decoder_cannot_discover_or_handoff_target(self):
        runtime = fake_runtime()
        result = runtime.install_decoded_state(3, {"state_mean": np.ones(6), "has_dynamic": True})
        self.assertFalse(result["target_known"])
        self.assertFalse(result["executor_assigned"])
        self.assertIsNone(runtime.locals[3].target)
        self.assertIsNone(runtime.locals[3].soft_state)
        self.assertEqual(np.asarray(runtime.observations).shape, (4, 28))

    def test_route_forecasts_ignore_hidden_target_and_other_agents(self):
        runtime = fake_runtime()
        runtime.candidate_count = 4
        env = runtime.physical
        env._v_xy_max = torch.ones(4)
        env.dt = .2
        env.prior_strength_search = env.prior_strength_executor = 1.
        env.path_subgoal_radius = .75
        contract = dict(dt=.2, eps=1e-8, use_residual_prior=True,
            prior_slow_radius_xy=2., prior_slow_radius_z=1., prior_kv_xy=1.,
            prior_kv_z=1., prior_strength_search=1., bounds=runtime.bounds.copy())
        spec = dict(v_xy_max=1., v_z_max=.5, a_xy_max=1., a_z_max=.5,
                    drag_xy=.1, drag_z=.1, buoyancy_bias=0.)
        runtime.base.get_public_control_contract = lambda: dict(contract)
        runtime.base.get_public_vehicle_specs = lambda: [dict(spec) for _ in range(4)]
        before = runtime.route_features(0)
        self.assertEqual(before["route_positions"].shape, (4, 9, 3))
        env._agent_pos[1:] = 999.
        env._agent_vel[1:] = -999.
        env.target_state = SimpleNamespace(position=np.ones(3) * 999.)
        env.map_module = SimpleNamespace(ground_truth_only=True)
        after = runtime.route_features(0)
        for key in ("route_features", "route_positions", "route_valid_steps", "route_goals"):
            np.testing.assert_array_equal(before[key], after[key])

    def test_latent_delivery_contains_only_sent_bytes_and_trace_identity(self):
        runtime = fake_runtime()
        runtime._metadata[7] = "training-log-only-17"
        codes = np.arange(32, dtype=np.int8)
        packet = Packet(Kind.DYNAMIC_LATENT, 0, 3, 7, 0, 0, codes.tobytes())
        runtime._deliver(packet, 4)
        delivery, = runtime.drain_neural_deliveries()
        self.assertEqual(delivery["metadata_id"], "training-log-only-17")
        self.assertEqual(delivery["packet_id"], 7)
        self.assertEqual(delivery["delivery_step"], 4)
        self.assertFalse(delivery["target_available"])
        np.testing.assert_array_equal(delivery["codes"], codes)
        self.assertEqual(runtime.drain_neural_deliveries(), [])

    def test_target_latent_flag_requires_legal_sender_evidence_and_costs_header(self):
        runtime = fake_runtime()
        runtime.channel = TDMAChannel(loss_probability=0., jitter_steps=0)
        runtime.begin_step = lambda: None
        runtime._learned_sent_step = -1
        codes = np.arange(32, dtype=np.int8)
        rejected = runtime.send_latent(0, 3, codes, 1, target_available=True)
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["reason"], "target_not_legally_known")
        self.assertEqual(runtime.channel.sent_bits, 0)
        self.assertEqual(len(runtime.channel.queue), 0)
        with self.assertRaises(ValueError):
            runtime.send_latent(0, 3, codes, 0, target_available=True)
        runtime.locals[0].target = SimpleNamespace(sample_step=0)
        sent = runtime.send_latent(0, 3, codes, 1, target_available=True, metadata_id="evidence-1")
        self.assertTrue(sent["accepted"])
        self.assertEqual(sent["bits"], 384)
        self.assertEqual(runtime.channel.sent_by_kind, {"TARGET_LATENT": 384})
        packet = Packet.from_bytes(runtime.channel.queue[0][2])
        self.assertEqual(packet.kind, Kind.TARGET_LATENT)
        self.assertEqual(len(packet.payload), 32)
        runtime._deliver(packet, sent["scheduled_delivery_step"])
        delivery, = runtime.drain_neural_deliveries()
        self.assertTrue(delivery["target_available"])
        self.assertEqual(delivery["stream_type"], 1)
        self.assertIsNone(runtime.locals[3].target)
        self.assertFalse(runtime.physical.executor_target_assigned)
        installed = runtime.install_decoded_state(3, {"state_mean": np.ones(6), "has_dynamic": True})
        self.assertFalse(installed["target_known"])


if __name__ == "__main__":
    unittest.main()
