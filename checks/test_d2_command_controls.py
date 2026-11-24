"""Synthetic packet/receipt checks; no simulated task or training."""
import unittest

import numpy as np

from chapter5_comm.protocol import COMMAND_CONTROLS, COMMAND_PADDING, D2CommandOverride, Kind, Packet, TDMAChannel
from chapter5_comm.runtime import CH5Runtime, GOAL3, _LocalAgent


class D2CommandControlTests(unittest.TestCase):
    def setUp(self):
        self.rt = object.__new__(CH5Runtime)
        self.rt.d2_allocator = object()
        self.rt.locals = [_LocalAgent(None, [3., 17., 6.5]) for _ in range(4)]
        self.rt.bounds = np.array([20., 20., 8.])
        self.rt._last_replan_request = {}
        self.local = self.rt.locals[1]
        self.deliver(Kind.COMMAND, 12, body=GOAL3.pack(3., 17., 6.5))
        self.local.last_plan = 53
        self.local.acks.clear()

    def deliver(self, kind, sample, *, seq=None, body=COMMAND_PADDING, sender=0, receiver=1):
        packet = Packet(kind, sender, receiver, sample if seq is None else seq, sample, 52, body)
        self.rt._deliver(Packet.from_bytes(packet.to_bytes()), 53)

    def test_pad_only_acks(self):
        command = self.local.command
        self.deliver(Kind.COMMAND_PAD, 52)
        self.assertIs(self.local.command, command)
        self.assertEqual((self.local.last_command_step, self.local.last_plan), (12, 53))
        self.assertEqual(self.rt._last_replan_request, {})
        self.assertEqual(list(self.local.acks), [(0, 52)])

    def test_replan_deduplicates_without_suppressing_a_real_command(self):
        command = self.local.command
        self.deliver(Kind.REPLAN_ONLY, 52)
        self.assertIs(self.local.command, command)
        self.assertEqual((self.local.last_command_step, self.local.last_plan), (12, -100000))
        self.local.last_plan = 54
        self.deliver(Kind.REPLAN_ONLY, 52)
        self.deliver(Kind.REPLAN_ONLY, 48, seq=54)
        self.assertEqual(self.local.last_plan, 54)
        self.deliver(Kind.COMMAND, 48, seq=55, body=GOAL3.pack(3., 15., 6.5))
        self.assertEqual(self.local.last_command_step, 48)
        np.testing.assert_array_equal(self.local.command[1], [3., 15., 6.5])
        self.assertEqual(len(self.local.acks), 4)

    def test_stale_request_or_invalid_source_body_and_recipient(self):
        self.deliver(Kind.REPLAN_ONLY, 11)
        self.assertEqual(self.local.last_plan, 53)
        self.local.acks.clear()
        for kind in COMMAND_CONTROLS:
            self.deliver(kind, 52, sender=2)
            self.deliver(kind, 52, body=bytes(11))
            self.deliver(kind, 52, body=GOAL3.pack(3., 15., 6.5))
            self.deliver(kind, 52, receiver=3)
        self.rt.d2_allocator = None
        self.deliver(Kind.REPLAN_ONLY, 52)
        self.assertEqual(self.local.last_plan, 53)
        self.assertFalse(self.local.acks)
        self.assertFalse(self.rt.locals[3].acks)
        self.assertEqual(self.rt._last_replan_request, {})

    def test_real_wire_budget_and_loss(self):
        for kind in COMMAND_CONTROLS:
            for budget, loss, accepted in ((223, 0., False), (224, 1., True), (224, 0., True)):
                with self.subTest(kind=kind, budget=budget, loss=loss):
                    channel = TDMAChannel(capacity_bits=budget, refill_bits=0, jitter_steps=0,
                                          delay_steps=0, loss_probability=loss)
                    sent = channel.send(kind, 0, 1, COMMAND_PADDING, sample_step=52, step=52)
                    self.assertEqual((sent['accepted'], sent['bits']), (accepted, 224))
                    arrivals = channel.receive(60)
                    self.assertEqual(len(arrivals), int(accepted and not loss))
                    self.assertEqual(channel.sent_bits, 224 * int(accepted))
                    self.assertEqual(channel.delivered_bits, len(arrivals) * 224)
                    if arrivals:
                        self.assertEqual(arrivals[0][0].kind, kind)
                        self.assertEqual(arrivals[0][0].payload, COMMAND_PADDING)

    def test_only_commander_slots_can_be_overridden(self):
        for step, receiver, kind in ((53, 1, Kind.COMMAND_PAD), (52, 3, Kind.REPLAN_ONLY), (52, 1, Kind.STATE)):
            with self.assertRaises(ValueError):
                D2CommandOverride(step, receiver, kind)


if __name__ == '__main__':
    unittest.main()
