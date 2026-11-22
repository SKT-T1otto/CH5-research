from unittest import TestCase
import struct
import torch
from types import SimpleNamespace

from chapter5_comm.history_evidence import encode, decode, fuse
from chapter5_comm.protocol import Packet, Kind


class EvidenceTest(TestCase):
    def mapper(self):
        return SimpleNamespace(occupancy_logodds=torch.tensor([0., 3., -2., 1.]),
            occupancy_last_observed_step=torch.tensor([-1, 7, 8, 9]),
            occupancy_observation_count=torch.tensor([0, 7, 8, 9]))

    def test_exact_budget_and_causal_roundtrip(self):
        body, rows = encode(self.mapper())
        self.assertEqual(len(body), 120)
        self.assertEqual(decode(body, 4, 9), rows)
        self.assertEqual(Packet(Kind.MAP_EVIDENCE,0,2,1,9,10,body).bits,1088)
        with self.assertRaises(ValueError):
            decode(body,4,6)
        with self.assertRaises(ValueError):
            decode(body,5,9)

    def test_duplicate_snapshots_do_not_accumulate_and_free_replaces(self):
        own = self.mapper()
        mapper = SimpleNamespace(occupancy_logodds=torch.zeros(4), occupancy_logodds_clip=8.,
                                 _refresh_online_masks=lambda:None)
        local = SimpleNamespace(mapper=mapper, own_evidence=own,
                                remote_evidence={0:{1:(7,3072,7)}})
        fuse(local)
        first = mapper.occupancy_logodds.clone()
        fuse(local)
        torch.testing.assert_close(first,mapper.occupancy_logodds)
        local.remote_evidence[0][1]=(8,-2048,8)
        fuse(local)
        self.assertEqual(float(mapper.occupancy_logodds[1]),1.)

    def test_changed_previous_cells_get_refresh_priority(self):
        mapper=self.mapper()
        body, rows=encode(mapper,[(2,1024)])
        self.assertEqual(rows[0][0],2)
        self.assertLess(rows[0][2],0)

    def test_runtime_uses_received_bytes_without_forwarding_or_duplicates(self):
        from chapter5_comm.runtime import CH5Runtime
        from chapter5_comm.pipeline import scenarios
        from chapter5_comm.experiment_config import read_experiment
        _,base=read_experiment('configs/chapter5/d2_check_experiment_v1.json')
        runtime=CH5Runtime(base,scenarios(1,699)[0],map_transport='own_history_v1')
        self.addCleanup(runtime.close)
        sender, receiver=runtime.locals[0],runtime.locals[2]
        own=receiver.own_evidence.occupancy_logodds.clone()
        payload,rows=encode(sender.own_evidence)
        packet=Packet(Kind.MAP_EVIDENCE,0,2,100,0,0,payload)
        runtime._deliver(packet,1)
        first=receiver.mapper.occupancy_logodds.clone()
        runtime._deliver(packet,1)
        torch.testing.assert_close(first,receiver.mapper.occupancy_logodds)
        torch.testing.assert_close(own,receiver.own_evidence.occupancy_logodds)
        torch.testing.assert_close(runtime.locals[1].mapper.occupancy_logodds,
                                   runtime.locals[1].own_evidence.occupancy_logodds)
