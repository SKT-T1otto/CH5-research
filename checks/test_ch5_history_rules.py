import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter5_comm import history_evidence, history_rules
from chapter5_comm.experiment_config import read_experiment
from chapter5_comm.pipeline import scenarios
from chapter5_comm.protocol import Kind, Packet
from chapter5_comm.runtime import CH5Runtime, QUERY9


class HistoryRuleTests(unittest.TestCase):
    def test_corrections_and_exact_ties(self):
        rows=[(i,5,1024,3) for i in range(20)]
        native=rows[:14]
        odds=np.ones(20)
        previous={i:0 for i in range(13)}
        dist=np.ones(20);dist[19]=0.
        result=history_rules.choose('route',native,rows,odds,previous,dist)
        self.assertEqual(result,native[:13]+[rows[19]])
        previous[13]=0
        self.assertEqual(history_rules.choose('route',native,rows,odds,previous,dist),native)
        self.assertEqual(history_rules.choose('route',native,rows,odds,{},np.ones(20)),native)
        self.assertEqual(history_rules.choose('route',native,rows,odds,{},None),native)
        first=history_rules.choose('native_tie',native,rows,odds,{},seed=29)
        self.assertEqual(first,history_rules.choose('native_tie',native,rows,odds,{},seed=29))
        self.assertEqual(first[:12],native[:12])

    def test_real_send_and_source_ledger_for_each_rule(self):
        _,base=read_experiment('configs/chapter5/d2_check_experiment_v1.json')
        scene=scenarios(1,699)[0]
        for rule in history_rules.RULES:
            with self.subTest(rule=rule):
                rt=CH5Runtime(base,scene,map_transport='own_history_v1',navigation='causal_d2_v1',
                              history_rule=rule,history_tie_seed=29)
                try:
                    s,r=0,1
                    rt._bootstrap_turn[s]=1
                    loc=rt.locals[s]
                    query=np.array([.17,.18,.5,.74,.72,.6,0.,1.,0.],np.float32)
                    rt._deliver(Packet(Kind.QUERY,r,s,999,0,0,QUERY9.pack(*query)),0)
                    source=loc.own_evidence
                    native_body,native=history_evidence.encode(source)
                    body,expected=(native_body,native) if rule=='native' else history_rules.encode(
                        source,loc.mapper,{},native_body,native,rule,query,s,r,0,29)
                    original=source.occupancy_logodds.clone()
                    context=patch.object(loc.mapper,'grid_astar_path',side_effect=AssertionError('query used by blind rule'))
                    if rule in ('no_query','native_tie'):
                        with context:
                            sent=rt._bootstrap()
                    else:
                        sent=rt._bootstrap()
                    self.assertTrue(sent['accepted'])
                    wire=Packet.from_bytes(next(item[2] for item in rt.channel.queue if item[3]['packet_id']==sent['packet_id']))
                    self.assertEqual((wire.kind,wire.receiver,wire.bits,wire.payload),(Kind.MAP_EVIDENCE,r,1088,body))
                    self.assertEqual(rt._evidence_sent[s,r],{i:q for i,t,q,n in expected})
                    rt._deliver(wire,1)
                    self.assertEqual(rt.locals[r].remote_evidence[s],{i:(t,q,n) for i,t,q,n in expected})
                    torch.testing.assert_close(original,source.occupancy_logodds)
                finally:
                    rt.close()

