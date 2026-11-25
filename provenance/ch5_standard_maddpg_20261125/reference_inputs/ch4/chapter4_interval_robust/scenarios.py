"""Paired physical scenarios with isolated train/development/final-test seed blocks."""
import copy
from core.scenarios.ch3_generator_impl import build_scenario_manifests


def make_scene(config,split,index):
    if split not in ('train','dev','test') or type(index) is not int or index<0:
        raise ValueError('explicit split and nonnegative scenario index required')
    seed=int(config['splits'][split])+index
    manifests=build_scenario_manifests(count=1,generator_seed=seed,
                                      split='train' if split=='train' else 'validation',
                                      profiles=['M20_MOVING_UNKNOWN_MULTI'])
    scene=copy.deepcopy(manifests['M20_MOVING_UNKNOWN_MULTI']['scenarios'][0])
    scene.update(scenario_id=f'ch4_{split}_{seed}',scenario_role=split,scenario_split=split,
                 max_steps=config['max_steps'],disturbance_seed=seed+5000000,
                 ch4_split_schema='ch4.scenes.v1')
    # The first domain varies amplitude, not the hidden spatial phase.
    if scene['flow_phase_x']!=0 or scene['flow_phase_y']!=0:
        raise ValueError('nonzero field phases need a separately declared model/domain')
    return scene
