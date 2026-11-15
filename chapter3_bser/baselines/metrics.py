"""Read-only physical/task metrics shared by pure MARL and the user's method."""
import copy
import numpy as np
import torch
from chapter3_bser.learning.config import scenario_fingerprint


def array(value):
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        value = torch.stack(value)
    return torch.as_tensor(value).detach().cpu().numpy().astype(np.float32, copy=True)


class EpisodeRecorder:
    """Observe an existing runtime after each physical step, without altering it.

    Both a PureMARLEnv and a D2/B2/B3/reach-avoid runtime can use this recorder.
    Pass their unwrapped simulator and final post-adapter four-agent rewards.
    """
    def __init__(self, scenario, reference_config):
        self.scenario, self.config = copy.deepcopy(scenario), copy.deepcopy(reference_config)

    def start(self, runtime):
        if runtime.step_count != 0:
            raise ValueError('metric recorder must start at physical step zero')
        self.position = array(runtime._agent_pos)
        self.steps, self.team_return, self.control_effort = 0, 0., 0.
        self.path = np.zeros(4, np.float64)
        self.clipped = 0

    def observe(self, runtime, final_rewards):
        if runtime.step_count != self.steps + 1:
            raise ValueError('record every physical transition exactly once')
        rewards = array(final_rewards).reshape(-1)
        if rewards.shape != (4,) or not np.isfinite(rewards).all():
            raise ValueError('four finite final rewards required')
        position = array(runtime._agent_pos)
        self.path += np.linalg.norm(position - self.position, axis=1)
        self.position = position
        self.team_return += float(rewards.mean())
        self.control_effort += float(np.square(array(runtime._agent_acc)).sum() * runtime.dt)
        self.steps += 1

    def finish(self, runtime):
        result = runtime.get_episode_result()
        if not result['terminated'] or runtime.step_count != self.steps:
            raise ValueError('complete, fully recorded terminal episode required')
        return dict(schema='auv.pure_marl.episode.v1', scenario_id=self.scenario['scenario_id'],
                    scenario_seed=self.scenario['scenario_seed'],
                    scenario_fingerprint=scenario_fingerprint(self.scenario), max_steps=self.config['max_steps'],
                    actual_length=self.steps, found_step=runtime.found_step, success_step=runtime.success_step,
                    handoff_step=runtime.executor_received_target_step, team_return=self.team_return,
                    path_length_by_agent=self.path.tolist(), path_length=float(self.path.sum()),
                    control_effort=self.control_effort, control_effort_is_physical_energy=False,
                    penalized_completion_steps=(runtime.success_step if result['success'] else
                        self.config['max_steps'] + runtime.failure_penalty_steps), **result)
