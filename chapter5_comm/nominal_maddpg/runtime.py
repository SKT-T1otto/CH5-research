"""Native CH5 decision boundaries for deterministic four-hull residual policies."""
import numpy as np
from chapter5_comm.runtime import CH5Runtime
from .config import task_config


class NominalRuntime:
    def __init__(self, config, scene, seed, replay, *, episode_id=0):
        base = task_config(config)
        options = config["runtime"]
        self.core = CH5Runtime(base, scene, seed=seed, episode_id=episode_id,
            channel_config={k: v for k, v in config["channel"].items() if k != "name"},
            **{k: options[k] for k in ("navigation", "map_transport", "history_rule", "history_tie_seed",
                                      "early_brake", "scan_first", "horizon", "candidate_count", "map_token_count")})
        self.replay = replay

    def observe(self):
        if self.core.terminal:
            raise RuntimeError("terminal mission has no next decision")
        self.core.begin_step()
        return np.stack(self.core.observations)

    def advance(self, policy):
        obs = self.observe()
        action = policy.step(obs, explore=False)
        if isinstance(action, (list, tuple)):  # inherited zero-residual policy
            action = np.stack([np.asarray(row).reshape(3) for row in action])
        transition = self.core.advance(actions=action)
        done = self.core.terminal
        # begin_step is idempotent. The next policy consumes this exact boundary;
        # the final physics observation is retained only for a terminal row.
        if not done:
            transition["next_observations"] = self.observe()
            transition["next_action_enabled"] = self.core.action_enabled().copy()
        transition["observations"] = obs
        transition["success_flags"] = [done and bool(self.core.physical.mission_complete)] * 4
        self.replay.push_transition(transition)
        return transition

    def close(self):
        self.core.close()

