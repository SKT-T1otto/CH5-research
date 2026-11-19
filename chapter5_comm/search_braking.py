"""Estimated stopping-corridor check using the boat's own scan and local map."""
import numpy as np


def stopping_risk(position, velocity, mapper, scan, *, dt, sensor_range, bounds,
                  prior_gain, prior_strength, acceleration_limit):
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-6:
        return None
    direction = velocity / speed
    # The hold prior brakes proportionally to velocity and saturates. The
    # envelope includes one decision interval and the planner's own clearance.
    limit = prior_strength * min(acceleration_limit)
    gain = prior_strength * min(prior_gain)
    proportional_speed = limit / gain
    saturated = max(0., speed**2 - proportional_speed**2) / (2 * limit)
    distance = speed * dt + saturated + min(speed, proportional_speed) / gain
    clearance = float(mapper.planner_obstacle_clearance)
    end = np.clip(position + min(distance, sensor_range) * direction, 0., bounds)
    # A hit near the forward corridor is usable before the voxel crosses its
    # occupied threshold. Do not use exact obstacle geometry or peer truth.
    _, origin, distances, hits = scan
    rays = mapper.sensor_directions.detach().cpu().numpy()
    offsets = origin + rays[hits] * distances[hits, None] - position
    projected = offsets @ direction
    lateral = np.linalg.norm(offsets - projected[:, None] * direction, axis=1)
    if np.any((projected > 0) & (projected <= distance + clearance) & (lateral <= clearance)):
        return "sonar"
    if mapper.segment_is_free(position, position) and not mapper.segment_is_free(position, end):
        return "local_map"
    return None
