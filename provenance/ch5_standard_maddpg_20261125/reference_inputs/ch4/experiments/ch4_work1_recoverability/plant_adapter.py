"""N0 relative-water actuator and drag for one freshly reset four-vehicle mission."""
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import torch


RHO = 0.15
# Median crossing speed from round01/calibration.json, in m/s.
CROSSING_SPEED = 0.6925365924835205
MIXED = "linear_quadratic_mix_0.5"


def bounded(value, low, high):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not low <= value <= high:
        raise ValueError(f"parameter must be finite and in [{low}, {high}]")
    return float(value)


@dataclass(frozen=True)
class WaterParameters:
    beta: float = 0.10
    drag: tuple = (1., 1., 1., 1.)
    thrust: tuple = (1., 1., 1., 1.)
    law: str = "linear"

    def __post_init__(self):
        object.__setattr__(self, "beta", bounded(self.beta, 0.05, 0.15))
        if self.law not in ("linear", MIXED):
            raise ValueError("unknown horizontal drag law")
        for name, low, high in (("drag", .85, 1.20), ("thrust", .85, 1.10)):
            values = tuple(bounded(v, low, high) for v in getattr(self, name))
            if len(values) != 4:
                raise ValueError(f"{name} requires one multiplier per vessel")
            object.__setattr__(self, name, values)


class RelativeWaterPlant:
    """Install on mission.unwrapped. Use one context per episode, after reset.

    The original environment owns the navigation prior, command limits, vertical
    damping/buoyancy, speed limits, contact handling, rewards and termination.
    """

    def __init__(self, core, parameters):
        self.core = core
        self.set_parameters(parameters)
        self._installed = False

    def set_parameters(self, parameters):
        """Change the physical coefficients at a decision boundary, retaining actuator memory."""
        if not isinstance(parameters, WaterParameters):
            raise TypeError("expected WaterParameters")
        self.parameters = parameters
        self.k = self.core._a_xy_max.new_tensor(parameters.drag)[:, None]
        self.s = self.core._a_xy_max.new_tensor(parameters.thrust)[:, None]

    def __enter__(self):
        core = self.core
        if self._installed or any(key in vars(core) for key in ("_flow_at", "_record_residual_prior_diagnostics")):
            raise RuntimeError("a physical hook is already installed")
        if core.step_count or (core._flow_phase_x, core._flow_phase_y) != (0., 0.) or not core.use_residual_prior:
            raise ValueError("requires a freshly reset, zero-phase mission with native command composition")
        self._episode = core.episode_outcome
        self._diagnostics = core._record_residual_prior_diagnostics
        self.native_drag = core._drag_xy
        self.limits = torch.stack((core._a_xy_max, core._a_xy_max, core._a_z_max), dim=1)
        if self.limits.shape != (4, 3) or not (self.limits > 0).all():
            raise ValueError("expected positive native four-vessel command limits")
        self.lag = torch.zeros_like(self.limits)
        core._drag_xy = torch.zeros_like(self.native_drag)
        core._record_residual_prior_diagnostics = self._apply_thrust
        core._flow_at = self._relative_drag
        self._installed = True
        return self

    def __exit__(self, *_):
        self.core._drag_xy = self.native_drag
        del self.core._flow_at
        del self.core._record_residual_prior_diagnostics
        self._installed = False

    def _apply_thrust(self, prior, residual, final):
        if self.core.episode_outcome is not self._episode:
            raise RuntimeError("reinstall the relative-water plant after reset")
        self._diagnostics(prior, residual, final)
        self.command = final.clone()                 # already clipped, m/s²
        self.sent = self.command / self.limits       # normalized applied command, d=0
        self.lag = RHO * self.lag + (1. - RHO) * self.sent
        self.thrust_acc = self.lag * self.limits    # before damping, m/s²
        self.thrust_acc[:, :2] *= self.s
        final.copy_(self.thrust_acc)

    def _relative_drag(self, position):
        core = self.core
        self.water = self.parameters.beta * torch.stack(
            (torch.sin(.25 * position[:, 1]), torch.cos(.22 * position[:, 0])), dim=1)
        relative = core._agent_vel[:, :2] + core.dt * core._agent_acc[:, :2] - self.water
        factor = 1.
        if self.parameters.law == MIXED:
            factor = .5 + .5 * torch.linalg.vector_norm(relative, dim=1, keepdim=True) / CROSSING_SPEED
        acceleration = torch.zeros_like(position)
        acceleration[:, :2] = -self.native_drag[:, None] * self.k * factor * relative
        self.drag_acc = acceleration[:, :2]
        return acceleration
