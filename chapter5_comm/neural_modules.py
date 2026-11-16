"""Trainable communication networks, independent of the simulator.

These modules do not import the simulator or third-chapter runtime. Integration
is supplied by runtime.py, data.py, training.py and pipeline.py. A finite tensor
test alone is not a task result.

Packet payloads use a fixed symmetric int8 quantizer with a straight-through
gradient. ``packet_bits`` includes the configured header length; protocol.py
serializes actual frames and pipeline.py verifies that their bit costs agree.
The caller must supply only delivered receiver queries and received packets.
Masking protects unavailable tensor values, not an incorrectly constructed
upstream observation. Decoded remote geometry is an estimate for learning; it
must not be written into an authoritative occupancy map as a true observation.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _mask(name: str, value: Tensor, shape: tuple) -> None:
    if value.dtype != torch.bool or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be a bool tensor of shape {shape}")


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite visible values")


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    safe = torch.where(valid[..., None], values, torch.zeros_like(values))
    count = valid.sum(dim=1, keepdim=True).to(values.dtype).clamp_min(1)
    return safe.sum(dim=1) / count


@dataclass
class PacketEncoding:
    """Quantized slots for a batch of sender/receiver pairs.

    ``tokens`` has shape [B,K,D], ``codes`` is int8 [B,K,D], ``stream_types``
    is long [K] (0 geometry, 1 dynamic), ``valid`` is bool [B,K], and
    ``packet_bits`` is long [K]. Empty streams produce invalid zero slots.
    """

    tokens: Tensor
    codes: Tensor
    stream_types: Tensor
    valid: Tensor
    packet_bits: Tensor


class TaskPacketEncoder(nn.Module):
    """Query-conditioned geometry/dynamic slot compression.

    A query is usable only when ``query_available`` is true. Otherwise its
    value (including NaNs) is ignored in favor of a learned unknown query.
    There is no handwritten task-stage priority or value score.
    """

    def __init__(
        self,
        map_dim: int,
        dynamic_dim: int,
        query_dim: int,
        latent_dim: int = 32,
        map_slots: int = 2,
        dynamic_slots: int = 2,
        num_heads: int = 4,
        header_bytes: int = 16,
    ):
        super().__init__()
        for name, value in (("map_dim", map_dim), ("dynamic_dim", dynamic_dim),
                            ("query_dim", query_dim), ("latent_dim", latent_dim),
                            ("num_heads", num_heads)):
            _positive_int(name, value)
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        if any(type(x) is not int or x < 0 for x in (map_slots, dynamic_slots, header_bytes)):
            raise ValueError("slot counts and header_bytes must be nonnegative integers")
        if map_slots + dynamic_slots < 1:
            raise ValueError("at least one packet slot is required")
        self.map_dim, self.dynamic_dim, self.query_dim = map_dim, dynamic_dim, query_dim
        self.latent_dim = latent_dim
        self.map_slots, self.dynamic_slots = map_slots, dynamic_slots
        self.header_bytes = header_bytes
        self.map_input = nn.Linear(map_dim, latent_dim)
        self.dynamic_input = nn.Linear(dynamic_dim, latent_dim)
        self.query_input = nn.Linear(query_dim, latent_dim)
        self.unknown_query = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.map_queries = nn.Parameter(torch.randn(map_slots, latent_dim) * 0.02)
        self.dynamic_queries = nn.Parameter(torch.randn(dynamic_slots, latent_dim) * 0.02)
        self.map_attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=0.0, batch_first=True)
        self.dynamic_attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=0.0, batch_first=True)
        self.map_output = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU())
        self.dynamic_output = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU())

    def _encode_stream(self, values, valid, query, slots, projection, attention, output):
        batch = values.shape[0]
        available = valid.any(dim=1)
        if slots.shape[0] == 0 or values.shape[1] == 0:
            return values.new_zeros(batch, slots.shape[0], self.latent_dim), available
        # Sanitize before projection, so inaccessible NaNs cannot contaminate
        # either the forward pass or parameter gradients.
        safe = torch.where(valid[..., None], values, torch.zeros_like(values))
        _finite("visible stream tokens", safe)
        keys = projection(safe)
        # MultiheadAttention has no defined all-key-masked softmax. Supply a
        # dummy zero key for an empty row and discard that row's result.
        keys = torch.where(available[:, None, None], keys, torch.zeros_like(keys))
        key_valid = valid.clone()
        key_valid[:, 0] |= ~available
        queries = slots[None] + query[:, None]
        encoded, _ = attention(queries, keys, keys, key_padding_mask=~key_valid, need_weights=False)
        encoded = output(encoded)
        return torch.where(available[:, None, None], encoded, torch.zeros_like(encoded)), available

    def forward(self, map_tokens, dynamic_tokens, receiver_query, query_available, map_mask, dynamic_mask):
        if map_tokens.ndim != 3 or map_tokens.shape[-1] != self.map_dim:
            raise ValueError("map_tokens must have shape [B,N,map_dim]")
        batch = map_tokens.shape[0]
        if batch < 1:
            raise ValueError("empty batches are not supported")
        if dynamic_tokens.ndim != 3 or dynamic_tokens.shape[0] != batch or dynamic_tokens.shape[-1] != self.dynamic_dim:
            raise ValueError("dynamic_tokens must have shape [B,M,dynamic_dim]")
        if tuple(receiver_query.shape) != (batch, self.query_dim):
            raise ValueError("receiver_query has an incompatible shape")
        _mask("query_available", query_available, (batch,))
        _mask("map_mask", map_mask, tuple(map_tokens.shape[:2]))
        _mask("dynamic_mask", dynamic_mask, tuple(dynamic_tokens.shape[:2]))
        safe_query = torch.where(query_available[:, None], receiver_query, torch.zeros_like(receiver_query))
        _finite("visible receiver_query", safe_query)
        query = torch.where(query_available[:, None], self.query_input(safe_query), self.unknown_query[None])
        geometry, has_geometry = self._encode_stream(
            map_tokens, map_mask, query, self.map_queries,
            self.map_input, self.map_attention, self.map_output,
        )
        dynamic, has_dynamic = self._encode_stream(
            dynamic_tokens, dynamic_mask, query, self.dynamic_queries,
            self.dynamic_input, self.dynamic_attention, self.dynamic_output,
        )
        bounded = torch.tanh(torch.cat((geometry, dynamic), dim=1))
        codes = torch.round(bounded * 127).to(torch.int8)
        dequantized = codes.to(bounded.dtype) / 127
        # Parentheses make the forward value exactly the quantized value.
        tokens = dequantized + (bounded - bounded.detach())
        kinds = torch.cat((
            torch.zeros(self.map_slots, dtype=torch.long, device=map_tokens.device),
            torch.ones(self.dynamic_slots, dtype=torch.long, device=map_tokens.device),
        ))
        valid = torch.cat((
            has_geometry[:, None].expand(-1, self.map_slots),
            has_dynamic[:, None].expand(-1, self.dynamic_slots),
        ), dim=1)
        bits = torch.full(
            (self.map_slots + self.dynamic_slots,),
            (self.latent_dim + self.header_bytes) * 8,
            dtype=torch.long,
            device=map_tokens.device,
        )
        return PacketEncoding(tokens, codes, kinds, valid, bits)


class GeometryTemporalDecoder(nn.Module):
    """Geometry-conditioned learned transport of delayed dynamic latents.

    Geometry slots do not move with message age. Dynamic slots evolve through
    a learned autonomous vector field explicitly conditioned on geometry.
    Euler integration and a composition loss do not establish a semigroup or
    a physically correct target-motion model. Prediction supervision is still
    required to avoid a trivial identity solution.
    """

    def __init__(self, latent_dim=32, geometry_dim=32, state_dim=6, integration_steps=4):
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("geometry_dim", geometry_dim),
                            ("state_dim", state_dim), ("integration_steps", integration_steps)):
            _positive_int(name, value)
        self.latent_dim, self.geometry_dim = latent_dim, geometry_dim
        self.integration_steps = integration_steps
        self.local_geometry = nn.Linear(geometry_dim, latent_dim)
        self.map_update = nn.Sequential(nn.Linear(3 * latent_dim, 2 * latent_dim), nn.GELU(), nn.Linear(2 * latent_dim, latent_dim))
        self.vector_field = nn.Sequential(nn.Linear(3 * latent_dim, 2 * latent_dim), nn.Tanh(), nn.Linear(2 * latent_dim, latent_dim), nn.Tanh())
        self.state_mean = nn.Sequential(nn.Linear(3 * latent_dim, 2 * latent_dim), nn.GELU(), nn.Linear(2 * latent_dim, state_dim))
        self.state_variance = nn.Linear(3 * latent_dim, state_dim)
        self.age_variance = nn.Linear(latent_dim, state_dim)

    def transport(self, z, geometry, elapsed):
        if z.ndim != 2 or z.shape[-1] != self.latent_dim or geometry.shape != z.shape:
            raise ValueError("z and geometry must have shape [N,latent_dim]")
        if tuple(elapsed.shape) != (z.shape[0],):
            raise ValueError("elapsed must have shape [N]")
        _finite("transport latent", z)
        _finite("transport geometry", geometry)
        _finite("elapsed", elapsed)
        if bool((elapsed < 0).any()):
            raise ValueError("elapsed cannot be negative")
        if not getattr(self, "transport_enabled", True):
            return z
        if not getattr(self, "geometry_conditioning", True):
            geometry = torch.zeros_like(geometry)
        dt = elapsed.to(dtype=z.dtype)[:, None] / self.integration_steps
        current = z
        for _ in range(self.integration_steps):
            drift = self.vector_field(torch.cat((current, geometry, current * geometry), dim=-1))
            current = current + dt * drift
        return torch.where((elapsed == 0)[:, None], z, current)

    def forward(self, packet_tokens, stream_types, received, ages, local_geometry):
        if packet_tokens.ndim != 3 or packet_tokens.shape[-1] != self.latent_dim:
            raise ValueError("packet_tokens must have shape [B,K,latent_dim]")
        batch, count, _ = packet_tokens.shape
        if batch < 1:
            raise ValueError("empty batches are not supported")
        if stream_types.dtype != torch.long or tuple(stream_types.shape) != (count,):
            raise ValueError("stream_types must be a long tensor of shape [K]")
        if bool(((stream_types != 0) & (stream_types != 1)).any()):
            raise ValueError("stream_types supports only 0 geometry and 1 dynamic")
        _mask("received", received, (batch, count))
        if tuple(ages.shape) != (batch, count) or tuple(local_geometry.shape) != (batch, self.geometry_dim):
            raise ValueError("ages or local_geometry has an incompatible shape")
        safe = torch.where(received[..., None], packet_tokens, torch.zeros_like(packet_tokens))
        safe_ages = torch.where(received, ages, torch.zeros_like(ages))
        _finite("received packet tokens", safe)
        _finite("received ages", safe_ages)
        _finite("local_geometry", local_geometry)
        if bool((safe_ages < 0).any()):
            raise ValueError("received ages cannot be negative")
        geometry_valid = received & (stream_types[None] == 0)
        dynamic_valid = received & (stream_types[None] == 1)
        geometry_mean = _masked_mean(safe, geometry_valid)
        local = self.local_geometry(local_geometry)
        update = self.map_update(torch.cat((local, geometry_mean, local * geometry_mean), dim=-1))
        has_map = geometry_valid.any(dim=1)
        map_context = local + torch.where(has_map[:, None], update, torch.zeros_like(update))
        geometry_for_packets = map_context[:, None].expand(-1, count, -1)
        # Do not evolve static or erased packet values, even internally.
        dynamic_input = torch.where(dynamic_valid[..., None], safe, torch.zeros_like(safe))
        dynamic_ages = torch.where(dynamic_valid, safe_ages, torch.zeros_like(safe_ages))
        aligned = self.transport(
            dynamic_input.reshape(batch * count, self.latent_dim),
            geometry_for_packets.reshape(batch * count, self.latent_dim),
            dynamic_ages.reshape(batch * count),
        ).reshape(batch, count, self.latent_dim)
        dynamic_context = _masked_mean(aligned, dynamic_valid)
        combined = torch.cat((map_context, dynamic_context, map_context * dynamic_context), dim=-1)
        mean = self.state_mean(combined)
        mean_age = _masked_mean(dynamic_ages[..., None], dynamic_valid)
        variance = F.softplus(self.state_variance(combined)) + 1e-4
        variance = variance + mean_age * F.softplus(self.age_variance(map_context))
        log_variance = variance.log().clamp(-10.0, 10.0)
        return {
            "map_context": map_context,
            "dynamic_context": dynamic_context,
            "state_mean": mean,
            "state_logvar": log_variance,
            "has_map": has_map,
            "has_dynamic": dynamic_valid.any(dim=1),
        }


class RouteUtilityHead(nn.Module):
    """Predict route encounter/utility logits from coupled geometry/dynamics.

    Actual candidate execution trajectories and geometric encounter labels
    must come from a separate third-chapter data provider, not this module.
    Encounter labels are approximate geometric targets, not full-task returns.
    """

    def __init__(self, latent_dim, route_dim, hidden_dim=64):
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("route_dim", route_dim), ("hidden_dim", hidden_dim)):
            _positive_int(name, value)
        self.latent_dim, self.route_dim = latent_dim, route_dim
        self.route = nn.Sequential(nn.Linear(route_dim, latent_dim), nn.GELU())
        self.utility = nn.Sequential(nn.Linear(5 * latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, map_context, dynamic_context, route_features):
        if map_context.ndim != 2 or map_context.shape[-1] != self.latent_dim or dynamic_context.shape != map_context.shape:
            raise ValueError("map/dynamic contexts must have shape [B,latent_dim]")
        if route_features.ndim != 3 or route_features.shape[0] != map_context.shape[0] or route_features.shape[-1] != self.route_dim:
            raise ValueError("route_features must have shape [B,C,route_dim]")
        route = self.route(route_features)
        geometry = map_context[:, None].expand_as(route)
        dynamic = dynamic_context[:, None].expand_as(route)
        features = torch.cat((route, geometry, dynamic, geometry * dynamic, route * dynamic), dim=-1)
        return self.utility(features).squeeze(-1)


class BudgetedPacketScheduler(nn.Module):
    """Learned single-packet categorical choice with an always-legal NOOP.

    The last category is NOOP. The caller must update budget and remove or
    otherwise account for a selected packet before another choice. This module
    alone does not implement a sequence budget, MAC, channel, or PPO learner.
    """

    def __init__(self, latent_dim, context_dim, hidden_dim=64):
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("context_dim", context_dim), ("hidden_dim", hidden_dim)):
            _positive_int(name, value)
        self.latent_dim, self.context_dim = latent_dim, context_dim
        self.packet_score = nn.Sequential(nn.Linear(latent_dim + context_dim + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.noop_score = nn.Sequential(nn.Linear(context_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, packet_tokens, context, packet_bits, budget_bits, valid):
        if packet_tokens.ndim != 3 or packet_tokens.shape[-1] != self.latent_dim:
            raise ValueError("packet_tokens must have shape [B,K,latent_dim]")
        batch, count, _ = packet_tokens.shape
        if tuple(context.shape) != (batch, self.context_dim) or tuple(budget_bits.shape) != (batch,):
            raise ValueError("context or budget_bits has an incompatible shape")
        _mask("valid", valid, (batch, count))
        if tuple(packet_bits.shape) == (count,):
            costs = packet_bits[None].expand(batch, -1)
        elif tuple(packet_bits.shape) == (batch, count):
            costs = packet_bits
        else:
            raise ValueError("packet_bits must have shape [K] or [B,K]")
        _finite("context", context)
        _finite("packet_bits", costs)
        _finite("budget_bits", budget_bits)
        if bool((costs <= 0).any()) or bool((budget_bits < 0).any()):
            raise ValueError("packet costs must be positive and budgets nonnegative")
        feasible = valid & (costs <= budget_bits[:, None])
        safe = torch.where(feasible[..., None], packet_tokens, torch.zeros_like(packet_tokens))
        _finite("available packet tokens", safe)
        budget = budget_bits.to(packet_tokens.dtype)
        cost_ratio = costs.to(packet_tokens.dtype) / budget[:, None].clamp_min(1)
        budget_feature = torch.log1p(budget)[:, None]
        features = torch.cat((safe, context[:, None].expand(-1, count, -1), cost_ratio[..., None], budget_feature[:, None].expand(-1, count, -1)), dim=-1)
        packet_logits = self.packet_score(features).squeeze(-1).masked_fill(~feasible, -torch.inf)
        noop = self.noop_score(torch.cat((context, budget_feature), dim=-1))
        return Categorical(logits=torch.cat((packet_logits, noop), dim=-1))


def route_distillation_loss(student_logits, teacher_logits, route_valid, temperature=1.0):
    """Masked route-distribution KL; the external teacher is stop-gradient."""
    if student_logits.ndim != 2 or teacher_logits.shape != student_logits.shape:
        raise ValueError("student and teacher route logits must have shape [B,C]")
    _mask("route_valid", route_valid, tuple(student_logits.shape))
    if student_logits.shape[0] < 1 or not bool(route_valid.any(dim=1).all()):
        raise ValueError("each sample requires at least one valid route")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    student = torch.where(route_valid, student_logits, torch.zeros_like(student_logits))
    teacher = torch.where(route_valid, teacher_logits.detach(), torch.zeros_like(teacher_logits))
    _finite("student route logits", student)
    _finite("teacher route logits", teacher)
    floor = torch.finfo(student.dtype).min
    log_student = F.log_softmax((student / temperature).masked_fill(~route_valid, floor), dim=-1)
    teacher_prob = F.softmax((teacher / temperature).masked_fill(~route_valid, floor), dim=-1)
    return F.kl_div(log_student, teacher_prob, reduction="batchmean") * temperature ** 2


def route_encounter_loss(logits, labels, route_valid):
    """BCE for externally supplied geometric route-encounter probabilities.

    Labels may be binary or soft values in [0,1]. Only valid entries contribute.
    A row without labels is allowed if some other row supplies supervision.
    The recording/trajectory label provider is not implemented here.
    """
    if logits.ndim != 2 or labels.shape != logits.shape:
        raise ValueError("route logits and labels must have shape [B,C]")
    _mask("route_valid", route_valid, tuple(logits.shape))
    if not bool(route_valid.any()):
        raise ValueError("at least one valid route label is required")
    prediction = torch.where(route_valid, logits, torch.zeros_like(logits))
    target = torch.where(route_valid, labels.detach(), torch.zeros_like(labels))
    _finite("route logits", prediction)
    _finite("route labels", target)
    if bool(((target < 0) | (target > 1)).any()):
        raise ValueError("valid route labels must lie in [0,1]")
    losses = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    return torch.where(route_valid, losses, torch.zeros_like(losses)).sum() / route_valid.sum()


def gaussian_prediction_nll(mean, logvar, target):
    """Diagonal Gaussian prediction NLL without its constant normalizer."""
    if mean.shape != logvar.shape or target.shape != mean.shape or mean.numel() == 0:
        raise ValueError("mean, logvar and target require equal nonempty shapes")
    _finite("prediction mean", mean)
    _finite("prediction logvar", logvar)
    _finite("prediction target", target)
    bounded_logvar = logvar.clamp(-10.0, 10.0)
    error = mean - target.detach()
    return 0.5 * (bounded_logvar + error.square() * torch.exp(-bounded_logvar)).mean()


def temporal_composition_loss(decoder, z, geometry, first_elapsed, second_elapsed):
    """Composition consistency at fixed geometry with no intervening message.

    This is an auxiliary objective, not an exact flow guarantee. It needs a
    prediction/task loss: the identity transport minimizes it trivially.
    """
    first = decoder.transport(z, geometry, first_elapsed)
    split = decoder.transport(first, geometry, second_elapsed)
    direct = decoder.transport(z, geometry, first_elapsed + second_elapsed)
    return F.mse_loss(split, direct)
