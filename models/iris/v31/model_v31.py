"""Real observation -> dense carrier -> protected continuous evidence paths.

Production capacity and target channels are explicit recipe data. Tiny test
capacities do not become scientific defaults. Frozen DINO features can be cached;
the trainable projection, carrier and endpoint states cannot be cached across
optimizer updates. Orthographic camera13 = origin/right/up/forward/half_extent.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class IRISConfigV31:
    carrier_resolution: int
    carrier_channels: int
    mechanical_channels: tuple[str, ...]
    pair_channels: tuple[str, ...]
    query_width: int = 192
    query_layers: int = 3
    fourier_bands: int = 4
    private_bottleneck: int = 64
    lift_chunk: int = 4096
    delta_bound: float = .03

    def validate(self):
        if min(self.carrier_resolution, self.carrier_channels, self.lift_chunk) <= 0:
            raise ValueError("IRIS_V31_CAPACITY_INVALID")
        if (self.query_width, self.query_layers, self.fourier_bands,
                self.private_bottleneck) != (192, 3, 4, 64):
            raise ValueError("IRIS_V31_FROZEN_QUERY_DRIFT")
        if not math.isfinite(self.delta_bound) or self.delta_bound <= 0:
            raise ValueError("IRIS_V31_DELTA_BOUND_INVALID")
        forbidden = {"attachment", "grip", "support", "same_drawable", "confidence"}
        for channels in (self.mechanical_channels, self.pair_channels):
            if (not channels or len(set(channels)) != len(channels)
                    or any(not isinstance(x, str) or not x or x.lower() in forbidden for x in channels)):
                raise ValueError("IRIS_V31_PRIMITIVE_CHANNELS_REQUIRED")


def admit_observations_v31(final, layer2, alpha, cameras):
    """Expensive validation belongs at input/cache admission, once per packet.

    This numerical court is not Platform source/raster/observability admission.
    An arbitrary tensor cannot become an admitted subject by passing this call.
    """
    if final.ndim != 5 or final.shape[-1] != 384 or layer2.shape != final.shape:
        raise ValueError("IRIS_V31_DINO_FEATURE_SHAPE")
    b, views, h, w, _ = final.shape
    if min(b, views, h, w) <= 0 or h != w:
        raise ValueError("IRIS_V31_DINO_GRID_SHAPE")
    if alpha.shape != (b, views, h, w, 196) or cameras.shape != (b, views, 13):
        raise ValueError("IRIS_V31_ALPHA_CAMERA_SHAPE")
    for value in (final, layer2, alpha, cameras):
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError("IRIS_V31_NONFINITE_OBSERVATIONS")
        if value.device != final.device:
            raise ValueError("IRIS_V31_OBSERVATION_DEVICE_MISMATCH")
    if (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("IRIS_V31_ALPHA_RANGE")
    axes = cameras[..., 3:12].reshape(b, views, 3, 3)
    gram = axes @ axes.transpose(-1, -2)
    if not torch.allclose(gram, torch.eye(3, device=gram.device, dtype=gram.dtype).expand_as(gram), atol=1e-5, rtol=0):
        raise ValueError("IRIS_V31_CAMERA_NONORTHONORMAL")
    # Either handedness is allowed only as an explicit right/up/forward frame.
    if (cameras[..., 12] <= 0).any():
        raise ValueError("IRIS_V31_CAMERA_EXTENT_NONPOSITIVE")


def alpha_patches_v31(alpha):
    if alpha.ndim != 5 or alpha.shape[2] != 1 or alpha.shape[-2:] != (518, 518):
        raise ValueError("IRIS_V31_ALPHA_RASTER_SHAPE")
    b, v = alpha.shape[:2]
    return F.unfold(alpha.reshape(b*v, 1, 518, 518), 14, stride=14).transpose(1, 2).reshape(b, v, 37, 37, 196)


class DirectDinoFusionV31(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = nn.Linear(384, 64)
        self.alpha = nn.Linear(196, 64)

    def forward(self, final, layer2, alpha):
        return torch.cat((final, self.layer2(layer2), self.alpha(alpha)), dim=-1)


class DenseObservationEncoderV31(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.carrier_channels
        self.projection = nn.Linear(512, c)
        self.refine = nn.Sequential(nn.Conv3d(2*c+4, c, 3, padding=1), nn.SiLU(),
                                    nn.Conv3d(c, c, 3, padding=1), nn.SiLU())

    def forward(self, tokens, cameras):
        b, views, h, w, _ = tokens.shape
        c, r = self.config.carrier_channels, self.config.carrier_resolution
        maps = self.projection(tokens).permute(0, 1, 4, 2, 3)
        centers = (torch.arange(r, device=tokens.device, dtype=tokens.dtype)+.5)*2/r-1
        z, y, x = torch.meshgrid(centers, centers, centers, indexing="ij")
        xyz = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        pieces = []
        # Reduce views online: no [B,V,512,R^3] or all-view lifted tensor.
        for points in xyz.split(self.config.lift_chunk):
            mean_sum = tokens.new_zeros((b, c, len(points)))
            square_sum = torch.zeros_like(mean_sum)
            count = tokens.new_zeros((b, 1, len(points)))
            for view in range(views):
                cam = cameras[:, view]
                relative = points[None] - cam[:, None, :3]
                u = (relative*cam[:, None, 3:6]).sum(-1)/cam[:, None, 12]
                v = -(relative*cam[:, None, 6:9]).sum(-1)/cam[:, None, 12]
                grid = torch.stack((u, v), -1)
                valid = (grid.abs() <= 1).all(-1)[:, None]
                sampled = F.grid_sample(maps[:, view], grid[:, None], mode="bilinear",
                                        padding_mode="zeros", align_corners=False).squeeze(2)
                sampled = sampled*valid
                mean_sum = mean_sum+sampled
                square_sum = square_sum+sampled.square()
                count = count+valid
            mean = mean_sum/count.clamp_min(1)
            variance = (square_sum/count.clamp_min(1)-mean.square()).clamp_min(0)
            pieces.append(torch.cat((mean, variance, points.T[None].expand(b, -1, -1), count/views), 1))
        return self.refine(torch.cat(pieces, -1).reshape(b, 2*c+4, r, r, r))


class PrivatePathV31(nn.Module):
    def __init__(self, width, bottleneck):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(width, bottleneck), nn.SiLU(), nn.Linear(bottleneck, width))

    def forward(self, state):
        return state+self.adapter(state)


class IRISV31(nn.Module):
    architecture_id = "RealSaS.IRIS.DenseObservationEvidence.v3_1"
    output_authority = "CONTINUOUS_EVIDENCE_ONLY"

    def __init__(self, config: IRISConfigV31):
        super().__init__()
        config.validate()
        self.config = config
        self.fusion = DirectDinoFusionV31()
        self.encoder = DenseObservationEncoderV31(config)
        width = config.query_width
        self.query = nn.Sequential(nn.Linear(config.carrier_channels+27, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.gm = PrivatePathV31(width, config.private_bottleneck)
        self.gp = PrivatePathV31(width, config.private_bottleneck)
        self.geometry = nn.Linear(width, 4)
        self.mechanics = nn.Linear(width, len(config.mechanical_channels))
        self.appearance = nn.Sequential(nn.Linear(width+13, width), nn.SiLU(), nn.Linear(width, 5))
        self.relation = nn.Sequential(nn.Linear(4*width+20, width), nn.SiLU(),
                                      nn.Linear(width, len(config.pair_channels)))

    def encode(self, *, final, layer2, alpha, cameras):
        return self.encoder(self.fusion(final, layer2, alpha), cameras)

    def query_states(self, carrier, xyz):
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.shape[0] != carrier.shape[0]:
            raise ValueError("IRIS_V31_XYZ_SHAPE")
        sampled = F.grid_sample(carrier, xyz[:, None, None], mode="bilinear",
                                padding_mode="zeros", align_corners=False).squeeze(2).squeeze(2).transpose(1, 2)
        bands = xyz.new_tensor([math.pi*2**i for i in range(4)])
        angle = xyz[..., None]*bands
        fourier = torch.cat((xyz, angle.sin().flatten(-2), angle.cos().flatten(-2)), -1)
        state = self.query(torch.cat((sampled, fourier), -1))
        return self.gm(state), self.gp(state)

    def continuous(self, carrier, xyz, cameras):
        gm, gp = self.query_states(carrier, xyz)
        geometry = self.geometry(gm)
        appearance = self.appearance(torch.cat((gp, cameras), -1))
        return {"signed_field": geometry[..., 0],
                "bounded_delta": geometry[..., 1:].tanh()*self.config.delta_bound,
                "mechanics": self.mechanics(gm),
                "rgba": appearance[..., :4].sigmoid(), "boundary_logit": appearance[..., 4],
                "gp_state": gp}

    def pair(self, gp_a, gp_b, xyz_a, xyz_b, cameras):
        # Address coordinates are detached as well: extraction cannot sneak
        # relation gradients into the signed-field/geometry path.
        a, b, xa, xb = (x.detach() for x in (gp_a, gp_b, xyz_a, xyz_b))
        d = xb-xa
        return self.relation(torch.cat((a, b, a-b, a*b, d, d.abs(), d.norm(dim=-1, keepdim=True), cameras), -1))
