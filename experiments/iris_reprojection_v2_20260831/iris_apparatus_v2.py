from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn

from .dinov2_foundation_v2 import (
    DINOv2FoundationAuthorityV2,
    DINOv2LoadedAuthorityV2,
    extract_dinov2_s_patch_map_v2,
    load_exact_dinov2_s_v2,
)
from .model_v2 import IrisReprojectionV2, IrisReprojectionOutputV2
from .q_domain_v2 import RayHypothesisDomainV2


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IrisFoundationRuntimeSealV2:
    foundation_contract_hash: str
    source_revision: str
    weight_sha256: str
    constructor: str
    output_level: str
    output_dim: int
    patch_grid_hw: tuple[int, int]
    preprocessing_id: str
    schema: str = "RealSaS.IRIS.FoundationRuntimeSeal.v2"

    @property
    def seal_hash(self) -> str:
        return _canonical_hash(self.__dict__)


class IrisDINOv2SApparatusV2(nn.Module):
    """Production IRIS V2 apparatus with exact frozen DINOv2-S authority inside it.

    The trainable learner never accepts arbitrary externally supplied foundation maps
    on this production path. Exact source/weight/preprocess validation happens before
    construction through ``from_authority_files``. The foundation remains eval-only
    and gradient-free while the native 1024 RGBA pyramid remains trainable.

    ``IrisReprojectionV2`` remains separately testable with injected maps for synthetic
    source tests; this apparatus is the scientific train/inference authority.
    """

    def __init__(
        self,
        learner: IrisReprojectionV2,
        frozen_dino: nn.Module,
        *,
        authority: DINOv2FoundationAuthorityV2 = DINOv2FoundationAuthorityV2(),
        loaded_authority: DINOv2LoadedAuthorityV2 | None = None,
    ):
        super().__init__()
        authority.validate()
        if learner.sampler.foundation_dims != (authority.embed_dim,):
            raise ValueError("IRIS learner foundation dimensions do not match exact DINO-S authority")
        if any(p.requires_grad for p in frozen_dino.parameters()):
            raise ValueError("DINO foundation must be frozen before apparatus construction")
        self.learner = learner
        self.foundation = frozen_dino
        self.foundation.eval()
        self.authority = authority
        contract = authority.feature_contract()
        if loaded_authority is not None and loaded_authority.feature_contract_hash != contract.contract_hash:
            raise ValueError("loaded DINO authority/feature contract mismatch")
        self.runtime_seal = IrisFoundationRuntimeSealV2(
            foundation_contract_hash=contract.contract_hash,
            source_revision=authority.source_revision,
            weight_sha256=authority.weight_sha256,
            constructor=authority.constructor,
            output_level=authority.output_level,
            output_dim=authority.embed_dim,
            patch_grid_hw=authority.patch_grid_hw,
            preprocessing_id=authority.preprocessing_id,
        )
        self.loaded_authority = loaded_authority

    @classmethod
    def from_authority_files(
        cls,
        source_dir: str | Path,
        weight_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        hidden_dim: int = 192,
        max_modes: int = 3,
        authority: DINOv2FoundationAuthorityV2 = DINOv2FoundationAuthorityV2(),
    ) -> "IrisDINOv2SApparatusV2":
        foundation, loaded = load_exact_dinov2_s_v2(source_dir, weight_path, device=device, authority=authority)
        learner = IrisReprojectionV2((authority.embed_dim,), hidden_dim=hidden_dim, max_modes=max_modes).to(device)
        return cls(learner, foundation, authority=authority, loaded_authority=loaded)

    def train(self, mode: bool = True):
        super().train(mode)
        self.foundation.eval()
        return self

    @staticmethod
    def _validate_and_scale_rgba(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or tuple(images.shape[1:3]) != (8, 4) or tuple(images.shape[-2:]) != (1024, 1024):
            raise ValueError("IRIS production images must be [B,8,4,1024,1024]")
        if images.dtype == torch.uint8:
            return images.to(torch.float32).div(255.0)
        if not images.is_floating_point():
            raise ValueError("IRIS images must be uint8 or floating point")
        x = images.to(torch.float32)
        if not torch.isfinite(x).all() or float(x.min()) < 0.0 or float(x.max()) > 1.0:
            raise ValueError("floating IRIS RGBA must be finite and scaled to [0,1]")
        return x

    @torch.no_grad()
    def extract_foundation_maps(self, native_rgba: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # DINO consumes exact native RGB only; alpha remains separate observation/support
        # evidence and continues through the learned native RGBA pyramid.
        if native_rgba.ndim != 5 or native_rgba.shape[2] != 4:
            raise ValueError("native_rgba shape drift")
        rgb = native_rgba[:, :, :3]
        patch_map = extract_dinov2_s_patch_map_v2(self.foundation, rgb)
        if patch_map.shape[2:] != (self.authority.embed_dim, *self.authority.patch_grid_hw):
            raise RuntimeError("exact DINO patch-map runtime contract drift")
        return (patch_map.detach(),)

    def forward(self, images: torch.Tensor, domain: RayHypothesisDomainV2) -> IrisReprojectionOutputV2:
        rgba = self._validate_and_scale_rgba(images)
        maps = self.extract_foundation_maps(rgba)
        return self.learner(rgba, maps, domain)

    def trainable_parameters(self):
        return self.learner.parameters()

    def trainable_state_dict(self) -> dict:
        return self.learner.state_dict()

    @property
    def source_contract_hash(self) -> str:
        payload = {
            "runtime_seal_hash": self.runtime_seal.seal_hash,
            "learner_foundation_dims": self.learner.sampler.foundation_dims,
            "native_widths": self.learner.native.widths,
            "max_modes": self.learner.max_modes,
        }
        return _canonical_hash(payload)
