from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FoundationFeatureContractV2:
    backbone_id: str
    backbone_source_hash: str
    level_ids: tuple[str, ...]
    level_dims: tuple[int, ...]
    preprocessing_id: str
    schema_version: str = "RealSaS.FoundationFeatureContract.v2"

    def __post_init__(self) -> None:
        if not self.backbone_id or not self.backbone_source_hash or not self.preprocessing_id:
            raise ValueError("foundation contract identity required")
        if not self.level_ids or len(self.level_ids) != len(self.level_dims):
            raise ValueError("foundation level contract mismatch")
        if len(set(self.level_ids)) != len(self.level_ids) or min(self.level_dims) <= 0:
            raise ValueError("invalid foundation levels")

    @property
    def contract_hash(self) -> str:
        return _canonical_hash(self.__dict__)


class FrozenFoundationAdapterV2(nn.Module):
    def __init__(self, backbone: nn.Module, *, contract: FoundationFeatureContractV2, extract_levels: Callable[[nn.Module, torch.Tensor], Sequence[torch.Tensor]]):
        super().__init__()
        self.backbone = backbone
        self.contract = contract
        self.extract_levels = extract_levels
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, images_rgb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if images_rgb.ndim != 5 or images_rgb.shape[1] != 8 or images_rgb.shape[2] != 3:
            raise ValueError("foundation input must be [B,8,3,H,W]")
        if any(p.requires_grad for p in self.backbone.parameters()):
            raise ValueError("foundation backbone trainability drift")
        B, V, C, H, W = images_rgb.shape
        levels = tuple(self.extract_levels(self.backbone, images_rgb.reshape(B * V, C, H, W)))
        if len(levels) != len(self.contract.level_dims):
            raise ValueError("foundation level count drift")
        out = []
        for level, dim in zip(levels, self.contract.level_dims):
            if level.ndim != 4 or level.shape[0] != B * V or level.shape[1] != dim:
                raise ValueError("foundation feature shape drift")
            out.append(level.detach().reshape(B, V, dim, level.shape[-2], level.shape[-1]))
        return tuple(out)


def save_foundation_cache_v2(path, *, maps: Sequence[torch.Tensor], contract: FoundationFeatureContractV2, observation_contract_hash: str) -> dict:
    path = Path(path)
    tensors = tuple(torch.as_tensor(x).detach().cpu().contiguous() for x in maps)
    for i, (x, dim) in enumerate(zip(tensors, contract.level_dims)):
        if x.ndim != 5 or x.shape[1] != 8 or x.shape[2] != dim:
            raise ValueError(f"foundation cache level shape drift:{i}")
    payload = {"schema": "RealSaS.FoundationFeatureCache.v2", "foundation_contract": contract.__dict__, "foundation_contract_hash": contract.contract_hash, "observation_contract_hash": str(observation_contract_hash), "maps": tensors}
    torch.save(payload, path)
    digest = sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "foundation_contract_hash": contract.contract_hash, "observation_contract_hash": str(observation_contract_hash)}


def load_foundation_cache_v2(path, *, expected_contract: FoundationFeatureContractV2, expected_observation_contract_hash: str) -> tuple[torch.Tensor, ...]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != "RealSaS.FoundationFeatureCache.v2":
        raise ValueError("foundation cache schema mismatch")
    if payload.get("foundation_contract_hash") != expected_contract.contract_hash:
        raise ValueError("foundation cache contract mismatch")
    if payload.get("observation_contract_hash") != expected_observation_contract_hash:
        raise ValueError("foundation cache observation mismatch")
    maps = tuple(payload.get("maps", ()))
    if len(maps) != len(expected_contract.level_dims):
        raise ValueError("foundation cache level count mismatch")
    for x, dim in zip(maps, expected_contract.level_dims):
        if x.ndim != 5 or x.shape[1] != 8 or x.shape[2] != dim:
            raise ValueError("foundation cache feature shape mismatch")
    return maps
