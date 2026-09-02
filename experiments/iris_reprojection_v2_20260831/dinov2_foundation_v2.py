from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import torch
import torch.nn.functional as F

from .foundation_adapter_v2 import FoundationFeatureContractV2


DINO_REPOSITORY = "facebookresearch/dinov2"
DINO_SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_S_CONSTRUCTOR = "dinov2_vits14"
DINO_S_EMBED_DIM = 384
DINO_PATCH_SIZE = 14
DINO_INPUT_HW = (518, 518)
DINO_PATCH_GRID_HW = (37, 37)
DINO_S_WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINO_S_WEIGHT_SIZE_BYTES = 88283115
DINO_PREPROCESS_FIXTURE_SHA256 = "04cb22cc31510331f780e4c9aaba1141776c1fe3b0f8e2570a00badd951aa968"
DINO_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
DINO_NORMALIZE_STD = (0.229, 0.224, 0.225)


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class DINOv2FoundationAuthorityV2:
    repository: str = DINO_REPOSITORY
    source_revision: str = DINO_SOURCE_REVISION
    constructor: str = DINO_S_CONSTRUCTOR
    embed_dim: int = DINO_S_EMBED_DIM
    patch_size: int = DINO_PATCH_SIZE
    input_hw: tuple[int, int] = DINO_INPUT_HW
    patch_grid_hw: tuple[int, int] = DINO_PATCH_GRID_HW
    weight_sha256: str = DINO_S_WEIGHT_SHA256
    weight_size_bytes: int = DINO_S_WEIGHT_SIZE_BYTES
    preprocess_fixture_sha256: str = DINO_PREPROCESS_FIXTURE_SHA256
    schema: str = "RealSaS.IRIS.DINOv2FoundationAuthority.v2"

    def validate(self) -> None:
        if self.repository != DINO_REPOSITORY or self.source_revision != DINO_SOURCE_REVISION:
            raise ValueError("DINO source authority drift")
        if self.constructor != DINO_S_CONSTRUCTOR or self.embed_dim != 384 or self.patch_size != 14:
            raise ValueError("DINO-S architecture authority drift")
        if tuple(self.input_hw) != (518, 518) or tuple(self.patch_grid_hw) != (37, 37):
            raise ValueError("DINO spatial contract drift")
        if self.weight_sha256 != DINO_S_WEIGHT_SHA256 or int(self.weight_size_bytes) != DINO_S_WEIGHT_SIZE_BYTES:
            raise ValueError("DINO-S weight authority drift")

    @property
    def authority_hash(self) -> str:
        self.validate()
        return _canonical_hash(asdict(self))

    @property
    def preprocessing_id(self) -> str:
        return "RGB_UINT8_OR_EXACT_FP01__WHOLE1024_TO518_TORCH_INTERPOLATE_BICUBIC_AA__IMAGENET_NORM__V2"

    def feature_contract(self) -> FoundationFeatureContractV2:
        return FoundationFeatureContractV2(
            backbone_id=f"dinov2_vits14@{self.source_revision}",
            backbone_source_hash=self.authority_hash,
            level_ids=("x_norm_patchtokens",),
            level_dims=(self.embed_dim,),
            preprocessing_id=self.preprocessing_id,
        )


@dataclass(frozen=True)
class DINOv2LoadedAuthorityV2:
    authority: DINOv2FoundationAuthorityV2
    source_dir: str
    weight_path: str
    source_head: str
    weight_sha256: str
    feature_contract_hash: str


def preprocess_dinov2_rgb_v2(native_rgb: torch.Tensor) -> torch.Tensor:
    """Exact whole-canvas DINOv2 input transform used by sealed RealSaS authority.

    Input is [B,8,3,1024,1024], either uint8 or floating point exactly in [0,1].
    Alpha is deliberately absent: support/alpha remains separate observation authority.
    Resize uses torch interpolate bicubic+antialias, mathematically matching the sealed
    torchvision functional operator for tensor inputs; real token parity remains the
    final executable authority against a frozen reference tensor.
    """
    if native_rgb.ndim != 5 or native_rgb.shape[1:3] != (8, 3) or tuple(native_rgb.shape[-2:]) != (1024, 1024):
        raise ValueError("DINO RGB input must be [B,8,3,1024,1024]")
    if native_rgb.dtype == torch.uint8:
        x = native_rgb.to(torch.float32).div(255.0)
    elif native_rgb.is_floating_point():
        x = native_rgb.to(torch.float32)
        if not torch.isfinite(x).all() or float(x.min()) < 0.0 or float(x.max()) > 1.0:
            raise ValueError("floating DINO RGB must be finite and exactly scaled to [0,1]")
    else:
        raise ValueError("DINO RGB must be uint8 or floating point")
    B, V = x.shape[:2]
    flat = x.reshape(B * V, 3, 1024, 1024)
    flat = F.interpolate(flat, size=DINO_INPUT_HW, mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(DINO_NORMALIZE_MEAN, device=flat.device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(DINO_NORMALIZE_STD, device=flat.device, dtype=torch.float32).view(1, 3, 1, 1)
    return ((flat - mean) / std).contiguous()


def _git_head(source_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"DINO source must be an exact git checkout: {source_dir}") from exc


def verify_dinov2_authority_files_v2(source_dir: str | Path, weight_path: str | Path, authority: DINOv2FoundationAuthorityV2 = DINOv2FoundationAuthorityV2()) -> DINOv2LoadedAuthorityV2:
    authority.validate()
    source_dir = Path(source_dir).resolve()
    weight_path = Path(weight_path).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"DINO source directory missing: {source_dir}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"DINO-S weight file missing: {weight_path}")
    head = _git_head(source_dir)
    if head != authority.source_revision:
        raise RuntimeError(f"DINO source revision mismatch expected={authority.source_revision} actual={head}")
    size = int(weight_path.stat().st_size)
    if size != authority.weight_size_bytes:
        raise RuntimeError(f"DINO-S weight size mismatch expected={authority.weight_size_bytes} actual={size}")
    digest = _sha256_file(weight_path)
    if digest != authority.weight_sha256:
        raise RuntimeError(f"DINO-S weight SHA mismatch expected={authority.weight_sha256} actual={digest}")
    return DINOv2LoadedAuthorityV2(
        authority=authority,
        source_dir=str(source_dir),
        weight_path=str(weight_path),
        source_head=head,
        weight_sha256=digest,
        feature_contract_hash=authority.feature_contract().contract_hash,
    )


def _import_exact_dinov2(source_dir: Path):
    source_dir = source_dir.resolve()
    root = str(source_dir)
    if root not in sys.path:
        sys.path.insert(0, root)
    # Fail closed if a different dinov2 package was already imported from elsewhere.
    loaded = sys.modules.get("dinov2")
    if loaded is not None:
        location = Path(getattr(loaded, "__file__", "")).resolve()
        if source_dir not in location.parents:
            raise RuntimeError(f"foreign dinov2 module already loaded from {location}")
    return importlib.import_module("dinov2.hub.backbones")


def load_exact_dinov2_s_v2(source_dir: str | Path, weight_path: str | Path, *, device: torch.device | str = "cpu", authority: DINOv2FoundationAuthorityV2 = DINOv2FoundationAuthorityV2()):
    seal = verify_dinov2_authority_files_v2(source_dir, weight_path, authority)
    backbones = _import_exact_dinov2(Path(seal.source_dir))
    constructor = getattr(backbones, authority.constructor, None)
    if constructor is None:
        raise RuntimeError(f"exact DINO source lacks constructor {authority.constructor}")
    model = constructor(pretrained=False)
    if int(getattr(model, "embed_dim", -1)) != authority.embed_dim:
        raise RuntimeError("DINO embed_dim drift")
    patch = getattr(model.patch_embed, "patch_size", None)
    patch_tuple = tuple(int(x) for x in patch) if isinstance(patch, tuple) else (int(patch), int(patch))
    if patch_tuple != (authority.patch_size, authority.patch_size):
        raise RuntimeError("DINO patch-size drift")
    if int(getattr(model, "num_register_tokens", 0)) != 0:
        raise RuntimeError("unexpected DINO register tokens")
    state = torch.load(seal.weight_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
        raise RuntimeError("DINO strict-load compatibility drift")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, seal


@torch.no_grad()
def extract_dinov2_s_patch_map_v2(model: torch.nn.Module, native_rgb: torch.Tensor) -> torch.Tensor:
    """Return exact final normalized DINO patch map [B,8,384,37,37]."""
    x = preprocess_dinov2_rgb_v2(native_rgb)
    B = native_rgb.shape[0]
    features = model.forward_features(x)
    if not isinstance(features, Mapping) or "x_norm_patchtokens" not in features:
        raise RuntimeError("DINO forward_features contract drift: x_norm_patchtokens missing")
    tokens = features["x_norm_patchtokens"].float()
    if tuple(tokens.shape) != (B * 8, 37 * 37, DINO_S_EMBED_DIM):
        raise RuntimeError(f"DINO patch token shape drift: {tuple(tokens.shape)}")
    return tokens.reshape(B, 8, 37, 37, DINO_S_EMBED_DIM).permute(0, 1, 4, 2, 3).contiguous()


def tensor_bytes_sha256_v2(tensor: torch.Tensor) -> str:
    x = tensor.detach().cpu().contiguous()
    return sha256(x.numpy().tobytes()).hexdigest()
