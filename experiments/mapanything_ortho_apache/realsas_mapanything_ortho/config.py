from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass(frozen=True)
class OrthoConfig:
    upstream_model_id: str = "facebook/map-anything-apache"
    upstream_model_revision: str = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
    upstream_code_commit: str = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"

    num_views: int = 8
    native_size: int = 1024
    global_size: int = 518
    output_size: int = 512
    target_xy_reference_size: int = 256
    depth_limit: float = 1.0

    detail_base_dim: int = 32
    detail_mid_dim: int = 64
    detail_deep_dim: int = 96
    detail_fused_dim: int = 128
    dropout: float = 0.0

    w_point: float = 1.0
    w_depth: float = 0.35
    w_normal: float = 0.30
    w_consistency: float = 0.25
    w_risk: float = 0.10
    huber_beta: float = 0.02

    lr_refiner: float = 2.0e-4
    lr_dense: float = 2.0e-5
    lr_info: float = 1.0e-5
    lr_encoder: float = 2.0e-6
    weight_decay: float = 1.0e-2
    grad_clip: float = 1.0

    epochs_adapter: int = 1
    epochs_geometry: int = 3
    epochs_full: int = 8
    grad_accum_steps: int = 4

    require_cuda: bool = True
    require_bf16: bool = True
    enable_gradient_checkpointing: bool = True
    seed: int = 20260827

    @property
    def total_epochs(self) -> int:
        return self.epochs_adapter + self.epochs_geometry + self.epochs_full

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
