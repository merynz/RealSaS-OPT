"""IRIS V3.1 exact F* teacher isolation from model mixed precision.

The V3.1 model is trained under BF16 autocast, but F* target construction is a
separate exact numerical authority.  It must not inherit the caller's autocast
state: at R1024, BF16 cannot represent the closed-edge clamp value 1023 and can
round it to 1024 before manual bilinear indexing.
"""
from __future__ import annotations

import numpy as np
import torch

from models.iris.v5.source_signed_field_v5 import (
    SourceSignedFieldPolicyV5,
    source_signed_field_torch_v5,
)


def exact_fstar_targets_v31(subject: dict, points: np.ndarray, *, device: str) -> np.ndarray:
    kwargs = {
        key: subject[key]
        for key in (
            "signed_2d",
            "center_xyz",
            "normalization_half_extent",
            "camera_origins",
            "camera_right",
            "camera_screen_up",
            "camera_half_extent",
        )
    }
    kwargs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
    }
    policy = SourceSignedFieldPolicyV5(required_resolution=subject["resolution"])
    device_type = torch.device(device).type
    chunks = np.array_split(points, max(1, (len(points) + 65535) // 65536))
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
        out = [
            source_signed_field_torch_v5(
                torch.from_numpy(chunk).to(device),
                policy=policy,
                **kwargs,
            ).cpu().numpy()
            for chunk in chunks
        ]
    return np.concatenate(out).astype(np.float32)
