from __future__ import annotations

from pathlib import Path
import torch

from .skin_field_codec_v1 import SkinFieldCodecV1

SCHEMA = "RealSaS.SkinFieldCodecCheckpoint.v1"


def save_skin_field_codec_checkpoint_v1(path: str | Path, codec: SkinFieldCodecV1, *, metadata: dict | None = None) -> None:
    payload = {
        "authority": {
            "schema": SCHEMA,
            "architecture_id": codec.config.architecture_id,
            "config_hash": codec.config.config_hash,
            "surface_feature_dim": codec.config.surface_feature_dim,
            "joint_feature_dim": codec.config.joint_feature_dim,
            "latent_dim": codec.config.latent_dim,
            "shared_decoder_authority": True,
            "full_3d_reconstruction_authority": False,
            "metadata": dict(metadata or {}),
        },
        "state_dict": {k: v.detach().cpu() for k, v in codec.state_dict().items()},
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp); tmp.replace(path)


def load_skin_field_codec_checkpoint_v1(path: str | Path, codec: SkinFieldCodecV1, *, map_location="cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    authority = dict(payload.get("authority", {})) if isinstance(payload, dict) else {}
    expected = {
        "schema": SCHEMA,
        "architecture_id": codec.config.architecture_id,
        "config_hash": codec.config.config_hash,
        "surface_feature_dim": codec.config.surface_feature_dim,
        "joint_feature_dim": codec.config.joint_feature_dim,
        "latent_dim": codec.config.latent_dim,
        "shared_decoder_authority": True,
        "full_3d_reconstruction_authority": False,
    }
    for key, value in expected.items():
        if authority.get(key) != value:
            raise ValueError(f"SkinFieldCodec checkpoint authority mismatch:{key}")
    codec.load_state_dict(payload["state_dict"], strict=True)
    return authority
