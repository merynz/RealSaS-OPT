from __future__ import annotations

"""Strong pretrained SDXL + MV-Adapter bootstrap, never a fabricated Stage A/B release.

This is an *offline, pinned* bridge to real upstream task weights.  It does not
pretend that a single-reference six-RGB MV-Adapter checkpoint is a trained
two-source RGB+normal BUSTED Stage B, or that a base SDXL is a trained Stage A.
The existing four-channel/latent-row product ABI is deliberately unchanged.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import importlib
import json

BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
BASE_LICENSE = "creativeml-openrail++-m"
ADAPTER_REPO = "huanngzh/mv-adapter"
ADAPTER_CODE_REPO = "huanngzh/MV-Adapter"
ADAPTER_LICENSE = "Apache-2.0"
ADAPTER_WEIGHT = "mvadapter_i2mv_sdxl.safetensors"
SCHEMA = "RealSaS.RSSvNext.StrongPretrainedStageABootstrap.v1"
AUTHORITY = "PRETRAINED_REFERENCE_NOT_PRODUCT_STAGE_AB_RELEASE"
BASE_UNET_WEIGHT = "unet/diffusion_pytorch_model.safetensors"


def _hex(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and all(
        ch in "0123456789abcdef" for ch in value
    )


def _sha256_file(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def pinned_source_tree_sha256_v1(
    root: str | Path, *, suffixes: tuple[str, ...],
) -> str:
    """Deterministic inventory with path, byte count and byte digest.

    This is an integrity record only, NOT third-party author/rights attestation.
    A local checkout or snapshot must be copied with REAL files (not symlinks).
    """
    root = Path(root).expanduser()
    if not root.is_dir() or root.is_symlink() or not suffixes:
        raise ValueError("RSS_VNEXT_PRETRAINED_PIN_ROOT_INVALID")
    items: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if ".git" in parts or ".cache" in parts or "__pycache__" in parts:
            continue
        if path.is_symlink():
            raise ValueError("RSS_VNEXT_PRETRAINED_PIN_SYMLINK_FORBIDDEN")
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if len(items) >= 10000 or path.stat().st_size > 16 * 1024**3:
            raise ValueError("RSS_VNEXT_PRETRAINED_PIN_FILE_BUDGET")
        items.append((
            path.relative_to(root).as_posix(),
            path.stat().st_size, _sha256_file(path),
        ))
    if not items:
        raise ValueError("RSS_VNEXT_PRETRAINED_PIN_FILES_MISSING")
    return sha256(json.dumps(
        items, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class StrongSDXLMVAdapterPinV1:
    """Immutable snapshots: exact upstream task weights, never random UNets.

    A reviewed license memo is a recorded HUMAN decision; its hash does not
    automatically prove rights. Product launch has another independent gate.
    """
    base_snapshot_dir: str
    base_revision: str
    base_tree_sha256: str
    base_unet_sha256: str
    adapter_snapshot_dir: str
    adapter_revision: str
    adapter_weight_sha256: str
    upstream_code_dir: str
    upstream_code_revision: str
    upstream_code_tree_sha256: str
    commercial_review_record_sha256: str
    base_repo: str = BASE_REPO
    base_license: str = BASE_LICENSE
    adapter_repo: str = ADAPTER_REPO
    adapter_code_repo: str = ADAPTER_CODE_REPO
    adapter_license: str = ADAPTER_LICENSE
    adapter_code_license: str = ADAPTER_LICENSE
    source_observation_count: int = 2
    generated_view_count: int = 6
    generated_modalities: tuple[str, str] = ("RGB", "NORMAL")
    pretrained_adapter_is_two_source: bool = False
    pretrained_adapter_generates_normals: bool = False
    product_release_qualified: bool = False
    schema: str = SCHEMA

    def validate_metadata(self) -> None:
        if (
            type(self) is not StrongSDXLMVAdapterPinV1
            or self.schema != SCHEMA
            or self.base_repo != BASE_REPO
            or self.base_license != BASE_LICENSE
            or self.adapter_repo != ADAPTER_REPO
            or self.adapter_code_repo != ADAPTER_CODE_REPO
            or self.adapter_license != ADAPTER_LICENSE
            or self.adapter_code_license != ADAPTER_LICENSE
            or self.source_observation_count != 2
            or self.generated_view_count != 6
            or self.generated_modalities != ("RGB", "NORMAL")
            or self.pretrained_adapter_is_two_source is not False
            or self.pretrained_adapter_generates_normals is not False
            or self.product_release_qualified is not False
            or not all(_hex(v, 40) for v in (
                self.base_revision, self.adapter_revision,
                self.upstream_code_revision,
            ))
            or not all(_hex(v, 64) for v in (
                self.base_tree_sha256, self.base_unet_sha256,
                self.adapter_weight_sha256, self.upstream_code_tree_sha256,
                self.commercial_review_record_sha256,
            ))
        ):
            raise ValueError("RSS_VNEXT_SDXL_MVADAPTER_SOURCE_OR_LICENSE_PIN_INVALID")


def verify_strong_pretrained_snapshots_v1(
    pin: StrongSDXLMVAdapterPinV1,
) -> dict[str, str | bool]:
    """Fail closed on missing snapshots, altered bytes or incompatible SDXL.

    All assets are local. This function never downloads or unpickles weights.
    """
    pin.validate_metadata()
    base = Path(pin.base_snapshot_dir).expanduser()
    adapter = Path(pin.adapter_snapshot_dir).expanduser()
    code = Path(pin.upstream_code_dir).expanduser()
    for root in (base, adapter, code):
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError("RSS_VNEXT_PRETRAINED_LOCAL_SNAPSHOT_REQUIRED")
    weight = adapter / ADAPTER_WEIGHT
    unet = base / BASE_UNET_WEIGHT
    if (
        not weight.is_file() or weight.is_symlink()
        or not unet.is_file() or unet.is_symlink()
        or _sha256_file(weight) != pin.adapter_weight_sha256
        or _sha256_file(unet) != pin.base_unet_sha256
    ):
        raise RuntimeError("RSS_VNEXT_PRETRAINED_CHECKPOINT_BYTES_DRIFT")
    base_digest = pinned_source_tree_sha256_v1(
        base, suffixes=(".safetensors", ".json", ".txt", ".md", ".model"),
    )
    code_digest = pinned_source_tree_sha256_v1(
        code / "mvadapter", suffixes=(".py",),
    )
    if (
        base_digest != pin.base_tree_sha256
        or code_digest != pin.upstream_code_tree_sha256
    ):
        raise RuntimeError("RSS_VNEXT_PRETRAINED_SNAPSHOT_OR_UPSTREAM_CODE_DRIFT")
    try:
        model = json.loads((base / "model_index.json").read_text())
        cfg = json.loads((base / "unet" / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("RSS_VNEXT_PRETRAINED_SDXL_ARCHITECTURE_MISSING") from exc
    if (
        model.get("_class_name") != "StableDiffusionXLPipeline"
        or cfg.get("in_channels") != 4 or cfg.get("out_channels") != 4
        or cfg.get("cross_attention_dim") != 2048
        or cfg.get("addition_embed_type") != "text_time"
        or not (base / "vae" / "config.json").is_file()
        or not (base / "text_encoder" / "config.json").is_file()
        or not (base / "text_encoder_2" / "config.json").is_file()
    ):
        raise RuntimeError("RSS_VNEXT_PRETRAINED_NOT_REAL_SDXL_BASE")
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "base_repo": BASE_REPO,
        "adapter_repo": ADAPTER_REPO,
        "base_revision": pin.base_revision,
        "adapter_revision": pin.adapter_revision,
        "upstream_code_revision": pin.upstream_code_revision,
        "actual_base_tree_sha256": base_digest,
        "actual_adapter_sha256": _sha256_file(weight),
        "actual_upstream_code_tree_sha256": code_digest,
        "pretrained_stage_b_single_reference_rgb_only": True,
        "trained_two_source_canonicalization_proven": False,
        "trained_two_source_rgb_normal_multiview_proven": False,
        "product_release_qualified": False,
    }


def _verified_upstream_module(
    name: str, *, pin: StrongSDXLMVAdapterPinV1,
) -> Any:
    module = importlib.import_module(name)
    actual = Path(getattr(module, "__file__", "")).resolve()
    expected = (Path(pin.upstream_code_dir).expanduser().resolve()
                / "mvadapter")
    if not _within(actual, expected):
        raise RuntimeError("RSS_VNEXT_PRETRAINED_UNPINNED_UPSTREAM_IMPORT")
    return module


def load_pinned_sdxl_stage_a_unet_v1(
    pin: StrongSDXLMVAdapterPinV1, *, lora_rank: int = 16,
):
    """Load true pretrained SDXL Stage A prior + fresh trainable LoRA.

    Stage A still needs the already implemented paired-image conditioner,
    source-rendered canonical pose targets and trained LoRA weights.
    This loader CANNOT mark an untrained canonicalizer ready.
    """
    verify_strong_pretrained_snapshots_v1(pin)
    from .paired_sdxl_lora_prior_v1 import (
        SDXLBackboneBindingV1, attach_pinned_sdxl_lora_unet_v1,
    )
    binding = SDXLBackboneBindingV1(
        unet_checkpoint_sha256=pin.base_unet_sha256,
        source_repository_revision=pin.base_revision,
        lora_adapter_sha256=None,
        lora_rank=lora_rank,
        has_commercial_license_clearance=False,
    )
    return attach_pinned_sdxl_lora_unet_v1(
        local_model_directory=pin.base_snapshot_dir, binding=binding,
    )


def load_official_pretrained_i2mv_stage_b_v1(
    pin: StrongSDXLMVAdapterPinV1, *, device: str, dtype: Any,
):
    """Load the REAL trained SDXL + MV-Adapter I2MV checkpoint, offline.

    Preserve upstream in-UNet attention and image-reference conditioning.
    Do not remap this checkpoint into BUSTED's *incompatible* latent-row
    four-channel Stage B release schema or synthesize normal-map weights.
    """
    verify_strong_pretrained_snapshots_v1(pin)
    module = _verified_upstream_module(
        "mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl", pin=pin,
    )
    pipe = module.MVAdapterI2MVSDXLPipeline.from_pretrained(
        str(Path(pin.base_snapshot_dir).expanduser().resolve()),
        local_files_only=True, use_safetensors=True, torch_dtype=dtype,
    )
    if (
        pipe.unet.config.in_channels != 4
        or pipe.unet.config.out_channels != 4
        or pipe.unet.config.cross_attention_dim != 2048
    ):
        raise RuntimeError("RSS_VNEXT_PRETRAINED_LOADED_UNET_ARCH_DRIFT")
    pipe.init_custom_adapter(num_views=6)
    pipe.load_custom_adapter(
        str(Path(pin.adapter_snapshot_dir).expanduser().resolve()),
        weight_name=ADAPTER_WEIGHT,
    )
    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)
    pipe.enable_vae_slicing()
    pipe.unet.eval()
    return pipe



def run_official_pretrained_six_rgb_v1(
    pin: StrongSDXLMVAdapterPinV1, *,
    reference_canonical_png: str | Path,
    prompt: str,
    device: str = "cuda",
    dtype: Any = None,
    seed: int = 0,
    inference_steps: int = 50,
    image_size: int = 768,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Full-strength six-view reference inference using actual upstream weights.

    This is NOT BUSTED Stage B E2E: the upstream checkpoint consumes ONE
    canonical image and outputs six RGB views, not two canonical images plus
    trained RGB/normal heads.  It is a diagnostic of real pretrained capability.
    """
    if (
        type(prompt) is not str or not prompt.strip()
        or type(device) is not str or not device
        or type(seed) is not int or not 0 <= seed < 2**63
        or type(inference_steps) is not int or not 1 <= inference_steps <= 100
        or image_size not in (768, 1024)
    ):
        raise ValueError("RSS_VNEXT_PRETRAINED_I2MV_EXECUTION_POLICY")
    import torch
    from PIL import Image

    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    source = Path(reference_canonical_png).expanduser()
    if not source.is_file() or source.suffix.lower() != ".png":
        raise ValueError("RSS_VNEXT_PRETRAINED_CANONICAL_REFERENCE_REQUIRED")
    with Image.open(source) as file:
        original = file.convert("RGBA")
        white = Image.new("RGBA", original.size, (255, 255, 255, 255))
        white.alpha_composite(original)
        reference = white.convert("RGB").resize(
            (image_size, image_size), resample=Image.Resampling.LANCZOS,
        )
    pipe = load_official_pretrained_i2mv_stage_b_v1(
        pin, device=device, dtype=dtype,
    )
    camera_lib = _verified_upstream_module(
        "mvadapter.utils.mesh_utils", pin=pin,
    )
    geometry_lib = _verified_upstream_module(
        "mvadapter.utils.geometry", pin=pin,
    )
    angles = (0, 45, 90, 180, 270, 315)
    cameras = camera_lib.get_orthogonal_camera(
        elevation_deg=[0] * 6,
        distance=[1.8] * 6,
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[x - 90 for x in angles],
        device=device,
    )
    plucker = geometry_lib.get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * 6, image_size,
    )
    controls = ((plucker + 1.0) / 2.0).clamp(0, 1)
    result = pipe(
        prompt,
        height=image_size, width=image_size,
        num_inference_steps=inference_steps,
        guidance_scale=3.0,
        num_images_per_prompt=6,
        control_image=controls,
        control_conditioning_scale=1.0,
        reference_image=reference,
        reference_conditioning_scale=1.0,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    images = tuple(result.images)
    if len(images) != 6 or any(
        not isinstance(im, Image.Image)
        or im.size != (image_size, image_size)
        for im in images
    ):
        raise RuntimeError("RSS_VNEXT_PRETRAINED_I2MV_VIEW_COUNT_OR_SIZE_DRIFT")
    return images, {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "model_weights_sha256": pin.adapter_weight_sha256,
        "reference_is_single_canonical_image": True,
        "six_rgb_images": True,
        "six_trained_normal_images": False,
        "two_original_inputs_conditioned": False,
        "source_or_product_truth": False,
        "product_release_qualified": False,
        "azimuth_deg": list(angles),
        "seed": seed,
        "image_size": image_size,
    }


def assert_pretrained_is_not_final_stage_ab_v1(
    *, canonical_trained_for_two_sources: bool,
    multiview_trained_for_two_sources: bool,
    multiview_trained_for_rgb_and_normals: bool,
) -> None:
    """Independent release blocker: the three new task skills require training.

    Intentionally fail closed until *real* task-weights, provenance and
    held-out tests can upgrade a future version of this contract.
    """
    if any(type(v) is not bool for v in (
        canonical_trained_for_two_sources,
        multiview_trained_for_two_sources,
        multiview_trained_for_rgb_and_normals,
    )):
        raise TypeError("RSS_VNEXT_PRETRAINED_STAGE_AB_QUALIFICATION_FLAGS_INVALID")
    raise RuntimeError(
        "RSS_VNEXT_PRETRAINED_ONLY_NOT_TRAINED_PAIRED_RGB_NORMAL_STAGE_AB"
    )


__all__ = [
    "SCHEMA", "BASE_REPO", "BASE_LICENSE", "ADAPTER_REPO",
    "ADAPTER_LICENSE", "ADAPTER_WEIGHT", "AUTHORITY",
    "StrongSDXLMVAdapterPinV1", "pinned_source_tree_sha256_v1",
    "verify_strong_pretrained_snapshots_v1", "load_pinned_sdxl_stage_a_unet_v1",
    "load_official_pretrained_i2mv_stage_b_v1",
    "run_official_pretrained_six_rgb_v1",
    "assert_pretrained_is_not_final_stage_ab_v1",
]
