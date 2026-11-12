from __future__ import annotations

"""Audited StdGEN reference backbone ancestry; no StdGEN/Era3D weight reuse.

Sources (verified 2026-11-12):
  StdGEN CVPR 2025 §3.1/3.2:
      SD + ReferenceNet -> Era3D joint 6-view RGB/normal -> S-LRM
      S-LRM starts from InstantMesh's InstantNeRF + attention LoRA.
  CharacterGen 2D_Stage/configs/infer.yaml:
      pretrained_model_path: stabilityai/stable-diffusion-2-1
  Era3D configs/train-unclip-512-6view.yaml:
      pretrained_model_name_or_path: stabilityai/stable-diffusion-2-1-unclip
      6 views, full UNet train, CLIP+VAE frozen, 512 training.
  StdGEN upgrades Era3D-style multiview to 1024 by progressive training.

The exact Stable Diffusion canonicalization base in the *StdGEN authors'
private training* is NOT published. SD2.1 canonicalization provenance comes
from CharacterGen's released reference config, so it is explicitly marked
inferred rather than certified as identical.
No third-party checkpoint is downloaded or accepted without a locally
attested immutable revision and a separate commercial-use rights review.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


CANONICAL = "CANONICALIZATION"
MULTIVIEW = "JOINT_MULTIVIEW_RGB_NORMAL"
RECONSTRUCTION = "SEMANTIC_LARGE_RECONSTRUCTION"
_SOURCE_NAME = "RealSaS.StdGENCleanroom.UpstreamBackbone.20261112.v1"
_REVISION = re.compile(r"^[a-f0-9]{40}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class StdGENReferenceAncestryV1:
    stage: str
    upstream_repo: str
    ancestral_base_repo: str
    adaptation: str
    original_training_provenance: str
    evidence: str
    default_tuning_policy: str
    released_task_checkpoint_not_used: bool = True
    released_era3d_weights_not_used: bool = True


_REFERENCE = {
    CANONICAL: StdGENReferenceAncestryV1(
        stage=CANONICAL, upstream_repo="zjp-shadow/CharacterGen",
        ancestral_base_repo="stabilityai/stable-diffusion-2-1",
        adaptation="Stable Diffusion + independently trained ReferenceNet; "
                   "single arbitrary reference -> canonical reference",
        original_training_provenance="CHARACTERGEN_REFERENCE_CONFIG; STDGEN_BASE_INFERRED_NOT_PROVEN",
        evidence="https://github.com/zjp-shadow/CharacterGen/blob/main/2D_Stage/configs/infer.yaml",
        default_tuning_policy="TASK_TRAINED_REFERENCE_UNET_AND_DENOISER",
    ),
    MULTIVIEW: StdGENReferenceAncestryV1(
        stage=MULTIVIEW, upstream_repo="pengHTYX/Era3D",
        ancestral_base_repo="stabilityai/stable-diffusion-2-1-unclip",
        adaptation="Era3D-derived full UNet training, camera-conditional "
                   "row-wise UNet feature attention; six aligned views and "
                   "joint RGB/normal with 512->1024 progressive training",
        original_training_provenance="ERA3D_TRAINING_CONFIG_DIRECT",
        evidence="https://github.com/pengHTYX/Era3D/blob/main/configs/train-unclip-512-6view.yaml",
        default_tuning_policy="FULL_UNET_FINETUNE_NOT_LORA_ONLY",
    ),
    RECONSTRUCTION: StdGENReferenceAncestryV1(
        stage=RECONSTRUCTION, upstream_repo="TencentARC/InstantMesh",
        ancestral_base_repo="TencentARC/InstantMesh:InstantNeRF",
        adaptation="S-LRM initialized from InstantNeRF; ViT, triplane, "
                   "LoRA in attention linear layers, new semantic decoder, "
                   "NeRF single-layer -> NeRF layered -> FlexiCubes mesh",
        original_training_provenance="STDGEN_PAPER_SECTION_3_2_DIRECT",
        evidence="https://openaccess.thecvf.com/content/CVPR2025/papers/"
                 "He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.pdf",
        default_tuning_policy="ATTENTION_LORA_AND_NEW_SEMANTIC_DECODER",
    ),
}


def reference_ancestry_v1(stage: str) -> StdGENReferenceAncestryV1:
    if stage not in _REFERENCE:
        raise ValueError("RSS_VNEXT_STDGEN_UNKNOWN_REFERENCE_STAGE")
    return _REFERENCE[stage]


@dataclass(frozen=True)
class StdGENBackbonePinV1:
    """Explicit, immutable local snapshot of a third-party ancestral base.

    A snapshot's SHA is not proof of its origin, commercial rights, or that
    authors trained on identical weights. Separate repository records are
    needed to establish origin and commercial-use permission.
    """
    stage: str
    upstream_repository_id: str
    upstream_revision_sha: str
    snapshot_path: str
    checkpoint_inventory_sha256: str
    license_id: str
    commercial_use_review_sha256: str
    mirror_of_original: bool = False
    mirror_equivalence_proven: bool = False


def _expected_components(stage: str) -> tuple[str, ...]:
    if stage == CANONICAL:
        return ("unet", "vae", "scheduler", "text_encoder", "tokenizer")
    if stage == MULTIVIEW:
        return (
            "unet", "vae", "scheduler", "text_encoder", "tokenizer",
            "image_encoder", "image_normalizer", "image_noising_scheduler",
        )
    raise ValueError("RSS_VNEXT_STDGEN_SLRM_REQUIRES_SEPARATE_INSTANTNERF_PIN")


def _stream_sha256(filename: Path) -> str:
    digest = sha256()
    with filename.open("rb") as file:
        while chunk := file.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def offline_snapshot_inventory_sha256_v1(
    snapshot_path: str, stage: str,
) -> str:
    """Hash the actual local assets; no download, symlinks, pickle weights.

    The canonical model_index.json and every required weights/config folder
    must exist. This hash changes when any config or weight file is replaced.
    """
    required = _expected_components(stage)
    root = Path(snapshot_path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("RSS_VNEXT_STDGEN_BASE_SNAPSHOT_DIR_MISSING")
    model_index = root / "model_index.json"
    if not model_index.is_file() or model_index.is_symlink():
        raise ValueError("RSS_VNEXT_STDGEN_BASE_MODEL_INDEX_MISSING")
    try:
        settings = json.loads(model_index.read_text("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("RSS_VNEXT_STDGEN_BASE_MODEL_INDEX_INVALID") from exc
    if type(settings) is not dict or not settings.get("_class_name"):
        raise ValueError("RSS_VNEXT_STDGEN_BASE_MODEL_INDEX_INVALID")
    for name in required:
        folder = root / name
        if not folder.is_dir() or folder.is_symlink() or not any(folder.iterdir()):
            raise ValueError("RSS_VNEXT_STDGEN_BASE_COMPONENT_MISSING:" + name)
    entries = []
    weight_count = 0
    for file in sorted(root.rglob("*")):
        if file.is_symlink():
            raise ValueError("RSS_VNEXT_STDGEN_BASE_SYMLINK_FORBIDDEN")
        if not file.is_file():
            continue
        relative = file.relative_to(root).as_posix()
        # Never unpickle an untrusted third-party checkpoint.
        if file.suffix in (".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle"):
            raise ValueError("RSS_VNEXT_STDGEN_BASE_PICKLE_CHECKPOINT_FORBIDDEN")
        if file.suffix == ".safetensors":
            weight_count += 1
        if file.stat().st_size > 16 * 1024**3:
            raise ValueError("RSS_VNEXT_STDGEN_BASE_FILE_BUDGET")
        entries.append([relative, file.stat().st_size, _stream_sha256(file)])
    if weight_count < 1:
        raise ValueError("RSS_VNEXT_STDGEN_BASE_SAFETENSORS_REQUIRED")
    payload = json.dumps(
        {"schema": _SOURCE_NAME, "stage": stage, "files": entries},
        separators=(",",":"), sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def verify_stdgen_backbone_pin_v1(pin: StdGENBackbonePinV1) -> None:
    if type(pin) is not StdGENBackbonePinV1 or pin.stage not in (CANONICAL,MULTIVIEW):
        raise ValueError("RSS_VNEXT_STDGEN_PIN_TYPE_OR_STAGE")
    lineage = reference_ancestry_v1(pin.stage)
    if (
        pin.upstream_repository_id != lineage.ancestral_base_repo
        or not _REVISION.fullmatch(pin.upstream_revision_sha)
        or not _DIGEST.fullmatch(pin.checkpoint_inventory_sha256)
        or not _DIGEST.fullmatch(pin.commercial_use_review_sha256)
        or not pin.license_id or pin.license_id.upper() in ("UNKNOWN","NONE","RESEARCH_ONLY")
    ):
        raise ValueError("RSS_VNEXT_STDGEN_PRETRAINED_OR_RIGHTS_PIN_INVALID")
    if pin.mirror_of_original and not pin.mirror_equivalence_proven:
        raise ValueError("RSS_VNEXT_STDGEN_UNVERIFIED_MIRROR_FORBIDDEN")
    if pin.mirror_equivalence_proven and not pin.mirror_of_original:
        raise ValueError("RSS_VNEXT_STDGEN_INVALID_MIRROR_ATTESTATION")
    actual = offline_snapshot_inventory_sha256_v1(pin.snapshot_path,pin.stage)
    if actual != pin.checkpoint_inventory_sha256:
        raise ValueError("RSS_VNEXT_STDGEN_SNAPSHOT_CONTENT_DRIFT")


def load_stdgen_ancestral_base_v1(
    pin: StdGENBackbonePinV1, *, torch_dtype: Any,
) -> Any:
    """Load the *upstream base only*, no StdGEN task weights or Era3D code.

    Canonical needs a separately trained cleanroom ReferenceNet and a pose
    target; multiview needs trained camera/rowwise UNet feature layers.
    Loading an upstream base never constitutes trained task inference.
    """
    verify_stdgen_backbone_pin_v1(pin)
    try:
        from diffusers import StableDiffusionPipeline, StableUnCLIPImg2ImgPipeline
    except ImportError as exc:
        raise RuntimeError("RSS_VNEXT_DIFFUSERS_NOT_INSTALLED") from exc
    pipeline_type = (
        StableDiffusionPipeline if pin.stage == CANONICAL
        else StableUnCLIPImg2ImgPipeline
    )
    return pipeline_type.from_pretrained(
        pin.snapshot_path, torch_dtype=torch_dtype,
        local_files_only=True, use_safetensors=True,
    )


@dataclass(frozen=True)
class StdGENCompatibleTrainingRecipeV1:
    """Published reference, not an invented parity or final GPU schedule."""
    canonical_reference_inputs: int = 1
    multiview_views: int = 6
    generate_rgb_and_normals_jointly: bool = True
    multiview_base_resolution: int = 512
    stdgen_progressive_target_resolution: int = 1024
    denoiser_policy: str = "FULL_UNET_FINETUNE"
    unclip_vae_frozen: bool = True
    unclip_image_encoder_frozen: bool = True
    unclip_text_encoder_frozen: bool = True
    row_attention_location: str = "UNET_FEATURE_TRANSFORMER"
    source_pose_canonicalization: str = "SOURCE_IMAGE_TO_NEUTRAL_POSE"
    reconstruction_lora_location: str = "SLRM_ATTENTION_LINEAR_LAYERS"
    exact_stdgen_diffusion_training_recipe_public: bool = False


def stdgen_reference_recipe_v1() -> StdGENCompatibleTrainingRecipeV1:
    return StdGENCompatibleTrainingRecipeV1()


def audit_stdgen_diffusion_training_architecture_v1(
    *,
    denoiser: Any,
    vae: Any,
    image_encoder: Any,
    text_encoder: Any,
    num_views: int,
    row_attention_location: str,
    num_generated_modalities: int,
) -> None:
    """Abstain if attempting to claim Era3D/StdGEN with LoRA-only UNet.

    Does not claim sufficient structural equivalence; it enforces P0
    necessary conditions taken directly from the upstream method.
    """
    recipe=stdgen_reference_recipe_v1()
    if num_views != recipe.multiview_views or num_generated_modalities != 2:
        raise ValueError("RSS_VNEXT_STDGEN_SIX_VIEW_RGB_NORMAL_REQUIRED")
    if row_attention_location != recipe.row_attention_location:
        raise ValueError("RSS_VNEXT_STDGEN_ROW_ATTENTION_MUST_BE_IN_UNET_FEATURES")
    if not any(p.requires_grad for p in denoiser.parameters()):
        raise ValueError("RSS_VNEXT_STDGEN_FINETUNABLE_UNET_REQUIRED")
    if any(not p.requires_grad for name,p in denoiser.named_parameters() if
           "lora_" not in name and "rowwise_" not in name and "camera_" not in name):
        raise ValueError("RSS_VNEXT_STDGEN_LORA_ONLY_DENOISER_NOT_BASELINE")
    if any(p.requires_grad for mod in (vae,image_encoder,text_encoder)
           for p in mod.parameters()):
        raise ValueError("RSS_VNEXT_STDGEN_PRETRAINED_ENCODERS_MUST_BE_FROZEN")


__all__=[
    "CANONICAL","MULTIVIEW","RECONSTRUCTION",
    "StdGENReferenceAncestryV1","reference_ancestry_v1",
    "StdGENBackbonePinV1","offline_snapshot_inventory_sha256_v1",
    "verify_stdgen_backbone_pin_v1","load_stdgen_ancestral_base_v1",
    "StdGENCompatibleTrainingRecipeV1","stdgen_reference_recipe_v1",
    "audit_stdgen_diffusion_training_architecture_v1",
]
