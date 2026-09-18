from __future__ import annotations

"""Reference-input admission for the complete Mage Runtime-v4 render path.

This stage deliberately does not run the full product proof engine and does not render.
It rebuilds the current bounded Mage product from its existing qualified artifacts,
derives the Compiler-qualified directional joint/view binding, evaluates the two
historical motion clips into qualification-owned bakes, and projects those exact bakes
into Runtime-v4 view-local directional attachments.

Renderer/input correctness is validated through the Runtime-v4 contract. Global
boundary-manifold and continuous-embedding theorems are not renderer prerequisites.
Dynamic geometry quality remains a separate Compiler/product-quality concern.

The output authorizes a reference/native render attempt. It is not PRODUCT_PASS.
"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
import zlib
from time import perf_counter

from compiler.realsas_compiler_core.directional_binding import (
    qualify_directional_joint_view_binding,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    validate_playback_runtime_v4_contract,
    validate_runtime_v4_clip,
)
from compiler.realsas_compiler_core.v4 import bind_proof_plan
from compiler.realsas_compiler_services.export.current_v4_directional_runtime_v4 import (
    BODY_COMPONENT_ID,
    project_directional_product_bakes_to_runtime_v4,
)
from compiler.realsas_compiler_services.export.current_v4_runtime_v2 import (
    RuntimeTexturePayloadV1,
)
from compiler.realsas_compiler_services.proof.directional_motion_provider import (
    make_qualified_directional_motion_provider,
)

import experiments.mage_demo_fit1_v5_p1.materialize_product_state_fit2_v3_motion_engine as product_v3

SCHEMA = "RealSaS.MageFullAssemblyRuntimeV4Admission.v2"
PASS_STATUS = "PASS__FULL_MAGE_RUNTIME_V4_REFERENCE_INPUT_ADMITTED"
REQUIRED_CLIPS = ("mage_fit1_idle_v2", "mage_fit1_run_v2")
VIEW_IDS = tuple(f"V{i}" for i in range(8))


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            value = zlib.crc32(block, value)
    return int(value & 0xFFFFFFFF)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"MAGE_V4_ADMISSION_EXPECTED_JSON_OBJECT:{path}")
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _texture_bindings(state, foreground_dir: Path):
    manifest_path = foreground_dir / "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"MAGE_V4_ADMISSION_FOREGROUND_MANIFEST_MISSING:{manifest_path}")
    manifest = _load_json(manifest_path)
    expected = str(state["source_hashes"].get("foreground_manifest") or "")
    if expected and _sha(manifest_path) != expected:
        raise RuntimeError("MAGE_V4_ADMISSION_FOREGROUND_MANIFEST_SHA_DRIFT")
    rows = {int(row["view"]): row for row in manifest.get("views") or ()}
    if set(rows) != set(range(8)):
        raise RuntimeError("MAGE_V4_ADMISSION_FOREGROUND_VIEW_SET_INCOMPLETE")
    out = []
    for view in range(8):
        row = rows[view]
        atlas_path = foreground_dir / str(row["atlas_file"])
        if not atlas_path.is_file():
            raise RuntimeError(f"MAGE_V4_ADMISSION_ATLAS_MISSING:V{view}:{atlas_path}")
        actual_sha = _sha(atlas_path)
        if actual_sha != str(row["atlas_sha256"]):
            raise RuntimeError(f"MAGE_V4_ADMISSION_ATLAS_SHA_DRIFT:V{view}")
        relpath = str(row["atlas_relpath"])
        expected_payload_hash = content_sha256({
            "image_sha256": actual_sha,
            "image_relpath": relpath,
        })
        if expected_payload_hash != str(row["atlas_payload_hash"]):
            raise RuntimeError(f"MAGE_V4_ADMISSION_ATLAS_PAYLOAD_DRIFT:V{view}")
        out.append(RuntimeTexturePayloadV1(
            view_index=view,
            image_relpath=relpath,
            image_sha256=actual_sha,
            image_crc32=_crc32(atlas_path),
            width=int(row["atlas_width"]),
            height=int(row["atlas_height"]),
            atlas_payload_hash=expected_payload_hash,
        ))
    return tuple(out)


def _camera_payloads(args):
    paths = tuple(Path(x).resolve() for x in args.cameras)
    if len(paths) != 8:
        raise RuntimeError("MAGE_V4_ADMISSION_REQUIRES_8_CAMERAS")
    return {f"V{i}": _load_json(path) for i, path in enumerate(paths)}


def _admission_bakes(product):
    binding = qualify_directional_joint_view_binding(product)
    provider = make_qualified_directional_motion_provider(binding)
    provider.assert_for_product(product)
    clips = {str(clip.clip_id): clip for clip in product.motion_state.clips}
    if set(REQUIRED_CLIPS) - set(clips):
        raise RuntimeError(
            "MAGE_V4_ADMISSION_REQUIRED_CLIPS_MISSING:"
            + ",".join(sorted(set(REQUIRED_CLIPS) - set(clips)))
        )

    bakes = []
    plans = []
    for clip_id in REQUIRED_CLIPS:
        plan = bind_proof_plan(
            product,
            proof_domain="MOTION",
            operator_policy_hashes=(
                provider.provider_hash,
                provider.evaluator_policy_hash,
            ),
            probe_specification={
                "schema": "RealSaS.MageRuntimeV4AdmissionMotionProbe.v1",
                "scope": "RENDER_FREE_RUNTIME_V4_ADMISSION",
                "clip_id": clip_id,
                "qualification_owned_bake_required": True,
                "full_product_proof_claimed": False,
            },
        )
        bake = provider(product, plan, clips[clip_id])
        plans.append(plan)
        bakes.append(bake)
    return binding, provider, tuple(plans), tuple(bakes)


def run(args) -> dict:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Reuse the current full-Mage product materializer. It consumes the exact qualified
    # P1Q BODY, typed foreground assembly, one-hot rigid carry, continuity underlay,
    # and historical idle/run motion authorities. No alternate puppet truth is built.
    stage_t0 = perf_counter()
    print("MAGE_V4_ADMISSION_STAGE=PRODUCT_MATERIALIZATION_START", flush=True)
    state = product_v3.build_final_state(args, persist=False)
    product = state["product"]
    print(
        f"MAGE_V4_ADMISSION_STAGE=PRODUCT_MATERIALIZATION_PASS elapsed_s={perf_counter()-stage_t0:.3f}",
        flush=True,
    )
    render_metadata = dict(product.directional_renderables.metadata or {})
    if render_metadata.get("continuity_underlay_qualified") is not True:
        raise RuntimeError("MAGE_V4_ADMISSION_CONTINUITY_UNDERLAY_NOT_BOUND")

    stage_t0 = perf_counter()
    print("MAGE_V4_ADMISSION_STAGE=MOTION_BAKES_START", flush=True)
    binding, provider, plans, bakes = _admission_bakes(product)
    print(
        f"MAGE_V4_ADMISSION_STAGE=MOTION_BAKES_PASS elapsed_s={perf_counter()-stage_t0:.3f}",
        flush=True,
    )

    stage_t0 = perf_counter()
    print("MAGE_V4_ADMISSION_STAGE=TEXTURE_CAMERA_BINDING_START", flush=True)
    textures = _texture_bindings(state, Path(args.foreground_dir).resolve())
    cameras = _camera_payloads(args)
    print(
        f"MAGE_V4_ADMISSION_STAGE=TEXTURE_CAMERA_BINDING_PASS elapsed_s={perf_counter()-stage_t0:.3f}",
        flush=True,
    )

    stage_t0 = perf_counter()
    print("MAGE_V4_ADMISSION_STAGE=RUNTIME_V4_PROJECTION_START", flush=True)
    projection = project_directional_product_bakes_to_runtime_v4(
        product=product,
        motion_bakes=bakes,
        texture_bindings=textures,
        cameras=cameras,
        required_view_ids=VIEW_IDS,
        body_component_id=BODY_COMPONENT_ID,
        runtime_qualified=False,
    )
    print(
        f"MAGE_V4_ADMISSION_STAGE=RUNTIME_V4_PROJECTION_PASS elapsed_s={perf_counter()-stage_t0:.3f}",
        flush=True,
    )

    stage_t0 = perf_counter()
    print("MAGE_V4_ADMISSION_STAGE=REFERENCE_INPUT_VALIDATION_START", flush=True)
    playback_contract_hash = validate_playback_runtime_v4_contract(
        projection.contract,
        required_view_ids=VIEW_IDS,
    )
    clip_ids = tuple(clip.clip_id for clip in projection.clips)
    if set(clip_ids) != set(REQUIRED_CLIPS) or len(clip_ids) != len(REQUIRED_CLIPS):
        raise RuntimeError("MAGE_V4_ADMISSION_RUNTIME_CLIP_SET_DRIFT")
    for clip in projection.clips:
        if clip.runtime_qualified:
            raise RuntimeError("MAGE_V4_ADMISSION_PREMATURE_RUNTIME_QUALIFICATION")
        validate_runtime_v4_clip(
            projection.contract,
            clip,
            required_view_ids=VIEW_IDS,
        )
    print(
        f"MAGE_V4_ADMISSION_STAGE=REFERENCE_INPUT_VALIDATION_PASS elapsed_s={perf_counter()-stage_t0:.3f}",
        flush=True,
    )

    # Persist typed admission evidence. Rendering is a downstream consumer and must
    # refuse to run unless this report is present and status is exact PASS.
    _write_json(out / "DIRECTIONAL_JOINT_VIEW_BINDING_SET.json", binding.to_dict())
    for plan, bake in zip(plans, bakes):
        _write_json(out / f"{bake.clip_id}_MOTION_ADMISSION_PLAN.json", plan.to_dict())
        _write_json(out / f"{bake.clip_id}_QUALIFICATION_OWNED_MOTION_BAKE.json", bake.to_dict())

    report = {
        "schema": SCHEMA,
        "status": PASS_STATUS,
        "subject_id": "MAGE_FIT2_CURRENT_FULL_ASSEMBLY",
        "scope": "RENDER_FREE_RUNTIME_V4_ADMISSION",
        "source_product_state_hash": product.product_state_hash,
        "source_mechanical_state_hash": product.mechanical_state_hash,
        "source_directional_visual_state_hash": product.directional_visual_state_hash,
        "source_motion_state_hash": product.motion_state_hash,
        "component_assembly_hash": state["assembly"].component_assembly_hash,
        "continuity_underlay_set_hash": state["underlay_set"].qualification_hash,
        "directional_binding_set_hash": binding.binding_set_hash,
        "directional_binding_policy_hash": binding.policy_hash,
        "qualified_motion_provider_hash": provider.provider_hash,
        "evaluator_policy_hash": provider.evaluator_policy_hash,
        "evaluator_semantic_version": provider.evaluator_semantic_version,
        "runtime_v4_projection_hash": projection.projection_hash,
        "runtime_v4_asset_count": len(projection.contract.assets),
        "runtime_v4_slot_count": len(projection.contract.slots),
        "runtime_v4_view_count": len(projection.contract.views),
        "body_component_id": projection.body_component_id,
        "motion_bakes": {
            bake.clip_id: {
                "proof_plan_hash": plan.proof_plan_hash,
                "bake_hash": bake.bake_hash,
                "frame_count": len(bake.frames),
                "sampling_policy": bake.sampling_policy,
            }
            for plan, bake in zip(plans, bakes)
        },
        "reference_render_input_validation": {
            "playback_contract_hash": playback_contract_hash,
            "clip_ids": list(clip_ids),
            "fixed_runtime_topology_required": True,
            "typed_source_provenance_required": True,
            "explicit_slot_draw_order_required": True,
            "completion_allowed": False,
            "global_boundary_manifold_required": False,
            "continuous_embedding_theorem_required": False,
            "dynamic_geometry_quality_claimed": False,
            "dynamic_geometry_quality_owner": "COMPILER_PRODUCT_QUALITY_SEPARATE_FROM_RASTER_CORRECTNESS",
        },
        "source_hashes": dict(state["source_hashes"]),
        "completion_used": False,
        "new_pixels_generated": False,
        "full_product_proof_claimed": False,
        "product_pass_claimed": False,
        "render_executed": False,
        "gif_executed": False,
        "render_gate": {
            "required_status": PASS_STATUS,
            "required_projection_hash": projection.projection_hash,
            "required_playback_contract_hash": playback_contract_hash,
            "topology_quality_theorem_required": False,
        },
    }
    report_path = out / "MAGE_FULL_ASSEMBLY_RUNTIME_V4_ADMISSION_V1.json"
    _write_json(report_path, report)
    print("MAGE_FULL_ASSEMBLY_RUNTIME_V4_ADMISSION_PASS")
    print(json.dumps({
        "report": str(report_path),
        "report_sha256": _sha(report_path),
        "product_state_hash": product.product_state_hash,
        "projection_hash": projection.projection_hash,
        "playback_contract_hash": playback_contract_hash,
    }, indent=2, sort_keys=True))
    return {
        "report": report,
        "report_path": str(report_path),
        "report_sha256": _sha(report_path),
        "projection": projection,
        "product": product,
        "state": state,
        "binding": binding,
        "provider": provider,
        "plans": plans,
        "bakes": bakes,
        "playback_contract_hash": playback_contract_hash,
        "textures": textures,
        "cameras": cameras,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--p1q-dir", required=True)
    p.add_argument("--foreground-dir", required=True)
    p.add_argument("--assembly-dir", required=True)
    p.add_argument("--fit2-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--expected-p1q-manifest", default="")
    p.add_argument("--expected-foreground-manifest", default="")
    p.add_argument("--expected-assembly-manifest", default="")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
