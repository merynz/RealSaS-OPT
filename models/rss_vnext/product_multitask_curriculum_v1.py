from __future__ import annotations

"""Machine-readable, trainer-enforced curriculum and source-truth obligations.

The five *macro* training domains are canonical diffusion, multiview diffusion,
S-LRM, embedded MIRA codec, and seeded unified structural reconstruction.
This schema selects SUBPHASES only within that last domain. It is not a new
ModelRelease or inference-time network, and it does not auto-create teachers.
"""
from dataclasses import asdict, dataclass
from typing import Mapping

SCHEMA = "RealSaS.RSSvNext.ProductSeededCurriculum.v1"
STAGES = frozenset({
    "COMPATIBILITY_UNSCHEDULED",
    "MSTAR_GEOMETRY_BOOTSTRAP",
    "IRIS_GEOMETRY_SLR_GRAPHICS",
    "AXIS_MIRA_MSTAR_BINDING",
    "SOURCE_POSED_CONSEQUENCE",
    "FULL_SHARED_H_FINETUNE",
})


@dataclass(frozen=True)
class ProductSeededCurriculumV1:
    stage: str = "COMPATIBILITY_UNSCHEDULED"

    def validate(
        self, *, seeded_mode: bool, semantic_weights,
        axis_mira_weights, graphics_rgb_weight: float,
        graphics_joint_ray_weight: float, component_udf_weight: float,
        anchor_face_weight: float,
        mstar_pose_raster_weight: float = 0.,
    ) -> None:
        if self.stage not in STAGES:
            raise ValueError("RSS_VNEXT_CURRICULUM_STAGE_INVALID")
        if mstar_pose_raster_weight > 0 and (
            not seeded_mode or axis_mira_weights.posed_surface <= 0
        ):
            raise ValueError("RSS_VNEXT_CURRICULUM_POSED_MSTAR_REQUIRES_NATIVE_POSE")
        if self.stage == "COMPATIBILITY_UNSCHEDULED":
            return
        if not seeded_mode:
            raise ValueError("RSS_VNEXT_CURRICULUM_REQUIRES_PRODUCT_SEEDED")
        sem = (
            semantic_weights.patch_axes + semantic_weights.patch_risk
            + semantic_weights.surface_roles + semantic_weights.region_pairs
            + semantic_weights.typed_relations
        ) > 0
        rgb = (graphics_rgb_weight > 0 or graphics_joint_ray_weight > 0
               or mstar_pose_raster_weight > 0)
        axis = axis_mira_weights.axis_program > 0
        posed = (
            axis_mira_weights.posed_surface > 0
            or axis_mira_weights.dynamic_surface > 0
        )
        if self.stage == "MSTAR_GEOMETRY_BOOTSTRAP":
            allowed = not sem and not rgb and not axis and anchor_face_weight == 0
        elif self.stage == "IRIS_GEOMETRY_SLR_GRAPHICS":
            # Gradients are jointly sourced from 3D SLR + original source ray
            # targets, never diffusion pseudo-ground-truth.
            allowed = sem and rgb and not axis and anchor_face_weight == 0
        elif self.stage == "AXIS_MIRA_MSTAR_BINDING":
            allowed = (
                axis and axis_mira_weights.source_skin > 0
                and not posed and anchor_face_weight >= 0
            )
        elif self.stage == "SOURCE_POSED_CONSEQUENCE":
            allowed = (
                axis and axis_mira_weights.source_skin > 0
                and posed
            )
        elif self.stage == "FULL_SHARED_H_FINETUNE":
            allowed = sem and rgb and axis and posed
        else:
            raise AssertionError("Unexpected curriculum stage")
        if not allowed:
            raise ValueError(
                "RSS_VNEXT_CURRICULUM_OBJECTIVE_STAGE_MISMATCH:" + self.stage
            )

    def required_source_teacher_families(
        self, *, semantic_weights, axis_mira_weights,
        graphics_rgb_weight: float, graphics_joint_ray_weight: float,
        component_udf_weight: float, anchor_face_weight: float,
        mstar_pose_raster_weight: float = 0.,
    ) -> dict[str, Mapping[str, str]]:
        """Loss->foundry witness; source observable != generated hypothesis.

        These requirements are conditional on ACTIVE objectives; per-asset
        qualification masks remain mandatory. The catalog is not a coverage
        claim and cannot override the exact source-receipt validators.
        """
        result: dict[str, Mapping[str, str]] = {
            "original_two_view_rgba": {
                "authority": "ORIGINAL_SOURCE_BYTES_ONLY",
                "purpose": "Student observations; source camera never inferred",
            },
            "generated_six_view_conditioning": {
                "authority": "PINNED_GENERATOR_HYPOTHESES_NOT_TEACHER",
                "purpose": "Student H(x) conditioning and exact checkpoint identity",
            },
            "seeded_mstar_surface": {
                "authority": "SOURCE_NATIVE_FACE_BARYCENTRIC_NORMAL_LAYER",
                "purpose": "Qualified mesh geometry/normal/topology targets",
            },
            "source_identity_and_rights": {
                "authority": "INDEPENDENT_SOURCE_LINEAGE_LICENSE_RECEIPT",
                "purpose": "Asset family and allowed train/val/unseen use",
            },
        }
        if component_udf_weight > 0:
            result["component_udf"] = {
                "authority": "SOURCE_NATIVE_OPEN_OR_COMPONENT_DOMAIN",
                "purpose": "Unsigned distance and component mask; never fake global sign",
            }
        if semantic_weights.patch_axes or semantic_weights.patch_risk:
            result["original_patch_semantics"] = {
                "authority": "SOURCE_2D_PATCH_AXIS_QUALIFIED",
                "purpose": "Original image evidence; unknown per-axis mask",
            }
        if semantic_weights.surface_roles:
            result["3d_surface_roles"] = {
                "authority": "SOURCE_3D_QUERY_ADDRESS_QUALIFIED",
                "purpose": "SLR surface/layer roles; no physical material inference",
            }
        if semantic_weights.region_pairs:
            result["3d_region_co_membership"] = {
                "authority": "SOURCE_3D_REGION_PAIR_QUALIFIED",
                "purpose": "Same source region only; not drawable/contact",
            }
        if semantic_weights.typed_relations:
            result["research_typed_relations"] = {
                "authority": "RESEARCH_ONLY_NOT_PRODUCT",
                "purpose": "Never require in default product curriculum",
            }
        if graphics_rgb_weight > 0:
            result["original_visible_rgb"] = {
                "authority": "SOURCE_FIRST_HIT_RGB_RAYS_VISIBILITY_MASK",
                "purpose": "View-conditioned RGB; not hidden UV/alpha truth",
            }
        if graphics_joint_ray_weight > 0:
            result["source_3d_ray_rgb_alpha_layers"] = {
                "authority": "SOURCE_3D_RENDERED_RAY_STACK_AND_LAYER_MASK",
                "purpose": "Joint color/opacity/visibility and SLR peeled targets",
            }
        if mstar_pose_raster_weight > 0:
            result["source_native_pose_mstar_2d_raster"] = {
                "authority": "SOURCE_NATIVE_POSE_ZBUFFER_RGB_ALPHA_CAMERA_AND_MSTAR_ADDRESS",
                "purpose": "Canonical-material first-hit M* posed raster; no teacher RGB into H",
            }
        if axis_mira_weights.enabled:
            result["native_axis_rig"] = {
                "authority": "SOURCE_NATIVE_JOINT_ORDER_PARENT_FRAMES",
                "purpose": "AXIS causal sequence on SAME final M*",
            }
        if (
            axis_mira_weights.source_skin > 0
            or axis_mira_weights.hard_skin > 0
            or axis_mira_weights.chain_skin > 0
            or axis_mira_weights.posed_surface > 0
        ):
            result["native_mstar_skin_transport"] = {
                "authority": "SOURCE_FACE_BARYCENTRIC_JOINT_ORDER_MATCHED",
                "purpose": "Source skin weights on final M*; no nearest relabel",
            }
        if axis_mira_weights.chain_skin > 0:
            result["native_joint_chain_partition"] = {
                "authority": "SOURCE_NATIVE_ANCESTRAL_CHAIN_AND_VERTEX_MASK",
                "purpose": "Chain mass and conditional skin; UNKNOWN masked",
            }
        if (
            axis_mira_weights.posed_surface > 0
            or axis_mira_weights.dynamic_surface > 0
        ):
            result["native_lbs_pose_consequence"] = {
                "authority": "SOURCE_NATIVE_LBS_POSES_AND_MSTAR_SAMPLE_MASK",
                "purpose": "Joint matrices, posed vertices, strain and fold evidence",
            }
        if anchor_face_weight > 0:
            result["native_attachment_face"] = {
                "authority": "SOURCE_NATIVE_DIRECTED_ANCHOR_FACE",
                "purpose": "Attachment witness, not inferred from geometry proximity",
            }
        return result

    def compiler_acceptance_gaps(
        self, *, mstar_pose_raster_weight: float = 0.,
    ) -> tuple[str, ...]:
        """Not extra source truths. Do not silently train what cannot be known."""
        return (
            ("POSED_MSTAR_STAGE3_TEACHER_FORCED_NOT_POST_XPBD_OR_FREE_RUNNING"
             if mstar_pose_raster_weight > 0 else
             "PRODUCT_MSTAR_STAGE3_NATIVE_MOVING_RGBA_RASTER_NOT_ENABLED"),
            "UNSEEN_TWO_SOURCE_FREE_RUNNING_MSTAR_AXIS_MIRA_NOT_QUALIFIED",
            "GARMENT_BODY_CLEARANCE_AND_POST_PHYSICS_RASTER_NOT_QUALIFIED",
            "XPBD_MATERIAL_PINS_CONTACT_REQUIRE_AUTHORED_SOURCE_OR_POLICY",
            "FINAL_PRODUCTPROOF_SOURCE_AND_RUNTIME_ADMISSION_OPEN",
        )


__all__ = ["SCHEMA", "STAGES", "ProductSeededCurriculumV1"]
