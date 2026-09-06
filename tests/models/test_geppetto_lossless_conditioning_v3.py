from __future__ import annotations

import unittest

import numpy as np

from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from models.geppetto.v2.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2
from models.geppetto.v3.conditioning_lossless_v3 import (
    AUTHORITY_REVOCATION_ID_V1,
    GeppettoProductConditioningAdapterV3,
    LEGACY_SUMMARY_AUTHORITY_V2,
    PRODUCT_CONDITIONING_AUTHORITY_V3,
)


RES = 1024


def _surface(*, swapped: bool) -> RiggingSurfaceIR:
    # Same support set and same unordered raster coordinates. Only the association
    # between view identity and raster coordinate changes. A mean/std summary cannot
    # distinguish the two surfaces; a lossless per-view contract must.
    a = (128.0, 256.0)
    b = (896.0, 768.0)
    binds0 = ((0, b), (1, a)) if swapped else ((0, a), (1, b))
    binds1 = ((0, a), (1, b)) if swapped else ((0, b), (1, a))

    nodes = (
        SurfaceNode(
            surface_id="SFS:000",
            P=(-0.25, 0.0, 0.0),
            support_views=(0, 1),
            provenance_refs=("TEST",),
            source_observation_ids=(),
            raster_bindings=binds0,
            persistence_group_id="G:0",
            derived_normal=(1.0, 0.0, 0.0),
            validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": False},
        ),
        SurfaceNode(
            surface_id="SFS:001",
            P=(0.25, 0.0, 0.0),
            support_views=(0, 1),
            provenance_refs=("TEST",),
            source_observation_ids=(),
            raster_bindings=binds1,
            persistence_group_id="G:1",
            derived_normal=(1.0, 0.0, 0.0),
            validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": False},
        ),
    )
    relation = SurfaceRelation(
        relation_id="SFSREL:0-1",
        a_surface_id="SFS:000",
        b_surface_id="SFS:001",
        relation_kind="SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",
        score=1.0,
        metadata={
            "world_distance": 0.5,
            "crosses_unknown": False,
            "unknown_bridge": False,
            "teacher_truth_used": False,
        },
    )
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=(relation,),
        geometry_lineage_hash="LINEAGE:SWAP" if swapped else "LINEAGE:BASE",
        builder_id="RealSaS.GeometricSubstrateAssembler.SceneFirstSigned.v1",
        schema_version="RealSaS.RiggingSurfaceIR.v1",
        metadata={
            "scene_first_signed_geometry": True,
            "teacher_truth_used": False,
            "character_gen_runtime_used": False,
            "raster_coordinate_system": "PIXEL_CENTER_XY",
            "resolution": RES,
            "source_run_id": "TEST_RUN",
            "source_checkpoint_sha256": "TEST_CHECKPOINT",
            "source_zero_surface_sha256": "TEST_ZERO_SURFACE",
            "Nd_operator_sha256": "TEST_ND",
            "compact_surface_node_count": 2,
            "observed_node_count": 2,
            "completed_node_count": 0,
        },
    )


class GeppettoLosslessConditioningV3Test(unittest.TestCase):
    def test_v2_summary_aliases_view_identity_but_v3_preserves_it(self) -> None:
        s0 = _surface(swapped=False)
        s1 = _surface(swapped=True)

        legacy0 = GeppettoConditioningAdapterV2()((s0,))
        legacy1 = GeppettoConditioningAdapterV2()((s1,))
        np.testing.assert_array_equal(legacy0.features, legacy1.features)

        lossless0 = GeppettoProductConditioningAdapterV3()((s0,))
        lossless1 = GeppettoProductConditioningAdapterV3()((s1,))
        self.assertFalse(
            np.array_equal(lossless0.raster_xy_by_view, lossless1.raster_xy_by_view)
        )
        self.assertNotEqual(lossless0.evidence_hashes[0], lossless1.evidence_hashes[0])

    def test_product_batch_carries_full_source_and_no_legacy_features_alias(self) -> None:
        surface = _surface(swapped=False)
        batch = GeppettoProductConditioningAdapterV3()((surface,))

        self.assertEqual(batch.conditioning_authority, PRODUCT_CONDITIONING_AUTHORITY_V3)
        self.assertFalse(batch.derived_feature_authority)
        self.assertEqual(batch.derived_features_24d.shape[-1], 24)
        self.assertFalse(hasattr(batch, "features"))
        self.assertEqual(batch.raw_surface_payloads[0], surface.to_dict())
        np.testing.assert_array_equal(batch.raster_valid_by_view, batch.support_by_view)

        cert = batch.product_certificate(0)
        self.assertEqual(cert["revoked_legacy_summary_authority"], LEGACY_SUMMARY_AUTHORITY_V2)
        self.assertEqual(cert["authority_revocation_id"], AUTHORITY_REVOCATION_ID_V1)
        self.assertFalse(cert["derived_feature_authority"])

    def test_legacy_collapse_is_fail_closed(self) -> None:
        batch = GeppettoProductConditioningAdapterV3()((_surface(swapped=False),))
        with self.assertRaises(RuntimeError):
            batch.as_legacy_v2_diagnostic()


if __name__ == "__main__":
    unittest.main()
