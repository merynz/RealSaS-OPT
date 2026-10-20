import unittest

from compiler.realsas_compiler_core.graphics_v2 import (
    GraphicsGlobalVisibilityEvidenceV2,
    GraphicsStaticRelationEvidenceV2,
    GraphicsViewPairEvidenceV2,
    build_graphics_evidence_v2,
    compile_graphics_v2,
)
from compiler.realsas_compiler_core.types import QualificationError


class GraphicsV2Tests(unittest.TestCase):
    def setUp(self):
        # Two disconnected triangles guarantee two fragments without requiring
        # any fabricated static ownership relation.
        self.faces = ((0, 1, 2), (3, 4, 5))
        self.mesh = "mesh-hash"
        self.skin = "skin-hash"
        self.source = "source-graphics-hash"

    def _compile(self, *, pairs=(), visibility=(), static=()):
        evidence = build_graphics_evidence_v2(
            mesh_binding_hash=self.mesh,
            static_relations=static,
            view_pair_relations=pairs,
            global_visibility=visibility,
        )
        return compile_graphics_v2(
            canonical_mesh_binding_hash=self.mesh,
            canonical_mesh_skin_binding_hash=self.skin,
            source_graphics_binding_hash=self.source,
            faces=self.faces,
            evidence=evidence,
        )

    def test_pair_order_can_exist_while_both_faces_globally_hidden(self):
        compiled = self._compile(
            pairs=(
                GraphicsViewPairEvidenceV2(
                    view_index=0,
                    face_a=0,
                    face_b=1,
                    front_probability=0.95,
                    front_valid=True,
                    occlusion_probability=1.0,
                    occlusion_valid=True,
                    evidence_ref="pair-depth",
                ),
            ),
            visibility=(
                GraphicsGlobalVisibilityEvidenceV2(0, 0, 0.0, True, True, "ray:A-behind"),
                GraphicsGlobalVisibilityEvidenceV2(0, 1, 0.0, True, True, "ray:B-behind"),
            ),
        )
        view = compiled.views[0]
        self.assertEqual(view.visible_fragment_ids, ())
        self.assertEqual(len(view.order_edges), 1)
        self.assertTrue(compiled.metadata["global_visibility_requires_complete_stack"])
        self.assertFalse(compiled.metadata["pairwise_visibility_used"])

    def test_visibility_only_evidence_changes_visible_set_without_pair_spoof(self):
        compiled = self._compile(
            visibility=(
                GraphicsGlobalVisibilityEvidenceV2(3, 0, 1.0, True, True, "ray:front"),
                GraphicsGlobalVisibilityEvidenceV2(3, 1, 0.0, True, True, "ray:hidden"),
            ),
        )
        view = compiled.views[0]
        self.assertEqual(view.view_index, 3)
        self.assertEqual(len(view.visible_fragment_ids), 1)
        self.assertEqual(view.order_edges, ())

    def test_incomplete_stack_cannot_be_valid_visibility_authority(self):
        with self.assertRaisesRegex(QualificationError, "REQUIRES_COMPLETE_STACK"):
            self._compile(
                visibility=(
                    GraphicsGlobalVisibilityEvidenceV2(0, 0, 1.0, True, False, "partial-stack"),
                ),
            )

    def test_invalid_visibility_abstains_instead_of_hiding(self):
        compiled = self._compile(
            visibility=(
                GraphicsGlobalVisibilityEvidenceV2(0, 0, 0.0, False, False, "unknown"),
            ),
        )
        view = compiled.views[0]
        self.assertEqual(len(view.visible_fragment_ids), 2)
        self.assertEqual(len(view.metadata["unknown_visibility_fragment_ids"]), 2)

    def test_unknown_static_relation_does_not_merge(self):
        # Make the two faces adjacent for this one case.
        self.faces = ((0, 1, 2), (2, 1, 3))
        row = GraphicsStaticRelationEvidenceV2(
            0,
            1,
            same_drawable_affinity=1.0,
            same_drawable_valid=False,
            cut_probability=0.0,
            cut_valid=False,
            alpha_boundary_probability=0.0,
            alpha_boundary_valid=False,
            appearance_boundary_probability=0.0,
            appearance_boundary_valid=False,
        )
        compiled = self._compile(static=(row,))
        self.assertEqual(len(compiled.fragments), 2)


if __name__ == "__main__":
    unittest.main()
