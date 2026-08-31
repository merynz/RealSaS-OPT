from __future__ import annotations

import unittest
from dataclasses import replace

from realsas_compiler_core.bundle_routes import route_for
from realsas_compiler_core.mesh_binding import (
    mesh_lineage_hash,
    mesh_skin_lineage_hash,
    validate_surface_support_binding,
)
from realsas_compiler_core.product import (
    assemble_product,
    assemble_product_v2,
    bind_proof,
    require_current_proof,
)
from realsas_compiler_core.types import (
    CanonicalPuppetGraph,
    CanonicalPuppetGraphV2,
    QualifiedEditableMeshIR,
    QualifiedJoint,
    QualifiedMeshSkinIR,
    QualifiedMeshSkinRow,
    QualifiedMeshVertex,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceSupportBinding,
    QualificationError,
)


class MWB0ContractTests(unittest.TestCase):
    def setUp(self):
        self.surface = RiggingSurfaceIR(
            surface_nodes=(
                SurfaceNode("S0", (0.0, 0.0, 0.0), (0,), ("p0",), ("o0",)),
                SurfaceNode("S1", (1.0, 0.0, 0.0), (0,), ("p1",), ("o1",)),
                SurfaceNode("S2", (0.0, 1.0, 0.0), (0,), ("p2",), ("o2",)),
            ),
            geometry_lineage_hash="SURFACE:fixture",
        )
        self.skeleton = QualifiedSkeletonIR(
            joints=(QualifiedJoint("J:0", (0.0, 0.0, 0.0), None, ("S0",), "P0"),),
            root_id="J:0",
            qualification_report={"passed": True},
            skeleton_lineage_hash="SKELETON:fixture",
        )
        self.skin = QualifiedSkinIR(
            rows=(
                QualifiedSkinRow("S0", (("J:0", 1.0),), 0.0, 0.0),
                QualifiedSkinRow("S1", (("J:0", 1.0),), 0.0, 0.0),
                QualifiedSkinRow("S2", (("J:0", 1.0),), 0.0, 0.0),
            ),
            surface_binding_hash=self.surface.geometry_lineage_hash,
            skeleton_binding_hash=self.skeleton.skeleton_lineage_hash,
            qualification_report={"passed": True},
            skin_lineage_hash="SKIN:fixture",
        )
        self.mesh = self._mesh(("MV0", "MV1", "MV2"))
        self.mesh_skin = self._mesh_skin(self.mesh)

    @staticmethod
    def _identity(surface_id: str) -> SurfaceSupportBinding:
        return SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((surface_id, 1.0),))

    def _mesh(self, face: tuple[str, str, str]) -> QualifiedEditableMeshIR:
        mesh = QualifiedEditableMeshIR(
            vertices=(
                QualifiedMeshVertex("MV0", (0.0, 0.0, 0.0), self._identity("S0"), "CV0"),
                QualifiedMeshVertex("MV1", (1.0, 0.0, 0.0), self._identity("S1"), "CV1"),
                QualifiedMeshVertex("MV2", (0.0, 1.0, 0.0), self._identity("S2"), "CV2"),
            ),
            faces=(face,),
            edges=(("MV0", "MV1"), ("MV1", "MV2"), ("MV2", "MV0")),
            surface_binding_hash=self.surface.geometry_lineage_hash,
            view_index=0,
            camera_binding_hash="CAM:0",
            qualification_report={"passed": True},
            mesh_lineage_hash="",
        )
        return replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))

    def _mesh_skin(self, mesh: QualifiedEditableMeshIR, *, drop_last: bool = False) -> QualifiedMeshSkinIR:
        rows = (
            QualifiedMeshSkinRow("MV0", (("J:0", 1.0),), (("S0", 1.0),), 0.0, 0.0),
            QualifiedMeshSkinRow("MV1", (("J:0", 1.0),), (("S1", 1.0),), 0.0, 0.0),
            QualifiedMeshSkinRow("MV2", (("J:0", 1.0),), (("S2", 1.0),), 0.0, 0.0),
        )
        if drop_last:
            rows = rows[:-1]
        mesh_skin = QualifiedMeshSkinIR(
            rows=rows,
            surface_binding_hash=self.surface.geometry_lineage_hash,
            skeleton_binding_hash=self.skeleton.skeleton_lineage_hash,
            skin_binding_hash=self.skin.skin_lineage_hash,
            mesh_binding_hash=mesh.mesh_lineage_hash,
            transfer_method="IDENTITY_SURFACE_NODE",
            qualification_report={"passed": True},
            mesh_skin_lineage_hash="",
        )
        return replace(mesh_skin, mesh_skin_lineage_hash=mesh_skin_lineage_hash(mesh_skin))

    def test_legacy_product_path_remains_v1(self):
        product = assemble_product(self.surface, self.skeleton, self.skin)
        self.assertIsInstance(product, CanonicalPuppetGraph)
        self.assertEqual(product.schema_version, "RealSaS.CanonicalPuppetGraph.v1")

    def test_v2_product_binds_mesh_and_mesh_skin_hashes(self):
        product = assemble_product_v2(self.surface, self.skeleton, self.skin, self.mesh, self.mesh_skin)
        self.assertIsInstance(product, CanonicalPuppetGraphV2)
        self.assertEqual(product.schema_version, "RealSaS.CanonicalPuppetGraph.v2")
        self.assertEqual(product.admitted_mesh_hash, self.mesh.mesh_lineage_hash)
        self.assertEqual(product.admitted_mesh_skin_hash, self.mesh_skin.mesh_skin_lineage_hash)
        self.assertEqual(route_for(self.mesh).section, "mesh")
        self.assertEqual(route_for(self.mesh_skin).filename, "qualified_mesh_skin_ir.json")
        self.assertEqual(route_for(product).filename, "canonical_puppet_graph_v2.json")

    def test_topology_mutation_invalidates_old_proof(self):
        product_a = assemble_product_v2(self.surface, self.skeleton, self.skin, self.mesh, self.mesh_skin)
        proof_a = bind_proof(product_a, probe_plan={"probe": 1}, measurements={"ok": True}, passed=True)

        mesh_b = self._mesh(("MV0", "MV2", "MV1"))
        self.assertNotEqual(mesh_b.mesh_lineage_hash, self.mesh.mesh_lineage_hash)
        mesh_skin_b = self._mesh_skin(mesh_b)
        product_b = assemble_product_v2(self.surface, self.skeleton, self.skin, mesh_b, mesh_skin_b)
        self.assertNotEqual(product_b.product_state_hash, product_a.product_state_hash)
        with self.assertRaises(QualificationError):
            require_current_proof(product_b, proof_a, require_pass=True)

    def test_mesh_skin_lineage_mismatch_fails_closed(self):
        bad = replace(self.mesh_skin, mesh_binding_hash="MESH:wrong")
        bad = replace(bad, mesh_skin_lineage_hash=mesh_skin_lineage_hash(bad))
        with self.assertRaisesRegex(QualificationError, "MESH_WEIGHT_MESH_LINEAGE_MISMATCH"):
            assemble_product_v2(self.surface, self.skeleton, self.skin, self.mesh, bad)

    def test_missing_mesh_vertex_weight_row_fails_closed(self):
        incomplete = self._mesh_skin(self.mesh, drop_last=True)
        with self.assertRaisesRegex(QualificationError, "MESH_WEIGHT_UNSUPPORTED_VERTEX"):
            assemble_product_v2(self.surface, self.skeleton, self.skin, self.mesh, incomplete)

    def test_support_binding_simplex_is_enforced(self):
        with self.assertRaisesRegex(QualificationError, "MESH_VERTEX_SUPPORT_INVALID"):
            validate_surface_support_binding(
                SurfaceSupportBinding("LOCAL_CONVEX_INTERPOLATION", (("S0", 0.4), ("S1", 0.4))),
                known_surface_ids={"S0", "S1"},
            )


if __name__ == "__main__":
    unittest.main()
