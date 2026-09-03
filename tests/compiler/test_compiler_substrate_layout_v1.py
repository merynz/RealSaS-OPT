from __future__ import annotations

from pathlib import Path
import unittest


class CompilerSubstrateLayoutTests(unittest.TestCase):
    def test_compatibility_paths_reflect_canonical_implementations(self):
        from compiler.realsas_compiler_core import surface as compat_surface
        from compiler.realsas_compiler_core import local_geometry as compat_local
        from compiler.realsas_compiler_core.substrate import surface as canonical_surface
        from compiler.realsas_compiler_core.substrate import local_geometry as canonical_local

        self.assertIs(compat_surface.build_surface_from_persistence, canonical_surface.build_surface_from_persistence)
        self.assertIs(compat_surface.rigging_surface_from_d2_arrays, canonical_surface.rigging_surface_from_d2_arrays)
        self.assertIs(compat_local.robust_local_plane_normals, canonical_local.robust_local_plane_normals)
        self.assertIs(compat_local.attach_dtb_nd1_normals, canonical_local.attach_dtb_nd1_normals)
        self.assertEqual(compat_local.dtb_nd1_operator_hash(), canonical_local.dtb_nd1_operator_hash())

    def test_iris_v2_adapter_is_byte_preserved(self):
        root = Path(__file__).resolve().parents[2]
        promoted = root / "compiler" / "realsas_compiler_core" / "substrate" / "iris_v2.py"
        provenance = root / "experiments" / "iris_reprojection_v2_20260831" / "persistence_adapter_v2.py"
        self.assertEqual(promoted.read_bytes(), provenance.read_bytes())

    def test_mainline_substrate_does_not_import_dated_experiments(self):
        root = Path(__file__).resolve().parents[2]
        substrate = root / "compiler" / "realsas_compiler_core" / "substrate"
        for path in substrate.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("experiments.", text, path.name)
            self.assertNotIn("experiments/", text, path.name)


if __name__ == "__main__":
    unittest.main()
