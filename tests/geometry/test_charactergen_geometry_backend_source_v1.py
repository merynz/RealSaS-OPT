from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "experiments" / "geometry_backends" / "charactergen_geometry_backend_v1.py"


def _load_backend():
    spec = importlib.util.spec_from_file_location("charactergen_geometry_backend_v1", BACKEND)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load CharacterGen geometry backend")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CharacterGenGeometryBackendSourceV1(unittest.TestCase):
    def test_pins_and_cardinal_contract_are_frozen(self):
        mod = _load_backend()
        self.assertEqual(mod.CHARACTERGEN_COMMIT, "f329a835dbd5003060a5653eafd83d4d8868b043")
        self.assertEqual(mod.CHARACTERGEN_LRM_SHA256, "ec0edc6eed553910bdf8a1ceb204d8837d7b823826dca55f673c84046078094b")
        self.assertEqual(mod.DEFAULT_REALSAS_CARDINAL_MAP, ("N", "S", "E", "W"))

    def test_missing_or_ambiguous_view_fails_closed(self):
        mod = _load_backend()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(FileNotFoundError):
                mod._find_view_image(root, "S")
            (root / "S.png").write_bytes(b"x")
            (root / "s.jpg").write_bytes(b"x")
            with self.assertRaises(RuntimeError):
                mod._find_view_image(root, "S")

    def test_complete_geometry_is_not_product_contract(self):
        source = BACKEND.read_text(encoding="utf-8")
        self.assertIn("only RiggingSurfaceIR persists", source)
        self.assertIn("retain_debug_mesh: bool = False", source)
        self.assertIn("Do not fabricate observational support_views", source)
        self.assertIn("append_experiment_ledger(args.realsas_root, report)", source)


if __name__ == "__main__":
    unittest.main()
