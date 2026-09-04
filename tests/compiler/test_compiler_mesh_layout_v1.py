from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import unittest


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


class CompilerMeshLayoutTests(unittest.TestCase):
    def test_canonical_implementation_blobs_are_preserved(self):
        root = Path(__file__).resolve().parents[2]
        base = root / "compiler" / "realsas_compiler_core" / "mesh"
        expected = {
            "mwb2.py": "ff1376c190988c7f746e22496854a7cbe5bceeef",
            "mesh_binding.py": "f9795c988084676e5abd1a12aad37ed5012732e2",
            # V2 pre-FIT hardening makes mesh-skin normalization an explicitly
            # bounded, fail-closed repair with correction telemetry. This blob is
            # the current sealed implementation; the previous pin predates that
            # repair-accounting authority.
            "mwb2_skin.py": "bd3b11d869123b90b0cc9bf7cf3e87dbd1956152",
        }
        for name, sha in expected.items():
            self.assertEqual(git_blob_sha(base / name), sha, name)

    def test_compatibility_paths_reflect_canonical_modules(self):
        from compiler.realsas_compiler_core import mwb2 as compat_mwb2
        from compiler.realsas_compiler_core import mesh_binding as compat_binding
        from compiler.realsas_compiler_core import mwb2_skin as compat_skin
        from compiler.realsas_compiler_core.mesh import mwb2, mesh_binding, mwb2_skin

        self.assertIs(compat_mwb2.build_mwb2_candidate, mwb2.build_mwb2_candidate)
        self.assertIs(compat_mwb2.qualify_mwb2_mesh, mwb2.qualify_mwb2_mesh)
        self.assertIs(compat_binding.validate_qualified_mesh, mesh_binding.validate_qualified_mesh)
        self.assertIs(compat_skin.bind_mwb2_mesh_skin, mwb2_skin.bind_mwb2_mesh_skin)


if __name__ == "__main__":
    unittest.main()
