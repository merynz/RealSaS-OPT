from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
STATIC_JS = ROOT / "product" / "living_compile" / "static" / "js"


class AwakenProductionWiringTests(unittest.TestCase):
    def test_awaken_submits_selected_front_back_views_and_hands_off_workspace(self):
        source = (STATIC_JS / "awaken.js").read_text(encoding="utf-8")
        self.assertIn("async function startProductionCompile()", source)
        self.assertIn("slot: index === 0 ? 'front' : 'back'", source)
        self.assertIn("bytes_base64", source)
        self.assertIn("body: JSON.stringify({views})", source)
        self.assertIn("/api/compile/start", source)
        self.assertIn("/api/compile/status?id=", source)
        self.assertIn("/api/compile/workspace?id=", source)
        self.assertIn("workspace.workspace_url", source)

    def test_studio_shell_cannot_start_a_source_compile(self):
        source = (STATIC_JS / "product_shell.js").read_text(encoding="utf-8")
        self.assertNotIn("/api/compile/start", source)
        self.assertNotIn("compileCharacter", source)
        self.assertIn("Input belongs to Awaken", source)


if __name__ == "__main__":
    unittest.main()
