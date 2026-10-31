from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "product" / "living_compile" / "static"
STATIC_JS = STATIC / "js"


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

    def test_front_back_authority_is_visible_and_swap_changes_real_file_order(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        slots = (STATIC_JS / "awaken_input_slots.js").read_text(encoding="utf-8")
        self.assertIn('data-input-slot="front"', html)
        self.assertIn('id="frontSlotFile">REQUIRED', html)
        self.assertIn('data-input-slot="back"', html)
        self.assertIn('id="backSlotFile">OPTIONAL', html)
        self.assertIn('id="swapInputViews"', html)
        self.assertIn("transfer.items.add(selected[1])", slots)
        self.assertIn("transfer.items.add(selected[0])", slots)
        self.assertIn("input.files = transfer.files", slots)
        self.assertIn("input.dispatchEvent(new Event('change'", slots)

    def test_studio_shell_cannot_start_a_source_compile(self):
        source = (STATIC_JS / "product_shell.js").read_text(encoding="utf-8")
        self.assertNotIn("/api/compile/start", source)
        self.assertNotIn("compileCharacter", source)
        self.assertIn("Input belongs to Awaken", source)


if __name__ == "__main__":
    unittest.main()
