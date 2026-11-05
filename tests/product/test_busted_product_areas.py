from pathlib import Path
import unittest
from product.living_compile.server import LivingCompileApplication

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "product" / "living_compile" / "static"

class BustedProductAreasTests(unittest.TestCase):
    def test_editor_uses_progressive_disclosure(self):
        page = (STATIC / "workspace.html").read_text("utf-8")
        self.assertIn('id="advancedStages"', page)
        self.assertIn('id="advancedTools"', page)
        self.assertIn('id="toggleTimeline"', page)
        self.assertIn('id="studioWelcome"', page)
        self.assertIn('id="exportRuntime" class="primary" hidden', page)
        self.assertIn('href="/motion"', page)
        self.assertIn('href="/export"', page)
        editor_js = (STATIC / "js" / "editor_v4.js").read_text("utf-8")
        self.assertIn('s?.scene?.proof?.passed===true', editor_js)
        self.assertIn('timeline-open', editor_js)

    def test_motion_and_export_are_separate_real_routes(self):
        app = LivingCompileApplication.default()
        for route, title, file in [
            ("/motion", "BUSTED · Motion Lab", "motion.html"),
            ("/export", "BUSTED · Export", "export.html"),
        ]:
            data, mime = app.static_bytes(route)
            self.assertIn(title.encode("utf-8"), data)
            self.assertEqual(mime, "text/html")
            status, page, mime = app.dispatch("GET", route)
            self.assertEqual(status, 200)
            self.assertEqual(page, data)
            self.assertEqual(mime, "text/html")
            self.assertTrue((STATIC / file).is_file())

    def test_motion_is_runtime_baked_and_export_is_proof_gated(self):
        motion = (STATIC / "js" / "motion_lab_v1.js").read_text("utf-8")
        export = (STATIC / "js" / "export_center_v1.js").read_text("utf-8")
        self.assertIn("/api/living/scene?bundle=", motion)
        self.assertIn("/api/runtime/clips?bundle=", motion)
        self.assertIn("/api/runtime/frame?bundle=", motion)
        self.assertIn("scene?.proof?.passed!==true", motion)
        self.assertIn("disabled=!matching.length", motion)
        self.assertIn("scene?.proof?.passed!==true", export)
        self.assertIn("'/api/export'", export)
        self.assertIn("EXISTING_PROOF_GATED_NATIVE_RUNTIME", export)
        self.assertIn("no file was downloaded", export)
        self.assertNotIn("runPresentationFlow", motion)
        self.assertNotIn("PROOF PASS", export)

    def test_home_code_is_varied_but_non_authoritative(self):
        js = (STATIC / "js" / "awaken_depth.js").read_text("utf-8")
        css = (STATIC / "css" / "busted_awaken_v3.css").read_text("utf-8")
        self.assertIn("Math.random", js)
        self.assertIn("translateZ(", js)
        self.assertIn("generateGlyphs", js)
        self.assertIn("desktopSlots", js)
        self.assertIn("mobileSlots", js)
        self.assertIn("matrix-trace", js)
        self.assertIn("numberOfLanes", js)
        self.assertIn("compensate", js)
        self.assertNotIn("requestAnimationFrame", js)
        self.assertNotIn("getContext(", js)
        matrix = (STATIC / "css" / "matrix_stage_v5.css").read_text("utf-8")
        self.assertIn(".depth-floor{display:none!important}", matrix)
        self.assertIn("matrixDescent", matrix)

if __name__ == "__main__":
    unittest.main()
