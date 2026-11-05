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

    def test_code_study_is_varied_and_not_compiler_telemetry(self):
        js=(STATIC/"js"/"busted_code_typing_v2.js").read_text("utf-8")
        rain=(STATIC/"js"/"busted_spatial_rain_v1.js").read_text("utf-8")
        self.assertIn("Math.random",js)
        self.assertIn("const nouns=",js)
        self.assertIn("const verbs=",js)
        self.assertIn("const fields=",js)
        self.assertIn("previous.push(output)",js)
        self.assertIn("trailing decode window",js.lower())
        self.assertIn("continuous downward glyph field",rain.lower())
        self.assertIn("ctx.fillText",rain)
        self.assertIn("document.hidden",rain)

if __name__ == "__main__":
    unittest.main()
