from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "product" / "living_compile" / "static"

class BustedAwakenStageContractTests(unittest.TestCase):
    def test_spatial_scene_is_non_canvas_and_static(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        app = (STATIC / "js" / "awaken.js").read_text(encoding="utf-8")
        stage = (STATIC / "js" / "awaken_depth.js").read_text(encoding="utf-8")
        self.assertIn('id="depthCode"', html)
        self.assertNotIn('id="depthScene"', html)
        self.assertNotIn('id="agentReach"', html)
        self.assertNotIn("agent_reach.js", html)
        self.assertNotIn("requestAnimationFrame", stage)
        self.assertNotIn("getContext(", stage)
        self.assertIn("Math.random", stage)
        self.assertIn("translateZ(", stage)
        self.assertIn("lane.lines=lane.rows.map(fmt)", stage)
        self.assertNotIn("runPresentationFlow", app)
        self.assertIn("visibilitychange", stage)
        self.assertIn("prefers-reduced-motion", stage)

    def test_navigation_and_fail_closed_gateway(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        app = (STATIC / "js" / "awaken.js").read_text(encoding="utf-8")
        studio = (STATIC / "workspace.html").read_text(encoding="utf-8")
        self.assertIn('class="studio-link" href="/studio"', html)
        self.assertIn('href="/"', studio)
        self.assertIn("BUSTED · Studio", studio)
        self.assertIn("workspace.workspace_url", app)
        self.assertIn("project?.compile_available && project?.public_compile_gateway", app)
        self.assertIn("COMPILER OFFLINE", app)
        self.assertIn('id="compileAvailability"', html)
        self.assertIn("awaken.disabled = running || !ready", app)

if __name__ == "__main__":
    unittest.main()
