from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
STATIC=ROOT/"product"/"living_compile"/"static"

class BustedAwakenStageContractTests(unittest.TestCase):
    def test_global_spatial_rain_and_scoped_code(self):
        html=(STATIC/"index.html").read_text("utf-8")
        rain=(STATIC/"js"/"busted_spatial_rain_v1.js").read_text("utf-8")
        typer=(STATIC/"js"/"busted_spatial_typing_v4.js").read_text("utf-8")
        specimen=(STATIC/"js"/"busted_geometry_core_v3.js").read_text("utf-8")
        stage_rain=(STATIC/"js"/"busted_stage_rain_v2.js").read_text("utf-8")
        self.assertIn('id="awakenChamber"',html)
        self.assertIn('id="matrixTypewriter"',html)
        self.assertNotIn('id="landingMatrixTypewriter"',html)
        self.assertNotIn('id="landingDepthCode"',html)
        self.assertIn('id="dropzone"',html)
        self.assertIn("busted_spatial_rain_v1.js",html)
        self.assertIn("busted_spatial_typing_v4.js",html)
        self.assertIn("busted_geometry_core_v3.js",html)
        self.assertIn("busted_stage_rain_v2.js",html)
        self.assertIn("busted_cinematic_hotfix_v3.css",html)
        self.assertIn("busted_cinematic_v2.css",html)
        self.assertIn("NOT COMPILED OUTPUT",html)
        self.assertNotIn("src=\"/static/js/awaken_depth.js",html)
        self.assertIn("className='busted-spatial-rain'",rain)
        self.assertIn("requestAnimationFrame",rain)
        self.assertIn("prefers-reduced-motion",rain)
        self.assertIn("document.hidden",rain)
        self.assertIn("visibilitychange",rain)
        self.assertIn("function line(max)",typer)
        self.assertIn("Math.random",typer)
        self.assertIn("traces",typer)
        self.assertIn("function relocate(t)",typer)
        self.assertIn("function permitted",typer)
        self.assertIn("terminal-hold",typer)
        self.assertIn("t.typeFinish+1050",typer)
        self.assertNotIn("animatePositions",typer)
        self.assertNotIn("travelling",typer)
        self.assertIn("IntersectionObserver",typer)
        self.assertIn("alphabet",typer)
        self.assertIn("not compiler telemetry or proof",typer)
        self.assertIn("const triangles=",specimen)
        self.assertIn("MORPHOLOGICAL FIELD",html)
        self.assertIn("AWAKEN",html)
        self.assertNotIn(">COMPILE CHARACTER",html)
        self.assertIn("const latitude=",specimen)
        self.assertIn("IntersectionObserver",stage_rain)
        # Performance budget: the home page must not render a redundant
        # full-screen rain animation above its stage-local canvases.
        self.assertIn("homeStage",rain)
        self.assertIn("!homeStage",rain)
        self.assertIn("scene.width<760?86:62",stage_rain)
        self.assertIn("scene.dpr=Math.min(devicePixelRatio||1,1.15)",stage_rain)
        self.assertIn("const latitude=26,longitude=42",specimen)
        self.assertIn("if(now-last>=52)",specimen)
        self.assertIn("prefers-reduced-motion",specimen)
        self.assertNotIn("product_proof::pass",specimen)
        self.assertNotIn("lastMutation",rain)
        self.assertIn("busted_visual_priority_v1.css",html)
        self.assertNotIn("POWERED BY REALSAS",html)
        self.assertNotIn("WE DON'T",html)

        self.assertNotIn("getContext(",typer)
        self.assertNotIn("runPresentationFlow",typer)

    def test_all_product_pages_share_rain_only_home_types(self):
        pages=["index.html","workspace.html","motion.html","export.html"]
        for page in pages:
            html=(STATIC/page).read_text("utf-8")
            self.assertIn("busted_spatial_rain_v1.js",html)
            self.assertIn("busted_spatial_rain_v1.css",html)
            if page=="index.html":
                self.assertIn("busted_spatial_typing_v4.js",html)
            else:
                self.assertNotIn("busted_spatial_typing_v4.js",html)

    def test_navigation_and_fail_closed_gateway(self):
        html=(STATIC/"index.html").read_text("utf-8")
        app=(STATIC/"js"/"awaken.js").read_text("utf-8")
        studio=(STATIC/"workspace.html").read_text("utf-8")
        self.assertIn('class="studio-link" href="/studio"',html)
        self.assertIn('href="/"',studio)
        self.assertIn("workspace.workspace_url",app)
        self.assertIn("project?.compile_available && project?.public_compile_gateway",app)
        self.assertIn("COMPILER OFFLINE",app)
        self.assertIn('id="compileAvailability"',html)
        self.assertIn("awaken.disabled = running || !ready",app)

if __name__=="__main__":
    unittest.main()
