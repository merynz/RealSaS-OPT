"""Static safety contracts for linked Studio workspaces + evidence-led progress."""
from pathlib import Path
import unittest

STATIC = Path(__file__).resolve().parents[2] / "product" / "living_compile" / "static"

class BustedDirectorProgressContractTest(unittest.TestCase):
    def test_studio_director_is_one_puppet_two_workspaces(self):
        html = (STATIC / "workspace.html").read_text("utf-8")
        director = (STATIC / "js" / "studio_director_v1.js").read_text("utf-8")
        css = (STATIC / "css" / "studio_director_v1.css").read_text("utf-8")
        self.assertIn('id="studioDirector"', html)
        self.assertIn('data-director-workspace="puppet"', html)
        self.assertIn('data-director-workspace="motion"', html)
        self.assertIn('data-director-command="new-clip"', html)
        self.assertIn('data-director-command="add-key"', html)
        self.assertIn('data-director-command="ik"', html)
        self.assertIn('disabled>ADD KEY', html)
        self.assertIn("Boolean(s?.bundle&&s?.scene?.puppet&&s?.scene?.proof?.passed===true)", director)
        self.assertIn("chooseMode(workspace==='motion'?'animation':'rig')", director)
        self.assertIn("click('#newClip')", director)
        self.assertIn("click('#addKey')", director)
        self.assertIn("click('#fitView')", director)
        self.assertIn("grid-row:8!important", css)
        self.assertNotIn("new StudioTimeline(", director)
        self.assertNotIn("product_proof::pass", director)

    def test_compiler_progress_has_no_fake_timer(self):
        html = (STATIC / "index.html").read_text("utf-8")
        script = (STATIC / "js" / "awaken.js").read_text("utf-8")
        style = (STATIC / "css" / "busted_compile_terminal_v1.css").read_text("utf-8")
        self.assertIn('id="compileProgressConsole"', html)
        self.assertIn('id="compileProgressPercent"', html)
        self.assertIn('id="compileProgressBlocks"', html)
        self.assertIn("const progressSegments=24", script)
        self.assertIn("function reportedPercent(payload)", script)
        self.assertIn("payload?.progress_percent", script)
        self.assertIn("onCompilerEvidence(payload,stageName,index,isTerminal)", script)
        self.assertIn("Math.floor(completed/9*100)", script)
        self.assertIn("Math.floor(p/100*progressSegments)", script)
        self.assertIn("STAGES SEALED", html)
        self.assertIn("prefers-reduced-motion", style)
        self.assertNotIn("setInterval(updateCompileProgress", script)

    def test_alive_is_only_claimed_after_real_idle_frame(self):
        html = (STATIC / "index.html").read_text("utf-8")
        script = (STATIC / "js" / "awaken.js").read_text("utf-8")
        self.assertIn("<h2>QUALIFIED</h2>", html)
        self.assertIn("if(!meshes||typeof meshes!=='object'||!Object.keys(meshes).length)", script)
        self.assertIn("result.classList.add('alive-reveal')", script)
        self.assertIn("idle runtime frame has no deformable mesh evidence", script)
        self.assertIn("status.querySelector('span').textContent='IDLE UNAVAILABLE'", script)
        self.assertIn("status.querySelector('span').textContent='PUPPET QUALIFIED'", script)
        self.assertNotIn("|| (runtime.clips || [])[0]", script)

if __name__ == "__main__":
    unittest.main()
