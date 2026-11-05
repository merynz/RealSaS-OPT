"""Browser-level visual and interaction contracts for the BUSTED product frontend.

Requires the product public_server at 127.0.0.1:8765 and Playwright Chromium.
Screenshots deliberately use reduced motion for reproducible inspection.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright

BASE = os.environ.get("BUSTED_VISUAL_BASE", "http://127.0.0.1:8765")
OUT = Path(os.environ.get("BUSTED_VISUAL_ARTIFACTS", ".artifacts/busted-visual"))
OUT.mkdir(parents=True, exist_ok=True)
VIEWPORTS = [(1440, 900), (1366, 768), (1280, 720), (960, 600), (390, 844)]


async def metrics(page, selector: str):
    return await page.locator(selector).evaluate("""el => {
      const r=el.getBoundingClientRect();
      return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom};
    }""")


async def run():
    failures = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        for width, height in VIEWPORTS:
            page = await browser.new_page(viewport={"width":width,"height":height}, device_scale_factor=1, reduced_motion="reduce")
            errors = []
            page.on("pageerror", lambda err: errors.append(str(err)))
            try:
                response = await page.goto(BASE + "/", wait_until="networkidle", timeout=25000)
                assert response and response.status == 200, f"Awaken HTTP failure at {width}x{height}"
                await page.locator("#compileAvailability").wait_for()
                await page.screenshot(path=str(OUT / f"awaken_{width}x{height}.png"), full_page=True)
                await page.locator("#awakenChamber").screenshot(path=str(OUT / f"chamber_{width}x{height}.png"))

                extent = await page.evaluate("({scroll:document.documentElement.scrollWidth,inner:window.innerWidth})")
                if extent["scroll"] > width+2:
                    offenders = await page.evaluate("""() => Array.from(document.querySelectorAll('*')).map(e => {
                      const r=e.getBoundingClientRect();
                      return {tag:e.tagName.toLowerCase(),className:typeof e.className==='string'?e.className:'',id:e.id,
                        x:Math.round(r.x),right:Math.round(r.right),width:Math.round(r.width),overflow:getComputedStyle(e).overflowX};
                    }).filter(x=>x.right>innerWidth+2).slice(0,24)""")
                    raise AssertionError(f"Awaken horizontal overflow: {extent} offenders={offenders} at {width}x{height}")
                title=await metrics(page, ".hero-intro h1")
                drop=await metrics(page, "#dropzone")
                assert title["width"] > 130 and drop["width"] > 180, f"Awaken content collapsed at {width}x{height}"
                assert title["right"] <= width+2 and drop["right"] <= width+2, f"Awaken content cut off at {width}x{height}"
                intro = await metrics(page, "#introduction")
                chamber = await metrics(page, "#awakenChamber")
                assert intro["height"] >= height-2, f"Landing no longer fills first viewport: {intro}"
                assert chamber["y"] >= intro["bottom"]-2, f"Chamber overlaps introductory section: {chamber}"
                assert drop["y"] >= chamber["y"], f"Source input not located in second section: {drop}"
                assert not await page.locator("#introduction .matrix-trace").count(), "Typed code belongs only to character intake"
                assert not await page.locator(".matrix-column").count(), "Old CSS rain must not compete with the global renderer"
                intake_title=(await page.locator(".chamber-header h2").inner_text()).strip()
                assert "START" in intake_title and "HERE." in intake_title, f"Intake copy drift: {intake_title}"
                headline=(await page.locator(".hero-intro h1").inner_text()).strip()
                assert "BRING YOUR" in headline and "TO LIFE." in headline, f"BUSTED consumer promise drift: {headline}"
                assert await page.locator("#structuralCanvas").is_visible(), "Procedural reference carrier absent"
                canvas_evidence=await page.locator("#structuralCanvas").evaluate("""canvas => {
                  const ctx=canvas.getContext('2d');
                  const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
                  let occupied=0;
                  for(let i=3;i<pixels.length;i+=4)if(pixels[i]>40)occupied++;
                  return {occupied,width:canvas.width,height:canvas.height};
                }""")
                assert canvas_evidence["occupied"]>350, f"Spatial carrier produced no geometry: {canvas_evidence}"
                assert await page.locator("#compileMonitor").is_hidden(), "Compiler console must remain contextual"
                assert await page.locator("#bustedSpatialRain").is_visible(), "Global data rain not created"
                rain=await page.locator("#bustedSpatialRain").evaluate("""canvas=>{
                  const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
                  let ink=0;
                  for(let i=3;i<pixels.length;i+=24)if(pixels[i]>8)ink++;
                  return {width:canvas.width,height:canvas.height,ink};
                }""")
                assert rain["ink"]>80, f"Data rain has no visible glyphs: {rain}"
                assert abs(drop["x"]+drop["width"]/2-(chamber["x"]+chamber["width"]/2))<=35, (
                    f"Character input is not centered: {drop} {chamber}"
                )
                assert await page.locator("#awakenChamber #matrixTypewriter").count()==1
                assert await page.locator("#introduction #matrixTypewriter").count()==0
                layers=await page.evaluate("""()=>{
                  const z=selector=>Number(getComputedStyle(document.querySelector(selector)).zIndex);
                  const bodyRain=document.querySelector('body > #bustedSpatialRain');
                  const userInterface=document.querySelector('#awakenChamber > .chamber-inner');
                  const typing=document.querySelector('#awakenChamber > #matrixTypewriter');
                  const shell=document.querySelector('main.awaken-shell');
                  return {rainBody:bodyRain?.parentElement===document.body,
                    rain:z('#bustedSpatialRain'),shell:z('.awaken-shell'),
                    typing:z('#awakenChamber > #matrixTypewriter'),
                    ui:z('#awakenChamber > .chamber-inner'),
                    pointerEvents:getComputedStyle(bodyRain).pointerEvents,
                    codePointerEvents:getComputedStyle(typing).pointerEvents,
                    labelAtCenter:!!document.elementFromPoint(
                      document.querySelector('#dropzone').getBoundingClientRect().left+
                      document.querySelector('#dropzone').getBoundingClientRect().width/2,
                      document.querySelector('#dropzone').getBoundingClientRect().top+
                      document.querySelector('#dropzone').getBoundingClientRect().height/2
                    )?.closest('#dropzone')};
                }""")
                assert layers["rainBody"] and layers["rain"] < layers["shell"], f"Rain is above main shell: {layers}"
                assert layers["typing"] < layers["ui"], f"Typing is above product UI: {layers}"
                assert layers["pointerEvents"]=="none" and layers["codePointerEvents"]=="none", (
                    f"Decorative layers intercept user interaction: {layers}"
                )
                code=await page.locator("#awakenChamber .matrix-trace-line").all_inner_texts()
                nonempty=[x.strip() for x in code if x.strip()]
                assert len(nonempty)>=2, f"No visible source-shaped code in Awaken: {code}"
                assert len(set(nonempty))>=2, f"Repeated fallback code in Awaken: {code}"
                safe=await page.locator("#awakenChamber").evaluate("""scope=>{
                  const blocks=['.chamber-header','#dropzone','#inputSlotLedger',
                    '.chamber-console .compile-action','.chamber-foot']
                    .map(s=>scope.querySelector(s)).filter(Boolean)
                    .map(el=>el.getBoundingClientRect());
                  const traces=Array.from(scope.querySelectorAll('.matrix-trace')).filter(el=>
                    getComputedStyle(el).display!=='none').map(el=>el.getBoundingClientRect());
                  const collisions=traces.flatMap(t=>blocks.filter(b=>
                    t.left<b.right+5&&t.right>b.left-5&&t.top<b.bottom+5&&t.bottom>b.top-5
                  ).map(b=>({code:{x:t.x,y:t.y},ui:{x:b.x,y:b.y}})));
                  return {visible:traces.length,collisions};
                }""")
                assert safe["visible"]>=1, f"No safe visible typing track: {safe}"
                assert not safe["collisions"], f"Typed code overprints intake UI: {safe}"
                # Previously legacy transform:none!important forced every spirit to
                # the same origin. Verify actual independently placed rectangles.
                spatial = await page.locator("#awakenChamber .spatial-code-spirit").evaluate_all("""nodes =>
                  nodes.map(node => {
                    const b=node.getBoundingClientRect();
                    return {x:Math.round(b.x), y:Math.round(b.y),
                      transform:getComputedStyle(node).transform};
                  })
                """)
                assert spatial and all(p["transform"] != "none" for p in spatial), (
                    f"Spatial code transforms are being overridden: {spatial}"
                )
                if width >= 960:
                    xs = {round(p["x"]/30) for p in spatial}
                    assert len(xs) >= 2, f"Spatial code collapsed into one column: {spatial}"
                stage_rain = await page.locator("#awakenChamber > canvas.busted-stage-rain").evaluate("""canvas => {
                  const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
                  let ink=0;
                  for(let i=3;i<pixels.length;i+=16) if(pixels[i]>20) ink++;
                  return {width:canvas.width,height:canvas.height,ink};
                }""")
                assert stage_rain["ink"] > 30, f"Stage-local Matrix rain rendered blank: {stage_rain}"
                # Offline CTA lets visitors view the interaction, but may
                # NEVER submit a real compile or present ProductProof.
                await page.wait_for_function("""() =>
                  document.querySelector('#awaken')?.dataset.ignitionPreview === 'true' &&
                  document.querySelector('#awakenIgnitionCanvas') !== null
                """, timeout=5000)
                await page.evaluate("""() => {
                  window.__ignitionCompilePosts=0;
                  const original=window.fetch.bind(window);
                  window.fetch=(url,options={})=>{
                    if(String(url).includes('/api/compile/start'))
                      window.__ignitionCompilePosts++;
                    return original(url,options);
                  };
                }""")
                button=page.locator("#awaken")
                assert await button.is_enabled(), "Offline visual ignition cannot be pressed"
                resting=await button.evaluate("""button => {
                  const face=button.getBoundingClientRect();
                  const source=document.querySelector('#dropzone').getBoundingClientRect();
                  const parent=button.closest('.compile-action').getBoundingClientRect();
                  const style=getComputedStyle(button);
                  return {
                    width:face.width,height:face.height,
                    sourceWidth:source.width,parentWidth:parent.width,
                    background:style.backgroundImage,phase:button.dataset.ignitionPhase || 'rest',
                    canvasHidden:document.querySelector('#awakenIgnitionCanvas').hidden
                  };
                }""")
                assert abs(resting["width"]-resting["sourceWidth"]) < 2.5, (
                    f"AWAKEN must match actual character source viewport, not the console: {resting}"
                )
                assert resting["width"] <= resting["parentWidth"]+2
                assert resting["canvasHidden"], "There must be NO ambient reactor rendering"
                assert resting["phase"] == "rest"
                assert "linear-gradient" in resting["background"]
                assert "20, 29, 66" in resting["background"], (
                    f"AWAKEN must rest in cold cobalt, not fiery gradients: {resting}"
                )

                # Reduced-motion mode: short static ignition, never a
                # stroboscopic Canvas, and always reset after the click.
                await page.evaluate("""() => {
                  const b=document.querySelector('#awaken');
                  window.__ignitionPhases=[];
                  new MutationObserver(() => {
                    window.__ignitionPhases.push(b.dataset.ignitionPhase || 'rest');
                  }).observe(b,{attributes:true,attributeFilter:['data-ignition-phase']});
                }""")
                await button.click()
                await page.wait_for_function("""() =>
                  window.__ignitionPhases?.includes('igniting') &&
                  window.__ignitionPhases?.includes('rest')
                """, timeout=5000)
                assert await page.locator("#awakenIgnitionCanvas").is_hidden(), (
                    "Reduced-motion ignition must not render a Canvas animation"
                )
                assert await page.evaluate("window.__ignitionCompilePosts") == 0, (
                    "Ignition preview must never call /api/compile/start"
                )
                assert await page.locator("#compileAvailability").inner_text() == "UI PREVIEW · COMPILER OFFLINE"
                assert not await page.locator("#result").evaluate("el=>el.classList.contains('show')"), (
                    "Visual ignition cannot invent a compiled puppet"
                )

                # Studio link is real navigation, even without a compiled bundle.
                await page.locator(".studio-link").click()
                await page.locator(".workspace").wait_for()
                await page.locator("#requalificationStrip").wait_for()
                assert "Studio" in (await page.title()), f"Studio navigation failed: {page.url}"
                assert "NO QUALIFIED CHARACTER" in (await page.locator("#requalTitle").inner_text()), "Empty Studio must never claim qualification"
                assert "PROOF PASS" not in (await page.locator("#requalDomains").inner_text()), "Empty Studio falsely reports ProductProof PASS"
                assert await page.locator("#exportRuntime").is_disabled(), "Empty Studio must not export unqualified content"
                status_rect=await metrics(page, "#requalificationStrip")
                viewport_rect=await metrics(page, ".workspace")
                assert status_rect["bottom"] <= viewport_rect["y"]+2, f"Requalification strip is outside Studio header: {status_rect} {viewport_rect}"
                assert await page.locator("#bustedSpatialRain").is_visible(), "Rain missing in Puppet Editor"
                assert await page.locator(".matrix-trace").count()==0, "Typed code leaked into Puppet Editor"
                await page.screenshot(path=str(OUT / f"studio_{width}x{height}.png"), full_page=True)

                rows=await page.evaluate("""() => ({
                  scroll:document.documentElement.scrollWidth,
                  inner:window.innerWidth,
                  height:window.innerHeight,
                  studio:document.querySelector('.workspace').getBoundingClientRect().toJSON(),
                  timeline:document.querySelector('.timeline').getBoundingClientRect().toJSON(),
                  topbar:document.querySelector('.topbar').getBoundingClientRect().toJSON()
                })""")
                assert rows["scroll"] <= width+2, f"Studio horizontal overflow {rows} at {width}x{height}"
                assert rows["timeline"]["bottom"] <= height+2, f"Timeline extends offscreen {rows} at {width}x{height}"
                assert rows["studio"]["height"] >= 180, f"Studio viewport too short {rows} at {width}x{height}"
                assert rows["topbar"]["height"] <= 60, f"Studio topbar too large {rows} at {width}x{height}"
                if width <= 870:
                    control=page.locator("#toggleScenePanel")
                    assert await control.is_visible(), "Mobile scene control hidden"
                    await control.click()
                    assert await control.get_attribute("aria-expanded") == "true"
                    panel=await metrics(page, "#scenePanel")
                    assert panel["x"] > -5 and panel["width"] >= 220, f"Scene panel inaccessible: {panel}"
                    await page.keyboard.press("Escape")
                    assert await control.get_attribute("aria-expanded") == "false"
                # Advanced tools are progressive, and the animation timeline is opt-in.
                assert await page.locator("#studioWelcome").is_visible(), "Empty editor requires understandable onboarding"
                canvas_rect = await metrics(page, ".viewport-panel")
                welcome_rect = await metrics(page, "#studioWelcome")
                assert canvas_rect["width"] >= width*.72, f"Editor canvas collapsed: {canvas_rect} at {width}x{height}"
                assert welcome_rect["width"] >= width*.67, f"Editor onboarding collapsed: {welcome_rect} at {width}x{height}"
                assert not await page.locator("#advancedStages").get_attribute("open"), "Stage inspector should default collapsed"
                assert await page.locator("#toggleTimeline").get_attribute("aria-pressed") == "false"
                await page.locator("#toggleTimeline").click()
                assert await page.locator("#toggleTimeline").get_attribute("aria-pressed") == "true"
                await page.locator("#toggleTimeline").click()

                # One qualified puppet will back two linked authoring workspaces.
                # The empty shell may switch contexts but must never fake authoring.
                director = page.locator("#studioDirector")
                assert await director.count() == 1, "Studio director tabs missing"
                assert await director.locator('[data-director-command="add-key"]').is_disabled()
                await director.locator('[data-director-workspace="motion"]').click()
                assert await page.locator("body").get_attribute("data-studio-workspace") == "motion"
                assert await page.locator("#toggleTimeline").get_attribute("aria-pressed") == "true", (
                    "Motion workspace must expose the real animation timeline"
                )
                assert await director.locator('[data-director-command="new-clip"]').is_disabled(), (
                    "Unqualified puppet must not be editable"
                )
                assert await director.locator('[data-director-command="ik-reach"]').is_disabled(), (
                    "IK reach cannot be offered without qualified rig, user clip and mapped controls"
                )
                assert await page.locator("#studioIKReachOverlay").count() == 1, (
                    "IK Reach interaction layer did not initialize"
                )
                await director.locator('[data-director-workspace="puppet"]').click()
                assert await page.locator("body").get_attribute("data-studio-workspace") == "puppet"
                assert await page.locator("#toggleTimeline").get_attribute("aria-pressed") == "false"
                assert await page.locator("#requalTitle").inner_text() == "NO QUALIFIED CHARACTER"

                # Motion Lab remains the baked-clip playback area, while Studio
                # Motion workspace authors user keys on the same puppet.
                for area, title in [("motion", "Motion Lab"), ("export", "Export")]:
                    await page.goto(BASE + "/" + area, wait_until="networkidle", timeout=25000)
                    assert title in (await page.title()), f"Missing {title} at {width}x{height}"
                    assert await page.locator("#bustedSpatialRain").is_visible(), f"Rain missing on {area}"
                    assert await page.locator(".matrix-trace").count()==0, f"Typed code leaked into {area}"
                    ext = await page.evaluate("document.documentElement.scrollWidth")
                    assert ext <= width + 2, f"{title} document overflow ({ext}) at {width}x{height}"
                    assert await page.locator("#proofStatus").inner_text() == "NO QUALIFIED PUPPET"
                    if area == "motion":
                        assert await page.locator("#motionPlay").is_disabled(), "Motion must be fail-closed without a puppet"
                        assert await page.locator("#presetShelf button").count() == 8
                        assert await page.locator("#presetShelf button:not([disabled])").count() == 0
                    else:
                        assert await page.locator("#prepareExport").is_disabled(), "Export must be fail-closed without proof"
                    await page.screenshot(path=str(OUT / f"{area}_{width}x{height}.png"), full_page=True)
                if errors:
                    raise AssertionError(f"Browser runtime errors at {width}x{height}: {errors[:4]}")
                print(f"PASS {width}x{height} Awaken + Puppet Editor + Motion Lab + Export")
            except Exception as error:
                failures.append(f"{width}x{height}: {error}")
                print("FAIL", failures[-1])
            finally:
                await page.close()
        # Verify actual progressive left-to-right typing with motion enabled.
        animated = await browser.new_page(
            viewport={"width": 1366, "height": 768},
            reduced_motion="no-preference"
        )
        try:
            await animated.goto(BASE + "/", wait_until="networkidle", timeout=25000)
            await animated.locator("#awakenChamber").scroll_into_view_if_needed()
            initial = await animated.locator('#awakenChamber .matrix-trace-line').all_inner_texts()
            await animated.wait_for_timeout(950)
            later = await animated.locator('#awakenChamber .matrix-trace-line').all_inner_texts()
            assert len(initial)==len(later) and any(a!=b for a,b in zip(initial,later)), (
                f"Code does not progress over time: {initial!r} -> {later!r}"
            )
            await animated.screenshot(path=str(OUT / "awaken_animated_chamber_1366x768.png"))
            # Normal-motion witness: the real VFX Canvas must contain energy,
            # in exactly the button's bounding box, for one click only.
            await animated.wait_for_function("""() =>
              document.querySelector('#awaken')?.dataset.ignitionPreview === 'true'
            """, timeout=5000)
            await animated.evaluate("""() => {
              const button=document.querySelector('#awaken');
              const canvas=document.querySelector('#awakenIgnitionCanvas');
              window.__reactorProof={phases:[],maxInk:0,maxAlpha:0,dimension:null};
              const witness=window.__reactorProof;
              new MutationObserver(()=>{
                witness.phases.push(button.dataset.ignitionPhase||'rest');
              }).observe(button,{attributes:true,attributeFilter:['data-ignition-phase']});
              const measure=()=>{
                if(!canvas.hidden&&canvas.width>0&&canvas.height>0){
                  const context=canvas.getContext('2d');
                  const w=canvas.width,h=canvas.height;
                  const px=context.getImageData(0,0,w,h).data;
                  let ink=0,maxAlpha=0;
                  for(let i=3;i<px.length;i+=32){
                    if(px[i]>16)ink++;
                    maxAlpha=Math.max(maxAlpha,px[i]);
                  }
                  witness.maxInk=Math.max(witness.maxInk,ink);
                  witness.maxAlpha=Math.max(witness.maxAlpha,maxAlpha);
                  const face=button.getBoundingClientRect();
                  witness.dimension={width:face.width,height:face.height,
                    canvasWidth:canvas.getBoundingClientRect().width,
                    canvasHeight:canvas.getBoundingClientRect().height};
                }
                if(button.dataset.ignitionPhase!=='rest' ||
                   !witness.phases.includes('rest'))
                  requestAnimationFrame(measure);
              };
              requestAnimationFrame(measure);
            }""")
            await animated.locator("#awaken").click()
            await animated.wait_for_function("""() => {
              const proof=window.__reactorProof;
              return proof?.phases.includes('charging') &&
                proof?.phases.includes('igniting') &&
                proof?.phases.includes('recovering') &&
                proof?.phases.includes('rest');
            }""", timeout=7000)
            reactor=await animated.evaluate("window.__reactorProof")
            assert reactor["maxInk"] > 50 and reactor["maxAlpha"] > 40, (
                f"Reactor never painted a visible energy sequence: {reactor}"
            )
            assert abs(reactor["dimension"]["width"]-reactor["dimension"]["canvasWidth"]) <= 2
            assert abs(reactor["dimension"]["height"]-reactor["dimension"]["canvasHeight"]) <= 2
            assert await animated.locator("#awakenIgnitionCanvas").is_hidden()
            print("PASS 1366x768 animated Matrix typing + one-shot reactor ignition")
        except Exception as error:
            failures.append(f"animated typewriter: {error}")
            print("FAIL", failures[-1])
        finally:
            await animated.close()
        await browser.close()
    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(run())
