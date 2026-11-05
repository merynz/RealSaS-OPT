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
                assert await page.locator("#knightShowcase").is_visible(), "Knight technical stage missing"
                await page.wait_for_function("window.bustedKnightShowcase?.nativeReady === true", timeout=12000)
                art=await page.locator("#knightMotionFrame").evaluate("""img => ({
                    complete:img.complete,naturalWidth:img.naturalWidth,naturalHeight:img.naturalHeight,
                    src:img.getAttribute('src')
                })""")
                assert art["complete"] and art["naturalWidth"]>0, f"Authentic Knight media not decoded: {art}"
                assert await page.locator('#knightBoneSelect option').count()==6, "Native sample bones unavailable"
                await page.locator('[data-knight-mode="rig"]').click()
                assert await page.locator("#knightShowcase").get_attribute("data-layer")=="rig"
                rig=await page.locator("#knightRigCanvas").evaluate("""canvas=>{
                    const pixels=canvas.getContext('2d').getImageData(0,0,512,512).data;
                    let ink=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i]>40)ink++;
                    return ink;
                }""")
                assert rig>75, f"Native rig did not render: {rig}"
                await page.locator('[data-knight-mode="weights"]').click()
                await page.locator("#knightBoneSelect").select_option("10")
                assert await page.locator("#knightShowcase").get_attribute("data-layer")=="weights"
                await page.locator('[data-knight-motion="run"]').click()
                assert "run.gif" in (await page.locator("#knightMotionFrame").get_attribute("src"))
                await page.locator('[data-knight-motion="slash"]').click()
                assert "slash.gif" in (await page.locator("#knightMotionFrame").get_attribute("src"))
                await page.locator('[data-knight-motion="idle"]').click()
                await page.locator('[data-knight-mode="art"]').click()
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
                assert await page.locator("#awaken").is_disabled(), "Offline preview must never offer a fake compile"


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

                # Two separate product areas, not fake features in a crowded editor.
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
            print("PASS 1366x768 animated Matrix typing: characters progressively appear")
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
