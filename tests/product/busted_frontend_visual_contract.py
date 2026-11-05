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
                assert await page.locator(".chamber-code-visual").count() == 0, "Unwanted separate decorative code panel"
                assert (await page.locator(".chamber-header h2").inner_text()).strip() == "YOUR ART. IN.", "Unexpected intake hierarchy"
                assert "WE COMPILE" in (await page.locator(".hero-intro h1").inner_text()), "Compiler identity missing"
                assert await page.locator("#structuralCanvas").is_visible(), "Procedural carrier absent"
                canvas_evidence=await page.locator("#structuralCanvas").evaluate("""canvas => {
                  const ctx=canvas.getContext('2d');
                  const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
                  let occupied=0;
                  for(let i=3;i<pixels.length;i+=4)if(pixels[i]>40)occupied++;
                  return {occupied,width:canvas.width,height:canvas.height};
                }""")
                assert canvas_evidence["occupied"]>350, f"Spatial carrier produced no geometry: {canvas_evidence}"
                assert await page.locator("#compileMonitor").is_hidden(), "Compiler console must not appear before real compilation"
                # Two independent ambient streams, measured from rendered text rather than markup only.
                for section, n in [("#introduction", 4 if width > 780 else 2),
                                   ("#awakenChamber", 8 if width > 780 else 4)]:
                    assert await page.locator(section+" .matrix-trace").count() == n, (
                        f"{section}: missing typing tracks"
                    )
                    assert await page.locator(section+" .matrix-column").count() == (30 if width > 780 else 13) if section == "#awakenChamber" else (
                        await page.locator(section+" .matrix-column").count() == (23 if width > 780 else 9)
                    )
                    code = await page.locator(section+" .matrix-trace-line").all_inner_texts()
                    nonempty = [line.strip() for line in code if line.strip()]
                    assert len(nonempty) >= n, f"{section}: empty code lanes: {code}"
                    assert len(set(nonempty)) >= max(3,n//2), f"{section}: repeated fallback code: {code}"
                    assert all("." in x or "(" in x for x in nonempty), f"{section}: code isn't readable: {code}"
                    tracks = await page.locator(section+" .matrix-trace").evaluate_all("""nodes => nodes.map(el=>{
                        const r=el.getBoundingClientRect();
                        const container=el.closest('section').getBoundingClientRect();
                        return {x:r.left,right:r.right,y:r.top,bottom:r.bottom,
                                left:container.left,limit:container.right,
                                top:container.top,bound:container.bottom}
                    })""")
                    assert all(v["x"] >= v["left"]-3 and v["right"] <= v["limit"]+3 for v in tracks), (
                        f"{section}: code is cut off at edges: {tracks}"
                    )
                    safety = await page.locator(section).evaluate("""(scope)=>{
                      const selectors=scope.id==='introduction'
                        ? ['.hero-eyebrow','.hero-intro h1','.hero-copy','.hero-description','.hero-meta','.descend-link','.landing-foot']
                        : ['.chamber-header','.chamber-docking-label','.dropzone',
                           '.input-slot-ledger','.chamber-console .console-heading',
                           '.chamber-console h3','.chamber-console .console-copy',
                           '#awaken','#compileAvailability','.chamber-console .telemetry'];
                      const blockers=selectors.map(s=>scope.querySelector(s)).filter(Boolean)
                        .flatMap(el=>{
                          const inkTargets=['.hero-eyebrow','.hero-intro h1','.hero-copy',
                            '.hero-description','.hero-meta'];
                          if(!inkTargets.some(selector=>el.matches(selector))){
                            return [el.getBoundingClientRect()];
                          }
                          // Range.selectNodeContents(element) includes block-wide child
                          // boxes. Measure actual individual text nodes instead.
                          const walker=document.createTreeWalker(el,NodeFilter.SHOW_TEXT);
                          const result=[];
                          let node;
                          while((node=walker.nextNode())){
                            if(!node.textContent.trim())continue;
                            const range=document.createRange();
                            range.selectNodeContents(node);
                            for(const ink of range.getClientRects()){
                              if(ink.width>0&&ink.height>0)result.push(ink);
                            }
                          }
                          return result.length?result:[el.getBoundingClientRect()];
                        });
                      const tracks=Array.from(scope.querySelectorAll('.matrix-trace'))
                        .filter(el=>getComputedStyle(el).display!=='none')
                        .map(el=>el.getBoundingClientRect());
                      const collisions=tracks.flatMap(track=>blockers.filter(b=>
                        track.left<b.right+3 && track.right>b.left-3 &&
                        track.top<b.bottom+3 && track.bottom>b.top-3
                      ).map(b=>({track:{x:track.x,y:track.y},block:{x:b.x,y:b.y}})));
                      return {visible:tracks.length,collisions};
                    }""")
                    assert safety["visible"]>=1, f"{section}: no ambient code remains visible"
                    assert not safety["collisions"], f"{section}: ambient code overprints UI: {safety}"

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
            first = animated.locator('#awakenChamber .matrix-trace[data-slot="0"] .matrix-trace-line').first
            initial = await first.inner_text()
            await animated.wait_for_timeout(850)
            later = await first.inner_text()
            assert len(later) > len(initial), (
                f"Code does not visibly type left-to-right: {initial!r} -> {later!r}"
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
