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
                if width >= 781:
                    assert title["right"] < drop["x"]+12, f"Desktop Awaken columns overlap at {width}x{height}: {title} {drop}"
                else:
                    assert title["bottom"] <= drop["y"]+12, f"Mobile Awaken hierarchy overlaps at {width}x{height}"
                # Matrix VFX uses reserved screen-space lanes: no typed-code collisions.
                # Motion reduction freezes the CSS streams for reproducible geometry.
                matrix = await page.locator(".matrix-column").evaluate_all("""elements => elements.map(el => {
                  const r=el.getBoundingClientRect();
                  return {lane:Number(el.dataset.lane), x:r.x, right:r.right, width:r.width,
                    depth:Number(el.dataset.depth)};
                }).sort((a,b)=>a.x-b.x)""")
                assert len(matrix) == (13 if width<=780 else 30), f"Unexpected number of matrix lanes at {width}: {len(matrix)}"
                assert all(0<=m["depth"]<=1 for m in matrix), "Invalid 3D depth parameter"
                for left,right in zip(matrix,matrix[1:]):
                    assert left["right"] <= right["x"] + 1, f"Matrix text lanes overlap at {width}: {left} vs {right}"
                assert await page.locator(".matrix-trace").count() == 0, "Unbounded floating code fragments should not be present"
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
        await browser.close()
    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(run())
