/* Read-only checks against an exported offline inspector, not a synthetic rig. */
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('playwright');
const args = process.argv.slice(2);
function option(name) { const i=args.indexOf(name); if(i<0)throw Error('Missing '+name);return args[i+1]; }
const root=path.resolve(option('--demo-path'));
const executablePath=option('--browser-executable');
function assert(value, message){if(!value)throw Error(message);}
(async()=>{
 const browser=await chromium.launch({executablePath,headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader','--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:1});
 const errors=[], external=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(/^https?:/.test(r.url()))external.push(r.url());});
 await page.goto(pathToFileURL(path.join(root,'START_DEMO.html')).href);
 await page.waitForFunction(()=>window.demo&&document.getElementById('loading').classList.contains('hidden'));
 const counts=await page.evaluate(()=>demo.counts);
 assert(counts.clips===7,'Seven native motions required');
 let maximum=0,samples=0;
 for(let ci=0;ci<counts.clips;ci++){
  for(let view=0;view<8;view++){
   await page.evaluate(({ci,view})=>demo.setState({clip:ci,view,playing:false}),{ci,view});
   await page.waitForFunction(()=>REALSAS_DEMO.clips[demo.state.clip].images[demo.state.view]?.naturalWidth>0);
   for(const mode of ['artwork','rig','mesh','weights','combined','depth']){
    await page.evaluate(mode=>demo.selectMode(mode),mode);
   }
  }
  const check=await page.evaluate(()=>{
   const c=REALSAS_DEMO.clips[demo.state.clip],out=[];
   for(const f of c.referenceFrames){const frame=Array.from(c.indices).indexOf(f);demo.setState({frame,playing:false});out.push(demo.referenceError());}
   return out;
  });
  check.forEach(e=>{assert(e!==null&&e<1e-9,'Browser pose differs from compiled mesh');maximum=Math.max(maximum,e);samples++;});
  await page.evaluate(()=>demo.setState({view:6,frame:Math.min(9,REALSAS_DEMO.clips[demo.state.clip].previewCount-1),mode:'combined',playing:false}));
  await page.waitForFunction(()=>REALSAS_DEMO.clips[demo.state.clip].images[6]?.naturalWidth>0);
  await page.screenshot({path:path.join(root,'verification',`inspector_${ci}.png`)});
 }
 const run=await page.evaluate(()=>REALSAS_DEMO.clips.findIndex(c=>c.kind==='RUN'));
 await page.evaluate(run=>demo.setState({clip:run,frame:0,mode:'artwork',playing:true}),run);
 await page.click('[data-mode="rig"]');
 const before=await page.evaluate(()=>demo.state.frame);
 await page.waitForTimeout(160);
 assert(await page.evaluate(before=>demo.state.playing&&demo.state.frame!==before,before),'Mode switching must not pause playback');
 await page.evaluate(()=>{document.getElementById('reveal').value=70;document.getElementById('reveal').dispatchEvent(new Event('input'));});
 assert(await page.evaluate(()=>demo.state.playing&&document.getElementById('rigToggle').checked&&document.getElementById('meshToggle').checked),'Reveal must retain playback');
 await page.evaluate(()=>demo.setState({frame:9,playing:false,mode:'rig',bone:10}));
 const segment=await page.evaluate(()=>demo.selectedBoneSegments()[0]);
 assert(segment,'Selected bone must have a selectable actual rig segment');
 const box=await page.locator('#rig').boundingBox();
 const px=await page.evaluate(()=>document.getElementById('rig').width);
 await page.mouse.click(box.x+(segment.a[0]+segment.b[0])/2/px*box.width,box.y+(segment.a[1]+segment.b[1])/2/px*box.height);
 assert(await page.evaluate(()=>Number(document.getElementById('bone').value)===demo.state.bone),'Bone click must select matching weight control');
 const death=await page.evaluate(()=>REALSAS_DEMO.clips.findIndex(c=>c.kind==='DEATH'));
 await page.evaluate(death=>{demo.setState({clip:death,frame:REALSAS_DEMO.clips[death].previewCount-2,playing:true});},death);
 await page.waitForTimeout(160);
 assert(await page.evaluate(()=>!demo.state.playing&&demo.state.frame===REALSAS_DEMO.clips[demo.state.clip].previewCount-1),'Death must finish without artificial loop');
 await page.click('#play');assert(await page.evaluate(()=>demo.state.playing),'Play must restart a finished take');
 assert(errors.length===0,'Page errors: '+errors.join(';'));
 assert(external.length===0,'Offline demo made an external request');
 const report={schema:'RealSaS.OfflineInspectorBrowserCheck.v1',counts,checkedDirections:8,checkedModes:6,poseComparisons:samples,maxBrowserPoseResidual:maximum,externalRequests:external.length,pageErrors:errors,modeSwitchPreservesPlayback:true,revealPreservesPlayback:true,boneSelection:true,nonloopTerminalStop:true,finishedTakeRestarts:true};
 fs.writeFileSync(path.join(root,'verification','browser_check.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
