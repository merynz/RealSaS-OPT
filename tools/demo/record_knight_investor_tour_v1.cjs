/* Deterministic 30 FPS capture of the real exported offline inspector. */
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url'),{spawn}=require('child_process'),{once}=require('events');
const {chromium}=require('playwright');
const args=process.argv.slice(2);function option(n){const i=args.indexOf(n);if(i<0)throw Error('Missing '+n);return args[i+1];}
(async()=>{
 const root=path.resolve(option('--demo-path')),destination=path.resolve(option('--output'));
 const browser=await chromium.launch({executablePath:option('--browser-executable'),headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader','--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:1});
 // Freeze wall-clock scheduling for export only. Every recorded frame still
 // executes the inspector's actual skinning, attachments, drawing and controls.
 await page.addInitScript(()=>{window.requestAnimationFrame=()=>0;});
 await page.goto(pathToFileURL(path.join(root,'START_DEMO.html')).href);
 await page.waitForFunction(()=>window.demo&&document.getElementById('loading').classList.contains('hidden'));
 const indices=await page.evaluate(()=>({run:REALSAS_DEMO.clips.findIndex(c=>c.kind==='RUN'),slash:REALSAS_DEMO.clips.findIndex(c=>c.kind==='SLASH')}));
 const temporary=destination.replace(/\.mp4$/,'.writing.mp4');
 const ffmpeg=spawn('ffmpeg',['-y','-v','error','-xerror','-f','image2pipe','-vcodec','png','-framerate','30','-i','pipe:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2',temporary],{stdio:['pipe','ignore','pipe']});
 let log='';ffmpeg.stderr.on('data',b=>log+=b.toString());const completed=once(ffmpeg,'close');
 for(let frame=0;frame<900;frame++){
  const second=frame/30,view=second<18?6:second<22?1:second<26?5:6,clip=second<26?indices.run:indices.slash;
  await page.evaluate(({frame,second,view,clip})=>{
   if(demo.state.clip!==clip)demo.setState({clip});
   if(demo.state.view!==view)demo.setState({view});
   let mode=second<2?'artwork':second<10?'combined':second<17?'weights':'combined';
   if(demo.state.mode!==mode)demo.selectMode(mode);
   if(second>=2&&second<7){let slider=document.getElementById('reveal');slider.value=Math.round((second-2)/5*78);slider.dispatchEvent(new Event('input'));}
   if(second>=10&&second<17){let input=document.getElementById('contribution');input.checked=second>=13;input.dispatchEvent(new Event('change'));}
   const index=second<26?frame%REALSAS_DEMO.clips[clip].previewCount:Math.round((second-26)*30);
   demo.setState({frame:index,playing:true,bone:10});
  },{frame,second,view,clip});
  await page.waitForFunction(()=>REALSAS_DEMO.clips[demo.state.clip].images[demo.state.view]?.naturalWidth>0);
  await page.evaluate(()=>demo.draw());
  const buffer=await page.screenshot({type:'png'});
  if(!ffmpeg.stdin.write(buffer))await once(ffmpeg.stdin,'drain');
  if(frame%150===0)console.log('TOUR_FRAME',frame);
 }
 ffmpeg.stdin.end();const [code]=await completed;if(code!==0)throw Error(log||'FFmpeg tour encoding failed');
 fs.renameSync(temporary,destination);await browser.close();console.log('TOUR_READY',destination);
})().catch(e=>{console.error(e);process.exit(1);});
