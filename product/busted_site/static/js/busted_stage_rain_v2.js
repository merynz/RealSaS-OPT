/* BUSTED / Stage Rain v2.2 — distant, rate-limited glyph field
 * Foreground atmospheric coding weather: stage-local canvas, safely under UI.
 * Ambient simulation only. Independent velocity, x-drift and depth. */
(() => {
  'use strict';
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const glyphs='0123456789ABCDEF<>[]{}:;/=+-';
  const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
  const random=(a,b)=>a+Math.random()*(b-a);
  const symbol=()=>glyphs[Math.floor(Math.random()*glyphs.length)];
  const scenes=[
    {host:document.querySelector('#introduction'),kind:'hero'},
    {host:document.querySelector('#awakenChamber'),kind:'intake'}
  ].filter(s=>Boolean(s.host));
  if(!scenes.length)return;
  const observer=new IntersectionObserver(entries=>{
    for(const entry of entries){
      const scene=scenes.find(s=>s.host===entry.target);
      if(scene){scene.visible=entry.isIntersecting;toggle()}
    }
  },{threshold:.01});

  function rebuild(scene){
    const rect=scene.host.getBoundingClientRect();
    scene.width=Math.max(1,Math.round(rect.width));
    scene.height=Math.max(1,Math.round(scene.host.offsetHeight));
    scene.dpr=Math.min(devicePixelRatio||1,1.15);
    scene.canvas.width=Math.round(scene.width*scene.dpr);
    scene.canvas.height=Math.round(scene.height*scene.dpr);
    scene.canvas.style.width=scene.width+'px';
    scene.canvas.style.height=scene.height+'px';
    scene.ctx.setTransform(scene.dpr,0,0,scene.dpr,0,0);
    scene.lanes=[];
    const narrow=scene.width<760;
    // Half the visible streams, not their velocity / typography timing.
    const count=narrow?(scene.kind==='hero'?13:12):(scene.kind==='hero'?36:32);
    for(let i=0;i<count;i++){
      const depth=Math.pow(Math.random(),.8);
      const font=(narrow?9:9.7)+depth*(narrow?3.2:4.7)+random(-.45,.45);
      const space=font*random(1.20,1.50);
      const length=Math.round(random(10,25)*(.78+depth*.40));
      scene.lanes.push({
        x:random(4,scene.width-4),offset:random(-length*space,scene.height+length*space),
        depth,font,space,length,
        speed:random(23,46)*(.52+depth*1.45),
        phase:random(0,Math.PI*2),
        sway:random(1,8),
        light:Math.random()>.84,
        letters:Array.from({length},symbol)
      });
    }
    draw(scene);
  }

  function draw(scene){
    const ctx=scene.ctx;
    ctx.clearRect(0,0,scene.width,scene.height);
    const t=reduced.matches?0:scene.clock/1000;
    ctx.textAlign='center';ctx.textBaseline='middle';
    for(const lane of scene.lanes){
      const extent=lane.length*lane.space;
      const span=scene.height+extent*1.65;
      const raw=lane.offset+t*lane.speed+extent;
      const head=((raw%span)+span)%span-extent;
      const px=lane.x+Math.sin(t*.2+lane.phase)*lane.sway*lane.depth;
      const col=lane.light?[207,234,255]:lane.depth>.58?[164,190,255]:[109,135,224];
      const opacityScale=lane.depth>.65?.80:lane.depth>.3?.64:.48;
      ctx.font=`500 ${lane.font.toFixed(1)}px "IBM Plex Mono", monospace`;
      for(let n=0;n<lane.length;n++){
        const y=head-n*lane.space;
        if(y<-25||y>scene.height+25)continue;
        const fade=Math.pow(1-n/lane.length,.78);
        let alpha=fade*opacityScale;
        if(n===0)alpha=Math.min(1,alpha*1.45);
        if(alpha<.018)continue;
        ctx.fillStyle=`rgba(${col.join(',')},${alpha.toFixed(3)})`;
        ctx.shadowBlur=n===0&&lane.light?4:0;
        ctx.shadowColor='rgba(161,167,255,.55)';
        ctx.fillText(lane.letters[n],px,y);
      }
    }
    ctx.shadowBlur=0;
  }

  let raf=0,prev=0;
  function render(now){
    raf=0;
    if(document.hidden||reduced.matches)return;
    const dt=prev?clamp(now-prev,0,90):0;
    prev=now;
    for(const scene of scenes){
      if(!scene.visible)continue;
      scene.clock+=dt;
      if(scene.clock-scene.lastPaint>(scene.width<760?86:62)){
        draw(scene);
        scene.lastPaint=scene.clock;
      }
    }
    if(scenes.some(s=>s.visible))raf=requestAnimationFrame(render);
  }
  function toggle(){
    if(raf){cancelAnimationFrame(raf);raf=0}
    prev=0;
    if(!document.hidden&&!reduced.matches&&scenes.some(s=>s.visible))
      raf=requestAnimationFrame(render);
    else scenes.forEach(draw);
  }
  const ro=new ResizeObserver(()=>{scenes.forEach(rebuild);toggle()});
  for(const scene of scenes){
    scene.canvas=document.createElement('canvas');
    scene.canvas.className='busted-stage-rain';
    scene.canvas.setAttribute('aria-hidden','true');
    scene.host.insertBefore(scene.canvas,scene.host.firstChild);
    scene.ctx=scene.canvas.getContext('2d',{alpha:true});
    scene.clock=0;scene.lastPaint=-1;scene.visible=false;
    rebuild(scene);observer.observe(scene.host);ro.observe(scene.host);
  }
  reduced.addEventListener?.('change',toggle);
  document.addEventListener('visibilitychange',toggle);
  addEventListener('pagehide',()=>{
    if(raf)cancelAnimationFrame(raf);
    ro.disconnect();observer.disconnect();
  },{once:true});
  toggle();
})();
