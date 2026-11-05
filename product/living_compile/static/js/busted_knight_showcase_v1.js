/* BUSTED Knight technical showcase.
 * Motion media: genuine RealSaS Knight preview GIF assets from repository.
 * Skeleton/weights: sampled native frame matrices and skin coefficients from
 * RealSaS_Knight_Investor_Demo.zip. Not a qualified ProductRevision.
 * No simulated compile success, no inferred backend model output.
 */
(() => {
  'use strict';
  const root=document.getElementById('knightShowcase');
  if(!root)return;
  const image=document.getElementById('knightMotionFrame');
  const canvas=document.getElementById('knightRigCanvas');
  const select=document.getElementById('knightBoneSelect');
  const readout=document.getElementById('knightModeReadout');
  const evidence=document.getElementById('knightEvidenceText');
  const ctx=canvas?.getContext('2d',{alpha:true});
  if(!image||!ctx||!select)return;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const clips=['idle','run','slash'];
  const names={idle:'Idle',run:'Run',slash:'Slash'};
  const assets={idle:'./static/media/knight/idle.gif',run:'./static/media/knight/run.gif',slash:'./static/media/knight/slash.gif'};
  const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
  const lerp=(a,b,f)=>a+(b-a)*f;
  let source=null,frames=null,motion='idle',layer='art',selected=10;
  let started=performance.now(),lastTick=0,frameID=0,visible=false;
  let sourceError=false;
  function setButtonState(name,value){
    root.querySelectorAll('[data-knight-'+name+']').forEach(el=>{
      const active=el.dataset['knight'+name[0].toUpperCase()+name.slice(1)]===value;
      el.classList.toggle('active',active);
      el.setAttribute('aria-pressed',String(active));
    });
  }
  function changeMotion(next){
    if(!clips.includes(next))return;
    motion=next;started=performance.now();
    if(image.getAttribute('src')!==assets[next])image.setAttribute('src',assets[next]);
    setButtonState('motion',next);
    render(started);
  }
  function changeLayer(next){
    if(!['art','rig','weights'].includes(next))return;
    if(next!=='art'&&(!source||!frames))return;
    layer=next;root.dataset.layer=next;
    readout.textContent=({art:'ARTWORK',rig:'RIG / NATIVE',weights:'WEIGHTS / SAMPLED'})[next];
    select.disabled=next!=='weights';
    setButtonState('mode',next);
    render(performance.now());
  }
  async function loadNativeSamples(){
    const response=await fetch('./static/media/knight/native_rig_samples.json',{cache:'force-cache'});
    if(!response.ok)throw Error('Native Knight data unavailable');
    const data=await response.json();
    if(data.schema!=='BUSTED.KnightShowcase.v1'||data.joints.length!==23||data.samples!==48||data.keyframes!==9){
      throw Error('Unrecognized native Knight sample contract');
    }
    if(typeof DecompressionStream==='undefined')throw Error('Native sample decompressor unavailable');
    const raw=Uint8Array.from(atob(data.compressed),c=>c.charCodeAt(0));
    const decompressed=new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate'));
    const buffer=await new Response(decompressed).arrayBuffer();
    const expected=data.clips.length*data.keyframes*(data.joints.length+data.samples)*2;
    if(buffer.byteLength!==expected*2)throw Error('Knight native sample length mismatch');
    const view=new DataView(buffer),native=new Int16Array(expected);
    for(let i=0;i<expected;i++)native[i]=view.getInt16(i*2,true);
    source=data;frames=native;
    select.replaceChildren();
    data.selected.forEach((bone,i)=>{
      const option=document.createElement('option');option.value=String(bone);
      option.textContent=data.selectedLabels[i];select.appendChild(option);
    });
    selected=data.selected[0];
    select.value=String(selected);
    evidence.textContent='Native rig · 23 joints · 48 skin samples';
    render(performance.now());
  }
  function xy(clip,frame,index){
    const per=(source.joints.length+source.samples)*2;
    const offset=(clip*source.keyframes+frame)*per+index*2;
    return [frames[offset]/16,frames[offset+1]/16];
  }
  function interpolate(clip,time,index){
    const descriptor=source.clips[clip];
    const normalized=descriptor.loop?((time%descriptor.duration)/descriptor.duration)
      :clamp(time/descriptor.duration,0,1);
    const cursor=normalized*(source.keyframes-1);
    const before=Math.floor(cursor),after=Math.min(source.keyframes-1,before+1),fraction=cursor-before;
    const a=xy(clip,before,index),b=xy(clip,after,index);
    return [lerp(a[0],b[0],fraction),lerp(a[1],b[1],fraction)];
  }
  function render(now){
    ctx.clearRect(0,0,512,512);
    if(layer==='art'||!source||!frames)return;
    const clip=clips.indexOf(motion);
    const elapsed=reduced.matches?0:(now-started)/1000;
    const count=source.joints.length,points=source.joints.map((_,i)=>interpolate(clip,elapsed,i));
    ctx.save();
    if(layer==='weights'){
      // These are sampled native skin weights, not an invented heatmap over
      // an artist-looking silhouette. Every light source follows a vertex
      // transformed by the actual frame-matrix skinning implementation.
      for(let i=0;i<source.samples;i++){
        if(source.sampleBone[i]!==selected)continue;
        const [x,y]=interpolate(clip,elapsed,count+i);
        const strength=source.sampleWeight[i]/255;
        const radius=14+16*strength;
        const gradient=ctx.createRadialGradient(x,y,0,x,y,radius);
        gradient.addColorStop(0,'rgba(237,176,255,'+(.55*strength).toFixed(3)+')');
        gradient.addColorStop(.28,'rgba(145,132,255,'+(.36*strength).toFixed(3)+')');
        gradient.addColorStop(1,'rgba(62,102,231,0)');
        ctx.fillStyle=gradient;
        ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fill();
        ctx.beginPath();ctx.arc(x,y,2.7+2*strength,0,Math.PI*2);
        ctx.fillStyle='rgba(224,216,255,'+(.38+.62*strength).toFixed(3)+')';ctx.fill();
      }
    }
    // Native rig hierarchy from the source demo, not guessed bone layout.
    ctx.lineWidth=layer==='weights'?.95:1.8;
    for(let i=0;i<count;i++){
      const p=source.parents[i];
      if(p<0||p>=count)continue;
      const a=points[p],b=points[i];
      const length=Math.hypot(a[0]-b[0],a[1]-b[1]);
      if(length<1.8)continue;
      ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);
      const hot=selected===i||selected===p;
      ctx.strokeStyle=hot?'rgba(226,194,255,.94)':layer==='weights'?'rgba(122,154,245,.24)':'rgba(167,191,255,.75)';
      ctx.stroke();
    }
    points.forEach(([x,y],i)=>{
      const active=selected===i;
      if(layer==='weights'&&!active)return;
      ctx.beginPath();ctx.arc(x,y,active?5.5:2.6,0,Math.PI*2);
      ctx.fillStyle=active?'#f4d8ff':'#cfddff';ctx.fill();
      if(active){
        ctx.beginPath();ctx.arc(x,y,11,0,Math.PI*2);
        ctx.strokeStyle='rgba(208,184,255,.45)';ctx.lineWidth=1;ctx.stroke();
      }
    });
    ctx.restore();
  }
  function frame(now){
    frameID=0;
    if(!visible||document.hidden||reduced.matches)return;
    if(now-lastTick>38){lastTick=now;render(now);}
    frameID=requestAnimationFrame(frame);
  }
  function sync(){
    if(frameID){cancelAnimationFrame(frameID);frameID=0}
    if(visible&&!document.hidden&&!reduced.matches)frameID=requestAnimationFrame(frame);
    else render(performance.now());
  }
  root.querySelectorAll('[data-knight-motion]').forEach(button=>
    button.addEventListener('click',()=>changeMotion(button.dataset.knightMotion)));
  root.querySelectorAll('[data-knight-mode]').forEach(button=>
    button.addEventListener('click',()=>changeLayer(button.dataset.knightMode)));
  select.addEventListener('change',()=>{
    selected=Number(select.value);
    render(performance.now());
  });
  canvas.addEventListener('click',event=>{
    if(!source||!frames||layer==='art')return;
    const bounds=canvas.getBoundingClientRect();
    const mouse=[(event.clientX-bounds.left)*512/bounds.width,(event.clientY-bounds.top)*512/bounds.height];
    const clip=clips.indexOf(motion),time=reduced.matches?0:(performance.now()-started)/1000;
    let best=16,index=-1;
    for(let i=0;i<source.joints.length;i++){
      const [x,y]=interpolate(clip,time,i);
      const dist=Math.hypot(x-mouse[0],y-mouse[1]);
      if(dist<best){best=dist;index=i}
    }
    if(index!==-1){
      const eligible=source.selected.includes(index)?index:source.selected.reduce((a,b)=>Math.abs(index-a)<Math.abs(index-b)?a:b);
      selected=eligible;select.value=String(eligible);render(performance.now());
    }
  });
  image.addEventListener('error',()=>{
    sourceError=true;
    evidence.textContent='Knight preview media unavailable';
    root.dataset.evidence='offline';
  });
  const intersection=new IntersectionObserver(entries=>{
    visible=Boolean(entries[0]?.isIntersecting);
    sync();
  },{threshold:.01});
  intersection.observe(root);
  document.addEventListener('visibilitychange',sync);
  reduced.addEventListener?.('change',sync);
  addEventListener('pagehide',()=>{
    if(frameID)cancelAnimationFrame(frameID);
    intersection.disconnect();
  },{once:true});
  root.dataset.layer='art';
  loadNativeSamples().catch(error=>{
    evidence.textContent='Artwork available · native rig layer unavailable';
    root.querySelectorAll('[data-knight-mode="rig"],[data-knight-mode="weights"]').forEach(el=>{
      el.disabled=true;el.title=error.message;
    });
  });
  window.bustedKnightShowcase={get motion(){return motion},get layer(){return layer},get nativeReady(){return !!source&&!sourceError}};
})();
