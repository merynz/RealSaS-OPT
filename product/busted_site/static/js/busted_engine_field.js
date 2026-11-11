/* BUSTED Structural Field / procedural spatial carrier visualization.
 * A deterministic, original geometry model, not a compiled character or Proof.
 * Mesh vertices are constructed, transformed, projected and depth-sorted.
 * Render is capped and paused when offscreen. No external shaders or images.
 */
(() => {
  'use strict';
  const surface=document.querySelector('#structuralField');
  const canvas=document.querySelector('#structuralCanvas');
  if(!surface||!canvas)return;
  const ctx=canvas.getContext('2d',{alpha:true,desynchronized:true});
  if(!ctx)return;
  const reduce=matchMedia('(prefers-reduced-motion: reduce)');
  let w=1,h=1,dpr=1,visible=true,running=false,frameId=0,last=0,epoch=0;
  let aimX=0,aimY=0,curX=0,curY=0;
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const mix=(a,b,t)=>a+(b-a)*t;
  const rings=26,segments=36;
  const vertices=[],edges=[],faces=[];
  // Carrier is an irregular, organic bounded manifold. The center axis
  // represents a structural reference rather than a rendered character.
  for(let j=0;j<=rings;j++){
    const v=j/rings,t=v*2-1;
    const base=.15+.85*Math.pow(Math.max(0,1-t*t),.63);
    for(let i=0;i<segments;i++){
      const theta=i/segments*Math.PI*2;
      const scallop=1+.085*Math.cos(3*theta+5*t)+.035*Math.sin(7*theta-3*t);
      vertices.push([
        105*base*scallop*Math.cos(theta),
        t*183 + 12*Math.sin(theta*2+t*3)*base,
        78*base*scallop*Math.sin(theta)+11*Math.sin(3*t+theta)
      ]);
      const current=j*segments+i;
      if(j>0)edges.push([current,(j-1)*segments+i]);
      edges.push([current,j*segments+(i+1)%segments]);
      if(j>0){
        const prev=(j-1)*segments+i;
        const next=(j-1)*segments+(i+1)%segments;
        const adjacent=j*segments+(i+1)%segments;
        faces.push([prev,next,current]);
        faces.push([next,adjacent,current]);
      }
    }
  }
  const refs=[[-.92,0],[-.67,0],[-.31,0],[.08,0],[.41,0],[.75,0],[.94,0]];
  function resize(){
    const rect=surface.getBoundingClientRect();
    w=Math.max(1,rect.width);h=Math.max(1,rect.height);
    dpr=Math.min(2,window.devicePixelRatio||1);
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    canvas.style.width=w+'px';canvas.style.height=h+'px';
    ctx.setTransform(dpr,0,0,dpr,0,0);
    draw(performance.now());
  }
  function turn(v,yaw,pitch,twist){
    const [x,y,z]=v;
    const c=Math.cos(yaw),s=Math.sin(yaw);
    const a=x*c-z*s,b=x*s+z*c;
    const cp=Math.cos(pitch),sp=Math.sin(pitch);
    const yy=y*cp-b*sp,zz=y*sp+b*cp;
    const cs=Math.cos(twist),ss=Math.sin(twist);
    return [a*cs-yy*ss,a*ss+yy*cs,zz];
  }
  function project(v,scale,cx,cy){
    const perspective=720/(720+v[2]);
    return [cx+v[0]*scale*perspective,cy+v[1]*scale*perspective,v[2],perspective];
  }
  function draw(now){
    ctx.clearRect(0,0,w,h);
    const t=reduce.matches?0:(now-epoch)*.001;
    curX=mix(curX,aimX,.036);curY=mix(curY,aimY,.036);
    const yaw=.34+Math.sin(t*.18)*.25+curX*.26;
    const pitch=-.17+Math.sin(t*.11)*.065+curY*.16;
    const twist=-.045+Math.sin(t*.26)*.035;
    const scale=Math.min(w/330,h/540)*.82;
    const cx=w*.53,cy=h*.49;
    // Instrument-space concentric contours, never a CSS floor grid.
    ctx.save();ctx.translate(cx,cy);ctx.rotate(-.12);
    for(let i=0;i<4;i++){
      const rx=(116+i*35)*scale,ry=(195+i*28)*scale;
      ctx.beginPath();ctx.ellipse(0,0,rx,ry,0,0,Math.PI*2);
      ctx.strokeStyle=i===0?'rgba(157,180,255,.14)':'rgba(102,124,200,.055)';
      ctx.lineWidth=i===0?1:.7;ctx.stroke();
    }
    ctx.restore();
    // Geometric wire model: genuine vertices/faces/edges and depth sort.
    const projected=vertices.map((v,i)=>{
      const phase=i%segments/segments*Math.PI*2;
      const flex=7*Math.sin(t*.42+v[1]*.014+phase*2);
      const displacement=[v[0]*(1+.025*Math.sin(t*.4)),v[1],v[2]+flex];
      return project(turn(displacement,yaw,pitch,twist),scale,cx,cy);
    });
    const faceList=faces.map(f=>({f,z:(projected[f[0]][2]+projected[f[1]][2]+projected[f[2]][2])/3}));
    faceList.sort((a,b)=>b.z-a.z);
    for(const item of faceList){
      const f=item.f,p=f.map(i=>projected[i]);
      const bright=clamp((70-item.z)/260,0,1);
      if(bright<.30)continue;
      ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);ctx.lineTo(p[1][0],p[1][1]);ctx.lineTo(p[2][0],p[2][1]);ctx.closePath();
      ctx.fillStyle='rgba(92,112,210,'+(.007+bright*.021).toFixed(3)+')';
      ctx.fill();
    }
    const lines=edges.map(e=>({e,z:(projected[e[0]][2]+projected[e[1]][2])/2})).sort((a,b)=>b.z-a.z);
    for(const {e,z} of lines){
      const a=projected[e[0]],b=projected[e[1]];
      const front=clamp((80-z)/245,0,1);
      ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);
      ctx.strokeStyle='rgba('+(front>.56?'164,188,255':'103,123,191')+','+(.045+front*.28).toFixed(3)+')';
      ctx.lineWidth=front>.5?.85:.63;ctx.stroke();
    }
    // Carrier central spine, projected using the same camera. Physical joints
    // join actual structural points rather than arbitrary decorative particles.
    const joints=refs.map(([norm])=>{
      const point=[0,norm*183,0];
      return project(turn(point,yaw,pitch,twist),scale,cx,cy);
    });
    ctx.lineWidth=1.5;ctx.strokeStyle='rgba(207,220,255,.45)';
    for(let i=1;i<joints.length;i++){
      ctx.beginPath();ctx.moveTo(joints[i-1][0],joints[i-1][1]);ctx.lineTo(joints[i][0],joints[i][1]);ctx.stroke();
    }
    joints.forEach((p,i)=>{
      const pulse=.64+.16*Math.sin(t*.8-i*.63);
      ctx.beginPath();ctx.arc(p[0],p[1],i===3?5.2:3.6,0,Math.PI*2);
      ctx.fillStyle='rgba(220,225,255,'+pulse.toFixed(2)+')';ctx.fill();
      ctx.beginPath();ctx.arc(p[0],p[1],i===3?11:8,0,Math.PI*2);
      ctx.strokeStyle='rgba(154,158,255,.22)';ctx.lineWidth=1;ctx.stroke();
    });
    // Isoline through the model, with a real projected path.
    ctx.beginPath();
    for(let j=0;j<=rings;j++){
      const p=projected[j*segments+2];
      if(j===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);
    }
    ctx.strokeStyle='rgba(205,199,255,.55)';ctx.lineWidth=1.4;ctx.stroke();
    // Crosshair anchored to the central reference joint.
    const anchor=joints[3];ctx.lineWidth=.7;ctx.strokeStyle='rgba(199,212,255,.3)';
    ctx.beginPath();ctx.moveTo(anchor[0]-22,anchor[1]);ctx.lineTo(anchor[0]-10,anchor[1]);
    ctx.moveTo(anchor[0]+10,anchor[1]);ctx.lineTo(anchor[0]+22,anchor[1]);
    ctx.moveTo(anchor[0],anchor[1]-22);ctx.lineTo(anchor[0],anchor[1]-10);
    ctx.moveTo(anchor[0],anchor[1]+10);ctx.lineTo(anchor[0],anchor[1]+22);
    ctx.stroke();
  }
  function update(now){
    frameId=0;
    if(!visible||document.hidden||reduce.matches){running=false;return}
    if(now-last>=32){last=now;draw(now)}
    frameId=requestAnimationFrame(update);
  }
  function sync(){
    if(frameId){cancelAnimationFrame(frameId);frameId=0}
    if(visible&&!document.hidden&&!reduce.matches){
      if(!epoch)epoch=performance.now();
      running=true;frameId=requestAnimationFrame(update);
    } else {running=false;draw(performance.now())}
  }
  surface.addEventListener('pointermove',e=>{
    const rect=surface.getBoundingClientRect();
    aimX=clamp((e.clientX-rect.left)/Math.max(1,rect.width)*2-1,-1,1);
    aimY=clamp((e.clientY-rect.top)/Math.max(1,rect.height)*2-1,-1,1);
    if(reduce.matches)draw(performance.now());
  },{passive:true});
  surface.addEventListener('pointerleave',()=>{aimX=0;aimY=0},{passive:true});
  const observer=new IntersectionObserver(records=>{
    visible=Boolean(records[0]?.isIntersecting);sync();
  },{threshold:.01});
  observer.observe(surface);
  const resizeObserver=new ResizeObserver(resize);resizeObserver.observe(surface);
  document.addEventListener('visibilitychange',sync);
  reduce.addEventListener?.('change',sync);
  window.addEventListener('pagehide',()=>{
    observer.disconnect();resizeObserver.disconnect();
    if(frameId)cancelAnimationFrame(frameId);
  },{once:true});
  resize();sync();
})();