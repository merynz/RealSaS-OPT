/* Static perspective geometry + low-frequency typewriter. Purely ambient, not proof telemetry. */
(() => {
  'use strict';
  const canvas = document.querySelector('#depthScene');
  if (!canvas) return;
  const ctx = canvas.getContext('2d', {alpha:true});
  if (!ctx) return;

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const lines = [
    'source.observe(front, back);',
    'scene.resolve(shape_memory);',
    'mesh.compile(canonical_topology);',
    'rig.bind(control_program);',
    'skin.attach(weight_field);',
    'mechanics.validate(deformation);',
    'graphics.resolve(appearance);',
    'motion.attach(clip_set);',
    'proof.require_authority();'
  ];
  const tracks = [
    {x:-320,y:-185,z:.27,first:0},
    {x:85,y:-146,z:.35,first:3},
    {x:-265,y:84,z:.22,first:6},
    {x:145,y:132,z:.29,first:1}
  ];
  let width=0, height=0, lastPaint=0, frame=0;
  const started=performance.now();

  function drawGrid() {
    const cx=width/2, horizon=height*.435, near=height+30, run=near-horizon;
    ctx.lineWidth=1;
    for(let i=-12;i<=12;i++){
      ctx.strokeStyle='rgba(119,160,184,.055)';
      ctx.beginPath();
      ctx.moveTo(cx+i*2.4,horizon);
      ctx.lineTo(cx+(i/12)*width*.92,near);
      ctx.stroke();
    }
    for(let i=1;i<=19;i++){
      const t=i/19, p=t*t, y=horizon+p*run, span=24+p*width*.92;
      ctx.strokeStyle='rgba(128,173,196,'+(.012+p*.06)+')';
      ctx.beginPath();ctx.moveTo(cx-span,y);ctx.lineTo(cx+span,y);ctx.stroke();
    }
    const farW=Math.min(width*.48,720),farH=Math.min(height*.25,195);
    ctx.strokeStyle='rgba(121,159,181,.03)';
    for(let i=-4;i<=4;i++){
      const x=cx+i*farW/8;
      ctx.beginPath();ctx.moveTo(x,horizon-farH);ctx.lineTo(x,horizon+farH*.18);ctx.stroke();
    }
    for(let i=-2;i<=2;i++){
      const y=horizon-i*farH/3;
      ctx.beginPath();ctx.moveTo(cx-farW/2,y);ctx.lineTo(cx+farW/2,y);ctx.stroke();
    }
  }

  function drawTyping(now) {
    const cx=width/2, horizon=height*.435;
    const elapsed=reduced?100000:Math.max(0,now-started);
    const mobile=width<720;
    for(let t=0;t<tracks.length;t++){
      const track=tracks[t], perspective=.38+track.z*.73;
      const x=cx+track.x*perspective, y=horizon+track.y*perspective;
      const fontSize=(mobile?7:10)*perspective+2, lineHeight=fontSize*1.8;
      const alpha=(mobile?.07:.135)*(.63+track.z);
      const localTime=reduced?6000:(elapsed+t*2900)%14500;
      ctx.save();ctx.textAlign='left';ctx.textBaseline='middle';
      ctx.font=fontSize+'px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      for(let row=0;row<3;row++){
        const text=lines[(track.first+row)%lines.length];
        const when=localTime-row*1450;
        const count=reduced?text.length:Math.max(0,Math.min(text.length,Math.floor(when/64)));
        if(!count)continue;
        const visible=text.slice(0,count), lineY=y+row*lineHeight;
        ctx.fillStyle='rgba(156,192,210,'+alpha+')';ctx.fillText(visible,x,lineY);
        if(!reduced&&count<text.length&&when>=0&&Math.floor(now/420)%2===0)
          ctx.fillRect(x+ctx.measureText(visible).width+3,lineY-fontSize*.55,1,fontSize*1.12);
      }
      ctx.restore();
    }
  }

  function paint(now){
    if(!width||!height)return;
    ctx.clearRect(0,0,width,height);
    drawGrid();
    drawTyping(now);
  }

  function resize(){
    // Cap backing resolution to prevent unnecessary high-DPI canvas fill cost.
    const dpr=Math.min(window.devicePixelRatio||1,1.5);
    width=window.innerWidth;height=window.innerHeight;
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    canvas.style.width=width+'px';canvas.style.height=height+'px';
    ctx.setTransform(dpr,0,0,dpr,0,0);
    paint(performance.now());
  }

  function tick(now){
    // Rendering at most 24 fps is sufficient for terminal typewriter ambience.
    if(now-lastPaint>=42&&!document.hidden){paint(now);lastPaint=now;frame++;}
    requestAnimationFrame(tick);
  }
  window.addEventListener('resize',resize,{passive:true});
  resize();
  if(!reduced)requestAnimationFrame(tick);
})();
