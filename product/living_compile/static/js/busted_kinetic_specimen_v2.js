/* BUSTED / Kinetic Specimen v2
 * Original procedural articulated mannequin: diagnostic ART STUDY, not a
 * compiled puppet, trained output, semantic proof, or character demo.
 * An actual joint hierarchy drives volumetric 3D surface topology each frame.
 */
(() => {
  'use strict';
  const field=document.querySelector('#structuralField');
  const canvas=document.querySelector('#structuralCanvas');
  if(!field||!canvas)return;
  const ctx=canvas.getContext('2d',{alpha:true,desynchronized:true});
  if(!ctx)return;
  const reduce=matchMedia('(prefers-reduced-motion: reduce)');
  const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
  const mix=(a,b,t)=>a+(b-a)*t;
  const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const mul=(a,k)=>[a[0]*k,a[1]*k,a[2]*k];
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const unit=a=>{const d=Math.hypot(...a)||1;return a.map(v=>v/d)};
  const particles=Array.from({length:62},(_,i)=>{
    const seed=(n)=>{const x=Math.sin((i+1)*n*127.1)*43758.5453;return x-Math.floor(x)};
    return {a:seed(1)*Math.PI*2,r:105+seed(3)*108,y:-188+seed(6)*420,
      speed:(.09+seed(8)*.23)*(i%2?-1:1),phase:seed(11)*Math.PI*2,
      size:.6+seed(15)*1.7,light:seed(17)};
  });
  let w=1,h=1,dpr=1,visible=false,raf=0,last=0,epoch=0;
  let targetX=0,targetY=0,lookX=0,lookY=0;
  const P={
    cobalt:[101,139,255],violet:[170,117,255],ice:[198,233,255],
    indigo:[49,68,153],cyan:[95,223,255]
  };

  function resize(){
    const r=field.getBoundingClientRect();
    w=Math.max(1,r.width);h=Math.max(1,r.height);
    dpr=Math.min(devicePixelRatio||1,1.7);
    canvas.width=Math.max(1,Math.round(w*dpr));
    canvas.height=Math.max(1,Math.round(h*dpr));
    canvas.style.width=w+'px';canvas.style.height=h+'px';
    ctx.setTransform(dpr,0,0,dpr,0,0);
    draw(performance.now());
  }

  function draw(now){
    ctx.clearRect(0,0,w,h);
    const sec=reduce.matches?0:(now-epoch)*.001;
    const idle=Math.sin(sec*.8),slow=Math.sin(sec*.31);
    lookX=mix(lookX,targetX,.055);
    lookY=mix(lookY,targetY,.055);
    const yaw=-.17+Math.sin(sec*.18)*.22+lookX*.52;
    const pitch=-.065+lookY*.17;
    const c=Math.cos(yaw),s=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    const scale=Math.min(w/355,h/600)*1.13;
    const cx=w*.51,cy=h*.49;
    function rotate(v){
      const x=v[0]*c-v[2]*s,z=v[0]*s+v[2]*c;
      return [x,v[1]*cp-z*sp,v[1]*sp+z*cp];
    }
    function project(v){
      const q=rotate(v),perspective=740/(740+q[2]);
      return {x:cx+q[0]*scale*perspective,y:cy+q[1]*scale*perspective,
        z:q[2],p:perspective};
    }
    function line(a,b,color,width=1,blur=0){
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
      ctx.lineWidth=width;ctx.strokeStyle=color;
      ctx.shadowColor=color;ctx.shadowBlur=blur;ctx.stroke();ctx.shadowBlur=0;
    }
    function ring(x,y,rx,ry,alpha,phase=0){
      ctx.beginPath();ctx.ellipse(x,y,rx,ry,phase,0,Math.PI*2);
      ctx.strokeStyle='rgba(135,160,255,'+alpha+')';ctx.lineWidth=.8;ctx.stroke();
    }

    // Stage volume: shaded holographic nebula, instrumental rings, and a floor.
    const halo=ctx.createRadialGradient(cx,cy-31*scale,2,cx,cy-31*scale,Math.max(w,h)*.49);
    halo.addColorStop(0,'rgba(78,71,192,.18)');
    halo.addColorStop(.38,'rgba(37,68,167,.095)');
    halo.addColorStop(.8,'rgba(17,22,65,.018)');
    halo.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle=halo;ctx.fillRect(0,0,w,h);
    const floor=project([0,268,0]);
    for(let i=0;i<5;i++)ring(floor.x,floor.y,(69+i*35)*scale,(13+i*12)*scale,.20-i*.032,-.075);
    const baseLight=ctx.createRadialGradient(floor.x,floor.y,0,floor.x,floor.y,125*scale);
    baseLight.addColorStop(0,'rgba(123,170,255,.19)');
    baseLight.addColorStop(.55,'rgba(85,87,215,.065)');
    baseLight.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle=baseLight;ctx.beginPath();
    ctx.ellipse(floor.x,floor.y,154*scale,51*scale,0,0,Math.PI*2);ctx.fill();

    const breathing=idle*2.5;
    const lean=slow*5;
    const j={
      root:[lean,96+breathing,0],pelvis:[lean,91+breathing,0],
      spine:[lean*.5,29+breathing,0],chest:[-lean*.3,-58+breathing,-1],
      neck:[-lean*.4,-119+breathing,0],head:[-lean*.4,-165+breathing+Math.sin(sec*.9)*1.8,0],
      crown:[-lean*.4,-207+breathing+Math.sin(sec*.9)*1.8,0],
      lShoulder:[-68-lean*.22,-70+breathing,2],
      rShoulder:[68-lean*.22,-70+breathing,2],
      lElbow:[-111+Math.sin(sec*.72)*5,-13+breathing,14],
      rElbow:[109+Math.sin(sec*.71+.7)*6,-4+breathing,-10],
      lWrist:[-99+Math.sin(sec*.61)*7,68+breathing,20],
      rWrist:[111+Math.sin(sec*.64+.9)*6,71+breathing,19],
      lKnuckle:[-99+Math.sin(sec*.61)*7,90+breathing,21],
      rKnuckle:[111+Math.sin(sec*.64+.9)*6,94+breathing,21],
      lHip:[-36,99+breathing,0],rHip:[36,99+breathing,0],
      lKnee:[-45+Math.sin(sec*.51)*2,182+breathing,18],
      rKnee:[47+Math.sin(sec*.51+1.7)*2,181+breathing,-14],
      lAnkle:[-48,259,3],rAnkle:[49,258,7],
      lFoot:[-48,269,27],rFoot:[49,269,27],
    };
    const bones=[
      ['pelvis','spine',43,34,'indigo'],['spine','chest',38,59,'cobalt'],
      ['chest','neck',58,20,'cobalt'],['neck','head',17,22,'ice'],
      ['head','crown',32,25,'ice'],['chest','lShoulder',33,28,'violet'],
      ['chest','rShoulder',33,28,'violet'],
      ['lShoulder','lElbow',23,16,'cobalt'],['rShoulder','rElbow',23,16,'cobalt'],
      ['lElbow','lWrist',17,10,'ice'],['rElbow','rWrist',17,10,'ice'],
      ['lWrist','lKnuckle',11,9,'violet'],['rWrist','rKnuckle',11,9,'violet'],
      ['pelvis','lHip',36,24,'violet'],['pelvis','rHip',36,24,'violet'],
      ['lHip','lKnee',30,20,'cobalt'],['rHip','rKnee',30,20,'cobalt'],
      ['lKnee','lAnkle',21,13,'ice'],['rKnee','rAnkle',21,13,'ice'],
      ['lAnkle','lFoot',14,12,'violet'],['rAnkle','rFoot',14,12,'violet']
    ];
    const points=[],faces=[],edges=[];
    // Real swept volumes with radial rings; not flat SVG lines.
    function segment(from,to,r0,r1,tone){
      const a=j[from],b=j[to],axis=unit(sub(b,a));
      let U=unit(cross(axis,[0,0,1]));
      if(Math.hypot(...U)<.1)U=[1,0,0];
      const V=unit(cross(axis,U)),S=10,R=3,index=points.length;
      for(let k=0;k<=R;k++){
        const f=k/R,center=add(a,mul(sub(b,a),f));
        const radius=mix(r0,r1,f)*(1+.035*Math.sin(sec*.9+f*2));
        for(let i=0;i<S;i++){
          const theta=i/S*Math.PI*2;
          const p=add(center,add(mul(U,Math.cos(theta)*radius),mul(V,Math.sin(theta)*radius*.82)));
          points.push(project(p));
          if(k>0){
            const cur=index+k*S+i,prev=index+(k-1)*S+i,next=index+k*S+(i+1)%S,prevNext=index+(k-1)*S+(i+1)%S;
            faces.push({v:[prev,prevNext,cur],tone});
            faces.push({v:[prevNext,next,cur],tone});
            edges.push([prev,cur,tone]);
          }
          edges.push([index+k*S+i,index+k*S+(i+1)%S,tone]);
        }
      }
    }
    bones.forEach(b=>segment(...b));
    faces.forEach(f=>{f.z=(points[f.v[0]].z+points[f.v[1]].z+points[f.v[2]].z)/3});
    faces.sort((a,b)=>b.z-a.z);
    for(const f of faces){
      const ps=f.v.map(i=>points[i]);
      const brightness=clamp((115-f.z)/275,.13,1);
      const rgb=P[f.tone];
      ctx.beginPath();ctx.moveTo(ps[0].x,ps[0].y);
      ctx.lineTo(ps[1].x,ps[1].y);ctx.lineTo(ps[2].x,ps[2].y);ctx.closePath();
      ctx.fillStyle='rgba('+rgb.join(',')+','+(.016+brightness*.047).toFixed(3)+')';ctx.fill();
    }
    // Topology edges retain near/far shading and occasional spectral scan.
    edges.sort((a,b)=>(points[b[0]].z+points[b[1]].z)-(points[a[0]].z+points[a[1]].z));
    for(const [a,b,tone] of edges){
      const p=points[a],q=points[b];
      const front=clamp((100-(p.z+q.z)*.5)/250,.08,1);
      const col=P[tone];
      const boost=(Math.sin(sec*.96+p.y*.034)> .96)? .17:0;
      const alpha=.045+front*.34+boost;
      line(p,q,'rgba('+col.join(',')+','+alpha.toFixed(3)+')',front>.65?.8:.6);
    }

    // Skeleton relationships share the exact articulated joints.
    const hierarchy=[
      ['pelvis','spine'],['spine','chest'],['chest','neck'],['neck','head'],
      ['chest','lShoulder'],['chest','rShoulder'],
      ['lShoulder','lElbow'],['lElbow','lWrist'],
      ['rShoulder','rElbow'],['rElbow','rWrist'],
      ['pelvis','lHip'],['lHip','lKnee'],['lKnee','lAnkle'],
      ['pelvis','rHip'],['rHip','rKnee'],['rKnee','rAnkle']
    ];
    for(const [a,b] of hierarchy){
      line(project(j[a]),project(j[b]),'rgba(214,229,255,.47)',1.15,3);
    }
    for(const [name,raw] of Object.entries(j)){
      if(name==='root'||name==='crown'||name.endsWith('Foot')||name.endsWith('Knuckle'))continue;
      const p=project(raw),major=['pelvis','spine','chest','head'].includes(name);
      const pulse=.70+.20*Math.sin(sec*1.8+raw[1]*.03);
      ctx.beginPath();ctx.arc(p.x,p.y,(major?4.5:2.8)*scale,p.p?0:0,Math.PI*2);
      ctx.fillStyle='rgba(224,238,255,'+pulse.toFixed(2)+')';
      ctx.shadowColor='#a5ccff';ctx.shadowBlur=major?11:7;ctx.fill();ctx.shadowBlur=0;
      ctx.beginPath();ctx.arc(p.x,p.y,(major?11:7)*scale,0,Math.PI*2);
      ctx.lineWidth=.75;ctx.strokeStyle='rgba(169,151,255,.36)';ctx.stroke();
    }

    // Visor establishes an unmistakable head orientation (front=positive z).
    const eyeA=project(add(j.head,[-17,-1,25]));
    const eyeB=project(add(j.head,[17,-1,25]));
    line(eyeA,eyeB,'rgba(145,228,255,.62)',2*scale,8);
    // Spectral heartbeat and non-grid instrumentation around the torso.
    const chest=project(j.chest),phase=sec*.36;
    for(let i=0;i<3;i++){
      const rr=(77+i*28)*scale;
      ctx.beginPath();
      ctx.ellipse(chest.x,chest.y,rr,rr*.83,-.19,phase+i*1.5,phase+i*1.5+Math.PI*.47);
      ctx.lineWidth=i===0?1.6:.7;
      ctx.strokeStyle=i===0?'rgba(168,173,255,.29)':'rgba(113,150,251,.14)';
      ctx.shadowColor='#7679ff';ctx.shadowBlur=i===0?8:0;ctx.stroke();ctx.shadowBlur=0;
    }
    const scanY=cy+Math.sin(sec*.54)*153*scale;
    const scan=ctx.createLinearGradient(cx-165*scale,scanY,cx+165*scale,scanY);
    scan.addColorStop(0,'rgba(88,136,255,0)');
    scan.addColorStop(.5,'rgba(132,180,255,.17)');
    scan.addColorStop(1,'rgba(151,116,255,0)');
    ctx.fillStyle=scan;ctx.fillRect(cx-165*scale,scanY,330*scale,1.1);

    // 3D orbit particles are depth-dependent, autonomous and non-repeating.
    for(const p of particles){
      const a=p.a+sec*p.speed,dx=Math.cos(a)*p.r,dz=Math.sin(a)*p.r;
      const y=p.y+Math.sin(sec*.45+p.phase)*12;
      const q=project([dx,y,dz]);
      const alpha=clamp((90-q.z)/240,.10,.75)*(.42+p.light*.36);
      const sz=p.size*scale*q.p;
      ctx.beginPath();ctx.arc(q.x,q.y,Math.max(.35,sz),0,Math.PI*2);
      ctx.fillStyle='rgba('+(p.light>.68?'184,214,255':'115,125,255')+','+alpha.toFixed(3)+')';
      ctx.fill();
      if(p.light>.86){
        line({x:q.x-4,y:q.y},{x:q.x+4,y:q.y},'rgba(170,191,255,'+(alpha*.35).toFixed(3)+')',.6);
      }
    }

    // Optical corners, calibration markers, and a restrained sweep.
    ctx.strokeStyle='rgba(153,177,255,.21)';ctx.lineWidth=1;
    const S=Math.min(w,h)*.068;
    for(const [x,y,dx,dy] of [[18,55,1,1],[w-18,55,-1,1],[18,h-47,1,-1],[w-18,h-47,-1,-1]]){
      ctx.beginPath();ctx.moveTo(x+dx*S,y);ctx.lineTo(x,y);ctx.lineTo(x,y+dy*S);ctx.stroke();
    }
    const anchor=project(j.chest);
    line({x:anchor.x-17,y:anchor.y},{x:anchor.x-8,y:anchor.y},'rgba(230,237,255,.24)',.8);
    line({x:anchor.x+8,y:anchor.y},{x:anchor.x+17,y:anchor.y},'rgba(230,237,255,.24)',.8);
  }

  function frame(now){
    raf=0;
    if(!visible||document.hidden||reduce.matches)return;
    if(now-last>33){last=now;draw(now)}
    raf=requestAnimationFrame(frame);
  }
  function sync(){
    if(raf)cancelAnimationFrame(raf);
    raf=0;
    if(visible&&!document.hidden&&!reduce.matches){
      if(!epoch)epoch=performance.now();
      raf=requestAnimationFrame(frame);
    }else draw(performance.now());
  }
  field.addEventListener('pointermove',event=>{
    const r=field.getBoundingClientRect();
    targetX=clamp(((event.clientX-r.left)/Math.max(1,r.width)-.5)*2,-1,1);
    targetY=clamp(((event.clientY-r.top)/Math.max(1,r.height)-.5)*2,-1,1);
    if(reduce.matches)draw(performance.now());
  },{passive:true});
  field.addEventListener('pointerleave',()=>{targetX=0;targetY=0},{passive:true});
  const io=new IntersectionObserver(entries=>{visible=Boolean(entries[0]?.isIntersecting);sync()},{threshold:.01});
  const ro=new ResizeObserver(resize);
  io.observe(field);ro.observe(field);
  document.addEventListener('visibilitychange',sync);
  reduce.addEventListener?.('change',sync);
  addEventListener('pagehide',()=>{if(raf)cancelAnimationFrame(raf);io.disconnect();ro.disconnect()},{once:true});
  resize();sync();
})();
