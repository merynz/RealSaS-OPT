/* BUSTED / Morphological Field v3
 * Original procedurally synthesized topology, NOT a compiled puppet.
 * Rotating triangulated 3D volume + real normal lighting + orbital diagnostics.
 */
(() => {
  'use strict';
  const field=document.querySelector('#structuralField');
  const canvas=document.querySelector('#structuralCanvas');
  if (!field||!canvas) return;
  const ctx=canvas.getContext('2d',{alpha:true,desynchronized:true});
  if (!ctx) return;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const TAU=Math.PI*2;
  const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
  const mix=(a,b,t)=>a+(b-a)*t;
  const norm=a=>{const d=Math.hypot(...a)||1;return a.map(v=>v/d)};
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  let width=1,height=1,dpr=1,frame=0,last=0,epoch=0,visible=false;
  let aimX=0,aimY=0,lookX=0,lookY=0;
  const latitude=39,longitude=62;
  const triangles=[];
  for(let j=0;j<latitude-1;j++) for(let i=0;i<longitude;i++){
    const a=j*longitude+i,b=j*longitude+(i+1)%longitude;
    const c=(j+1)*longitude+i,d=(j+1)*longitude+(i+1)%longitude;
    triangles.push([a,b,c],[b,d,c]);
  }
  const dust=Array.from({length:95},(_,i)=>{
    const hash=x=>{const y=Math.sin((i+23)*x*127.1)*43758.5453;return y-Math.floor(y)};
    return {a:hash(1)*TAU,r:165+hash(2)*145,
      y:-200+hash(3)*400,velocity:(.05+hash(4)*.16)*(i%2?-1:1),
      sparkle:hash(5),size:.55+hash(6)*1.5,phase:hash(7)*TAU};
  });

  function size(){
    const r=field.getBoundingClientRect();
    width=Math.max(1,r.width);height=Math.max(1,r.height);
    dpr=Math.min(devicePixelRatio||1,1.7);
    canvas.width=Math.round(width*dpr);
    canvas.height=Math.round(height*dpr);
    canvas.style.width=width+'px';canvas.style.height=height+'px';
    ctx.setTransform(dpr,0,0,dpr,0,0);
    draw(performance.now());
  }
  function draw(now){
    ctx.clearRect(0,0,width,height);
    const t=reduced.matches?0:(now-epoch)*.001;
    lookX=mix(lookX,aimX,.07);
    lookY=mix(lookY,aimY,.07);
    const yaw=-.34+t*.085+lookX*.49;
    const pitch=-.12+Math.sin(t*.17)*.12+lookY*.23;
    const C=Math.cos(yaw),S=Math.sin(yaw),CP=Math.cos(pitch),SP=Math.sin(pitch);
    const scale=Math.min(width/410,height/580)*1.02;
    const cx=width*.505,cy=height*.485;
    const rmx=v=> {
      const x=v[0]*C-v[2]*S,z=v[0]*S+v[2]*C;
      return [x,v[1]*CP-z*SP,v[1]*SP+z*CP];
    };
    const proj=v=>{
      const p=rmx(v),factor=890/(890+p[2]);
      return {x:cx+p[0]*scale*factor,y:cy+p[1]*scale*factor,z:p[2],factor,p};
    };
    function stroke(a,b,rgb,alpha,width=1,shadow=0){
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
      ctx.lineWidth=width;ctx.strokeStyle='rgba('+rgb+','+alpha+')';
      ctx.shadowBlur=shadow;ctx.shadowColor='rgba('+rgb+',.5)';
      ctx.stroke();ctx.shadowBlur=0;
    }

    // Volumetric aura is independent of the topology's source authority.
    const aura=ctx.createRadialGradient(cx,cy,6,cx,cy,Math.min(width,height)*.47);
    aura.addColorStop(0,'rgba(81,93,221,.23)');
    aura.addColorStop(.38,'rgba(45,72,167,.13)');
    aura.addColorStop(.72,'rgba(60,35,137,.055)');
    aura.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle=aura;ctx.fillRect(0,0,width,height);

    // Elliptical calibrated particle orbit, with real 3D perspective.
    for(const particle of dust){
      const a=particle.a+t*particle.velocity;
      const p=proj([Math.cos(a)*particle.r,particle.y+Math.sin(t*.27+particle.phase)*9,Math.sin(a)*particle.r]);
      const depth=clamp((130-p.z)/380,.13,.89);
      const alpha=depth*(.15+.55*particle.sparkle);
      ctx.beginPath();ctx.arc(p.x,p.y,Math.max(.3,particle.size*p.factor*scale),0,TAU);
      ctx.fillStyle='rgba('+(particle.sparkle>.67?'183,209,255':'119,130,252')+','+alpha.toFixed(3)+')';
      ctx.fill();
    }

    // Internal faceted volume: triangulation shares vertices with actual 3D surface.
    const vertices=[];
    for(let j=0;j<latitude;j++){
      const theta=.025+(Math.PI-.05)*j/(latitude-1);
      for(let i=0;i<longitude;i++){
        const phi=i/longitude*TAU;
        const s=Math.sin(theta),c=Math.cos(theta);
        const ripple=1+.11*Math.sin(theta*6+phi*3+t*.55)*Math.sin(phi*4-t*.39)
          +.06*Math.cos(theta*10-phi*5+t*.33);
        const radius=119*ripple;
        const twist=phi+.26*c+Math.sin(t*.25+theta*3)*.033;
        vertices.push(proj([
          radius*s*Math.cos(twist),
          radius*c*1.23,
          radius*s*Math.sin(twist)*.91
        ]));
      }
    }
    const faces=[];
    const lightDir=norm([-.45,-.46,-1]);
    for(let i=0;i<triangles.length;i++){
      const tr=triangles[i],a=vertices[tr[0]],b=vertices[tr[1]],c=vertices[tr[2]];
      const normal=norm(cross(sub(b.p,a.p),sub(c.p,a.p)));
      const camera=Math.abs(normal[2]);
      const light=Math.abs(dot(normal,lightDir));
      const depth=(a.z+b.z+c.z)/3;
      const front=clamp((135-depth)/310,.05,1);
      faces.push({a,b,c,depth,front,light,camera,i});
    }
    faces.sort((a,b)=>b.depth-a.depth);
    for(const f of faces){
      const {a,b,c,front,light,camera,i}=f;
      const magenta=Math.sin(t*.26+i*.031)*.5+.5;
      const r=Math.round(mix(49,134,light)*.72+52*magenta);
      const g=Math.round(55+123*light+23*front);
      const bl=Math.round(138+96*light);
      const shade=(.14+.47*light+.17*camera)*front;
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.lineTo(c.x,c.y);ctx.closePath();
      ctx.fillStyle='rgba('+r+','+g+','+bl+','+clamp(shade,.06,.68).toFixed(3)+')';
      ctx.fill();
      if(i%4===0){
        ctx.strokeStyle='rgba(166,180,255,'+(.04+light*.15).toFixed(3)+')';
        ctx.lineWidth=.5;ctx.stroke();
      }
    }

    // Geodesic latitudes + dynamic deformations expose the actual topology.
    for(let j=3;j<latitude-2;j+=4){
      for(let i=0;i<longitude;i++){
        const a=vertices[j*longitude+i],b=vertices[j*longitude+(i+1)%longitude];
        const near=clamp((110-(a.z+b.z)/2)/270,.0,.75);
        if(near>.10)stroke(a,b,'161,194,255',(.04+near*.31).toFixed(3),.75);
      }
    }
    // Tri-axis gyroscopes are sampled as transformed world-space circles.
    function gimbal(radius,tilt,spin,color,widthLine){
      const count=164;
      let previous=null;
      for(let i=0;i<=count;i++){
        const a=i/count*TAU+spin;
        const p=proj([
          radius*Math.cos(a),
          radius*Math.sin(a)*Math.cos(tilt),
          radius*Math.sin(a)*Math.sin(tilt)
        ]);
        if(previous){
          const alpha=clamp((170-(p.z+previous.z)*.5)/350,.05,.65);
          stroke(previous,p,color,alpha.toFixed(3),widthLine,alpha>.5?4:0);
        }
        previous=p;
      }
    }
    gimbal(177,.74+t*.055,t*.08,'170,151,255',1.25);
    gimbal(192,-.88-t*.038,-t*.045,'103,203,255',.85);
    gimbal(214,1.43+t*.03,t*.042,'145,159,246',.68);

    // Long-axis scanning beam, with a refracted inner highlight.
    const y=cy+Math.sin(t*.39)*106*scale;
    const beam=ctx.createLinearGradient(cx-170*scale,y,cx+170*scale,y);
    beam.addColorStop(0,'rgba(130,159,252,0)');
    beam.addColorStop(.5,'rgba(189,214,255,.22)');
    beam.addColorStop(1,'rgba(146,121,255,0)');
    ctx.fillStyle=beam;ctx.fillRect(cx-170*scale,y,340*scale,1.3);
    const core=ctx.createRadialGradient(cx,cy,0,cx,cy,55*scale);
    core.addColorStop(0,'rgba(236,245,255,.26)');
    core.addColorStop(.3,'rgba(164,174,255,.13)');
    core.addColorStop(1,'rgba(110,92,234,0)');
    ctx.fillStyle=core;ctx.beginPath();ctx.arc(cx,cy,55*scale,0,TAU);ctx.fill();

    // Surgical framing calibrates scale without looking like a humanoid.
    const tickColor='rgba(163,186,255,.20)';
    ctx.lineWidth=.7;ctx.strokeStyle=tickColor;
    for(let k=0;k<4;k++){
      const y0=cy+(-150+k*100)*scale;
      ctx.beginPath();ctx.moveTo(cx-183*scale,y0);ctx.lineTo(cx-176*scale,y0);ctx.stroke();
      ctx.beginPath();ctx.moveTo(cx+176*scale,y0);ctx.lineTo(cx+183*scale,y0);ctx.stroke();
    }
  }
  function animate(now){
    frame=0;
    if(!visible||document.hidden||reduced.matches)return;
    if(now-last>=34){last=now;draw(now)}
    frame=requestAnimationFrame(animate);
  }
  function sync(){
    if(frame)cancelAnimationFrame(frame);
    frame=0;
    if(visible&&!document.hidden&&!reduced.matches){
      if(!epoch)epoch=performance.now();
      frame=requestAnimationFrame(animate);
    }else draw(performance.now());
  }
  field.addEventListener('pointermove',e=>{
    const r=field.getBoundingClientRect();
    aimX=clamp((e.clientX-r.left)/Math.max(1,r.width)*2-1,-1,1);
    aimY=clamp((e.clientY-r.top)/Math.max(1,r.height)*2-1,-1,1);
    if(reduced.matches)draw(performance.now());
  },{passive:true});
  field.addEventListener('pointerleave',()=>{aimX=0;aimY=0},{passive:true});
  const observer=new IntersectionObserver(entries=>{visible=Boolean(entries[0]?.isIntersecting);sync()},{threshold:.01});
  const ro=new ResizeObserver(size);
  observer.observe(field);ro.observe(field);
  reduced.addEventListener?.('change',sync);
  document.addEventListener('visibilitychange',sync);
  addEventListener('pagehide',()=>{
    if(frame)cancelAnimationFrame(frame);observer.disconnect();ro.disconnect();
  },{once:true});
  size();sync();
})();
