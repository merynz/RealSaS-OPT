/* BUSTED Spatial Data Rain v1: a shared, multi-depth visual simulation.
   Decorative computation, never live compiler telemetry or product proof. */
(() => {
 'use strict';
 if(document.getElementById('bustedSpatialRain'))return;
 const canvas=document.createElement('canvas');
 canvas.id='bustedSpatialRain';canvas.className='busted-spatial-rain';
 canvas.setAttribute('aria-hidden','true');document.body.prepend(canvas);
 const ctx=canvas.getContext('2d',{alpha:true,desynchronized:true});
 if(!ctx)return;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const alphabet='0123456789ABCDEF<>[]{}:/=+-アイウエオカキクケコ';
 const rand=(a,b)=>a+Math.random()*(b-a);
 const char=()=>alphabet.charAt(Math.floor(Math.random()*alphabet.length));
 let width=0,height=0,dpr=1,lanes=[],raf=0,last=0,clock=0,prev=0,shift=0,aim=0,lastMutation=-1;
 const product=Boolean(document.querySelector('.workspace,.motion-layout,.export-layout'));
 function rebuild(){
   width=Math.max(1,innerWidth);height=Math.max(1,innerHeight);
   dpr=Math.min(devicePixelRatio||1,1.6);
   canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
   canvas.style.width=width+'px';canvas.style.height=height+'px';
   ctx.setTransform(dpr,0,0,dpr,0,0);
   const mobile=width<760;
   const counts=mobile?[13,10,5]:[39,26,17];lanes=[];
   for(let layer=0;layer<3;layer++){
     const cells=counts[layer],stepX=width/cells;
     for(let i=0;i<cells;i++){
       const depth=[.17,.52,.87][layer]+rand(-.045,.045);
       const font=[10,12.5,15][layer]*(mobile?.87:1);
       const step=font*1.43;
       const length=Math.floor(rand(...([[10,17],[15,24],[18,30]][layer])));
       lanes.push({
         x:(i+.17+Math.random()*.66)*stepX,depth,font,step,length,
         speed:rand(...([[18,33],[36,57],[70,97]][layer])),
         offset:rand(0,height+length*step),layer,
         symbols:Array.from({length},char)
       });
     }
   }
   draw();
 }
 function draw(){
   ctx.clearRect(0,0,width,height);
   const sec=(reduced.matches?0:clock)*.001;
   const dim=product?.65:1;
   ctx.textAlign='center';ctx.textBaseline='middle';
   for(const lane of lanes){
     const tailHeight=lane.length*lane.step;
     const cycle=height+tailHeight;
     const head=reduced.matches?lane.offset%cycle-tailHeight*.4:
       (lane.offset+sec*lane.speed)%cycle-tailHeight;
     const x=lane.x+shift*lane.depth*8;
     const color=lane.layer===0?[98,110,176]:lane.layer===1?[137,155,220]:[181,196,255];
     ctx.font=lane.font.toFixed(1)+'px IBM Plex Mono, monospace';
     for(let j=0;j<lane.length;j++){
       const y=head-j*lane.step;
       if(y<-20||y>height+20)continue;
       const tail=Math.pow(1-j/lane.length,1.4);
       const opacity=[.28,.46,.72][lane.layer]*tail*dim;
       if(opacity<.008)continue;
       if(j===0&&lane.layer===2){
         ctx.fillStyle='rgba(227,235,255,'+(.86*dim).toFixed(3)+')';
         ctx.shadowColor='rgba(135,153,244,.38)';ctx.shadowBlur=6;
       }else{
         ctx.fillStyle='rgba('+color.join(',')+','+opacity.toFixed(3)+')';
         ctx.shadowBlur=0;
       }
       ctx.fillText(lane.symbols[j],x,y);
     }
     ctx.shadowBlur=0;
   }
 }
 function frame(now){
   raf=0;if(document.hidden||reduced.matches)return;
   const dt=prev?Math.min(100,now-prev):0;prev=now;clock+=dt;
   shift+=(aim-shift)*.045;
   // Sparse glyph replacement gives each tail the characteristic decoded
   // Matrix shimmer without a separate DOM animation or full text rewrite.
   const mutation=Math.floor(clock/180);
   if(mutation!==lastMutation){
     lastMutation=mutation;
     for(let k=0;k<Math.min(12,lanes.length);k++){
       const stream=lanes[Math.floor(Math.random()*lanes.length)];
       stream.symbols[Math.floor(Math.random()*stream.length)]=char();
     }
   }
   if(now-last>=((width<760||product)?67:43)){draw();last=now}
   raf=requestAnimationFrame(frame);
 }
 function sync(){
   if(raf)cancelAnimationFrame(raf);raf=0;prev=0;
   if(!document.hidden&&!reduced.matches)raf=requestAnimationFrame(frame);
   else draw();
 }
 addEventListener('resize',()=>{if(raf)cancelAnimationFrame(raf);raf=0;rebuild();sync()},{passive:true});
 addEventListener('pointermove',e=>{aim=(e.clientX/Math.max(1,width)-.5)*2},{passive:true});
 document.addEventListener('visibilitychange',sync);
 reduced.addEventListener?.('change',sync);
 addEventListener('pagehide',()=>{if(raf)cancelAnimationFrame(raf)},{once:true});
 rebuild();sync();
})();
