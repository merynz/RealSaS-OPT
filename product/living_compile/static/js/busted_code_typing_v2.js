/* BUSTED: typed compiler-language study.
 * Scoped ONLY to #awakenChamber. The snippets are illustrative source-shaped
 * fragments, NOT live status, proof, actual source excerpts or backend activity.
 * Thousands of unique combinations, never a repeated static fallback string.
 */
(() => {
 'use strict';
 const area=document.querySelector('#awakenChamber');
 const root=area?.querySelector('#matrixTypewriter');
 if(!area||!root)return;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const nouns=['carrier','mesh','meshIR','skinIR','scene','source','topology','surface','normals','joints','rig','weights','pose','deform','charts','atlas','attachment','occlusion','identity','projection','visibility','motion','camera','evidence','geometry','mechanics','constraint','skeleton','sourceView','renderChart','depth','product'];
 const verbs=['project','compile','resolve','validate','observe','bind','attach','seal','trace','compute','sample','stitch','decode','verify','normalize','reconcile','fit','measure','map','estimate','transform','propagate','address','solve'];
 const fields=['canonical','world','local','normal','depth','owner','parent','weights','joint','edge','vertex','surface','chart','alpha','visible','camera','confidence','topology','identity','pose','index','order','material','constraint'];
 const pick=a=>a[Math.floor(Math.random()*a.length)];
 const decodeAlphabet='01ABCDEF<>[]{}';
 const number=()=>String(Math.floor(Math.random()*256)).padStart(2,'0');
 const scalar=()=>Math.random().toFixed(3);
 let previous=[],traces=[],timer=null,time=0,observed=false,last=performance.now();
 function generate(max){
   const candidates=[];
   for(let i=0;i<35;i++){
     const n=pick(nouns),v=pick(verbs),f=pick(fields),id=number(),q=scalar();
     candidates.push(pick([
       n+'.'+v+'('+f+');',
       n+'.'+f+' = '+q+';',
       'const '+n+' = '+v+'('+id+');',
       'assert('+n+'.'+f+');',
       n+'['+id+'].'+v+'();',
       'bind('+n+', '+f+');',
       'if ('+n+'.visible) '+v+'();',
       'tensor.'+v+'('+n+');',
       'proof.'+v+'('+f+');',
       'scene.'+n+'.'+v+'();'
     ]));
   }
   const fit=candidates.filter(s=>s.length<=max&&!previous.includes(s));
   const output=fit.length?pick(fit):pick(nouns).slice(0,Math.max(2,max-8))+'.fit();';
   previous.push(output);if(previous.length>25)previous.shift();
   return output;
 }
 function collide(a,b,pad){
   return a.left<b.right+pad&&a.right>b.left-pad&&a.top<b.bottom+pad&&a.bottom>b.top-pad;
 }
 function layout(){
   root.replaceChildren();traces=[];
   const bounds=area.getBoundingClientRect();
   const width=bounds.width,height=bounds.height;
   if(width<100||height<140)return;
   const mobile=width<760,tablet=width<1170;
   const lineHeight=mobile?43:47;
   const slots=[];
   if(mobile){
     // The stage explicitly reserves mobile code space below upload controls.
     // Use actual layout geometry instead of guessing percentage positions.
     const reserve=area.querySelector('#matrixMobileCodeReserve');
     if(reserve){
       const r=reserve.getBoundingClientRect();
       slots.push({
         left:r.left-bounds.left+12,
         top:r.top-bounds.top+12,
         width:Math.max(0,r.width-24),
         height:70
       });
     }
   }else{
     const trackWidth=Math.min(tablet?225:292,width*(tablet?.25:.26));
     for(const frac of [.24,.39,.57,.75,.87]){
       slots.push({left:width*.035,top:height*frac,width:trackWidth,height:lineHeight});
       slots.push({left:width-trackWidth-width*.035,top:height*frac,width:trackWidth,height:lineHeight});
     }
   }
   const selector=['.chamber-header','.chamber-docking-label','#dropzone','#inputSlotLedger',
     '.chamber-console .compile-action','.chamber-foot','.compile-monitor-heading','#compileMonitor'];
   const obstacles=selector.map(s=>area.querySelector(s)).filter(el=>{
     if(!el)return false;
     const st=getComputedStyle(el);
     return st.display!=='none'&&st.visibility!=='hidden';
   }).map(el=>{
     const b=el.getBoundingClientRect();
     return {left:b.left-bounds.left,right:b.right-bounds.left,top:b.top-bounds.top,bottom:b.bottom-bounds.top};
   });
   const occupied=[];
   for(const [i,slot] of slots.entries()){
     if(slot.left<12||slot.left+slot.width>width-11||slot.top+slot.height>height-12)continue;
     const rect={left:slot.left,top:slot.top,right:slot.left+slot.width,bottom:slot.top+slot.height};
     if(obstacles.some(b=>collide(rect,b,12)))continue;
     if(occupied.some(b=>collide(rect,b,12)))continue;
     occupied.push(rect);
     const depth=.28+Math.random()*.62;
     const font=mobile?8.5+Math.random()*1.0:10+Math.random()*1.8;
     const lane=document.createElement('div');
     lane.className='matrix-trace';
     lane.dataset.depth=depth.toFixed(3);
     lane.style.left=slot.left.toFixed(1)+'px';
     lane.style.top=slot.top.toFixed(1)+'px';
     lane.style.width=slot.width.toFixed(1)+'px';
     lane.style.height=slot.height.toFixed(1)+'px';
     lane.style.setProperty('--trace-depth',depth.toFixed(3));
     lane.style.setProperty('--trace-font',font.toFixed(1)+'px');
     lane.style.setProperty('--trace-opacity',(.32+depth*.32).toFixed(2));
     const rowA=document.createElement('span'),rowB=document.createElement('span');
     rowA.className='matrix-trace-line';rowB.className='matrix-trace-line';
     lane.append(rowA,rowB);root.append(lane);
     const max=Math.max(12,Math.floor((slot.width-14)/(font*.64)));
     const initial=i*780+1500;
     traces.push({lane,rows:[rowA,rowB],max,rate:55+Math.random()*50,
       interval:12500+Math.random()*5600,offset:initial,lastCycle:-1,
       lines:[generate(max),generate(max)],gap:850+Math.random()*450});
   }
   if(reduced.matches)paint();
 }
 function paint(){
   for(const t of traces){
     const total=time+t.offset;
     const cycle=Math.floor(total/t.interval);
     if(cycle!==t.lastCycle){t.lines=[generate(t.max),generate(t.max)];t.lastCycle=cycle}
     const local=total%t.interval;
     t.rows.forEach((row,i)=>{
       const age=local-i*t.gap;
       const n=reduced.matches?t.lines[i].length:Math.max(0,Math.min(t.lines[i].length,Math.floor(age/t.rate)));
       const activelyTyping=!reduced.matches&&age>=0&&n<t.lines[i].length;
       const decoding=activelyTyping&&n>1&&Math.floor((time+t.offset)/180)%5===0;
       const txt=decoding
         ?t.lines[i].slice(0,n-1)+decodeAlphabet[(Math.floor(time/90)+i)%decodeAlphabet.length]
         :t.lines[i].slice(0,n);
       if(row.textContent!==txt)row.textContent=txt;
       row.classList.toggle('typing',activelyTyping);
       row.classList.toggle('decoding',decoding);
     });
   }
 }
 function tick(){
   if(document.hidden||reduced.matches||!observed)return;
   const now=performance.now();
   time+=Math.max(0,Math.min(300,now-last));last=now;
   paint();
 }
 function sync(){
   if(timer!==null){clearInterval(timer);timer=null}
   last=performance.now();
   if(!document.hidden&&observed&&!reduced.matches)timer=setInterval(tick,90);
   else paint();
 }
 let resizeJob=null;
 const resize=()=>{clearTimeout(resizeJob);resizeJob=setTimeout(()=>{layout();sync()},110)};
 addEventListener('resize',resize,{passive:true});
 const io=new IntersectionObserver(entries=>{
   observed=Boolean(entries[0]?.isIntersecting);sync();
 },{threshold:.01});
 io.observe(area);
 reduced.addEventListener?.('change',sync);
 document.addEventListener('visibilitychange',sync);
 document.fonts?.ready?.then(()=>{layout();sync()});
 addEventListener('pagehide',()=>{if(timer!==null)clearInterval(timer);io.disconnect()},{once:true});
 layout();sync();
})();
