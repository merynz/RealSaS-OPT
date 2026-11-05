/* BUSTED / Spatial Typing v4
 * Source-shaped AMBIENT TYPOGRAPHY ONLY; not compiler telemetry or proof.
 * Every spirit is stationary while typing. The text completes, the final caret
 * blinks for ~1 second, then the line dissolves. The NEXT cycle spawns at a
 * different collision-safe location with NO translation animation.
 */
(() => {
  'use strict';
  const chamber=document.querySelector('#awakenChamber');
  const stage=chamber?.querySelector('#matrixTypewriter');
  if(!chamber||!stage)return;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const nouns=['carrier','meshIR','skinIR','scene','source','topology','surface','joints','rig','weights','pose','charts','atlas','identity','occlusion','projection','geometry','mechanics','skeleton','motion','evidence','constraint','visibility'];
  const verbs=['project','compile','resolve','validate','observe','bind','attach','trace','sample','stitch','decode','verify','normalize','reconcile','fit','measure','transform','propagate','address','solve'];
  const fields=['canonical','world','local','depth','owner','parent','weights','joint','edge','surface','alpha','visible','camera','confidence','topology','identity','order','constraint'];
  const alphabet='01ABCDEF<>[]{}';
  const choose=a=>a[Math.floor(Math.random()*a.length)];
  const random=(a,b)=>a+Math.random()*(b-a);
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const collides=(a,b,pad=10)=>a.left<b.right+pad&&a.right>b.left-pad&&a.top<b.bottom+pad&&a.bottom>b.top-pad;
  const rect=(x,y,w,h)=>({left:x,top:y,right:x+w,bottom:y+h});
  let traces=[],bounds={width:0,height:0},obstacles=[],reserved=null;
  let raf=0,elapsed=0,last=0,paintAt=0,observed=false,resizeTimer=0;
  let previous=[];

  function line(max){
    const candidates=[];
    for(let i=0;i<35;i++){
      const n=choose(nouns),v=choose(verbs),f=choose(fields),id=Math.floor(random(1,256));
      candidates.push(choose([
        n+'.'+v+'('+f+');','bind('+n+', '+f+');',
        'const '+n+' = '+v+'('+id+');',
        'if ('+n+'.visible) '+v+'();',
        'scene.'+n+'.'+v+'();',
        n+'['+id+'].'+v+'();',
        'assert('+n+'.'+f+');',
        'tensor.'+v+'('+n+');'
      ]));
    }
    const eligible=candidates.filter(x=>x.length<=max&&!previous.includes(x));
    const text=eligible.length?choose(eligible):choose(['skin.bind();','rig.solve();','mesh.decode();']);
    previous.push(text);
    if(previous.length>32)previous.shift();
    return text;
  }

  function measure(){
    const c=chamber.getBoundingClientRect();
    bounds={width:c.width,height:c.height};
    const selectors=['.chamber-header','.chamber-docking-label','#dropzone','#inputSlotLedger',
      '.chamber-console .compile-action','.chamber-foot','.compile-monitor-heading','#compileMonitor'];
    obstacles=selectors.map(x=>chamber.querySelector(x)).filter(el=>{
      if(!el)return false;
      const cs=getComputedStyle(el);
      return cs.display!=='none'&&cs.visibility!=='hidden';
    }).map(el=>{
      const r=el.getBoundingClientRect();
      return {left:r.left-c.left,right:r.right-c.left,top:r.top-c.top,bottom:r.bottom-c.top};
    });
    reserved=null;
    if(bounds.width<760){
      const reserve=chamber.querySelector('#matrixMobileCodeReserve');
      if(reserve){
        const b=reserve.getBoundingClientRect();
        reserved={x:b.left-c.left+14,y:b.top-c.top+8,
          w:Math.max(100,b.width-28),h:Math.max(46,b.height-14)};
      }
    }
  }
  function permitted(t,x,y,pad=10){
    const r=rect(x,y,t.w,t.h);
    if(x<12||y<12||r.right>bounds.width-12||r.bottom>bounds.height-10)return false;
    if(obstacles.some(o=>collides(r,o,pad)))return false;
    return !traces.some(other=>other!==t&&collides(r,rect(other.x,other.y,other.w,other.h),12));
  }
  function randomPosition(t){
    if(reserved)return {
      x:reserved.x+random(0,Math.max(1,reserved.w-t.w)),
      y:reserved.y+random(0,Math.max(1,reserved.h-t.h))};
    // Two diffuse, irregular volumes around the protected central UI.
    const left=t.side==='left';
    const x=left?random(.015,.30)*bounds.width:random(.70,.97)*bounds.width-t.w*.35;
    const y=random(.13,.88)*bounds.height;
    return {x:clamp(x,14,bounds.width-t.w-14),y:clamp(y,15,bounds.height-t.h-15)};
  }
  function relocate(t){
    for(let n=0;n<100;n++){
      const p=randomPosition(t);
      const displacement=Math.hypot(p.x-t.x,p.y-t.y);
      if(!permitted(t,p.x,p.y))continue;
      if(t.hasPlaced&&displacement<45)continue;
      t.x=p.x;t.y=p.y;t.hasPlaced=true;
      t.el.style.setProperty('transform',`translate3d(${p.x.toFixed(1)}px,${p.y.toFixed(1)}px,0)`,'important');
      return true;
    }
    return false;
  }

  function beginCycle(t,initial=false){
    if(!initial)relocate(t); // change position only while the previous text is invisible
    t.lines=[line(t.max),line(t.max)];
    t.typeFinish=t.gap+t.lines[1].length*t.rate;
    t.holdFinish=t.typeFinish+1050; // ~1 second of completed text + blinking terminal mark
    t.fadeFinish=t.holdFinish+340;
    t.duration=t.fadeFinish+random(450,850);
  }
  function paint(){
    for(const t of traces){
      let local=elapsed-t.started;
      if(!reduced.matches){
        while(local>=t.duration){
          t.started+=t.duration;
          beginCycle(t);
          local=elapsed-t.started;
        }
      }
      if(reduced.matches){
        t.rows.forEach((row,i)=>{
          if(row.textContent!==t.lines[i])row.textContent=t.lines[i];
          row.classList.remove('typing','decoding','terminal-hold');
        });
        t.el.style.setProperty('--trace-opacity',t.baseOpacity.toFixed(3));
        continue;
      }
      let alpha=1;
      if(local>=t.holdFinish)alpha=1-clamp((local-t.holdFinish)/340,0,1);
      t.el.style.setProperty('--trace-opacity',(t.baseOpacity*alpha).toFixed(3));
      t.rows.forEach((row,i)=>{
        const age=local-(i?t.gap:0);
        const n=clamp(Math.floor(age/t.rate),0,t.lines[i].length);
        const typing=age>=0&&n<t.lines[i].length;
        let txt=t.lines[i].slice(0,n);
        if(typing&&n>2){
          const tail=Math.min(3,n),stable=n-tail,seed=Math.floor((elapsed+i*71)/90);
          txt=t.lines[i].slice(0,stable)+Array.from({length:tail},(_,k)=>
            (seed+k)%4===0?t.lines[i][stable+k]:alphabet[(seed*5+k*3)%alphabet.length]).join('');
        }
        if(local>=t.fadeFinish)txt='';
        if(row.textContent!==txt)row.textContent=txt;
        row.classList.toggle('typing',typing);
        row.classList.toggle('decoding',typing&&n>2);
        row.classList.toggle('terminal-hold',i===1&&local>=t.typeFinish&&local<t.holdFinish);
      });
    }
  }
  function build(){
    stage.replaceChildren();traces=[];measure();
    if(bounds.width<120||bounds.height<150)return;
    const small=Boolean(reserved),tablet=bounds.width<1170;
    const count=small?1:tablet?8:11;
    for(let i=0;i<count;i++){
      const depth=random(.25,.98),side=i%2?'right':'left';
      const w=small?Math.min(330,reserved.w):tablet?175:Math.min(248,Math.max(160,bounds.width*.18));
      const h=small?42:54;
      const font=small?9.4:tablet?random(9.2,11):random(10.6,12.4);
      const el=document.createElement('div');
      el.className='matrix-trace spatial-code-spirit';
      el.dataset.depth=depth.toFixed(2);
      el.style.cssText='left:0px;top:0px;position:absolute;';
      el.style.width=w+'px';el.style.height=h+'px';
      el.style.setProperty('--trace-font',font.toFixed(1)+'px');
      const rows=[document.createElement('span'),document.createElement('span')];
      for(const row of rows){row.className='matrix-trace-line';el.append(row)}
      const max=Math.max(12,Math.floor((w-12)/(font*.67)));
      const t={el,rows,depth,side,w,h,x:0,y:0,max,hasPlaced:false,
        baseOpacity:.36+depth*.35,rate:random(54,88),gap:random(450,850),
        started:-random(400,1550)-i*120,lines:[],duration:1};
      if(!relocate(t))continue;
      beginCycle(t,true);
      traces.push(t);stage.append(el);
    }
    paint();
  }

  function tick(now){
    raf=0;
    if(document.hidden||!observed||reduced.matches)return;
    elapsed+=Math.min(100,Math.max(0,now-last));last=now;
    if(now-paintAt>=40){paint();paintAt=now}
    raf=requestAnimationFrame(tick);
  }
  function sync(){
    if(raf)cancelAnimationFrame(raf);
    raf=0;last=performance.now();
    if(observed&&!document.hidden&&!reduced.matches)raf=requestAnimationFrame(tick);
    else paint();
  }
  function resize(){
    clearTimeout(resizeTimer);
    resizeTimer=setTimeout(()=>{build();sync()},120);
  }
  const io=new IntersectionObserver(entries=>{
    observed=Boolean(entries[0]?.isIntersecting);sync();
  },{threshold:.01});
  io.observe(chamber);
  addEventListener('resize',resize,{passive:true});
  document.fonts?.ready?.then(resize);
  reduced.addEventListener?.('change',()=>{build();sync()});
  document.addEventListener('visibilitychange',sync);
  addEventListener('pagehide',()=>{
    if(raf)cancelAnimationFrame(raf);
    clearTimeout(resizeTimer);io.disconnect();
  },{once:true});
  build();sync();
})();
