/* BUSTED / Spatial Code Drift v3
 * Decorative source-shaped typing, NOT live compiler telemetry.
 * Free-positioned, obstacle-aware, asynchronous code spirits:
 * no fixed rows, no regular columns, and no animation over product controls.
 */
(() => {
  'use strict';
  const chamber = document.querySelector('#awakenChamber');
  const stage = chamber?.querySelector('#matrixTypewriter');
  if (!chamber || !stage) return;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const nouns = ['carrier','meshIR','skinIR','scene','source','topology','surface','joints','rig','weights','pose','charts','atlas','identity','occlusion','projection','geometry','mechanics','skeleton','motion','evidence','constraint','visibility'];
  const verbs = ['project','compile','resolve','validate','observe','bind','attach','trace','sample','stitch','decode','verify','normalize','reconcile','fit','measure','transform','propagate','address','solve'];
  const fields = ['canonical','world','local','depth','owner','parent','weights','joint','edge','surface','alpha','visible','camera','confidence','topology','identity','order','constraint'];
  const symbols = '01ABCDEF<>[]{}';
  const random = (a,b) => a + Math.random() * (b-a);
  const choose = a => a[Math.floor(Math.random()*a.length)];
  const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
  const ease = t => t*t*(3-2*t);
  const traces = [];
  let obstacles = [], bounds = {width:0,height:0}, reserved = null;
  let raf = 0, elapsed = 0, last = 0, paintAt = 0, visible = false, previous = [];
  const mobile = () => chamber.clientWidth < 760;
  const rect = (x,y,w,h) => ({left:x,top:y,right:x+w,bottom:y+h});
  const overlap = (a,b,pad=12) => a.left < b.right+pad && a.right > b.left-pad &&
    a.top < b.bottom+pad && a.bottom > b.top-pad;

  function line(max) {
    const candidates = [];
    for (let i=0; i<32; i++) {
      const a=choose(nouns),b=choose(verbs),f=choose(fields),n=String(Math.floor(random(1,256)));
      candidates.push(choose([
        a+'.'+b+'('+f+');', 'const '+a+' = '+b+'('+n+');',
        'bind('+a+', '+f+');', 'if ('+a+'.visible) '+b+'();',
        'scene.'+a+'.'+b+'();', 'assert('+a+'.'+f+');',
        'tensor.'+b+'('+a+');', a+'['+n+'].'+b+'();'
      ]));
    }
    const eligible = candidates.filter(s => s.length<=max && !previous.includes(s));
    const result = eligible.length ? choose(eligible) : choose(['rig.solve();','mesh.decode();','skin.bind();','scene.trace();']);
    previous.push(result);
    if (previous.length > 35) previous.shift();
    return result;
  }

  function measure() {
    const b=chamber.getBoundingClientRect();
    bounds={width:b.width,height:b.height};
    const selectors=['.chamber-header','.chamber-docking-label','#dropzone','#inputSlotLedger',
      '.chamber-console .compile-action','.chamber-foot','.compile-monitor-heading','#compileMonitor'];
    obstacles=selectors.map(sel=>chamber.querySelector(sel)).filter(el=>{
      if (!el) return false;
      const style=getComputedStyle(el);
      return style.display!=='none' && style.visibility!=='hidden';
    }).map(el=>{
      const r=el.getBoundingClientRect();
      return {left:r.left-b.left,right:r.right-b.left,top:r.top-b.top,bottom:r.bottom-b.top};
    });
    const rail=chamber.querySelector('#matrixMobileCodeReserve');
    if (mobile() && rail) {
      const r=rail.getBoundingClientRect();
      reserved={left:r.left-b.left+12,top:r.top-b.top+8,
        width:Math.max(100,r.width-24),height:Math.max(55,r.height-14)};
    } else reserved=null;
  }

  function otherRects(except) {
    return traces.filter(t=>t!==except).map(t=>rect(t.x,t.y,t.w,t.h));
  }
  function permitted(x,y,w,h,except,pad=9) {
    const candidate=rect(x,y,w,h);
    if (candidate.left<12 || candidate.right>bounds.width-12 ||
        candidate.top<12 || candidate.bottom>bounds.height-10) return false;
    if (obstacles.some(b=>overlap(candidate,b,pad))) return false;
    if (otherRects(except).some(b=>overlap(candidate,b,12))) return false;
    return true;
  }

  function candidate(t,initial=false) {
    if (reserved) {
      return {x:reserved.left+random(0,Math.max(1,reserved.width-t.w)),
        y:reserved.top+random(0,Math.max(1,reserved.height-t.h))};
    }
    const W=bounds.width,H=bounds.height;
    const isLeft=t.side==='left';
    const lo=isLeft?.025:.73, hi=isLeft?.28:.975;
    let x=random(lo,hi)*W;
    x=clamp(x,16,W-t.w-16);
    let y=random(.12,.91)*H;
    y=clamp(y,16,H-t.h-16);
    if (!initial) {
      // Wander locally, rather than teleporting between numbered grid cells.
      const reach=Math.min(105,Math.max(34,W*.095));
      x=clamp(t.x+random(-reach,reach),16,W-t.w-16);
      y=clamp(t.y+random(-93,93),16,H-t.h-16);
    }
    return {x,y};
  }

  function routeClear(t,target) {
    for (let i=1;i<=12;i++) {
      const s=i/12;
      const x=t.x+(target.x-t.x)*s,y=t.y+(target.y-t.y)*s;
      // Other moving spirits are handled again at each rendered frame.
      const box=rect(x,y,t.w,t.h);
      if (box.left<12 || box.right>bounds.width-12 || box.top<12 || box.bottom>bounds.height-10 ||
          obstacles.some(o=>overlap(box,o,13))) return false;
    }
    return true;
  }

  function findLocation(t,initial=false) {
    for(let i=0;i<110;i++) {
      const c=candidate(t,initial);
      if (!permitted(c.x,c.y,t.w,t.h,t)) continue;
      if (!initial && (!routeClear(t,c) || Math.hypot(c.x-t.x,c.y-t.y)<20)) continue;
      return c;
    }
    return null;
  }

  function put(t) {
    t.el.style.setProperty('transform','translate3d('+t.x.toFixed(1)+'px,'+t.y.toFixed(1)+'px,0)','important');
    t.el.style.setProperty('--trace-opacity',t.opacity.toFixed(3));
  }

  function build() {
    stage.replaceChildren();
    traces.length=0;
    measure();
    if(bounds.width<120||bounds.height<150)return;
    const small=Boolean(reserved),tablet=bounds.width<1170;
    const count=small?1:tablet?7:11;
    for(let i=0;i<count;i++) {
      const side=i%2?'right':'left';
      const depth=random(.18,.95);
      const w=small?Math.min(340,reserved.width):Math.min(tablet?180:250,Math.max(150,bounds.width*(tablet?.185:.18)));
      const h=small?42:54;
      const el=document.createElement('div');
      el.className='matrix-trace spatial-code-spirit';
      el.dataset.depth=depth.toFixed(3);
      el.style.left='0px';el.style.top='0px';
      el.style.width=w.toFixed(1)+'px';el.style.height=h+'px';
      el.style.setProperty('--trace-font',(small?9:random(tablet?9.5:10.5,tablet?11.5:12.5)).toFixed(1)+'px');
      const rows=[document.createElement('span'),document.createElement('span')];
      rows.forEach(row=>{row.className='matrix-trace-line';el.append(row)});
      const max=Math.max(13,Math.floor((w-10)/((small?9:11.2)*.68)));
      const t={el,rows,side,depth,w,h,x:0,y:0,startX:0,startY:0,targetX:0,targetY:0,
        travelling:false,hold:0,movedAt:0,duration:random(11000,21000),opacity:.26+depth*.39,
        offset:300+i*430+random(0,650),interval:random(11200,19000),
        rate:random(65,115),gap:random(900,1550),lastCycle:-1,lines:[line(max),line(max)],max};
      const start=findLocation(t,true);
      if (!start) continue;
      t.x=t.startX=t.targetX=start.x;t.y=t.startY=t.targetY=start.y;
      t.hold=elapsed+random(2200,8500);
      stage.append(el);
      traces.push(t);put(t);
    }
    renderText();
  }

  function animatePositions() {
    for(const t of traces) {
      if (reserved || reduced.matches) continue;
      if (!t.travelling) {
        if (elapsed<t.hold) continue;
        const target=findLocation(t);
        if (!target) {t.hold=elapsed+random(2000,5000);continue}
        t.startX=t.x;t.startY=t.y;t.targetX=target.x;t.targetY=target.y;
        t.movedAt=elapsed;t.duration=random(10000,23000);t.travelling=true;
      }
      const progress=clamp((elapsed-t.movedAt)/t.duration,0,1);
      const k=ease(progress);
      const x=t.startX+(t.targetX-t.startX)*k;
      const y=t.startY+(t.targetY-t.startY)*k;
      if (permitted(x,y,t.w,t.h,t,10)) {
        t.x=x;t.y=y;put(t);
      } else {
        // Fail closed: never paint animated code over the input UI.
        t.travelling=false;t.hold=elapsed+random(500,2000);
      }
      if(progress>=1){t.travelling=false;t.hold=elapsed+random(1700,6200)}
    }
  }

  function renderText() {
    traces.forEach(t=>{
      const total=elapsed+t.offset;
      const cycle=Math.floor(total/t.interval);
      if(cycle!==t.lastCycle){t.lines=[line(t.max),line(t.max)];t.lastCycle=cycle}
      const local=total%t.interval;
      t.rows.forEach((row,i)=>{
        const age=local-i*t.gap;
        const n=reduced.matches?t.lines[i].length:clamp(Math.floor(age/t.rate),0,t.lines[i].length);
        const typing=!reduced.matches&&age>=0&&n<t.lines[i].length;
        let txt=t.lines[i].slice(0,n);
        if(typing && n>2){
          const scramble=Math.min(3,n),base=n-scramble,shift=Math.floor((elapsed+i*89)/100);
          let tail='';
          for(let k=0;k<scramble;k++){
            tail+=((shift+k)%4===0)?t.lines[i][base+k]:symbols[(shift*3+k*7)%symbols.length];
          }
          txt=t.lines[i].slice(0,base)+tail;
        }
        if(row.textContent!==txt)row.textContent=txt;
        row.classList.toggle('typing',typing);
        row.classList.toggle('decoding',typing && n>2);
      });
    });
  }

  function frame(now) {
    raf=0;
    if(document.hidden || !visible || reduced.matches) return;
    elapsed+=Math.max(0,Math.min(100,now-last));
    last=now;
    if(now-paintAt>=45){animatePositions();renderText();paintAt=now}
    raf=requestAnimationFrame(frame);
  }
  function sync() {
    if(raf)cancelAnimationFrame(raf);
    raf=0;
    last=performance.now();
    if(visible && !document.hidden && !reduced.matches)raf=requestAnimationFrame(frame);
    else renderText();
  }
  let resizeTimer=0;
  function resize() {
    clearTimeout(resizeTimer);
    resizeTimer=setTimeout(()=>{build();sync()},130);
  }
  const observer=new IntersectionObserver(entries=>{
    visible=Boolean(entries[0]?.isIntersecting);sync();
  },{threshold:.01});
  observer.observe(chamber);
  addEventListener('resize',resize,{passive:true});
  document.fonts?.ready?.then(resize);
  reduced.addEventListener?.('change',()=>{build();sync()});
  document.addEventListener('visibilitychange',sync);
  addEventListener('pagehide',()=>{if(raf)cancelAnimationFrame(raf);clearTimeout(resizeTimer);observer.disconnect()},{once:true});
  build();sync();
})();
