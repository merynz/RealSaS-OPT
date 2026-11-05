/**
 * BUSTED / Reactor Ignition v3
 *
 * A one-shot procedural VFX, driven only by a genuine AWAKEN click.
 * Ambient rendering: ZERO. Canvas is clear and RAF stops at the end.
 * The VFX is not compiler progress, a ProductProof signal or puppet evidence.
 *
 * Sequence: ion prime → plasma accumulation → white-hot ignition → afterglow.
 * Use a ResizeObserver to tie the ACTUAL CTA geometry to the source viewport,
 * not merely to the outer .compile-action container.
 */
const DURATION_MS = 1680;
const CHARGE_END_MS = 900;
const IGNITION_END_MS = 1190;
const TWO_PI = Math.PI * 2;
const clamp = (x, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
const smooth = (x) => {const a=clamp(x);return a*a*(3-2*a)};
const mix = (a,b,t)=>a+(b-a)*t;
const frac = n=>n-Math.floor(n);
function noise(seed) {return frac(Math.sin(seed * 127.1 + seed * seed * .031) * 43758.5453)}
function rgba(r,g,b,a){return `rgba(${r},${g},${b},${clamp(a)})`;}

export function installAwakenReactor(button, sourceViewport) {
  if (!(button instanceof HTMLButtonElement) || !(sourceViewport instanceof Element))
    throw new TypeError('Reactor needs the real AWAKEN button and source viewport');
  const canvas=document.createElement('canvas');
  canvas.id='awakenIgnitionCanvas';
  canvas.className='awaken-ignition-canvas';
  canvas.setAttribute('aria-hidden','true');
  button.prepend(canvas);
  const ctx=canvas.getContext('2d', {alpha:true});
  if (!ctx) throw new Error('2D canvas unavailable');
  let running=false,frame=0,begin=0,resolveCycle=null,generation=0;
  let width=0,height=0,dpr=1;
  const reduced=()=>window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const sparks=Array.from({length:54},(_,i)=>{
    const angle=noise(i*1.33+2.3)*TWO_PI;
    const speed=mix(30,200,noise(i*3.17+7.2));
    return {
      x:noise(i*2.1+5.7)*.25-.125,
      y:noise(i*4.7+1.3)*.9+.05,
      vx:Math.cos(angle)*speed,vy:Math.sin(angle)*speed*.65,
      offset:noise(i*3.43+9.7)*.13,
      life:mix(.33,.75,noise(i*6.9+5.0)),
      hot:noise(i*2.4+4.7)>.32,
    };
  });

  function fitToArtwork(){
    const card=sourceViewport.getBoundingClientRect();
    if(card.width>0 && Number.isFinite(card.width))
      button.style.setProperty('--awaken-fit-width',`${card.width.toFixed(2)}px`);
  }
  const resizeObserver=new ResizeObserver(fitToArtwork);
  resizeObserver.observe(sourceViewport);
  addEventListener('resize',fitToArtwork,{passive:true});
  fitToArtwork();

  function measure(){
    const r=button.getBoundingClientRect();
    const w=Math.max(1,Math.round(r.width)),h=Math.max(1,Math.round(r.height));
    dpr=Math.min(1.6,Math.max(1,window.devicePixelRatio||1));
    const aw=Math.round(w*dpr),ah=Math.round(h*dpr);
    if(canvas.width!==aw || canvas.height!==ah) {canvas.width=aw;canvas.height=ah}
    ctx.setTransform(dpr,0,0,dpr,0,0);
    width=w;height=h;
    return [w,h];
  }

  function glowLine(x1,y1,x2,y2,color,blur,lineWidth){
    ctx.save();
    ctx.strokeStyle=color;ctx.lineWidth=lineWidth;
    ctx.shadowBlur=blur;ctx.shadowColor=color;
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
    ctx.restore();
  }

  function plasmaFilaments(front, intensity, t) {
    if(front<4 || intensity<=0) return;
    const len=front;
    ctx.save();ctx.beginPath();ctx.rect(1,1,len,height-2);ctx.clip();
    for(let strand=0;strand<9;strand++){
      const y0=height*(.12+strand*.096);
      const hot=strand%3===0;
      const gradient=ctx.createLinearGradient(0,y0,Math.max(1,len),y0);
      gradient.addColorStop(0,rgba(65,105,232,.12*intensity));
      gradient.addColorStop(.55,rgba(161,124,251,(hot?.49:.33)*intensity));
      gradient.addColorStop(1,rgba(255,hot?218:145,hot?164:236,(hot?.8:.52)*intensity));
      ctx.beginPath();
      for(let x=0;x<=len+8;x+=8){
        const y=y0+
          Math.sin(x*.046+strand*2.9+t*12)*mix(1.1,3.7,intensity)+
          Math.sin(x*.112-strand*1.8-t*27)*1.5+
          Math.sin(x*.18+strand*1.2+t*39)*.56;
        if(x===0)ctx.moveTo(0,y);else ctx.lineTo(Math.min(len,x),y);
      }
      ctx.lineWidth=hot?1.3:.75;ctx.strokeStyle=gradient;ctx.stroke();
    }
    ctx.restore();
  }

  function chargeField(t) {
    const p=smooth((t-.025)/.665);
    if(p<=0)return;
    const front=mix(0,width-2,p);
    const visible=clamp(p*1.65+.12);
    const gradient=ctx.createLinearGradient(0,0,width,0);
    gradient.addColorStop(0,rgba(53,86,189,.16*visible));
    gradient.addColorStop(.25,rgba(89,85,215,.34*visible));
    gradient.addColorStop(.55,rgba(148,70,193,.42*visible));
    gradient.addColorStop(.78,rgba(213,81,124,.46*visible));
    gradient.addColorStop(1,rgba(248,145,73,.64*visible));
    ctx.fillStyle=gradient;
    ctx.fillRect(1,1,front,height-2);

    // Spectral caustics: long narrow light pools, not a rectangular wipe.
    const caustic=ctx.createRadialGradient(front, height*.5,1,front,height*.5,height*.9);
    caustic.addColorStop(0,rgba(255,211,142,.52*visible));
    caustic.addColorStop(.26,rgba(255,130,93,.31*visible));
    caustic.addColorStop(.7,rgba(180,84,239,.14*visible));
    caustic.addColorStop(1,'rgba(0,0,0,0)');
    ctx.save();ctx.beginPath();ctx.rect(0,0,front,height);ctx.clip();
    ctx.fillStyle=caustic;ctx.fillRect(Math.max(0,front-height),0,height*2,height);
    ctx.restore();

    plasmaFilaments(front,visible,t);
    // Three discrete charging traces around the inside perimeter.
    const traceGradient=ctx.createLinearGradient(0,0,width,0);
    traceGradient.addColorStop(0,rgba(110,148,255,.18));
    traceGradient.addColorStop(.7,rgba(213,144,255,.75));
    traceGradient.addColorStop(1,rgba(255,229,189,.96));
    for(const y of [2,height-3]){
      glowLine(2,y,front,y,traceGradient,5,1.1);
    }
    glowLine(front,3,front,height-3,rgba(255,235,190,.88),10,1.45);
    glowLine(front-2,7,front-2,height-7,rgba(255,141,99,.53),16,5);

    // Ion particles concentrate at the moving leading edge.
    for(let k=0;k<23;k++){
      const phase=noise(k*3.31+t*10),x=front-mix(4,45,noise(k*5.1+1.3));
      const y=4+(height-8)*noise(k*8.81+t*3.4);
      if(x<=3 || x>front || phase<.28)continue;
      const r=mix(.32,1.4,noise(k*9.2+4.6));
      ctx.fillStyle=rgba(225+k%2*30,k%3?175:235,255,(phase-.28)*1.1*visible);
      ctx.fillRect(x,y,r,r);
    }
  }

  function ignitionBlast(t){
    const attack=smooth((t-.67)/.12);
    const decay=1-smooth((t-.83)/.13);
    const heat=attack*decay;
    if(heat<=0)return;
    const centerX=width*.64,centerY=height*.5;
    // Compression into a white-hot core, then an expanding shock front.
    const size=mix(12,width*.9,smooth((t-.72)/.18));
    const flash=ctx.createRadialGradient(centerX,centerY,2,centerX,centerY,size);
    flash.addColorStop(0,rgba(255,249,233,1.0*heat));
    flash.addColorStop(.12,rgba(255,221,158,.98*heat));
    flash.addColorStop(.34,rgba(255,123,65,.75*heat));
    flash.addColorStop(.68,rgba(205,58,128,.26*heat));
    flash.addColorStop(1,'rgba(0,0,0,0)');
    ctx.save();
    ctx.translate(centerX,centerY);
    ctx.scale(1,.48);
    ctx.translate(-centerX,-centerY);
    ctx.fillStyle=flash;
    ctx.fillRect(-width,-height,width*3,height*3);
    ctx.restore();

    const spread=smooth((t-.76)/.15);
    const ringW=mix(3,width*.68,spread);
    ctx.save();ctx.globalAlpha=heat*.9;
    ctx.strokeStyle=rgba(255,218,173,.86);ctx.lineWidth=mix(2,.6,spread);
    ctx.shadowBlur=14;ctx.shadowColor='rgba(255,136,76,.92)';
    ctx.beginPath();ctx.ellipse(centerX,centerY,ringW,height*(.22+.5*spread),0,0,TWO_PI);
    ctx.stroke();ctx.restore();

    // High-temperature horizontal lances flick out of the ignition core.
    for(let i=0;i<5;i++){
      const y=centerY+(i-2)*height*.12;
      const reach=width*(.24+.18*noise(i*4.2));
      const phase=1-smooth(Math.abs(t-(.79+i*.013))/.10);
      if(phase<=0)continue;
      glowLine(centerX-reach*.65,y,centerX+reach,y,rgba(255,234-i*17,202-i*29,.6*phase),11,i===2?2.8:1.15);
    }
    for(let i=0;i<sparks.length;i++){
      const p=sparks[i],life=(t-(.76+p.offset))/.25;
      if(life<0||life>p.life)continue;
      const fade=Math.pow(1-life/p.life,1.4);
      const x=centerX+p.x*width+p.vx*life*.34;
      const y=centerY+(p.y-.5)*height*.68+p.vy*life*.22;
      const tail=mix(2,8,fade);
      glowLine(x-p.vx*.02*tail,y-p.vy*.015*tail,x,y,
        p.hot?rgba(255,201,125,fade*.93):rgba(190,165,255,fade*.75),3,p.hot?1.15:.75);
    }
  }

  function afterglow(t){
    if(t<.80)return;
    const fade=1-smooth((t-.84)/.16);
    if(fade<=0)return;
    const coreX=width*.64;
    const halo=ctx.createRadialGradient(coreX,height*.5,2,coreX,height*.5,width*.65);
    halo.addColorStop(0,rgba(255,150,92,.25*fade));
    halo.addColorStop(.28,rgba(204,99,136,.15*fade));
    halo.addColorStop(.76,rgba(99,90,203,.05*fade));
    halo.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle=halo;ctx.fillRect(0,0,width,height);
    for(let i=0;i<17;i++){
      const x=noise(i*2.3+3.7)*width;
      const y=noise(i*7.1+4.4)*height;
      ctx.fillStyle=i%3?rgba(155,151,255,fade*.39):rgba(251,149,117,fade*.61);
      ctx.fillRect(x,y,.9+(i%3)*.28,.8);
    }
  }

  function draw(now){
    if(!running)return;
    const t=clamp((now-begin)/DURATION_MS);
    const dt=now-begin;
    ctx.clearRect(0,0,width,height);
    ctx.save();ctx.beginPath();ctx.rect(1,1,width-2,height-2);ctx.clip();
    ctx.globalCompositeOperation='screen';
    if(dt<=CHARGE_END_MS)chargeField(t);
    else if(dt<=IGNITION_END_MS){
      chargeField(.70);ignitionBlast(t);
    }else{
      ctx.save();
      ctx.globalAlpha=1-smooth((t-.72)/.24);
      chargeField(.70);
      ctx.restore();
      afterglow(t);
    }
    ctx.restore();
    if(dt>=DURATION_MS)finish();
  }
  function finish(){
    if(frame)cancelAnimationFrame(frame);
    frame=0;
    running=false;
    ctx.clearRect(0,0,width,height);
    canvas.hidden=true;
    button.classList.remove('is-priming','is-charging','is-igniting','is-recovering');
    button.dataset.ignitionPhase='rest';
    const done=resolveCycle;
    resolveCycle=null;
    if(done)done();
  }

  function play(){
    if(running)return Promise.resolve(false);
    if(reduced()){
      button.dataset.ignitionPhase='igniting';
      button.classList.add('is-igniting');
      return new Promise(resolve=>setTimeout(()=>{
        button.classList.remove('is-igniting');
        button.dataset.ignitionPhase='rest';
        resolve(true);
      },220));
    }
    fitToArtwork();measure();
    generation++;
    button.dataset.ignitionGeneration=String(generation);
    running=true;canvas.hidden=false;
    button.dataset.ignitionPhase='charging';
    button.classList.add('is-charging');
    begin=performance.now();
    // Full canvas phase is time-driven; CSS phase changes are presentation only.
    const complete=new Promise(resolve=>{resolveCycle=()=>resolve(true)});
    let lastPhase='charging';
    const runner=(now)=>{
      if(!running)return;
      const ms=now-begin;
      const newPhase=ms<CHARGE_END_MS?'charging':
        ms<IGNITION_END_MS?'igniting':'recovering';
      if(lastPhase!==newPhase){
        lastPhase=newPhase;
        button.classList.remove('is-charging','is-igniting','is-recovering');
        button.classList.add('is-'+newPhase);
        button.dataset.ignitionPhase=newPhase;
      }
      draw(now);
      // RAF scheduling is owned by the wrapper; no duplicated animation loop.
      if(running)frame=requestAnimationFrame(runner);
    };
    // Separate draw from scheduling; one RAF drives the entire event.
    frame=requestAnimationFrame(runner);
    return complete;
  }
  return {play,fitToArtwork,get active(){return running}};
}
