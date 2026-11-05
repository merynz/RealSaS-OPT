/* BUSTED / layered computational atmosphere; decorative, not live compiler evidence. */
(() => {
  'use strict';
  function createScene(root,typeRoot,role){
  if(!root||!typeRoot)return;
  const narrow=window.matchMedia('(max-width: 780px)');
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  const glyphs='0123456789ABCDEFアイウエオカキクケコサシスセソタチツテト<>[]{}:/';
  const nouns=['puppet','carrier','scene','mesh','rig','skin','joint','topology','surface','depth','source','appearance','occlusion','motion','constraint','identity','frame','axis','weight','projection','evidence','shape'];
  const verbs=['resolve','compile','bind','observe','project','validate','sample','stitch','attach','trace','seal','decode','query'];
  const fields=['canonical','owner','slot','normal','visible','local','world','clip','pose','parent','weight','confidence','revision','order'];
  const random=(min,max)=>min+Math.random()*(max-min);
  const pick=a=>a[Math.floor(Math.random()*a.length)];
  const streams=[];
  const traces=[];
  let interval=null,elapsed=0,last=performance.now();

  function generateGlyphs(count){
    let result='';
    for(let i=0;i<count;i++)result+=glyphs.charAt(Math.floor(Math.random()*glyphs.length));
    return result;
  }
  function generateCode(){
    const noun=pick(nouns),verb=pick(verbs),field=pick(fields),num=random(.1,.99).toFixed(3),id=Math.floor(random(1,240));
    const variants=[
      noun+'.'+verb+'('+field+');',
      'const '+noun+' = '+verb+'('+num+');',
      noun+'.'+field+' = '+num+';',
      'bind('+noun+', '+field+', '+id+');',
      'assert('+noun+'.'+field+');',
      'scene['+id+'].'+field+' = '+num+';',
      'vec3 '+noun+' = project('+id+');',
      'if ('+noun+'.visible) '+verb+'();'
    ];
    return pick(variants);
  }

  // These are intentionally separate screen-space reservations, not randomly
  // placed text. Each reservation has a unique vertical row and bounded width.
  // Perspective compensation preserves x/y while changing real z depth.
  const desktopSlots=[
    {x:3,y:9,w:43},{x:54,y:9,w:43},
    {x:3,y:32,w:43},{x:54,y:32,w:43},
    {x:3,y:55,w:43},{x:54,y:55,w:43},
    {x:3,y:78,w:43},{x:54,y:78,w:43}
  ];
  const landingDesktopSlots=[
    {x:68,y:15,w:28},{x:68,y:36,w:28},
    {x:68,y:57,w:28},{x:68,y:77,w:28}
  ];
  const landingMobileSlots=[
    {x:57,y:13,w:39},{x:6,y:76,w:40}
  ];
  const mobileSlots=[
    {x:5,y:7,w:90},{x:5,y:30,w:90},
    {x:5,y:53,w:90},{x:5,y:76,w:90}
  ];

  function build(){
    root.replaceChildren();typeRoot.replaceChildren();streams.length=0;traces.length=0;elapsed=0;
    const mobile=narrow.matches,numberOfLanes=role==="landing"?(mobile?9:23):(mobile?13:30),stagePerspective=mobile?850:1100;
    const fragment=document.createDocumentFragment();
    for(let i=0;i<numberOfLanes;i++){
      const depth=random(.04,.96);
      const z=Math.round(-770+depth*220); // atmospheric rain is now well behind the typing plane
      const target=(i+.5)/numberOfLanes;
      const compensate=.5+(target-.5)*(stagePerspective-z)/stagePerspective;
      const lane=document.createElement('div');
      lane.className='matrix-column';
      lane.dataset.lane=String(i);lane.dataset.depth=depth.toFixed(3);
      lane.style.left=(compensate*100).toFixed(3)+'%';
      lane.style.transform='translateZ('+z+'px)';
      lane.style.setProperty('--matrix-size',random(mobile?9:10,mobile?11.5:14).toFixed(2)+'px');
      lane.style.setProperty('--matrix-opacity',(random(.12,.24)*(0.7+depth*.35)).toFixed(3));
      const duration=random(20,42);
      lane.style.setProperty('--matrix-duration',duration.toFixed(2)+'s');
      lane.style.setProperty('--matrix-delay',(-random(0,duration)).toFixed(2)+'s');
      lane.style.setProperty('--matrix-glow',random(.08,.24).toFixed(2));
      if(i%7===0)lane.classList.add('matrix-highlight');
      if(depth<.32)lane.classList.add('matrix-far');
      const stream=document.createElement('span');
      stream.className='matrix-stream';
      const length=Math.floor(random(16,36));
      stream.textContent=generateGlyphs(length);
      lane.appendChild(stream);streams.push({node:stream,length});
      fragment.appendChild(lane);
    }

    const initialSlots=role==="landing"?(mobile?landingMobileSlots:landingDesktopSlots):(mobile?mobileSlots:desktopSlots);
    for(const [i,slot] of initialSlots.entries()){
      const depth=random(.22,.84), z=Math.round(-325+depth*220);
      const scaleCorrection=(stagePerspective-z)/stagePerspective;
      const lane=document.createElement('div');
      lane.className='matrix-trace';
      lane.dataset.slot=String(i);lane.dataset.depth=depth.toFixed(3);
      lane.style.left=(50+(slot.x-50)*scaleCorrection).toFixed(3)+'%';
      lane.style.top=(47+(slot.y-47)*scaleCorrection).toFixed(3)+'%';
      lane.style.width=(slot.w*scaleCorrection).toFixed(3)+'%';
      lane.style.transform='translateZ('+z+'px)';
      lane.style.setProperty('--trace-font',random(mobile?10:12,mobile?12:14).toFixed(1)+'px');
      lane.style.setProperty('--trace-alpha',random(.72,.90).toFixed(3));
      lane.style.setProperty('--trace-tone',pick(['#c7d4ff','#b5ddff','#d0c1ff','#bfd0ff']));
      const rows=Array.from({length:2},()=>{
        const span=document.createElement('span');
        span.className='matrix-trace-line';lane.appendChild(span);return span;
      });
      const state={lane,rows,lines:rows.map(generateCode),rate:random(65,118),offset:(i*1350)%7100,cycle:random(13000,18000),stagger:random(850,1500),lastCycle:-1};
      traces.push(state);typeRoot.appendChild(lane);
    }
    root.appendChild(fragment);
    layoutTraces();
    paint();
  }


  // Code length is determined from explicit slot geometry, never from a hidden
  // DOM node's 0px clientWidth. This fixes the repeated rig.bind() fallback.
  let previousCode = '';
  function fitCode(maxChars) {
    const candidates=[];
    for(let i=0;i<22;i++)candidates.push(generateCode());
    candidates.push(
      'rig.solve();','mesh.decode();','skin.bind();','axis.resolve();',
      'shape.fit();','scene.observe();','joint.attach();','motion.sample();',
      'mesh.compile();','source.project();','vertex.trace();','atlas.stitch();'
    );
    const fitting=candidates.filter(text=>text.length<=maxChars && text!==previousCode);
    const result=fitting.length ? pick(fitting) : pick(['rig.solve();','skin.bind();','mesh.fit();']);
    previousCode=result;
    return result;
  }

  function layoutTraces() {
    // This code belongs exclusively to the Chamber's reserved visualization.
    // No status text, real input, header or primary control can be painted here.
    const bounds=typeRoot.getBoundingClientRect();
    const width=bounds.width,height=bounds.height;
    if(width<=0||height<=0)return;
    const mobile=narrow.matches;
    const perspective=mobile?850:1100;
    const slots=role==="landing"?(mobile?landingMobileSlots:landingDesktopSlots):(mobile?mobileSlots:desktopSlots);
    for(let i=0;i<traces.length;i++){
      const trace=traces[i],slot=slots[i];
      if(!slot){trace.lane.hidden=true;continue}
      const x=slot.x/100*width,y=slot.y/100*height;
      const w=slot.w/100*width,h=mobile?58:59;
      const z=Number((trace.lane.style.transform.match(/-?\d+/)||[])[0])||-150;
      const ratio=(perspective-z)/perspective;
      trace.lane.hidden=false;
      trace.lane.style.left=(width/2+(x-width/2)*ratio).toFixed(2)+'px';
      trace.lane.style.top=(height/2+(y-height/2)*ratio).toFixed(2)+'px';
      trace.lane.style.width=(w*ratio).toFixed(2)+'px';
      trace.lane.style.height=(h*ratio).toFixed(2)+'px';
      const fontPx=Number.parseFloat(trace.lane.style.getPropertyValue('--trace-font'))||12;
      const maxChars=Math.max(12,Math.floor((w-20)/(fontPx*.67)));
      trace.maxChars=maxChars;
      trace.lines=trace.rows.map(()=>fitCode(maxChars));
      trace.lastCycle=Math.floor((elapsed+trace.offset)/trace.cycle);
    }

    // Draw decorative text only where it cannot compete with product copy.
    // Actual DOM bounds, not guessed coordinates, govern the decision.
    // Landing uses pre-reserved right-side lanes. Container bounds around the
    // hero include empty whitespace and must not hide all ambient code.
    const blockSelectors = role==='landing'
      ? []
      : ['.masthead','.chamber-header','.chamber-docking-label','.chamber-intake .dropzone',
         '.chamber-intake .input-slot-ledger','.chamber-console .console-heading',
         '.chamber-console h3','.chamber-console .console-copy',
         '#awaken','#compileAvailability','.chamber-console .telemetry'];
    const blockers=blockSelectors.map(sel=>document.querySelector(sel)).filter(el=>{
      if(!el)return false;
      const style=getComputedStyle(el);
      return style.display!=='none'&&style.visibility!=='hidden';
    }).map(el=>el.getBoundingClientRect());
    const padding=role==='landing'?16:10;
    for(const trace of traces){
      const box=trace.lane.getBoundingClientRect();
      trace.lane.hidden=blockers.some(block=>
        box.left < block.right+padding && box.right > block.left-padding &&
        box.top < block.bottom+padding && box.bottom > block.top-padding
      );
    }
  }

  function paint(){
    for(const trace of traces){
      const t=elapsed+trace.offset,cycle=Math.floor(t/trace.cycle);
      if(cycle!==trace.lastCycle){
        trace.lines=trace.rows.map(()=>fitCode(trace.maxChars||20));
        trace.lastCycle=cycle;
      }
      const local=t%trace.cycle;
      trace.rows.forEach((row,i)=>{
        const age=local-i*trace.stagger;
        const count=reduced.matches?trace.lines[i].length:Math.min(trace.lines[i].length,Math.max(0,Math.floor(age/trace.rate)));
        const text=reduced.matches?trace.lines[i]:local>trace.cycle-950?'':trace.lines[i].slice(0,count);
        if(row.textContent!==text)row.textContent=text;
        row.classList.toggle('typing',!reduced.matches&&age>=0&&count<trace.lines[i].length&&local<trace.cycle-950);
      });
    }
  }

  function tick(){
    if(document.hidden||reduced.matches)return;
    const now=performance.now();
    elapsed+=Math.max(0,Math.min(350,now-last));last=now;paint();
    if(Math.random()<.025&&streams.length){
      const index=Math.floor(Math.random()*streams.length);
      streams[index].node.textContent=generateGlyphs(streams[index].length);
    }
  }
  function start(){
    if(interval!==null)clearInterval(interval);
    interval=null;last=performance.now();
    if(!document.hidden&&!reduced.matches)interval=setInterval(tick,100);
    else paint();
  }
  build();start();
  document.fonts?.ready?.then(()=>{layoutTraces();paint()});
  let resizeQueued=false;
  window.addEventListener('resize',()=>{
    if(resizeQueued)return;
    resizeQueued=true;
    setTimeout(()=>{resizeQueued=false;layoutTraces();paint()},80);
  });
  narrow.addEventListener?.('change',()=>{build();start()});
  reduced.addEventListener?.('change',start);
  document.addEventListener('visibilitychange',start);
  window.addEventListener('pagehide',()=>{if(interval!==null)clearInterval(interval);interval=null},{once:true});
  }
  createScene(document.querySelector('#landingDepthCode'),document.querySelector('#landingMatrixTypewriter'),'landing');
  createScene(document.querySelector('#depthCode'),document.querySelector('#matrixTypewriter'),'chamber');
})();