/* Ambient depth only; these snippets are decorative, not real compiler telemetry. */
(() => {
  'use strict';
  const root = document.querySelector('#depthCode');
  if (!root) return;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const rng = () => Math.random();
  const choose = items => items[Math.floor(rng() * items.length)];
  const fragments = [
    ['scene', 'query', 'resolve', 'compile', 'observe', 'validate'],
    ['mesh', 'shape', 'surface', 'rig', 'weights', 'motion', 'visibility', 'attachment', 'occlusion', 'topology', 'identity', 'constraint'],
    ['canonical', 'source', 'deformation', 'appearance', 'render_order', 'geometry', 'rest_pose', 'projection', 'evidence', 'frame', 'joint', 'skin'],
    ['tracked', 'bounded', 'editable', 'persistent', 'qualified', 'source_owned', 'local', 'typed', 'spatial', 'canonical']
  ];
  const fmt = () => {
    const n = (rng() * 0.7 + 0.18).toFixed(3);
    const depth = (rng() * 0.8 + 0.1).toFixed(2);
    const op = choose(fragments[0]), subject = choose(fragments[1]);
    const field = choose(fragments[2]), value = choose(fragments[3]);
    return choose([
      subject + '.' + op + '(' + field + ');',
      'const ' + subject + ' = ' + op + '(' + value + ');',
      'context.' + subject + '.' + field + ' = ' + n + ';',
      'bind(' + subject + ', ' + field + ', ' + depth + ');',
      'assert(' + subject + '.' + field + ' !== null);',
      'revision.' + subject + ' = ' + value + ';',
      'sample.' + subject + ' = vec3(' + depth + ', ' + n + ', 0);',
      'if (' + subject + '.visible) ' + op + '();'
    ]);
  };

  // Independent attributes per group: position, depth, intensity, typewriter rate,
  // scale and hue. Position is constrained to the periphery to keep the hero readable.
  const lanes = Array.from({length:7}, (_,i) => {
    const lane = document.createElement('div');
    lane.className = 'depth-code-track';
    const leftSide = i % 2 === 0;
    const depth = .15 + rng() * .8;
    const x = leftSide ? 2 + rng()*21 : 63 + rng()*15;
    const y = 8 + (i % 4)*22 + rng()*8;
    const alpha = .16 + rng()*.32;
    lane.style.left = x + '%';
    lane.style.top = y + '%';
    lane.style.opacity = alpha.toFixed(3);
    lane.style.fontSize = (9 + depth*7).toFixed(1)+'px';
    lane.style.transform = 'translateZ(' + Math.round(-130 + depth*190) + 'px) scale(' + (.7+depth*.35).toFixed(2)+')';
    lane.style.color = choose(['#a7d6ee','#a8c0dd','#91bde0','#a8d4df']);
    const rows = Array.from({length:2+(rng()>.55?1:0)}, () => {
      const span = document.createElement('span');
      lane.appendChild(span);
      return span;
    });
    root.appendChild(lane);
    return {lane,rows,lines:rows.map(fmt),interval:Math.round(64 + rng()*110),stagger:1100+rng()*2000,offset:rng()*17000,cycle:15000+rng()*9500};
  });
  let elapsed = 0, last = performance.now(),timer=null;
  function paint() {
    for (const lane of lanes) {
      const local=(elapsed+lane.offset)%lane.cycle;
      const hold=lane.cycle-2400;
      for(let i=0;i<lane.rows.length;i++){
        const row=lane.rows[i];
        const start=local-i*lane.stagger;
        const count=reduced?lane.lines[i].length:Math.min(lane.lines[i].length,Math.max(0,Math.floor(start/lane.interval)));
        const text=!reduced&&local>hold?'':lane.lines[i].slice(0,count);
        if(row.textContent!==text)row.textContent=text;
        row.classList.toggle('typing',!reduced&&local<hold&&start>0&&count>0&&count<lane.lines[i].length);
      }
      if(!reduced&&local>hold&&lane.rows.every(row=>!row.textContent)){
        lane.lines=lane.rows.map(fmt); // new code, not the same four repeating terminal snippets
      }
    }
  }
  function tick() {
    timer=null;
    if(document.hidden)return;
    const now=performance.now();
    elapsed+=Math.min(450,Math.max(0,now-last));
    last=now;
    paint();
    timer=setTimeout(tick,150); // no Canvas and no frame-by-frame full-page redraws
  }
  function visibility(){
    if(timer!==null)clearTimeout(timer);
    timer=null;last=performance.now();
    if(!document.hidden&&!reduced)timer=setTimeout(tick,150);
  }
  paint();
  if(!reduced){
    document.addEventListener('visibilitychange',visibility);
    window.addEventListener('pagehide',()=>{if(timer!==null)clearTimeout(timer)},{once:true});
    visibility();
  }
})();
