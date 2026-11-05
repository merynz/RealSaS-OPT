/* BUSTED Matrix depth atmosphere. Decorative, never real compile telemetry. */
(() => {
  'use strict';
  const root = document.querySelector('#depthCode');
  if (!root) return;
  const narrow = window.matchMedia('(max-width: 780px)');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const glyphs = '0123456789ABCDEFアイウエオカキクケコサシスセソタチツテト<>[]{}:;/';
  const nouns = ['mesh','rig','skin','joint','topology','depth','proof','source','surface','motion','axis','atlas','pose','frame','weight','occlusion'];
  const verbs = ['bind','compile','resolve','sample','seal','observe','verify','project','trace','validate'];
  const random = (low,high) => low + Math.random() * (high-low);
  const pick = list => list[Math.floor(Math.random()*list.length)];
  const streams = [];
  let interval = null;

  function generateGlyphs(count) {
    const semantic = pick(nouns) + '.' + pick(verbs) + '(' + Math.floor(random(0,9999)) + ');';
    let result='';
    for (let j=0;j<count;j++) {
      const token = Math.random()<.24 ? semantic[j%semantic.length] : glyphs[Math.floor(Math.random()*glyphs.length)];
      result += token;
    }
    return result;
  }

  function build() {
    root.replaceChildren();
    streams.length = 0;
    const mobile = narrow.matches;
    const numberOfLanes = mobile ? 13 : 30;
    const fragment = document.createDocumentFragment();
    for (let i=0;i<numberOfLanes;i++) {
      const depth = random(.04,.96);
      const z = Math.round(-470 + depth*460);
      // Undo perspective contraction in positioning, so every lane occupies
      // a non-overlapping screen-space cell regardless of its unique depth.
      const target = (i+.5) / numberOfLanes;
      const compensate = .5 + (target-.5) * (1100-z)/1100;
      const lane = document.createElement('div');
      lane.className = 'matrix-column';
      lane.dataset.lane = String(i);
      lane.dataset.depth = depth.toFixed(3);
      lane.style.left = (compensate*100).toFixed(3) + '%';
      lane.style.transform = 'translateZ(' + z + 'px)';
      lane.style.setProperty('--matrix-size',random(mobile?8.0:9.2,mobile?10.0:12.6).toFixed(2)+'px');
      lane.style.setProperty('--matrix-opacity',(random(.20,.34)*(0.58+depth*.64)).toFixed(3));
      const duration = random(16,32)*(1.24-depth*.3);
      lane.style.setProperty('--matrix-duration',duration.toFixed(2)+'s');
      lane.style.setProperty('--matrix-delay',(-random(0,duration)).toFixed(2)+'s');
      lane.style.setProperty('--matrix-glow',(depth>.7 ? .3:.09).toFixed(2));
      if (i%8===0) lane.classList.add('matrix-highlight');
      if (depth<.3) lane.classList.add('matrix-far');
      const stream = document.createElement('span');
      stream.className = 'matrix-stream';
      const length = Math.floor(random(17,34));
      stream.textContent = generateGlyphs(length);
      lane.appendChild(stream);
      streams.push({node:stream,length});
      fragment.appendChild(lane);
    }
    root.appendChild(fragment);
  }

  function refresh() {
    if (document.hidden || reduced.matches) return;
    // Update only one distant strip occasionally. The rendering animation
    // runs via CSS compositor; JavaScript never repaints a full-screen canvas.
    const index = Math.floor(Math.random()*streams.length);
    if (streams[index]) streams[index].node.textContent=generateGlyphs(streams[index].length);
  }
  function syncTimer() {
    if (interval!==null) clearInterval(interval);
    interval=null;
    if (!document.hidden && !reduced.matches) interval=setInterval(refresh,6000);
  }
  function rebuild() { build(); syncTimer(); }
  rebuild();
  narrow.addEventListener?.('change',rebuild);
  reduced.addEventListener?.('change',syncTimer);
  document.addEventListener('visibilitychange',syncTimer);
  window.addEventListener('pagehide',()=>{
    if(interval!==null) clearInterval(interval);
    interval=null;
  },{once:true});
})();
