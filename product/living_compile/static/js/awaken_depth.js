/* BUSTED spatial code: writes text on a fixed far plane, not a moving FX canvas. */
(() => {
  'use strict';
  const root = document.querySelector('#depthCode');
  if (!root) return;

  const sequences = [
    ['source.observe(front, back);','scene.resolve(shape_memory);','mesh.compile(canonical_topology);'],
    ['rig.bind(control_program);','skin.attach(weight_field);','mechanics.validate(deformation);'],
    ['graphics.resolve(appearance);','motion.attach(clip_set);','proof.require_authority();'],
    ['scene.memory → surface','topology → constraints','puppet → editable']
  ];
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const lanes = sequences.map((sequence, index) => {
    const lane = document.createElement('div');
    lane.className = 'depth-code-track';
    const rows = sequence.map(() => {
      const row = document.createElement('span');
      lane.appendChild(row);
      return row;
    });
    root.appendChild(lane);
    return {sequence, rows, offset:index * 2100};
  });

  // One scheduled callback updates only changed text nodes. No animation loop,
  // no random positions, no GPU-sized canvas clears; tabs hidden -> no timer.
  const cycleMs = 17000;
  const charMs = 88;
  const lineDelayMs = 2550;
  const holdMs = 4000;
  let clock = 0;
  let last = performance.now();
  let timer = null;

  function render() {
    for (const lane of lanes) {
      const local = (clock + lane.offset) % cycleMs;
      lane.rows.forEach((row, line) => {
        const original = lane.sequence[line];
        const age = local - line * lineDelayMs;
        const characters = reduceMotion
          ? original.length
          : Math.max(0,Math.min(original.length,Math.floor(age / charMs)));
        const next = original.slice(0,characters);
        if (row.textContent !== next) row.textContent = next;
      });
      // A brief empty beat before the next writing cycle.
      if (!reduceMotion && local > cycleMs - holdMs) {
        lane.rows.forEach(row => {if (row.textContent) row.textContent = '';});
      }
    }
  }

  function pulse() {
    timer = null;
    if (document.hidden) return;
    const now = performance.now();
    clock += Math.min(500,Math.max(0,now-last));
    last = now;
    render();
    timer = window.setTimeout(pulse,110);
  }

  function visibility() {
    if (timer !== null) window.clearTimeout(timer);
    timer = null;
    last = performance.now();
    if (!document.hidden && !reduceMotion) timer = window.setTimeout(pulse,110);
  }

  render();
  if (!reduceMotion) {
    document.addEventListener('visibilitychange',visibility);
    visibility();
    window.addEventListener('pagehide',() => {
      if (timer !== null) window.clearTimeout(timer);
      timer = null;
    },{once:true});
  }
})();
