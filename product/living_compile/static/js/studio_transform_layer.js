import {StudioSelection} from './studio_selection.js';
import {StudioToolContext} from './studio_tool_context.js';

const $ = (selector) => document.querySelector(selector);
const MODES_WITH_CONTROLS = new Set(['rig', 'topology', 'ik', 'animation', 'bindings']);

function boot() {
  const studio = window.RealSaSStudio;
  const viewport = $('#viewport');
  const panel = document.querySelector('.viewport-panel');
  if (!studio || !viewport || !panel) {
    requestAnimationFrame(boot);
    return;
  }

  const S = studio.state;
  const documentModel = studio.document;
  const journal = studio.journal;
  const dirty = studio.dirty;
  const selection = new StudioSelection(`V${S.view}`);
  const tools = new StudioToolContext();
  studio.selection = selection;
  studio.toolContext = tools;

  installToolbar();
  const overlay = installOverlay();
  const ctx = overlay.getContext('2d');
  let interaction = null;
  let saveTimer = null;

  syncOverlaySize();
  window.addEventListener('resize', syncOverlaySize);
  selection.addEventListener('change', () => {
    syncLegacySelection();
    renderToolbar();
    drawOverlay();
  });
  tools.addEventListener('change', () => {
    renderToolbar();
    drawOverlay();
  });

  overlay.oncontextmenu = (event) => event.preventDefault();
  overlay.onpointerdown = pointerDown;
  overlay.onpointermove = pointerMove;
  overlay.onpointerup = pointerUp;
  overlay.onpointercancel = cancelInteraction;
  overlay.onwheel = wheel;

  window.addEventListener('keydown', (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return;
    if (event.key === 'q') tools.setTool('select');
    if (event.key === 'w') tools.setTool('translate');
    if (event.key === 'e') tools.setTool('rotate');
    if (event.key === 'r') tools.setTool('scale');
    if (event.key === 'b') tools.setTool('box');
    if (event.key.toLowerCase() === 'x') tools.toggleTransformSpace();
    if (event.key.toLowerCase() === 'g') tools.setSnapEnabled(!tools.snapEnabled);
    if (event.key === 'Escape') cancelInteraction();
  });

  const observer = new MutationObserver(() => {
    const nextView = `V${S.view}`;
    if (selection.viewID !== nextView) selection.setView(nextView);
    drawOverlay();
  });
  observer.observe($('#viewLabel'), {childList: true, characterData: true, subtree: true});

  requestAnimationFrame(frame);

  function currentView() {
    return `V${S.view}`;
  }

  function controls() {
    return S.scene?.puppet?.rig?.controls?.filter((item) => item.view_id === currentView()) || [];
  }

  function meshes() {
    return S.scene?.puppet?.meshes?.filter((item) => item.view_id === currentView()) || [];
  }

  function rigValue(id) {
    return S.ed.rig.find((item) => item.control_id === id) || {control_id: id, rotation_degrees: 0, translation_xy: [0, 0]};
  }

  function rigEdit(id) {
    let edit = S.ed.rig.find((item) => item.control_id === id);
    if (!edit) {
      edit = {control_id: id, rotation_degrees: 0, translation_xy: [0, 0]};
      S.ed.rig.push(edit);
    }
    return edit;
  }

  function meshDelta(meshID, index) {
    return S.ed.mesh.find((item) => item.mesh_id === meshID && item.vertex_index === index)?.translation_xy || [0, 0];
  }

  function meshEdit(meshID, index) {
    let edit = S.ed.mesh.find((item) => item.mesh_id === meshID && item.vertex_index === index);
    if (!edit) {
      edit = {mesh_id: meshID, vertex_index: index, translation_xy: [0, 0]};
      S.ed.mesh.push(edit);
    }
    return edit;
  }

  function itemWorld(item) {
    if (item.kind === 'control') {
      const control = controls().find((candidate) => candidate.control_id === item.control_id);
      if (!control) return null;
      const edit = rigValue(item.control_id);
      return [control.anchor[0] + Number(edit.translation_xy?.[0] || 0), control.anchor[1] + Number(edit.translation_xy?.[1] || 0)];
    }
    if (item.kind === 'vertex') {
      const mesh = meshes().find((candidate) => candidate.mesh_id === item.mesh_id);
      const vertex = mesh?.vertices?.[item.vertex_index];
      if (!vertex) return null;
      const delta = meshDelta(item.mesh_id, item.vertex_index);
      return [vertex[0] + delta[0], vertex[1] + delta[1]];
    }
    return null;
  }

  function intrinsicFromEvent(event) {
    const bounds = viewport.getBoundingClientRect();
    return [
      (event.clientX - bounds.left) * viewport.width / Math.max(1, bounds.width),
      (event.clientY - bounds.top) * viewport.height / Math.max(1, bounds.height),
    ];
  }

  function worldFromEvent(event) {
    const p = intrinsicFromEvent(event);
    return [(p[0] - S.cam.x) / S.cam.s, (p[1] - S.cam.y) / S.cam.s];
  }

  function screenFromWorld(point) {
    const intrinsic = [point[0] * S.cam.s + S.cam.x, point[1] * S.cam.s + S.cam.y];
    return [intrinsic[0] * overlay.width / viewport.width, intrinsic[1] * overlay.height / viewport.height];
  }

  function candidates() {
    if (MODES_WITH_CONTROLS.has(S.mode)) {
      return controls().map((item) => ({kind: 'control', view_id: currentView(), control_id: item.control_id}));
    }
    const result = [];
    for (const mesh of meshes()) {
      mesh.vertices.forEach((_, index) => result.push({kind: 'vertex', view_id: currentView(), mesh_id: mesh.mesh_id, vertex_index: index}));
    }
    return result;
  }

  function hit(event) {
    const pointer = intrinsicFromEvent(event);
    const bounds = viewport.getBoundingClientRect();
    const threshold = 13 * viewport.width / Math.max(1, bounds.width);
    let best = null;
    let distance = threshold;
    for (const item of candidates()) {
      const world = itemWorld(item);
      if (!world) continue;
      const screen = [world[0] * S.cam.s + S.cam.x, world[1] * S.cam.s + S.cam.y];
      const candidate = Math.hypot(screen[0] - pointer[0], screen[1] - pointer[1]);
      if (candidate < distance) {
        distance = candidate;
        best = item;
      }
    }
    return best;
  }

  function pointerDown(event) {
    if (!S.scene) return;
    overlay.setPointerCapture(event.pointerId);
    const world = worldFromEvent(event);
    const pan = event.button === 1 || event.button === 2 || tools.tool === 'pan' || event.altKey && event.button === 0;
    if (pan) {
      interaction = {kind: 'pan', start: intrinsicFromEvent(event), cam: {...S.cam}};
      return;
    }

    if (tools.tool === 'box') {
      interaction = {kind: 'box', start: world, current: world, additive: event.shiftKey || event.ctrlKey || event.metaKey};
      drawOverlay();
      return;
    }

    const target = hit(event);
    const additive = event.shiftKey || event.ctrlKey || event.metaKey;
    if (target) {
      if (event.ctrlKey || event.metaKey) selection.toggle(target);
      else if (additive) selection.add(target, {primary: true});
      else if (!selection.has(target)) selection.replace(target);
    } else if (!additive && tools.tool === 'select') {
      selection.clear({viewOnly: true});
      return;
    }

    if (!selection.size()) return;
    const before = documentModel.snapshot();
    const positions = selection.active().map((item) => ({item, point: itemWorld(item)})).filter((row) => row.point);
    const pivot = resolvePivot(positions);
    const vector = [world[0] - pivot[0], world[1] - pivot[1]];
    interaction = {
      kind: tools.tool === 'rotate' ? 'rotate' : tools.tool === 'scale' ? 'scale' : 'translate',
      start: world,
      before,
      positions,
      pivot,
      startAngle: Math.atan2(vector[1], vector[0]),
      startRadius: Math.max(1e-6, Math.hypot(vector[0], vector[1])),
      changed: false,
    };
  }

  function pointerMove(event) {
    if (!interaction) return;
    if (interaction.kind === 'pan') {
      const point = intrinsicFromEvent(event);
      S.cam.x = interaction.cam.x + point[0] - interaction.start[0];
      S.cam.y = interaction.cam.y + point[1] - interaction.start[1];
      return;
    }
    const world = worldFromEvent(event);
    if (interaction.kind === 'box') {
      interaction.current = world;
      drawOverlay();
      return;
    }

    documentModel.replace(interaction.before);
    if (interaction.kind === 'translate') {
      let delta = [world[0] - interaction.start[0], world[1] - interaction.start[1]];
      delta = tools.snapTranslation(delta, event.altKey);
      for (const row of interaction.positions) setItemWorld(row.item, [row.point[0] + delta[0], row.point[1] + delta[1]]);
    } else if (interaction.kind === 'rotate') {
      const vector = [world[0] - interaction.pivot[0], world[1] - interaction.pivot[1]];
      let angle = Math.atan2(vector[1], vector[0]) - interaction.startAngle;
      angle = tools.snapAngleRadians(angle, event.altKey);
      for (const row of interaction.positions) {
        const next = rotateAround(row.point, interaction.pivot, angle);
        setItemWorld(row.item, next);
        if (row.item.kind === 'control') rigEdit(row.item.control_id).rotation_degrees = Number(rigValue(row.item.control_id).rotation_degrees || 0) + angle * 180 / Math.PI;
      }
    } else if (interaction.kind === 'scale') {
      const radius = Math.hypot(world[0] - interaction.pivot[0], world[1] - interaction.pivot[1]);
      let factor = radius / interaction.startRadius;
      factor = tools.snapScaleFactor(factor, event.altKey);
      for (const row of interaction.positions) {
        const next = [interaction.pivot[0] + (row.point[0] - interaction.pivot[0]) * factor, interaction.pivot[1] + (row.point[1] - interaction.pivot[1]) * factor];
        setItemWorld(row.item, next);
      }
    }
    interaction.changed = true;
    dirty.markDirty();
    drawOverlay();
  }

  function pointerUp(event) {
    if (!interaction) return;
    if (interaction.kind === 'box') finishBox();
    else if (interaction.changed) {
      journal.commit(`${capitalize(interaction.kind)} selection`, interaction.before, documentModel.snapshot());
      dirty.markDirty();
      scheduleSave();
    }
    interaction = null;
    try { overlay.releasePointerCapture(event.pointerId); } catch {}
    drawOverlay();
  }

  function cancelInteraction() {
    if (interaction?.before) documentModel.replace(interaction.before);
    interaction = null;
    drawOverlay();
  }

  function finishBox() {
    const x0 = Math.min(interaction.start[0], interaction.current[0]);
    const x1 = Math.max(interaction.start[0], interaction.current[0]);
    const y0 = Math.min(interaction.start[1], interaction.current[1]);
    const y1 = Math.max(interaction.start[1], interaction.current[1]);
    const inside = candidates().filter((item) => {
      const point = itemWorld(item);
      return point && point[0] >= x0 && point[0] <= x1 && point[1] >= y0 && point[1] <= y1;
    });
    if (!interaction.additive) selection.replaceMany(inside);
    else inside.forEach((item) => selection.add(item, {primary: false}));
  }

  function setItemWorld(item, point) {
    if (item.kind === 'control') {
      const control = controls().find((candidate) => candidate.control_id === item.control_id);
      if (!control) return;
      rigEdit(item.control_id).translation_xy = [point[0] - control.anchor[0], point[1] - control.anchor[1]];
      return;
    }
    if (item.kind === 'vertex') {
      const mesh = meshes().find((candidate) => candidate.mesh_id === item.mesh_id);
      const base = mesh?.vertices?.[item.vertex_index];
      if (!base) return;
      meshEdit(item.mesh_id, item.vertex_index).translation_xy = [point[0] - base[0], point[1] - base[1]];
    }
  }

  function resolvePivot(rows) {
    if (tools.pivotMode === 'custom') return [...tools.customPivot];
    if (tools.pivotMode === 'primary') {
      const primary = selection.primary();
      const point = primary && itemWorld(primary);
      if (point) return point;
    }
    if (!rows.length) return [0, 0];
    return [rows.reduce((sum, row) => sum + row.point[0], 0) / rows.length, rows.reduce((sum, row) => sum + row.point[1], 0) / rows.length];
  }

  function rotateAround(point, pivot, angle) {
    const x = point[0] - pivot[0];
    const y = point[1] - pivot[1];
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return [pivot[0] + x * c - y * s, pivot[1] + x * s + y * c];
  }

  function wheel(event) {
    event.preventDefault();
    const before = worldFromEvent(event);
    const factor = event.deltaY > 0 ? 0.9 : 1.1;
    S.cam.s = Math.max(0.05, Math.min(20, S.cam.s * factor));
    const afterScreen = intrinsicFromEvent(event);
    S.cam.x = afterScreen[0] - before[0] * S.cam.s;
    S.cam.y = afterScreen[1] - before[1] * S.cam.s;
  }

  function syncLegacySelection() {
    const primary = selection.primary();
    if (!primary) {
      S.sel = {control: null, mesh: null, vertex: -1};
    } else if (primary.kind === 'control') {
      S.sel.control = primary.control_id;
    } else if (primary.kind === 'vertex') {
      S.sel.mesh = primary.mesh_id;
      S.sel.vertex = primary.vertex_index;
    }
  }

  function scheduleSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      if (dirty.status === 'dirty') $('#saveLayer')?.click();
    }, 1200);
  }

  function installToolbar() {
    if ($('#studioTransformTools')) return;
    const bar = document.querySelector('.viewport-toolbar');
    const group = document.createElement('div');
    group.id = 'studioTransformTools';
    group.className = 'studio-transform-tools';
    group.innerHTML = `
      <span class="tool-separator"></span>
      <button data-studio-tool="select" title="Select · Q">SELECT</button>
      <button data-studio-tool="translate" title="Move · W">MOVE</button>
      <button data-studio-tool="rotate" title="Rotate · E">ROTATE</button>
      <button data-studio-tool="scale" title="Scale · R">SCALE</button>
      <button data-studio-tool="box" title="Box select · B">BOX</button>
      <button data-studio-tool="pan">PAN</button>
      <button id="transformSpace">WORLD</button>
      <label class="snap-toggle"><input id="snapEnabled" type="checkbox" checked> SNAP</label>
      <input id="translateSnap" class="mini-number" type="number" min=".001" step=".25" value="1" title="Translate snap">
      <input id="rotateSnap" class="mini-number" type="number" min=".1" step="1" value="15" title="Rotate snap degrees">
    `;
    bar.append(group);
    group.querySelectorAll('[data-studio-tool]').forEach((button) => button.onclick = () => tools.setTool(button.dataset.studioTool));
    $('#transformSpace').onclick = () => tools.toggleTransformSpace();
    $('#snapEnabled').onchange = (event) => tools.setSnapEnabled(event.target.checked);
    $('#translateSnap').onchange = (event) => tools.configureSnapping({translate: Number(event.target.value)});
    $('#rotateSnap').onchange = (event) => tools.configureSnapping({rotateDegrees: Number(event.target.value)});
  }

  function installOverlay() {
    let canvas = $('#studioOverlay');
    if (canvas) return canvas;
    canvas = document.createElement('canvas');
    canvas.id = 'studioOverlay';
    canvas.setAttribute('aria-label', 'Studio selection and transform overlay');
    viewport.insertAdjacentElement('afterend', canvas);
    const style = document.createElement('style');
    style.textContent = `
      .viewport-panel{position:relative}.viewport-panel #studioOverlay{position:absolute;inset:42px 0 0 0;width:100%;height:calc(100% - 42px);touch-action:none;z-index:5}
      .studio-transform-tools{display:flex;align-items:center;gap:4px;margin-left:auto}.studio-transform-tools button.active{border-color:#8dd2ff;color:#fff;background:#183047}.tool-separator{width:1px;height:20px;background:#253342;margin:0 4px}.mini-number{width:52px}.snap-toggle{display:inline-flex;align-items:center;gap:3px}
      .anim-authoring-tools{display:flex;align-items:center;gap:5px;flex-wrap:wrap}.anim-authoring-tools input[type=text]{width:90px}.anim-authoring-tools input[type=number]{width:66px}.anim-divider{width:1px;height:22px;background:#253342;margin:0 3px}.timeline-body{grid-template-columns:minmax(180px,240px) 1fr 240px}.curve-editor{border-left:1px solid #1d2a36;padding:8px;background:#0b1016}.curve-editor canvas{width:100%;height:140px}.curve-fields{display:grid;grid-template-columns:1fr 1fr;gap:4px}.curve-fields label{font-size:9px}.curve-fields input{width:56px}.curve-actions{display:flex;gap:4px;margin:7px 0}
    `;
    document.head.append(style);
    return canvas;
  }

  function syncOverlaySize() {
    const rect = viewport.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    overlay.width = Math.max(1, Math.round(rect.width * dpr));
    overlay.height = Math.max(1, Math.round(rect.height * dpr));
    drawOverlay();
  }

  function renderToolbar() {
    document.querySelectorAll('[data-studio-tool]').forEach((button) => button.classList.toggle('active', button.dataset.studioTool === tools.tool));
    if ($('#transformSpace')) $('#transformSpace').textContent = tools.transformSpace.toUpperCase();
    if ($('#snapEnabled')) $('#snapEnabled').checked = tools.snapEnabled;
  }

  function drawOverlay() {
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    for (const item of selection.active()) {
      const world = itemWorld(item);
      if (!world) continue;
      const [x, y] = screenFromWorld(world);
      ctx.strokeStyle = '#ffffff';
      ctx.fillStyle = 'rgba(141,210,255,.22)';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, item.kind === 'control' ? 9 : 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    const rows = selection.active().map((item) => ({item, point: itemWorld(item)})).filter((row) => row.point);
    if (rows.length && ['translate', 'rotate', 'scale'].includes(tools.tool)) drawGizmo(resolvePivot(rows));
    if (interaction?.kind === 'box') {
      const a = screenFromWorld(interaction.start);
      const b = screenFromWorld(interaction.current);
      ctx.strokeStyle = '#8dd2ff';
      ctx.fillStyle = 'rgba(141,210,255,.08)';
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
      ctx.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
      ctx.setLineDash([]);
    }
  }

  function drawGizmo(pivot) {
    const [x, y] = screenFromWorld(pivot);
    ctx.lineWidth = 2;
    if (tools.tool === 'rotate') {
      ctx.strokeStyle = '#ffd36b';
      ctx.beginPath(); ctx.arc(x, y, 34, 0, Math.PI * 2); ctx.stroke();
      return;
    }
    if (tools.tool === 'scale') {
      ctx.strokeStyle = '#c79cff';
      ctx.strokeRect(x - 22, y - 22, 44, 44);
      return;
    }
    ctx.strokeStyle = '#ff6c77';
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 45, y); ctx.stroke();
    ctx.strokeStyle = '#6ee7a8';
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - 45); ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
  }

  function frame() {
    const nextView = `V${S.view}`;
    if (selection.viewID !== nextView) selection.setView(nextView);
    drawOverlay();
    requestAnimationFrame(frame);
  }

  function capitalize(value) {
    return value.charAt(0).toUpperCase() + value.slice(1);
  }

  renderToolbar();
}

boot();
