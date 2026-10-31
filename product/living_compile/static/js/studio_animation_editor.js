import {StudioTimeline, keyID} from './studio_timeline.js';
import {normalizePose} from './studio_control_semantics.js';

const $ = (selector) => document.querySelector(selector);
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

function boot() {
  const studio = window.RealSaSStudio;
  if (!studio || !$('#dope') || !$('#clipSelect')) {
    requestAnimationFrame(boot);
    return;
  }

  const S = studio.state;
  const documentModel = studio.document;
  const journal = studio.journal;
  const dirty = studio.dirty;
  const timeline = new StudioTimeline(S.ed.animation);
  studio.timeline = timeline;

  let autosaveTimer = null;
  let playing = false;
  let lastTime = 0;

  const legacy = {
    clipChange: $('#clipSelect').onchange,
    play: $('#play').onclick,
    stop: $('#stop').onclick,
  };

  installUI();
  wire();
  refresh();
  requestAnimationFrame(tick);

  function bindTimeline() {
    if (timeline.animationEdits !== S.ed.animation) timeline.bind(S.ed.animation);
  }

  function userClip() {
    bindTimeline();
    return timeline.clip(S.clip);
  }

  function animationControls() {
    const view = `V${S.view}`;
    const rig = S.scene?.puppet?.rig || {};
    const canonical = (rig.animator_controls || []).filter((item) => item.view_id === view);
    if (canonical.length) return canonical;
    return (rig.deform_controls || rig.controls || []).filter((item) => item.view_id === view);
  }

  function selectedAnimationControlID() {
    const allowed = new Set(animationControls().map((item) => item.control_id));
    const primary = studio.selection?.primary()?.control_id;
    if (primary && allowed.has(primary)) return primary;
    if (S.sel?.control && allowed.has(S.sel.control)) return S.sel.control;
    return null;
  }

  function transaction(label, mutate) {
    bindTimeline();
    const before = documentModel.snapshot();
    mutate();
    const after = documentModel.snapshot();
    if (!journal.commit(label, before, after)) return false;
    dirty.markDirty();
    scheduleSave();
    refresh();
    previewAt(S.t);
    return true;
  }

  function scheduleSave() {
    clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(() => {
      if (dirty.status === 'dirty' && !dirty.lastError) $('#saveLayer')?.click();
    }, 1200);
  }

  function installUI() {
    if ($('#animAuthoringTools')) return;
    const head = document.querySelector('.timeline-head');
    const tools = document.createElement('div');
    tools.id = 'animAuthoringTools';
    tools.className = 'anim-authoring-tools';
    tools.innerHTML = `
      <span class="anim-divider"></span>
      <label>NAME <input id="animClipName" type="text" value=""></label>
      <label>DURATION <input id="animDuration" type="number" min="0.033" step="0.033" value="1"></label>
      <label><input id="animLoop" type="checkbox" checked> LOOP</label>
      <button id="duplicateClip">DUP CLIP</button>
      <button id="deleteClip" class="danger">DEL CLIP</button>
      <span class="anim-divider"></span>
      <button id="copyKeys">COPY</button>
      <button id="pasteKeys">PASTE</button>
      <button id="nudgeKeyLeft" title="Move selected keys one frame left">◀ KEY</button>
      <button id="nudgeKeyRight" title="Move selected keys one frame right">KEY ▶</button>
    `;
    head.append(tools);

    const body = document.querySelector('.timeline-body');
    const curve = document.createElement('aside');
    curve.id = 'curveEditor';
    curve.className = 'curve-editor';
    curve.innerHTML = `
      <div class="panel-title">CURVE</div>
      <canvas id="curveCanvas" width="220" height="140"></canvas>
      <div class="curve-fields">
        <label>X1 <input id="curveX1" type="number" min="0" max="1" step=".01" value=".25"></label>
        <label>Y1 <input id="curveY1" type="number" min="0" max="1" step=".01" value=".10"></label>
        <label>X2 <input id="curveX2" type="number" min="0" max="1" step=".01" value=".25"></label>
        <label>Y2 <input id="curveY2" type="number" min="0" max="1" step=".01" value="1"></label>
      </div>
      <div class="curve-actions">
        <button data-curve="step">STEP</button>
        <button data-curve="linear">LINEAR</button>
        <button data-curve="bezier">BEZIER</button>
      </div>
      <small id="keySelectionReadout">0 keys selected</small>
    `;
    body.append(curve);
  }

  function wire() {
    $('#newClip').onclick = () => transaction('Create animation clip', () => {
      const clip = timeline.createClip({fps: Number($('#fps').value) || 30});
      S.clip = clip.clip_id;
      S.t = 0;
      S.frame = null;
      S.previewPose.clear();
    });

    $('#duplicateClip').onclick = () => {
      if (!userClip()) return toast('Select a user clip first', true);
      transaction('Duplicate animation clip', () => {
        const clip = timeline.duplicateClip();
        S.clip = clip.clip_id;
        S.t = 0;
      });
    };

    $('#deleteClip').onclick = () => {
      if (!userClip()) return toast('Runtime clips are read-only', true);
      transaction('Delete animation clip', () => {
        timeline.deleteClip();
        S.clip = timeline.clipID || S.clips[0]?.clip_id || null;
        S.t = 0;
        S.frame = null;
        S.previewPose.clear();
      });
    };

    $('#animClipName').onchange = (event) => {
      if (userClip()) transaction('Rename animation clip', () => timeline.renameClip(event.target.value));
    };
    $('#animDuration').onchange = (event) => {
      if (userClip()) transaction('Change animation duration', () => timeline.setDuration(Number(event.target.value), {scaleKeys: event.shiftKey}));
    };
    $('#animLoop').onchange = (event) => {
      if (userClip()) transaction('Change animation loop', () => timeline.setLoop(event.target.checked));
    };
    $('#fps').onchange = (event) => {
      if (userClip()) transaction('Change animation FPS', () => timeline.setFPS(Number(event.target.value)));
    };

    $('#addKey').onclick = addCurrentTransformKey;
    $('#deleteKey').onclick = () => {
      if (!userClip()) return toast('Runtime clips are read-only', true);
      transaction('Delete animation keys', () => timeline.deleteKeys());
    };
    $('#copyKeys').onclick = () => toast(timeline.copySelectedKeys() ? 'Keys copied' : 'Select keys first', !timeline.clipboard.length);
    $('#pasteKeys').onclick = () => {
      if (!userClip()) return toast('Select a user clip first', true);
      transaction('Paste animation keys', () => timeline.pasteKeys(S.t));
    };
    $('#nudgeKeyLeft').onclick = () => nudge(-1);
    $('#nudgeKeyRight').onclick = () => nudge(1);

    document.querySelectorAll('[data-curve]').forEach((button) => {
      button.onclick = () => {
        if (userClip()) transaction(`Set ${button.dataset.curve} interpolation`, () => timeline.setSelectedCurve(button.dataset.curve));
      };
    });
    for (const id of ['curveX1', 'curveY1', 'curveX2', 'curveY2']) {
      $(`#${id}`).onchange = () => {
        if (!userClip()) return;
        const bezier = ['#curveX1', '#curveY1', '#curveX2', '#curveY2'].map((selector) => Number($(selector).value));
        transaction('Edit Bezier curve', () => timeline.setSelectedBezier(bezier));
      };
    }

    $('#clipSelect').onchange = async (event) => {
      stopUserPlayback();
      S.clip = event.target.value;
      S.t = 0;
      S.frame = null;
      S.previewPose.clear();
      bindTimeline();
      if (timeline.clip(S.clip)) {
        timeline.selectClip(S.clip);
        previewAt(0);
        refresh();
      } else if (legacy.clipChange) {
        await legacy.clipChange.call($('#clipSelect'), event);
      }
    };

    $('#play').onclick = async () => {
      if (!userClip()) {
        if (legacy.play) return legacy.play.call($('#play'));
        return;
      }
      playing = !playing;
      S.play = false;
      lastTime = performance.now();
      $('#play').textContent = playing ? '❚❚' : '▶';
    };

    $('#stop').onclick = () => {
      if (!userClip()) {
        if (legacy.stop) return legacy.stop.call($('#stop'));
        return;
      }
      stopUserPlayback();
      S.t = 0;
      timeline.setPlayhead(0);
      previewAt(0);
      refresh();
    };

    const dope = $('#dope');
    dope.onpointerdown = (event) => {
      if (!userClip()) return;
      const hit = hitKey(event);
      if (hit) {
        timeline.selectKey(hit.controlID, hit.time, {
          add: event.shiftKey || event.ctrlKey || event.metaKey,
          toggle: event.ctrlKey || event.metaKey,
        });
      } else {
        timeline.clearKeySelection();
        S.t = timeline.setPlayhead(timeFromEvent(event), {snapToFrame: event.shiftKey});
        previewAt(S.t);
      }
      drawDopesheet();
      drawCurve();
    };

    timeline.addEventListener('change', () => {
      S.t = timeline.playhead;
      refreshControls();
      drawDopesheet();
      drawCurve();
    });

    window.addEventListener('keydown', (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return;
      if (!userClip()) return;
      const command = event.ctrlKey || event.metaKey;
      if (command && event.key.toLowerCase() === 'c') {
        event.preventDefault();
        timeline.copySelectedKeys();
      } else if (command && event.key.toLowerCase() === 'v') {
        event.preventDefault();
        transaction('Paste animation keys', () => timeline.pasteKeys(S.t));
      } else if (event.key === 'Delete' || event.key === 'Backspace') {
        if (timeline.selectedKeys.size) {
          event.preventDefault();
          transaction('Delete animation keys', () => timeline.deleteKeys());
        }
      } else if (event.key === ',' || event.key === '.') {
        event.preventDefault();
        nudge(event.key === ',' ? -1 : 1);
      }
    });
  }

  function addCurrentTransformKey() {
    if (!userClip()) return toast('Create or select a user clip', true);
    const controlID = selectedAnimationControlID();
    if (!controlID) return toast('Select an animator control to key', true);
    const current = normalizePose(studio.animationPreviewPose?.get(controlID) || timeline.sample(controlID, S.t));
    transaction('Add animation key', () => {
      timeline.setPlayhead(S.t, {snapToFrame: true});
      timeline.upsertTransformKey(controlID, current, {curve: normalizedCurve($('#curveSelect').value)});
    });
  }

  function nudge(direction) {
    const clip = userClip();
    if (!clip || !timeline.selectedKeys.size) return;
    const fps = Number(clip.fps || $('#fps').value || 30);
    transaction('Nudge animation keys', () => timeline.moveSelectedKeys(direction / fps, {snapToFrames: true}));
  }

  function normalizedCurve(value) {
    return value === 'cubic' ? 'bezier' : (['step', 'linear', 'bezier'].includes(value) ? value : 'linear');
  }

  function stopUserPlayback() {
    playing = false;
    S.play = false;
    $('#play').textContent = '▶';
  }

  function tick(now) {
    if (playing && userClip()) {
      const clip = timeline.clip();
      const delta = Math.max(0, (now - lastTime) / 1000);
      lastTime = now;
      let time = S.t + delta;
      if (time > clip.duration_seconds) {
        if (clip.loop) time %= clip.duration_seconds;
        else {
          time = clip.duration_seconds;
          stopUserPlayback();
        }
      }
      S.t = timeline.setPlayhead(time);
      previewAt(S.t);
      drawDopesheet();
    }
    requestAnimationFrame(tick);
  }

  function previewAt(time) {
    const clip = userClip();
    if (!clip) return;
    window.dispatchEvent(new CustomEvent('realsas:studio-animation-pose-changed', {
      detail: {clip_id: S.clip, time_seconds: Number(time)},
    }));
    $('#timeReadout').textContent = `${Number(time).toFixed(3)} / ${Number(clip.duration_seconds).toFixed(3)}`;
  }

  function refresh() {
    bindTimeline();
    if (S.clip && timeline.clip(S.clip) && timeline.clipID !== S.clip) timeline.selectClip(S.clip);
    refreshControls();
    drawDopesheet();
    drawCurve();
  }

  function refreshControls() {
    const select = $('#clipSelect');
    const all = [
      ...(S.clips || []).map((item) => ({...item, user: false})),
      ...(S.ed.animation || []).map((item) => ({...item, user: true})),
    ];
    select.innerHTML = all.map((item) => `<option value="${escapeHTML(item.clip_id)}">${item.user ? '✎' : '▶'} ${escapeHTML(item.display_name || item.clip_id)}</option>`).join('');
    if (S.clip) select.value = S.clip;
    const clip = userClip();
    $('#animClipName').disabled = !clip;
    $('#animDuration').disabled = !clip;
    $('#animLoop').disabled = !clip;
    $('#duplicateClip').disabled = !clip;
    $('#deleteClip').disabled = !clip;
    if (clip) {
      $('#animClipName').value = clip.display_name || clip.clip_id;
      $('#animDuration').value = Number(clip.duration_seconds || 1).toFixed(3);
      $('#animLoop').checked = Boolean(clip.loop);
      $('#fps').value = Number(clip.fps || 30);
    }
    $('#keySelectionReadout').textContent = `${timeline.selectedKeys.size} key${timeline.selectedKeys.size === 1 ? '' : 's'} selected`;
  }

  function drawDopesheet() {
    const canvas = $('#dope');
    const ctx = canvas.getContext('2d');
    const clip = userClip();
    if (!clip) return;
    const tracks = clip.tracks || [];
    const rowHeight = 24;
    ctx.fillStyle = '#090d12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#243140';
    ctx.lineWidth = 1;
    const fps = Number(clip.fps || 30);
    const major = Math.max(1, Math.round(fps / 2));
    const frameCount = Math.max(1, Math.round(clip.duration_seconds * fps));
    for (let frame = 0; frame <= frameCount; frame += 1) {
      if (frame % major !== 0 && frameCount > 240) continue;
      const x = frame / frameCount * canvas.width;
      ctx.globalAlpha = frame % fps === 0 ? 0.8 : 0.25;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    tracks.forEach((track, row) => {
      const y = 28 + row * rowHeight;
      const controlID = track.control_id || track.bone_id;
      ctx.fillStyle = '#71879a';
      ctx.font = '11px Consolas, monospace';
      ctx.fillText(controlID, 6, y + 4);
      for (const key of track.keys || []) {
        const x = key.time_seconds / Math.max(0.0001, clip.duration_seconds) * canvas.width;
        const selected = timeline.selectedKeys.has(keyID(controlID, key.time_seconds));
        ctx.save(); ctx.translate(x, y); ctx.rotate(Math.PI / 4);
        ctx.fillStyle = selected ? '#ffffff' : '#ffd36b';
        ctx.fillRect(-5, -5, 10, 10); ctx.restore();
      }
    });
    const playheadX = S.t / Math.max(0.0001, clip.duration_seconds) * canvas.width;
    ctx.strokeStyle = '#ff6c77'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(playheadX, 0); ctx.lineTo(playheadX, canvas.height); ctx.stroke();
    $('#timeReadout').textContent = `${S.t.toFixed(3)} / ${clip.duration_seconds.toFixed(3)}`;
  }

  function hitKey(event) {
    const clip = userClip();
    if (!clip) return null;
    const canvas = $('#dope');
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width * canvas.width;
    const y = (event.clientY - bounds.top) / bounds.height * canvas.height;
    const rowHeight = 24;
    let best = null;
    let distance = 12;
    (clip.tracks || []).forEach((track, row) => {
      const ky = 28 + row * rowHeight;
      const controlID = track.control_id || track.bone_id;
      for (const key of track.keys || []) {
        const kx = key.time_seconds / Math.max(0.0001, clip.duration_seconds) * canvas.width;
        const d = Math.hypot(kx - x, ky - y);
        if (d < distance) { distance = d; best = {controlID, time: key.time_seconds}; }
      }
    });
    return best;
  }

  function timeFromEvent(event) {
    const clip = userClip();
    const bounds = $('#dope').getBoundingClientRect();
    return clamp((event.clientX - bounds.left) / Math.max(1, bounds.width) * clip.duration_seconds, 0, clip.duration_seconds);
  }

  function drawCurve() {
    const canvas = $('#curveCanvas');
    const ctx = canvas.getContext('2d');
    const selected = timeline.selectedKeyObjects()[0]?.key;
    const bezier = selected?.bezier || [0.25, 0.1, 0.25, 1];
    ['#curveX1', '#curveY1', '#curveX2', '#curveY2'].forEach((selector, index) => { $(selector).value = Number(bezier[index]).toFixed(2); });
    ctx.fillStyle = '#090d12'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#253342'; ctx.strokeRect(12, 12, canvas.width - 24, canvas.height - 24);
    ctx.strokeStyle = '#6e8296'; ctx.beginPath(); ctx.moveTo(12, canvas.height - 12); ctx.lineTo(canvas.width - 12, 12); ctx.stroke();
    if (selected?.curve === 'step') {
      ctx.strokeStyle = '#ffd36b'; ctx.beginPath(); ctx.moveTo(12, canvas.height - 12); ctx.lineTo(canvas.width - 12, canvas.height - 12); ctx.lineTo(canvas.width - 12, 12); ctx.stroke();
      return;
    }
    ctx.strokeStyle = selected?.curve === 'bezier' ? '#ffd36b' : '#8dd2ff';
    ctx.beginPath();
    for (let i = 0; i <= 60; i += 1) {
      const t = i / 60;
      const q = selected?.curve === 'bezier' ? cubic(t, 0, bezier[1], bezier[3], 1) : t;
      const x = 12 + t * (canvas.width - 24);
      const y = canvas.height - 12 - q * (canvas.height - 24);
      if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function cubic(t, p0, p1, p2, p3) {
    const u = 1 - t;
    return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
  }

  function escapeHTML(value) {
    return String(value).replace(/[&<>'"]/g, (char) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[char]));
  }

  function toast(message, bad = false) {
    const element = $('#toast');
    if (!element) return;
    element.textContent = message;
    element.className = `toast show${bad ? ' error' : ''}`;
    clearTimeout(toast.handle);
    toast.handle = setTimeout(() => { element.className = 'toast'; }, 2200);
  }
}

boot();
