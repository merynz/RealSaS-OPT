import {capturePoseSnapshot, pastePoseSnapshot} from './studio_pose_onion_core.js';

const $ = (selector) => document.querySelector(selector);

let poseClipboard = [];
let saveTimer = null;
let lastView = null;

function boot() {
  const studio = window.RealSaSStudio;
  if (!studio?.timeline || !studio?.document || !studio?.journal || !$('#animAuthoringTools') || !$('#viewport')) {
    requestAnimationFrame(boot);
    return;
  }
  installControls(studio);
  installGhostOverlay(studio);
  requestAnimationFrame(() => frame(studio));
}

function activeUserClip(studio) {
  const timeline = studio.timeline;
  const S = studio.state;
  if (timeline.animationEdits !== S.ed.animation) timeline.bind(S.ed.animation);
  if (S.clip && timeline.clip(S.clip) && timeline.clipID !== S.clip) timeline.selectClip(S.clip);
  return timeline.clip(S.clip);
}

function installControls(studio) {
  if ($('#poseOnionTools')) return;
  const host = $('#animAuthoringTools');
  const tools = document.createElement('span');
  tools.id = 'poseOnionTools';
  tools.className = 'pose-onion-tools';
  tools.innerHTML = `
    <span class="anim-divider"></span>
    <button id="copyPose" title="Copy the sampled animator-control pose at the playhead">COPY POSE</button>
    <button id="pastePose" title="Paste the copied pose as keys at the playhead">PASTE POSE</button>
    <label class="onion-toggle"><input id="onionSkin" type="checkbox"> ONION</label>
    <label>±F <input id="onionFrames" type="number" min="1" max="12" step="1" value="1"></label>
  `;
  host.append(tools);

  $('#copyPose').onclick = () => {
    const clip = activeUserClip(studio);
    if (!clip) return notify('Pose copy requires a user clip', true);
    poseClipboard = capturePoseSnapshot(studio.timeline, clip.clip_id, studio.state.t);
    notify(poseClipboard.length ? `Pose copied · ${poseClipboard.length} controls` : 'No animated controls at this pose', !poseClipboard.length);
    refreshControls(studio);
  };

  $('#pastePose').onclick = () => {
    const clip = activeUserClip(studio);
    if (!clip) return notify('Pose paste requires a user clip', true);
    if (!poseClipboard.length) return notify('Copy a pose first', true);
    const curve = String($('#curveSelect')?.value || 'linear');
    transaction(studio, 'Paste animation pose', () => pastePoseSnapshot(
      studio.timeline,
      poseClipboard,
      studio.state.t,
      {curve},
    ));
  };

  $('#onionSkin').onchange = () => applyOnionSettings(studio);
  $('#onionFrames').onchange = () => applyOnionSettings(studio);
  refreshControls(studio);
}

function transaction(studio, label, mutate) {
  const before = studio.document.snapshot();
  const changed = Number(mutate() || 0);
  if (!changed) return false;
  const after = studio.document.snapshot();
  if (!studio.journal.commit(label, before, after)) return false;
  studio.dirty.markDirty();
  preview(studio);
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    if (studio.dirty.status === 'dirty' && !studio.dirty.lastError) $('#saveLayer')?.click();
  }, 1200);
  notify(`Pose pasted · ${changed} controls`);
  return true;
}

function applyOnionSettings(studio) {
  const enabled = Boolean($('#onionSkin')?.checked) && Boolean(activeUserClip(studio));
  const frameOffset = Math.max(1, Math.min(12, Math.round(Number($('#onionFrames')?.value) || 1)));
  $('#onionFrames').value = String(frameOffset);
  if (typeof studio.setAnimationOnionSkin === 'function') {
    studio.setAnimationOnionSkin({enabled, frameOffset});
  } else {
    studio.animationOnionSkin = {enabled, frameOffset};
    preview(studio);
  }
}

function preview(studio) {
  const clip = activeUserClip(studio);
  if (!clip) return;
  window.dispatchEvent(new CustomEvent('realsas:studio-animation-pose-changed', {
    detail: {clip_id: clip.clip_id, time_seconds: Number(studio.state.t || 0)},
  }));
}

function refreshControls(studio) {
  const clip = activeUserClip(studio);
  if ($('#copyPose')) $('#copyPose').disabled = !clip;
  if ($('#pastePose')) $('#pastePose').disabled = !clip || !poseClipboard.length;
  if ($('#onionSkin')) $('#onionSkin').disabled = !clip;
  if ($('#onionFrames')) $('#onionFrames').disabled = !clip || !$('#onionSkin')?.checked;
}

function installGhostOverlay(studio) {
  if ($('#animationGhostOverlay')) return;
  const viewport = $('#viewport');
  const panel = document.querySelector('.viewport-panel');
  if (!viewport || !panel) return;
  const canvas = document.createElement('canvas');
  canvas.id = 'animationGhostOverlay';
  canvas.className = 'animation-ghost-overlay';
  canvas.setAttribute('aria-hidden', 'true');
  panel.append(canvas);
  studio.animationGhostOverlay = canvas;
}

function frame(studio) {
  refreshControls(studio);
  const view = Number(studio.state.view || 0);
  if (view !== lastView) {
    lastView = view;
    if (studio.animationOnionSkin?.enabled) preview(studio);
  }
  drawGhostOverlay(studio);
  requestAnimationFrame(() => frame(studio));
}

function drawGhostOverlay(studio) {
  const canvas = $('#animationGhostOverlay');
  const viewport = $('#viewport');
  if (!canvas || !viewport) return;
  if (canvas.width !== viewport.width) canvas.width = viewport.width;
  if (canvas.height !== viewport.height) canvas.height = viewport.height;
  canvas.style.left = `${viewport.offsetLeft}px`;
  canvas.style.top = `${viewport.offsetTop}px`;
  canvas.style.width = `${viewport.clientWidth}px`;
  canvas.style.height = `${viewport.clientHeight}px`;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const S = studio.state;
  if (S.mode !== 'animation' || !studio.animationOnionSkin?.enabled) return;
  const viewID = `V${S.view}`;
  const frames = (studio.animationGhostFrames || []).filter((frame) => frame.view_id === viewID);
  if (!frames.length) return;
  const meshes = (S.scene?.puppet?.meshes || []).filter((mesh) => mesh.view_id === viewID);
  const meshByID = new Map(meshes.map((mesh) => [mesh.mesh_id, mesh]));
  const screen = (point) => [point[0] * S.cam.s + S.cam.x, point[1] * S.cam.s + S.cam.y];

  for (const ghost of frames) {
    const previous = ghost.kind === 'previous';
    ctx.strokeStyle = previous ? 'rgba(90,190,255,.78)' : 'rgba(255,115,165,.78)';
    ctx.fillStyle = previous ? 'rgba(90,190,255,.035)' : 'rgba(255,115,165,.035)';
    ctx.lineWidth = 1.25;
    for (const [meshID, vertices] of Object.entries(ghost.mesh_vertices_by_id || {})) {
      const mesh = meshByID.get(meshID);
      if (!mesh || !Array.isArray(vertices)) continue;
      for (const triangle of mesh.triangles || []) {
        const points = triangle.map((index) => vertices[index]).filter(Boolean).map(screen);
        if (points.length !== 3) continue;
        ctx.beginPath();
        ctx.moveTo(...points[0]);
        ctx.lineTo(...points[1]);
        ctx.lineTo(...points[2]);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }
  }
}

function notify(message, bad = false) {
  const element = $('#toast');
  if (!element) return;
  element.textContent = String(message || '');
  element.className = `toast show${bad ? ' error' : ''}`;
  clearTimeout(notify.handle);
  notify.handle = setTimeout(() => { element.className = 'toast'; }, 2200);
}

boot();
