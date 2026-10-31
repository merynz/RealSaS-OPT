import {derivePreviewJointPoses, normalizePose} from './studio_control_semantics.js';
import {ghostSampleTimes} from './studio_pose_onion_core.js';

const $ = (selector) => document.querySelector(selector);
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

function boot() {
  const studio = window.RealSaSStudio;
  if (!studio?.timeline) {
    requestAnimationFrame(boot);
    return;
  }

  const S = studio.state;
  const timeline = studio.timeline;
  studio.animationPreviewPose = new Map();
  studio.animationOnionSkin = studio.animationOnionSkin || {enabled: false, frameOffset: 1};
  studio.animationGhostFrames = [];
  studio.refreshAnimationPreview = applyPreview;
  studio.setAnimationOnionSkin = ({enabled = false, frameOffset = 1} = {}) => {
    studio.animationOnionSkin = {
      enabled: Boolean(enabled),
      frameOffset: Math.max(1, Math.min(12, Math.round(Number(frameOffset) || 1))),
    };
    if (!studio.animationOnionSkin.enabled) studio.animationGhostFrames = [];
    applyPreview(S.t);
  };

  window.addEventListener('realsas:studio-animation-pose-changed', (event) => {
    const clipID = String(event.detail?.clip_id || S.clip || '');
    const time = Number(event.detail?.time_seconds ?? S.t ?? 0);
    if (timeline.animationEdits !== S.ed.animation) timeline.bind(S.ed.animation);
    if (!timeline.clip(clipID)) return;
    if (timeline.clipID !== clipID) timeline.selectClip(clipID);
    timeline.playhead = time;
    applyPreview(time);
  });

  function currentView() {
    return `V${S.view}`;
  }

  function animatorControls() {
    return S.scene?.puppet?.rig?.animator_controls?.filter((item) => item.view_id === currentView()) || [];
  }

  function deformControls() {
    const rig = S.scene?.puppet?.rig || {};
    return (rig.deform_controls || rig.controls || []).filter((item) => item.view_id === currentView());
  }

  function meshes() {
    return S.scene?.puppet?.meshes?.filter((item) => item.view_id === currentView()) || [];
  }

  function baseRig(jointID) {
    return normalizePose(S.ed.rig.find((item) => item.control_id === jointID));
  }

  function meshDelta(meshID, vertexIndex) {
    return S.ed.mesh.find((item) => item.mesh_id === meshID && item.vertex_index === vertexIndex)?.translation_xy || [0, 0];
  }

  function jointPoseStateAt(time) {
    const animatorPoseMap = new Map();
    for (const control of animatorControls()) {
      const sampled = timeline.sample(control.control_id, time);
      if (sampled) animatorPoseMap.set(control.control_id, normalizePose(sampled));
    }
    const baseJointPoseMap = new Map();
    for (const joint of deformControls()) baseJointPoseMap.set(joint.control_id, baseRig(joint.control_id));
    return {
      animatorPoseMap,
      jointPoses: derivePreviewJointPoses(animatorControls(), animatorPoseMap, baseJointPoseMap),
    };
  }

  function applyPreview(time = S.t) {
    if (timeline.animationEdits !== S.ed.animation) timeline.bind(S.ed.animation);
    const clip = timeline.clip(S.clip);
    if (!clip) {
      studio.animationGhostFrames = [];
      return;
    }

    const state = jointPoseStateAt(time);
    studio.animationPreviewPose = state.animatorPoseMap;

    S.previewPose.clear();
    for (const [jointID, pose] of state.jointPoses) S.previewPose.set(jointID, pose);

    // app.js applies sibling mesh_vertex_edits after reading any runtime frame.
    // The Studio animation deformation below already starts from those edited
    // source vertices, so compensate the handoff once to avoid applying the
    // same mesh delta twice in the viewport. Runtime proof frames are unchanged.
    const actualMeshes = deformMeshes(state.jointPoses);
    S.frame = {frame: {mesh_vertices_by_id: viewportFrameMeshes(actualMeshes)}};
    updateGhostFrames(clip, time);

    const readout = $('#timeReadout');
    if (readout) readout.textContent = `${Number(time).toFixed(3)} / ${Number(clip.duration_seconds || 0).toFixed(3)}`;
  }

  function viewportFrameMeshes(actualMeshes) {
    const result = {};
    for (const mesh of meshes()) {
      const vertices = actualMeshes[mesh.mesh_id];
      if (!vertices) continue;
      result[mesh.mesh_id] = vertices.map((point, vertexIndex) => {
        const delta = meshDelta(mesh.mesh_id, vertexIndex);
        return [point[0] - Number(delta[0] || 0), point[1] - Number(delta[1] || 0)];
      });
    }
    return result;
  }

  function updateGhostFrames(clip, time) {
    if (!studio.animationOnionSkin?.enabled) {
      studio.animationGhostFrames = [];
      return;
    }
    const times = ghostSampleTimes(
      time,
      Number(clip.duration_seconds || 0),
      Number(clip.fps || 30),
      {loop: Boolean(clip.loop), frameOffset: studio.animationOnionSkin.frameOffset},
    );
    const rows = [];
    for (const [kind, sampleTime] of [['previous', times.previous], ['next', times.next]]) {
      if (sampleTime === null) continue;
      const state = jointPoseStateAt(sampleTime);
      rows.push({
        kind,
        time_seconds: sampleTime,
        view_id: currentView(),
        mesh_vertices_by_id: deformMeshes(state.jointPoses),
      });
    }
    studio.animationGhostFrames = rows;
  }

  function controlMap() {
    return new Map(deformControls().map((item) => [item.control_id, item]));
  }

  function parentMap() {
    const valid = new Set(deformControls().map((item) => item.control_id));
    const parents = new Map();
    for (const edge of S.scene?.puppet?.rig?.edges || []) {
      if (edge.entity_kind && edge.entity_kind !== 'DEFORM_JOINT') continue;
      if (valid.has(edge.parent_control_id) && valid.has(edge.child_control_id)) parents.set(edge.child_control_id, edge.parent_control_id);
    }
    return parents;
  }

  function chain(controlID, parents) {
    const result = [];
    const seen = new Set();
    let current = controlID;
    while (current && !seen.has(current)) {
      seen.add(current);
      result.unshift(current);
      current = parents.get(current);
    }
    return result;
  }

  function pose(controlID, jointPoses) {
    return normalizePose(jointPoses.get(controlID) || baseRig(controlID));
  }

  function applyLocal(point, anchor, value) {
    const sx = Number(value.scale_xy?.[0] ?? 1);
    const sy = Number(value.scale_xy?.[1] ?? 1);
    const angle = Number(value.rotation_degrees || 0) * Math.PI / 180;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const x = (point[0] - anchor[0]) * sx;
    const y = (point[1] - anchor[1]) * sy;
    return [
      anchor[0] + x * c - y * s + Number(value.translation_xy?.[0] || 0),
      anchor[1] + x * s + y * c + Number(value.translation_xy?.[1] || 0),
    ];
  }

  function applyControlChain(point, controlID, byControl, parents, jointPoses) {
    let transformed = [...point];
    const prior = [];
    for (const id of chain(controlID, parents)) {
      const control = byControl.get(id);
      if (!control) continue;
      let anchor = [...control.anchor];
      for (const row of prior) anchor = applyLocal(anchor, row.anchor, row.pose);
      const currentPose = pose(id, jointPoses);
      transformed = applyLocal(transformed, anchor, currentPose);
      prior.push({anchor, pose: currentPose});
    }
    return transformed;
  }

  function deformMeshes(jointPoses) {
    const result = {};
    const byControl = controlMap();
    const parents = parentMap();
    for (const mesh of meshes()) {
      result[mesh.mesh_id] = mesh.vertices.map((base, vertexIndex) => {
        const delta = meshDelta(mesh.mesh_id, vertexIndex);
        const source = [base[0] + Number(delta[0] || 0), base[1] + Number(delta[1] || 0)];
        const weights = mesh.weight_solution?.per_vertex_weights?.[vertexIndex] || [];
        let total = 0;
        let x = 0;
        let y = 0;
        for (const [jointID, rawWeight] of weights) {
          const weight = clamp(Number(rawWeight || 0), 0, 1);
          if (!weight || !byControl.has(jointID)) continue;
          const point = applyControlChain(source, jointID, byControl, parents, jointPoses);
          x += point[0] * weight;
          y += point[1] * weight;
          total += weight;
        }
        if (total < 1) {
          x += source[0] * (1 - total);
          y += source[1] * (1 - total);
          total = 1;
        }
        return total > 0 ? [x / total, y / total] : source;
      });
    }
    return result;
  }
}

boot();
