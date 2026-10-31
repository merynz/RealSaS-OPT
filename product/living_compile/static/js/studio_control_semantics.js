export const StudioControlClass = Object.freeze({
  DEFORM_JOINT: 'DEFORM_JOINT',
  ANIMATOR_CONTROL: 'ANIMATOR_CONTROL',
});

function clonePair(value, fallback) {
  return Array.isArray(value) && value.length === 2
    ? [Number(value[0]) || 0, Number(value[1]) || 0]
    : [...fallback];
}

export function isAnimatorControl(row) {
  return row?.entity_kind === StudioControlClass.ANIMATOR_CONTROL
    || row?.control_class === StudioControlClass.ANIMATOR_CONTROL
    || row?.participation === 'animation';
}

export function isDeformJoint(row) {
  if (!row) return false;
  if (row.entity_kind === StudioControlClass.DEFORM_JOINT || row.control_class === StudioControlClass.DEFORM_JOINT) return true;
  if (row.participation === 'deform') return true;
  return !isAnimatorControl(row); // legacy V4 bundles exposed joints as generic controls
}

export function animatorControls(rows) {
  return (rows || []).filter(isAnimatorControl);
}

export function deformControls(rows) {
  return (rows || []).filter(isDeformJoint);
}

export function controlsForMode(rows, mode) {
  const all = rows || [];
  if (mode === 'animation') {
    const animator = animatorControls(all);
    return animator.length ? animator : deformControls(all); // legacy bundle fallback only
  }
  return deformControls(all);
}

export function identityPose() {
  return {translation_xy: [0, 0], rotation_degrees: 0, scale_xy: [1, 1]};
}

export function normalizePose(value) {
  return {
    translation_xy: clonePair(value?.translation_xy, [0, 0]),
    rotation_degrees: Number(value?.rotation_degrees) || 0,
    scale_xy: clonePair(value?.scale_xy, [1, 1]),
  };
}

/**
 * Local Studio preview only. Canonical constraint execution remains Mechanics
 * authority. We deterministically compose animator-control poses into their
 * declared target deform joints so key authoring produces immediate visual
 * feedback without minting joint-track authority.
 */
export function derivePreviewJointPoses(animatorRows, animatorPoseMap, baseJointPoseMap = new Map()) {
  const result = new Map();
  for (const [jointID, pose] of baseJointPoseMap || []) result.set(jointID, normalizePose(pose));

  const ordered = [...(animatorRows || [])].sort((a, b) => String(a.control_id).localeCompare(String(b.control_id)));
  for (const control of ordered) {
    const pose = animatorPoseMap?.get(control.control_id);
    if (!pose) continue;
    const normalized = normalizePose(pose);
    const targets = [...(control.target_joint_ids || [])].map(String).sort();
    for (const jointID of targets) {
      const current = result.get(jointID) || identityPose();
      result.set(jointID, {
        translation_xy: [
          current.translation_xy[0] + normalized.translation_xy[0],
          current.translation_xy[1] + normalized.translation_xy[1],
        ],
        rotation_degrees: current.rotation_degrees + normalized.rotation_degrees,
        scale_xy: [
          current.scale_xy[0] * normalized.scale_xy[0],
          current.scale_xy[1] * normalized.scale_xy[1],
        ],
        preview_metadata: {
          constraint_execution_approximation: true,
          authoritative_constraint_owner: 'MECHANICS',
        },
      });
    }
  }
  return result;
}
