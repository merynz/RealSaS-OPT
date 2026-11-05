const TOOLS = new Set(['select', 'translate', 'rotate', 'scale', 'pan', 'box', 'lasso', 'pivot']);
const SPACES = new Set(['world', 'local']);
const PIVOT_MODES = new Set(['selection', 'primary', 'custom']);

export class StudioToolContext extends EventTarget {
  constructor() {
    super();
    this.tool = 'select';
    this.transformSpace = 'world';
    this.snapEnabled = true;
    this.translateSnap = 1;
    this.rotateSnapDegrees = 15;
    this.scaleSnap = 0.1;
    this.pivotMode = 'selection';
    this.customPivot = [0, 0];
    this.interaction = null;
  }

  setTool(tool) {
    if (!TOOLS.has(tool)) throw new RangeError(`unsupported Studio tool: ${tool}`);
    if (tool === this.tool) return;
    this.tool = tool;
    this.interaction = null;
    this.emit();
  }

  setTransformSpace(space) {
    if (!SPACES.has(space)) throw new RangeError(`unsupported transform space: ${space}`);
    this.transformSpace = space;
    this.emit();
  }

  toggleTransformSpace() {
    this.setTransformSpace(this.transformSpace === 'world' ? 'local' : 'world');
  }

  setSnapEnabled(enabled) {
    this.snapEnabled = Boolean(enabled);
    this.emit();
  }

  configureSnapping({translate, rotateDegrees, scale} = {}) {
    if (translate !== undefined) this.translateSnap = positiveFinite(translate, 'translate snap');
    if (rotateDegrees !== undefined) this.rotateSnapDegrees = positiveFinite(rotateDegrees, 'rotate snap');
    if (scale !== undefined) this.scaleSnap = positiveFinite(scale, 'scale snap');
    this.emit();
  }

  setPivotMode(mode) {
    if (!PIVOT_MODES.has(mode)) throw new RangeError(`unsupported pivot mode: ${mode}`);
    this.pivotMode = mode;
    this.emit();
  }

  setCustomPivot(point) {
    if (!Array.isArray(point) || point.length !== 2 || point.some((value) => !Number.isFinite(value))) {
      throw new TypeError('custom pivot must be finite [x,y]');
    }
    this.customPivot = [Number(point[0]), Number(point[1])];
    this.pivotMode = 'custom';
    this.emit();
  }

  snapTranslation(delta, bypass = false) {
    if (!this.snapEnabled || bypass) return [...delta];
    return delta.map((value) => snap(value, this.translateSnap));
  }

  snapAngleRadians(angle, bypass = false) {
    if (!this.snapEnabled || bypass) return angle;
    const step = this.rotateSnapDegrees * Math.PI / 180;
    return snap(angle, step);
  }

  snapScaleFactor(factor, bypass = false) {
    if (!this.snapEnabled || bypass) return factor;
    return Math.max(this.scaleSnap, snap(factor, this.scaleSnap));
  }

  beginInteraction(kind, payload = {}) {
    this.interaction = {kind, ...payload};
    this.emit();
    return this.interaction;
  }

  updateInteraction(patch = {}) {
    if (!this.interaction) return null;
    Object.assign(this.interaction, patch);
    this.emit();
    return this.interaction;
  }

  endInteraction() {
    const finished = this.interaction;
    this.interaction = null;
    this.emit();
    return finished;
  }

  cancelInteraction() {
    return this.endInteraction();
  }

  snapshot() {
    return {
      tool: this.tool,
      transform_space: this.transformSpace,
      snap_enabled: this.snapEnabled,
      translate_snap: this.translateSnap,
      rotate_snap_degrees: this.rotateSnapDegrees,
      scale_snap: this.scaleSnap,
      pivot_mode: this.pivotMode,
      custom_pivot: [...this.customPivot],
      interaction: this.interaction ? {...this.interaction} : null,
    };
  }

  emit() {
    this.dispatchEvent(new CustomEvent('change', {detail: this.snapshot()}));
  }
}

function snap(value, step) {
  return Math.round(value / step) * step;
}

function positiveFinite(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new RangeError(`${label} must be > 0`);
  return number;
}
