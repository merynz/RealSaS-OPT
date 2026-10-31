const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

export class StudioTimelineView extends EventTarget {
  constructor(durationSeconds = 1) {
    super();
    this.duration = positive(durationSeconds, 1);
    this.zoom = 1;
    this.start = 0;
    this.loopStart = null;
    this.loopEnd = null;
  }

  bindDuration(durationSeconds) {
    this.duration = positive(durationSeconds, 1);
    this.zoom = clamp(Number(this.zoom) || 1, 1, 64);
    this.start = clamp(this.start, 0, this.maxStart());
    if (this.loopStart !== null) this.loopStart = clamp(this.loopStart, 0, this.duration);
    if (this.loopEnd !== null) this.loopEnd = clamp(this.loopEnd, 0, this.duration);
    if (this.loopStart !== null && this.loopEnd !== null && this.loopEnd <= this.loopStart) {
      this.loopStart = null;
      this.loopEnd = null;
    }
    this.emit();
  }

  span() {
    return this.duration / this.zoom;
  }

  end() {
    return Math.min(this.duration, this.start + this.span());
  }

  maxStart() {
    return Math.max(0, this.duration - this.span());
  }

  setZoom(zoom, anchorTime = this.start + this.span() / 2) {
    const previousSpan = this.span();
    const previousStart = this.start;
    const anchor = clamp(Number(anchorTime) || 0, 0, this.duration);
    const ratio = previousSpan > 0 ? clamp((anchor - previousStart) / previousSpan, 0, 1) : 0.5;
    this.zoom = clamp(Number(zoom) || 1, 1, 64);
    const nextSpan = this.span();
    this.start = clamp(anchor - ratio * nextSpan, 0, this.maxStart());
    this.emit();
    return this.zoom;
  }

  panBy(seconds) {
    this.start = clamp(this.start + Number(seconds || 0), 0, this.maxStart());
    this.emit();
    return this.start;
  }

  setStart(seconds) {
    this.start = clamp(Number(seconds || 0), 0, this.maxStart());
    this.emit();
    return this.start;
  }

  reset() {
    this.zoom = 1;
    this.start = 0;
    this.emit();
  }

  timeToUnit(timeSeconds) {
    const span = Math.max(1e-9, this.span());
    return (Number(timeSeconds || 0) - this.start) / span;
  }

  unitToTime(unit) {
    return clamp(this.start + clamp(Number(unit) || 0, 0, 1) * this.span(), 0, this.duration);
  }

  contains(timeSeconds) {
    const time = Number(timeSeconds || 0);
    return time >= this.start - 1e-9 && time <= this.end() + 1e-9;
  }

  setLoopIn(timeSeconds) {
    const time = clamp(Number(timeSeconds || 0), 0, this.duration);
    this.loopStart = time;
    if (this.loopEnd !== null && this.loopEnd <= time) this.loopEnd = null;
    this.emit();
  }

  setLoopOut(timeSeconds) {
    const time = clamp(Number(timeSeconds || 0), 0, this.duration);
    this.loopEnd = time;
    if (this.loopStart !== null && this.loopStart >= time) this.loopStart = null;
    this.emit();
  }

  setLoopRange(startSeconds, endSeconds) {
    const start = clamp(Number(startSeconds || 0), 0, this.duration);
    const end = clamp(Number(endSeconds || 0), 0, this.duration);
    if (end <= start) throw new RangeError('loop range end must be after start');
    this.loopStart = start;
    this.loopEnd = end;
    this.emit();
  }

  clearLoopRange() {
    this.loopStart = null;
    this.loopEnd = null;
    this.emit();
  }

  loopBounds(defaultStart = 0, defaultEnd = this.duration) {
    if (this.loopStart !== null && this.loopEnd !== null && this.loopEnd > this.loopStart) {
      return [this.loopStart, this.loopEnd];
    }
    return [clamp(defaultStart, 0, this.duration), clamp(defaultEnd, 0, this.duration)];
  }

  snapshot() {
    return {
      duration_seconds: this.duration,
      zoom: this.zoom,
      visible_start_seconds: this.start,
      visible_end_seconds: this.end(),
      loop_start_seconds: this.loopStart,
      loop_end_seconds: this.loopEnd,
    };
  }

  emit() {
    this.dispatchEvent(new CustomEvent('change', {detail: this.snapshot()}));
  }
}

function positive(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}
