function clone(value) {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value));
}

export function studioSelectionKey(item) {
  if (!item || typeof item !== 'object') throw new TypeError('selection item required');
  const view = String(item.view_id || '*');
  switch (item.kind) {
    case 'control':
      if (!item.control_id) throw new TypeError('control_id required');
      return `${view}:control:${item.control_id}`;
    case 'vertex':
      if (!item.mesh_id || !Number.isInteger(item.vertex_index)) throw new TypeError('mesh vertex identity required');
      return `${view}:vertex:${item.mesh_id}:${item.vertex_index}`;
    case 'face':
      if (!item.mesh_id || !Number.isInteger(item.face_index)) throw new TypeError('mesh face identity required');
      return `${view}:face:${item.mesh_id}:${item.face_index}`;
    case 'attachment':
      if (!item.attachment_id) throw new TypeError('attachment_id required');
      return `${view}:attachment:${item.attachment_id}`;
    case 'slot':
      if (!item.slot_id) throw new TypeError('slot_id required');
      return `${view}:slot:${item.slot_id}`;
    case 'key':
      if (!item.clip_id || !item.track_id || !Number.isFinite(item.time_seconds)) throw new TypeError('animation key identity required');
      return `${view}:key:${item.clip_id}:${item.track_id}:${item.time_seconds}`;
    default:
      throw new TypeError(`unsupported selection kind: ${item.kind}`);
  }
}

export class StudioSelection extends EventTarget {
  constructor(viewID = 'V0') {
    super();
    this.viewID = String(viewID || 'V0');
    this.items = new Map();
    this.primaryKey = null;
  }

  setView(viewID) {
    const next = String(viewID || 'V0');
    if (next === this.viewID) return;
    this.viewID = next;
    const primary = this.primary();
    if (!primary || (primary.view_id && primary.view_id !== next)) {
      this.primaryKey = this.activeKeys()[0] || null;
    }
    this.emit();
  }

  activeEntries() {
    return [...this.items.entries()].filter(([, item]) => !item.view_id || item.view_id === this.viewID);
  }

  active() {
    return this.activeEntries().map(([, item]) => clone(item));
  }

  activeKeys() {
    return this.activeEntries().map(([key]) => key);
  }

  primary() {
    if (this.primaryKey) {
      const item = this.items.get(this.primaryKey);
      if (item && (!item.view_id || item.view_id === this.viewID)) return clone(item);
    }
    const first = this.activeEntries()[0];
    return first ? clone(first[1]) : null;
  }

  has(item) {
    return this.items.has(studioSelectionKey(item));
  }

  size() {
    return this.activeEntries().length;
  }

  replace(item) {
    this.clear({emit: false, viewOnly: true});
    if (item) this.add(item, {emit: false, primary: true});
    this.emit();
  }

  replaceMany(items, {primary = null} = {}) {
    this.clear({emit: false, viewOnly: true});
    for (const item of items || []) this.add(item, {emit: false, primary: false});
    if (primary) {
      const key = studioSelectionKey(primary);
      if (this.items.has(key)) this.primaryKey = key;
    }
    if (!this.primaryKey || !this.items.has(this.primaryKey)) this.primaryKey = this.activeKeys()[0] || null;
    this.emit();
  }

  add(item, {emit = true, primary = true} = {}) {
    const value = clone(item);
    const key = studioSelectionKey(value);
    this.items.set(key, value);
    if (primary || !this.primaryKey) this.primaryKey = key;
    if (emit) this.emit();
    return key;
  }

  remove(item, {emit = true} = {}) {
    const key = typeof item === 'string' ? item : studioSelectionKey(item);
    const removed = this.items.delete(key);
    if (this.primaryKey === key) this.primaryKey = this.activeKeys()[0] || null;
    if (removed && emit) this.emit();
    return removed;
  }

  toggle(item) {
    const key = studioSelectionKey(item);
    if (this.items.has(key)) this.remove(key);
    else this.add(item);
  }

  clear({emit = true, viewOnly = false} = {}) {
    if (viewOnly) {
      for (const [key, item] of this.items) {
        if (!item.view_id || item.view_id === this.viewID) this.items.delete(key);
      }
    } else {
      this.items.clear();
    }
    this.primaryKey = this.activeKeys()[0] || null;
    if (emit) this.emit();
  }

  snapshot() {
    return {
      view_id: this.viewID,
      primary_key: this.primaryKey,
      items: [...this.items.values()].map(clone),
    };
  }

  restore(snapshot) {
    this.items.clear();
    this.viewID = String(snapshot?.view_id || 'V0');
    for (const item of snapshot?.items || []) this.items.set(studioSelectionKey(item), clone(item));
    this.primaryKey = snapshot?.primary_key && this.items.has(snapshot.primary_key)
      ? snapshot.primary_key
      : this.activeKeys()[0] || null;
    this.emit();
  }

  emit() {
    this.dispatchEvent(new CustomEvent('change', {detail: this.snapshot()}));
  }
}
