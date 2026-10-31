export class StudioDirtyState extends EventTarget {
  constructor() {
    super();
    this.status = 'clean';
    this.revisionNumber = 0;
    this.lockVersion = 0;
    this.requiresRequalification = false;
    this.lastError = null;
  }

  emit() {
    this.dispatchEvent(new CustomEvent('change', {detail: this.snapshot()}));
  }

  snapshot() {
    return {
      status: this.status,
      revision_number: this.revisionNumber,
      lock_version: this.lockVersion,
      requires_requalification: this.requiresRequalification,
      last_error: this.lastError,
    };
  }

  hydrate(head = {}) {
    this.status = 'clean';
    this.revisionNumber = Number(head.revision_number || 0);
    this.lockVersion = Number(head.lock_version || 0);
    this.requiresRequalification = Boolean(head.requires_requalification);
    this.lastError = null;
    this.emit();
  }

  markDirty() {
    if (this.status !== 'conflict') this.status = 'dirty';
    this.requiresRequalification = true;
    this.lastError = null;
    this.emit();
  }

  markSaving() {
    if (this.status !== 'conflict') this.status = 'saving';
    this.emit();
  }

  markSaved(revision = {}) {
    this.status = 'clean';
    this.revisionNumber = Number(revision.revision_number ?? this.revisionNumber);
    this.lockVersion = Number(revision.lock_version ?? this.lockVersion);
    this.requiresRequalification = Boolean(
      revision.requires_requalification ?? this.requiresRequalification,
    );
    this.lastError = null;
    this.emit();
  }

  markConflict(message = 'Studio revision conflict') {
    this.status = 'conflict';
    this.requiresRequalification = true;
    this.lastError = String(message);
    this.emit();
  }

  markError(message) {
    this.status = 'dirty';
    this.lastError = String(message || 'Save failed');
    this.emit();
  }
}
