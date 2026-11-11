function clone(value) {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value));
}

function signature(value) {
  return JSON.stringify(value);
}

export class StudioOperationJournal {
  constructor(limit = 100) {
    this.limit = Math.max(10, Number(limit) || 100);
    this.undoStack = [];
    this.redoStack = [];
  }

  clear() {
    this.undoStack.length = 0;
    this.redoStack.length = 0;
  }

  commit(label, before, after) {
    if (signature(before) === signature(after)) return false;
    this.undoStack.push({label: String(label || 'Edit'), before: clone(before), after: clone(after)});
    if (this.undoStack.length > this.limit) this.undoStack.shift();
    this.redoStack.length = 0;
    return true;
  }

  canUndo() {
    return this.undoStack.length > 0;
  }

  canRedo() {
    return this.redoStack.length > 0;
  }

  undo(document) {
    const operation = this.undoStack.pop();
    if (!operation) return null;
    document.replace(operation.before);
    this.redoStack.push(operation);
    return operation;
  }

  redo(document) {
    const operation = this.redoStack.pop();
    if (!operation) return null;
    document.replace(operation.after);
    this.undoStack.push(operation);
    return operation;
  }
}
