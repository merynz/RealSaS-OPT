const input = document.querySelector('#sourceInput');
const dropzone = document.querySelector('#dropzone');
const front = document.querySelector('#frontSlotFile');
const back = document.querySelector('#backSlotFile');
const frontRow = document.querySelector('[data-input-slot="front"]');
const backRow = document.querySelector('[data-input-slot="back"]');
const swap = document.querySelector('#swapInputViews');

function imageFiles(files) {
  return [...(files || [])].filter((file) => String(file.type || '').startsWith('image/')).slice(0, 2);
}

function assignInputFiles(files) {
  const selected = imageFiles(files);
  if (!input || typeof DataTransfer === 'undefined') return selected;
  const transfer = new DataTransfer();
  selected.forEach((file) => transfer.items.add(file));
  input.files = transfer.files;
  return selected;
}

function render(files) {
  const selected = imageFiles(files);
  const frontFile = selected[0] || null;
  const backFile = selected[1] || null;
  if (front) {
    front.textContent = frontFile?.name || 'REQUIRED';
    front.title = frontFile?.name || 'A front source view is required.';
  }
  if (back) {
    back.textContent = backFile?.name || 'OPTIONAL';
    back.title = backFile?.name || 'A back source view is optional but recommended.';
  }
  frontRow?.classList.toggle('assigned', Boolean(frontFile));
  backRow?.classList.toggle('assigned', Boolean(backFile));
  if (swap) swap.hidden = !frontFile || !backFile;
}

input?.addEventListener('change', () => render(input.files));

dropzone?.addEventListener('drop', (event) => {
  const selected = assignInputFiles(event.dataTransfer?.files);
  render(selected);
});

swap?.addEventListener('click', (event) => {
  event.preventDefault();
  event.stopPropagation();
  const selected = imageFiles(input?.files);
  if (selected.length !== 2 || !input || typeof DataTransfer === 'undefined') return;
  const transfer = new DataTransfer();
  transfer.items.add(selected[1]);
  transfer.items.add(selected[0]);
  input.files = transfer.files;
  render(input.files);
  input.dispatchEvent(new Event('change', {bubbles: true}));
});

render(input?.files);
