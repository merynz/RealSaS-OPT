const canvas = document.querySelector('#depthScene');
const ctx = canvas.getContext('2d');
const shell = document.querySelector('.awaken-shell');
const input = document.querySelector('#sourceInput');
const dropzone = document.querySelector('#dropzone');
const preview = document.querySelector('#sourcePreview');
const resultPreview = document.querySelector('#resultPreview');
const awaken = document.querySelector('#awaken');
const pipeline = [...document.querySelectorAll('.pipe-step')];
const log = document.querySelector('#compilerLog');
const status = document.querySelector('#mastStatus');
const viewCount = document.querySelector('#viewCount');
const result = document.querySelector('#result');
const modeLabel = document.querySelector('#modeLabel');

const productStages = [
  ['INPUT', 'source_observation::qualified'],
  ['SHAPE', 'scene_memory::resolved'],
  ['MESH', 'canonical_mesh::sealed'],
  ['RIG', 'rig_program::compiled'],
  ['SKIN', 'weight_field::bound'],
  ['MECHANICS', 'deformation::validated'],
  ['GRAPHICS', 'appearance_domain::resolved'],
  ['MOTION', 'motion_program::attached'],
  ['PROOF', 'product_proof::pass'],
];

let project = null;
let selectedFiles = [];
let running = false;
let particles = [];
let codeColumns = [];
let startTime = performance.now();

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(innerWidth * dpr);
  canvas.height = Math.floor(innerHeight * dpr);
  canvas.style.width = `${innerWidth}px`;
  canvas.style.height = `${innerHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  seedDepth();
}

function seedDepth() {
  const words = [
    'scene.memory', 'surface.query', 'mesh.compile', 'rig.program', 'weights.solve',
    'mechanics.validate', 'appearance.authority', 'motion.compile', 'proof.pass',
    'typed.evidence', 'persistent.identity', 'source.wins', 'product.revision',
  ];
  codeColumns = Array.from({length: Math.max(16, Math.floor(innerWidth / 90))}, (_, i) => ({
    x: (i + 0.5) * innerWidth / Math.max(16, Math.floor(innerWidth / 90)),
    z: Math.random(),
    speed: 0.000035 + Math.random() * 0.00005,
    phrase: words[i % words.length],
    phase: Math.random() * 100,
  }));
  particles = Array.from({length: 56}, () => ({
    x: Math.random() * innerWidth,
    y: Math.random() * innerHeight,
    z: Math.random(),
    speed: 0.015 + Math.random() * 0.045,
  }));
}

function drawDepth(now) {
  const w = innerWidth;
  const h = innerHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const column of codeColumns) {
    column.z += column.speed * (now - startTime);
    if (column.z > 1.2) column.z = 0.02;
    const depth = Math.max(0.06, column.z);
    const horizon = h * 0.46;
    const x = w * 0.5 + (column.x - w * 0.5) * (0.18 + depth * 1.34);
    const y = horizon + (depth * depth) * h * 0.52 + Math.sin(now * 0.00025 + column.phase) * 7;
    const alpha = Math.max(0, Math.min(0.12, depth * 0.11));
    const size = 7 + depth * 5;
    ctx.font = `${size}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
    ctx.fillStyle = `rgba(143,176,195,${alpha})`;
    ctx.fillText(column.phrase, x, y);
  }
  for (const p of particles) {
    p.y -= p.speed;
    if (p.y < -4) {
      p.y = h + 4;
      p.x = Math.random() * w;
    }
    const a = 0.03 + p.z * 0.12;
    ctx.fillStyle = `rgba(180,213,230,${a})`;
    ctx.fillRect(p.x, p.y, 1 + p.z, 1 + p.z);
  }
  ctx.restore();
  startTime = now;
  requestAnimationFrame(drawDepth);
}

function addLog(message, hot = false) {
  const row = document.createElement('div');
  row.textContent = `> ${message}`;
  if (hot) row.classList.add('hot');
  log.prepend(row);
  while (log.children.length > 8) log.removeChild(log.lastElementChild);
}

function setFiles(files) {
  selectedFiles = [...files].filter((file) => file.type.startsWith('image/')).slice(0, 2);
  if (!selectedFiles.length) return;
  const url = URL.createObjectURL(selectedFiles[0]);
  preview.src = url;
  resultPreview.src = url;
  dropzone.classList.add('has-image');
  viewCount.textContent = `${selectedFiles.length} VIEW${selectedFiles.length > 1 ? 'S' : ''}`;
  awaken.disabled = false;
  addLog(`source intake armed · ${selectedFiles.length} view${selectedFiles.length > 1 ? 's' : ''}`, true);
}

for (const event of ['dragenter', 'dragover']) {
  dropzone.addEventListener(event, (e) => {
    e.preventDefault();
    dropzone.classList.add('drag');
  });
}
for (const event of ['dragleave', 'drop']) {
  dropzone.addEventListener(event, (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag');
  });
}
dropzone.addEventListener('drop', (e) => setFiles(e.dataTransfer.files));
input.addEventListener('change', () => setFiles(input.files));

function resetPipeline() {
  pipeline.forEach((step) => step.classList.remove('active', 'done'));
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runPresentationFlow() {
  resetPipeline();
  shell.classList.add('compiling');
  status.classList.add('ready');
  status.querySelector('span').textContent = 'COMPILING';
  awaken.disabled = true;
  addLog('compile graph admitted', true);
  for (let i = 0; i < productStages.length; i += 1) {
    const [name, event] = productStages[i];
    if (i > 0) {
      pipeline[i - 1].classList.remove('active');
      pipeline[i - 1].classList.add('done');
    }
    pipeline[i].classList.add('active');
    addLog(`${String(i + 1).padStart(2, '0')} ${name.toLowerCase()} · ${event}`, true);
    await delay(i === 0 ? 720 : 560 + Math.random() * 380);
  }
  pipeline.at(-1).classList.remove('active');
  pipeline.at(-1).classList.add('done');
  addLog('ProductRevision sealed · proof authority PASS', true);
  await delay(500);
  result.classList.add('show');
  status.querySelector('span').textContent = 'ALIVE';
}

async function startProductionCompile() {
  const views = await Promise.all(selectedFiles.map(async (file, index) => ({
    slot: index === 0 ? 'front' : 'back',
    name: file.name,
    mime_type: file.type,
    bytes_base64: await file.arrayBuffer().then((buffer) => {
      let binary = '';
      const bytes = new Uint8Array(buffer);
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      return btoa(binary);
    }),
  })));
  const response = await fetch('/api/compile/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({views}),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.message || `HTTP ${response.status}`);
  return payload;
}

async function runProductionFlow() {
  resetPipeline();
  shell.classList.add('compiling');
  status.classList.add('ready');
  status.querySelector('span').textContent = 'SUBMITTING';
  awaken.disabled = true;
  addLog('submitting public compile session', true);
  const session = await startProductionCompile();
  const id = encodeURIComponent(session.compile_id || session.session_id || '');
  if (!id) throw new Error('compile gateway returned no public compile id');
  let terminal = false;
  let lastStage = -1;
  while (!terminal) {
    const response = await fetch(`/api/compile/status?id=${id}`, {cache: 'no-store'});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.message || `HTTP ${response.status}`);
    const stageName = String(payload.stage || payload.capability || '').toUpperCase();
    const index = productStages.findIndex(([name]) => stageName.includes(name));
    if (index >= 0 && index !== lastStage) {
      pipeline.forEach((step, i) => {
        step.classList.toggle('active', i === index);
        step.classList.toggle('done', i < index);
      });
      addLog(`${String(index + 1).padStart(2, '0')} ${stageName.toLowerCase()} · ${payload.detail || 'qualified execution'}`, true);
      lastStage = index;
    }
    terminal = Boolean(payload.terminal) || ['PASS', 'FAILED', 'ERROR'].includes(String(payload.status || '').toUpperCase());
    if (!terminal) await delay(700);
    if (terminal && String(payload.status || '').toUpperCase() !== 'PASS') {
      throw new Error(payload.message || 'compile did not qualify');
    }
  }
  pipeline.forEach((step) => {
    step.classList.remove('active');
    step.classList.add('done');
  });
  addLog('ProductRevision sealed · proof authority PASS', true);
  result.classList.add('show');
  status.querySelector('span').textContent = 'ALIVE';
}

awaken.addEventListener('click', async () => {
  if (running || !selectedFiles.length) return;
  running = true;
  try {
    if (project?.compile_available && project?.public_compile_gateway) {
      await runProductionFlow();
    } else {
      await runPresentationFlow();
    }
  } catch (error) {
    shell.classList.remove('compiling');
    status.classList.remove('ready');
    status.querySelector('span').textContent = 'INTAKE READY';
    awaken.disabled = false;
    addLog(`compile rejected · ${error.message}`, true);
  } finally {
    running = false;
  }
});

async function loadProject() {
  try {
    const response = await fetch('/api/project', {cache: 'no-store'});
    project = await response.json();
    const production = response.ok && project?.compile_available && project?.public_compile_gateway;
    modeLabel.textContent = production ? 'PRODUCTION GATEWAY' : 'PRESENTATION PREVIEW · NO PRODUCT AUTHORITY';
    if (production) {
      status.classList.add('ready');
      status.querySelector('span').textContent = 'PRODUCTION READY';
    }
  } catch {
    modeLabel.textContent = 'PRESENTATION PREVIEW · NO PRODUCT AUTHORITY';
  }
}

window.addEventListener('resize', resize);
resize();
requestAnimationFrame(drawDepth);
loadProject();
addLog('living compile surface online');
addLog('awaiting source observation');
