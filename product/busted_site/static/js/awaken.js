import {installAwakenReactor} from './busted_ignition_reactor_v3.js';
import {apiFetch} from './api_gateway_v1.js';
const shell = document.querySelector('.awaken-shell');
const input = document.querySelector('#sourceInput');
const dropzone = document.querySelector('#dropzone');
const preview = document.querySelector('#sourcePreview');
const resultPreview = document.querySelector('#resultPreview');
const resultArt = document.querySelector('.result-art');
const openWorkspace = document.querySelector('.open-workspace');
const awaken = document.querySelector('#awaken');
const pipeline = [...document.querySelectorAll('.pipe-step')];
const log = document.querySelector('#compilerLog');
const status = document.querySelector('#mastStatus');
const viewCount = document.querySelector('#viewCount');
const result = document.querySelector('#result');
const modeLabel = document.querySelector('#modeLabel');
const compileLiveStage = document.querySelector('#compileLiveStage');
const compileStatusMessage = document.querySelector('#compileStatusMessage');
const ignitionReactor = installAwakenReactor(awaken, dropzone);

function showLiveCompileStatus(stage, detail) {
  if (compileLiveStage) compileLiveStage.textContent = String(stage || 'AWAITING SOURCE');
  if (compileStatusMessage) compileStatusMessage.textContent = String(detail || 'Waiting for compiler events.');
}

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

const progressConsole=document.querySelector('#compileProgressConsole');
const progressPercent=document.querySelector('#compileProgressPercent');
const progressContext=document.querySelector('#compileProgressContext');
const progressSource=document.querySelector('#compileProgressSource');
const progressBlocks=document.querySelector('#compileProgressBlocks');
const progressSegments=24;
if(progressBlocks)for(let i=0;i<progressSegments;i++)progressBlocks.append(document.createElement('span'));
const progressCells=progressBlocks?[...progressBlocks.children]:[];
let lastCompletedStage=0;
/* The percent is NEVER elapsed-time estimation. It is either reported by the
 * gateway explicitly, or the discrete share of *completed* compiler stages.
 * Unknown progress stays —% until the compiler reports real evidence. */
function updateCompileProgress({percent=null,completed=null,stage='AWAITING SIGNAL',running=false,source='STAGE COMPLETION'}={}){
  const p=Number.isFinite(percent)?Math.max(0,Math.min(100,percent)):null;
  if(progressPercent)progressPercent.textContent=p===null?'—%':Math.floor(p)+'%';
  if(progressContext)progressContext.textContent=String(stage).slice(0,80);
  if(progressSource)progressSource.textContent=completed===null?'AWAITING EVIDENCE':
    String(source).slice(0,32)+' · '+completed+'/9';
  if(progressConsole){
    progressConsole.dataset.verified=String(p!==null);
    progressConsole.dataset.running=String(running);
  }
  const lit=p===null?0:Math.floor(p/100*progressSegments);
  progressCells.forEach((cell,i)=>{
    cell.classList.toggle('done',i<lit);
    cell.classList.toggle('leading',i===lit-1&&lit>0&&running);
  });
}
function reportedPercent(payload){
  const candidates=[payload?.progress_percent,payload?.progress?.percent_complete,payload?.progress?.percent];
  for(const value of candidates){
    if(typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=100)return value;
  }
  return null;
}
function onCompilerEvidence(payload,stageName,index,terminal){
  const failed=terminal&&String(payload?.status||'').toUpperCase()!=='PASS';
  const passed=terminal&&!failed;
  if(index>=0)lastCompletedStage=Math.max(lastCompletedStage,index);
  const completed=passed?9:Math.min(8,lastCompletedStage);
  const explicit=reportedPercent(payload);
  const percent=passed?100:explicit!==null?explicit:index>=0||lastCompletedStage>0?
    Math.floor(completed/9*100):null;
  updateCompileProgress({percent,completed,stage:stageName||String(payload?.status||'AWAITING EVIDENCE').toUpperCase(),
    running:!terminal,source:explicit!==null?'SERVER PROGRESS':'STAGES SEALED'});
}

const resultCanvas = document.createElement('canvas');
resultCanvas.width = 360;
resultCanvas.height = 420;
resultCanvas.setAttribute('aria-label', 'Proof-qualified compiled character idle preview');
Object.assign(resultCanvas.style, {
  width: '100%',
  height: '100%',
  objectFit: 'contain',
  display: 'none',
});
resultArt.append(resultCanvas);
const resultCtx = resultCanvas.getContext('2d');

let project = null;
let selectedFiles = [];
let activeSourceObjectURL = null;
const compileAvailability = document.querySelector('#compileAvailability');
let running = false;
let revealGeneration = 0;

function addLog(message, hot = false) {
  const row = document.createElement('div');
  row.textContent = `> ${message}`;
  if (hot) row.classList.add('hot');
  log.prepend(row);
  while (log.children.length > 8) log.removeChild(log.lastElementChild);
}

async function apiJson(url, options = {}) {
  const response = await apiFetch(url, {cache: 'no-store', ...options});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.message || payload?.error || `HTTP ${response.status}`);
  return payload;
}

function syncGatewayUI() {
  const ready = Boolean(project?.compile_available && project?.public_compile_gateway);
  // Offline users can inspect the one-shot visual interaction, but no compile
  // action is offered or submitted: the adjacent status is always explicit.
  awaken.disabled = running || (ready && selectedFiles.length === 0);
  awaken.dataset.ignitionPreview = String(!ready);
  awaken.setAttribute('aria-label', ready ? 'Awaken your character' :
    'Preview ignition interaction — compiler offline');
  if (status) {
    status.querySelector('span').textContent = ready ? 'COMPILER READY' : 'COMPILER OFFLINE';
    status.classList.toggle('ready', ready);
  }
  awaken.title = ready ? (selectedFiles.length ? 'Compile selected source views' : 'Select a front view first') :
    'Preview ignition effect only · compiler offline';
  if (compileAvailability) {
    compileAvailability.textContent = ready ? 'COMPILER READY · VERIFIED GATEWAY' : 'UI PREVIEW · COMPILER OFFLINE';
    compileAvailability.classList.toggle('ready', ready);
  }
}

function setFiles(files) {
  selectedFiles = [...files].filter((file) => file.type.startsWith('image/')).slice(0, 2);
  if (activeSourceObjectURL) URL.revokeObjectURL(activeSourceObjectURL);
  activeSourceObjectURL = null;
  revealGeneration += 1;
  result.classList.remove('show');
  if (!selectedFiles.length) {
    dropzone.classList.remove('has-image');
    preview.removeAttribute('src');
    resultPreview.removeAttribute('src');
    viewCount.textContent = '0 VIEWS';
    syncGatewayUI();
    showLiveCompileStatus('AWAITING SOURCE', 'Select a front character view to begin.');
    return;
  }
  activeSourceObjectURL = URL.createObjectURL(selectedFiles[0]);
  preview.src = activeSourceObjectURL;
  resultPreview.src = activeSourceObjectURL;
  resultPreview.style.display = '';
  resultCanvas.style.display = 'none';
  dropzone.classList.add('has-image');
  viewCount.textContent = `${selectedFiles.length} VIEW${selectedFiles.length > 1 ? 'S' : ''}`;
  syncGatewayUI();
  showLiveCompileStatus('SOURCE ATTACHED', project?.compile_available && project?.public_compile_gateway
    ? 'Source views selected. Use AWAKEN to submit them to the verified compiler gateway.'
    : 'Source views selected. This static preview has no production compiler connection.');
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

function affine(dst, src) {
  const [p0, p1, p2] = dst;
  const [t0, t1, t2] = src;
  const d = t0[0] * (t1[1] - t2[1]) + t1[0] * (t2[1] - t0[1]) + t2[0] * (t0[1] - t1[1]);
  if (Math.abs(d) < 1e-9) return null;
  const f = (q, k) => (q[0][k] * (t1[1] - t2[1]) + q[1][k] * (t2[1] - t0[1]) + q[2][k] * (t0[1] - t1[1])) / d;
  const g = (q, k) => (q[0][k] * (t2[0] - t1[0]) + q[1][k] * (t0[0] - t2[0]) + q[2][k] * (t1[0] - t0[0])) / d;
  const h = (q, k) => (q[0][k] * (t1[0] * t2[1] - t2[0] * t1[1]) + q[1][k] * (t2[0] * t0[1] - t0[0] * t2[1]) + q[2][k] * (t0[0] * t1[1] - t1[0] * t0[1])) / d;
  return [f(dst, 0), f(dst, 1), g(dst, 0), g(dst, 1), h(dst, 0), h(dst, 1)];
}

function drawTexturedTriangle(context, image, dst, src) {
  const matrix = affine(dst, src);
  if (!matrix) return;
  context.save();
  context.beginPath();
  context.moveTo(...dst[0]);
  context.lineTo(...dst[1]);
  context.lineTo(...dst[2]);
  context.closePath();
  context.clip();
  context.transform(...matrix);
  context.drawImage(image, 0, 0);
  context.restore();
}

async function loadBundleImage(bundle, scene) {
  const source = scene?.index?.source_views?.find((row) => Number(row.view_index) === 0);
  const path = source?.assets?.clean_cutout?.path || source?.assets?.raw_image?.path;
  if (!path) return null;
  const response = await apiFetch(`/api/blob?bundle=${encodeURIComponent(bundle)}&path=${encodeURIComponent(path)}`, {cache: 'no-store'});
  if (!response.ok) return null;
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const image = new Image();
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = reject;
    image.src = url;
  });
  image.dataset.objectUrl = url;
  return image;
}

function renderCompiledFrame(scene, runtimeFrame, sourceImage) {
  const meshes = (scene?.puppet?.meshes || []).filter((mesh) => mesh.view_id === 'V0');
  const frameMeshes = runtimeFrame?.frame?.mesh_vertices_by_id || {};
  const order = runtimeFrame?.frame?.render_order_by_view?.V0 || meshes.map((mesh) => mesh.mesh_id);
  const rank = new Map(order.map((id, index) => [id, index]));
  meshes.sort((a, b) => (rank.get(a.mesh_id) ?? 9999) - (rank.get(b.mesh_id) ?? 9999));

  const points = [];
  for (const mesh of meshes) {
    const animated = frameMeshes[mesh.mesh_id] || mesh.vertices;
    for (const point of animated) points.push(point);
  }
  if (!points.length) return;
  const xs = points.map((point) => Number(point[0]));
  const ys = points.map((point) => Number(point[1]));
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const pad = 18;
  const scale = Math.min(
    (resultCanvas.width - pad * 2) / Math.max(1, x1 - x0),
    (resultCanvas.height - pad * 2) / Math.max(1, y1 - y0),
  );
  const ox = resultCanvas.width / 2 - ((x0 + x1) / 2) * scale;
  const oy = resultCanvas.height / 2 - ((y0 + y1) / 2) * scale;
  const fit = (point) => [Number(point[0]) * scale + ox, Number(point[1]) * scale + oy];

  resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
  for (const mesh of meshes) {
    const animated = frameMeshes[mesh.mesh_id] || mesh.vertices;
    for (const triangle of mesh.triangles || []) {
      const dst = triangle.map((index) => fit(animated[index]));
      const src = triangle.map((index) => mesh.vertices[index]);
      if (sourceImage) {
        drawTexturedTriangle(resultCtx, sourceImage, dst, src);
      } else {
        resultCtx.beginPath();
        resultCtx.moveTo(...dst[0]);
        resultCtx.lineTo(...dst[1]);
        resultCtx.lineTo(...dst[2]);
        resultCtx.closePath();
        resultCtx.fillStyle = 'rgba(174,215,235,.16)';
        resultCtx.fill();
        resultCtx.strokeStyle = 'rgba(207,235,248,.28)';
        resultCtx.stroke();
      }
    }
  }
}

async function materializeWorkspace(compileID) {
  addLog('proof passed · preparing editable workspace', true);
  return apiJson(`/api/compile/workspace?id=${encodeURIComponent(compileID)}`);
}

async function startIdleReveal(bundle) {
  const generation = ++revealGeneration;
  const scene = await apiJson(`/api/living/scene?bundle=${encodeURIComponent(bundle)}`);
  const runtime = await apiJson(`/api/runtime/clips?bundle=${encodeURIComponent(bundle)}`);
  const idle = (runtime.clips || []).find((clip) => String(clip.intent || '').toLowerCase() === 'idle')
    || (runtime.clips || []).find((clip) => String(clip.clip_id || '').toLowerCase().includes('idle'));
  if (!idle) throw new Error('proof-qualified workspace has no runtime motion bake');
  const sourceImage = await loadBundleImage(bundle, scene);
  addLog(`runtime reveal · ${idle.clip_id} · qualification-owned bake`, true);
  let firstFrame=true;
  const started = performance.now();
  while (generation === revealGeneration) {
    const duration = Math.max(0.001, Number(idle.duration_seconds) || 1);
    const t = ((performance.now() - started) / 1000) % duration;
    const frame = await apiJson(`/api/runtime/frame?bundle=${encodeURIComponent(bundle)}&clip=${encodeURIComponent(idle.clip_id)}&time=${encodeURIComponent(t)}`);
    if (generation !== revealGeneration) break;
    if(firstFrame){
      const meshes=frame?.frame?.mesh_vertices_by_id;
      if(!meshes||typeof meshes!=='object'||!Object.keys(meshes).length){
        throw new Error('idle runtime frame has no deformable mesh evidence');
      }
    }
    renderCompiledFrame(scene, frame, sourceImage);
    if(firstFrame){
      firstFrame=false;
      resultPreview.style.display='none';
      resultCanvas.style.display='block';
      result.classList.add('alive-reveal');
      const heading=result.querySelector('.result-stream h2');
      if(heading)heading.textContent='ALIVE';
      status.querySelector('span').textContent='ALIVE';
      showLiveCompileStatus('PUPPET ALIVE','Verified editable puppet and actual Idle runtime frame are ready.');
      addLog('idle verified · the puppet is alive',true);
    }
    await delay(70);
  }
  if (sourceImage?.dataset?.objectUrl) URL.revokeObjectURL(sourceImage.dataset.objectUrl);
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
  return apiJson('/api/compile/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({views}),
  });
}

async function runProductionFlow() {
  resetPipeline();
  lastCompletedStage=0;
  updateCompileProgress({percent:0,completed:0,stage:'SUBMITTING',running:true});
  shell.classList.add('compiling');
  status.classList.add('ready');
  status.querySelector('span').textContent = 'SUBMITTING';
  showLiveCompileStatus('SUBMITTING', 'Sending selected source views to the production compiler.');
  awaken.disabled = true;
  addLog('submitting public compile session', true);
  const session = await startProductionCompile();
  const compileID = String(session.compile_id || session.session_id || '').trim();
  if (!compileID) throw new Error('compile gateway returned no public compile id');
  let terminal = false;
  let lastStage = -1;
  let finalizingLogged = false;
  while (!terminal) {
    const payload = await apiJson(`/api/compile/status?id=${encodeURIComponent(compileID)}`);
    const stageName = String(payload.stage || payload.capability || '').toUpperCase();
    const serverDetail = String(payload.detail || payload.message || '').trim();
    showLiveCompileStatus(
      stageName || String(payload.status || 'RUNNING').toUpperCase(),
      serverDetail || (stageName ? `Compiler reports stage: ${stageName.toLowerCase()}.` : 'Waiting for further compiler status.')
    );
    const index = productStages.findIndex(([name]) => stageName.includes(name));
    const isTerminal = Boolean(payload.terminal)||['PASS','FAILED','ERROR'].includes(String(payload.status||'').toUpperCase());
    onCompilerEvidence(payload,stageName,index,isTerminal);
    if (index >= 0 && index !== lastStage) {
      pipeline.forEach((step, i) => {
        step.classList.toggle('active', i === index);
        step.classList.toggle('done', i < index);
      });
      addLog(`${String(index + 1).padStart(2, '0')} ${stageName.toLowerCase()} · ${payload.detail || 'qualified execution'}`, true);
      lastStage = index;
    } else if (stageName === 'FINALIZE' && !finalizingLogged) {
      pipeline.forEach((step) => {
        step.classList.remove('active');
        step.classList.add('done');
      });
      addLog(`finalize · ${payload.detail || 'sealing editable product workspace'}`, true);
      finalizingLogged = true;
    }
    terminal = Boolean(payload.terminal) || ['PASS', 'FAILED', 'ERROR'].includes(String(payload.status || '').toUpperCase());
    if (!terminal) await delay(700);
    if (terminal && String(payload.status || '').toUpperCase() !== 'PASS') {
      throw new Error(payload.message || payload.detail || 'compile did not qualify');
    }
  }

  pipeline.forEach((step) => {
    step.classList.remove('active');
    step.classList.add('done');
  });
  const workspace = await materializeWorkspace(compileID);
  openWorkspace.href = `./studio.html?bundle=${encodeURIComponent(workspace.bundle)}`;
  openWorkspace.textContent = 'OPEN STUDIO';
  addLog('ProductRevision sealed · proof authority PASS', true);
  updateCompileProgress({percent:100,completed:9,stage:'PROOF PASSED · WORKSPACE READY',running:false});
  showLiveCompileStatus('PRODUCT PROOF · PASS', 'Qualified puppet sealed. Verifying actual Idle runtime.');
  const heading=result.querySelector('.result-stream h2');
  if(heading)heading.textContent='QUALIFIED';
  result.classList.remove('alive-reveal');
  result.classList.add('show');
  status.querySelector('span').textContent='PUPPET QUALIFIED';
  void startIdleReveal(workspace.bundle).catch((error) => {
    resultPreview.style.display = '';
    resultCanvas.style.display = 'none';
    status.querySelector('span').textContent='IDLE UNAVAILABLE';
    showLiveCompileStatus('IDLE UNAVAILABLE','The qualified puppet exists, but the Idle preview could not be verified.');
    addLog(`idle reveal unavailable · ${error.message}`, true);
  });
}

// The click VFX is its own one-shot render surface, not compiler progress.
async function playAwakenIgnition() {
  return ignitionReactor.play();
}

awaken.addEventListener('click', async () => {
  const ready = Boolean(project?.compile_available && project?.public_compile_gateway);
  if (running || (ready && !selectedFiles.length)) return;
  running = true;
  try {
    // The charge belongs to a click, not to page load, hover or scrolling.
    await playAwakenIgnition();
    if (ready) {
      result.classList.remove('show','alive-reveal');
      await runProductionFlow();
    } else {
      // This is a visual-only preview, not a fake compile or a simulated result.
      modeLabel.textContent = 'UI PREVIEW · COMPILER OFFLINE';
      showLiveCompileStatus('COMPILER OFFLINE',
        'Ignition interaction preview only. No compile session was started.');
    }
  } catch (error) {
    shell.classList.remove('compiling');
    status.classList.remove('ready');
    status.querySelector('span').textContent = 'INTAKE READY';
    showLiveCompileStatus('COMPILE FAILED', error.message);
    updateCompileProgress({percent:null,completed:null,stage:'COMPILER FAILED',running:false});
    addLog(`compile rejected · ${error.message}`, true);
  } finally {
    running = false;
    if (!ready || !shell.classList.contains('compiling')) syncGatewayUI();
  }
});

async function loadProject() {
  try {
    const response = await apiFetch('/api/project', {cache: 'no-store'});
    project = await response.json();
    const production = response.ok && project?.compile_available && project?.public_compile_gateway;
    modeLabel.textContent = production ? 'PRODUCTION GATEWAY' : 'UI PREVIEW · COMPILER OFFLINE';
    if (production) {
      status.classList.add('ready');
      status.querySelector('span').textContent = 'PRODUCTION READY';
    }
    syncGatewayUI();
    if (!production) showLiveCompileStatus('COMPILER OFFLINE', 'This preview shows the experience; real compilation requires the production gateway.');
  } catch {
    project = null;
    syncGatewayUI();
    modeLabel.textContent = 'UI PREVIEW · COMPILER OFFLINE';
    showLiveCompileStatus('COMPILER OFFLINE', 'This preview does not generate or claim a compiled puppet.');
  }
}

loadProject();
addLog('living compile surface online');
addLog('awaiting source observation');
