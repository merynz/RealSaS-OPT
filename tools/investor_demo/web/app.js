(() => {
  "use strict";

  const data = window.REALSAS_DEMO_DATA;
  const $ = (id) => document.getElementById(id);
  const missing = $("missingData");
  if (!data) {
    missing.classList.remove("hidden");
    return;
  }

  $("title").textContent = data.title || "RealSaS — KNIGHT";
  $("subtitle").textContent = data.subtitle || "Compiled Character Structure";
  $("vertexCount").textContent = data.mesh.rest_xyz.length.toLocaleString();
  $("faceCount").textContent = data.mesh.faces.length.toLocaleString();
  $("jointCount").textContent = data.skeleton.joints.length.toLocaleString();
  $("sourceSha").textContent = `main ${String(data.source_sha || "unknown").slice(0, 12)}`;

  const stage = $("stage");
  const artwork = $("artwork");
  const fieldCanvas = $("fieldCanvas");
  const overlayCanvas = $("overlayCanvas");
  const overlay = overlayCanvas.getContext("2d");
  const gl = fieldCanvas.getContext("webgl2", {
    alpha: true,
    antialias: true,
    premultipliedAlpha: true,
  });

  const controls = {
    view: $("viewSelect"),
    joint: $("jointSelect"),
    artwork: $("artworkToggle"),
    rig: $("rigToggle"),
    mesh: $("meshToggle"),
    weights: $("weightToggle"),
    artworkOpacity: $("artworkOpacity"),
    rigOpacity: $("rigOpacity"),
    meshOpacity: $("meshOpacity"),
    weightOpacity: $("weightOpacity"),
    play: $("playButton"),
    timeline: $("timeline"),
  };

  const state = {
    view: 0,
    joint: 0,
    frame: 0,
    playing: false,
    clipIndex: -1,
    lastTick: 0,
    positions: data.mesh.rest_xyz,
    joints: data.skeleton.joints.map((joint) => joint.rest),
    selectedWeights: null,
    projected: null,
  };

  const weightCache = new Map();
  const imageCache = new Map();

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function projectPoint(point, camera, width, height) {
    const q = [
      point[0] - camera.origin[0],
      point[1] - camera.origin[1],
      point[2] - camera.origin[2],
    ];
    const gx = dot(q, camera.right) / camera.half_extent;
    const gy = -dot(q, camera.screen_up) / camera.half_extent;
    const depth = dot(q, camera.forward);
    return {
      gx,
      gy,
      clipX: gx,
      clipY: -gy,
      x: (gx + 1) * 0.5 * width,
      y: (gy + 1) * 0.5 * height,
      depth,
    };
  }

  function currentClip() {
    const clips = (data.motion && data.motion.clips) || [];
    return state.clipIndex >= 0 && state.clipIndex < clips.length ? clips[state.clipIndex] : null;
  }

  function currentFrame() {
    const clip = currentClip();
    if (!clip || !clip.frames || !clip.frames.length) return null;
    return clip.frames[Math.min(state.frame, clip.frames.length - 1)];
  }

  function framePositions(frame) {
    return frame && frame.posed_xyz ? frame.posed_xyz : data.mesh.rest_xyz;
  }

  function frameJointPositions(frame) {
    if (!frame || !frame.joint_world_positions) {
      return data.skeleton.joints.map((joint) => joint.rest);
    }
    if (Array.isArray(frame.joint_world_positions)) {
      return frame.joint_world_positions;
    }
    return data.skeleton.joints.map(
      (joint) => frame.joint_world_positions[joint.id] || joint.rest
    );
  }

  function frameArtwork(frame, view) {
    if (frame && frame.artwork_by_view) {
      return frame.artwork_by_view[String(view)] || frame.artwork_by_view[view] || null;
    }
    return data.appearance.rest_frames[String(view)] || null;
  }

  function selectedWeights() {
    if (weightCache.has(state.joint)) return weightCache.get(state.joint);
    const values = new Float32Array(data.mesh.rest_xyz.length);
    const rows = data.weights.influences;
    for (let vertex = 0; vertex < rows.length; vertex += 1) {
      for (const pair of rows[vertex]) {
        if (pair[0] === state.joint) {
          values[vertex] = Number(pair[1]);
          break;
        }
      }
    }
    weightCache.set(state.joint, values);
    return values;
  }

  function prepareControls() {
    for (const camera of data.cameras) {
      const option = document.createElement("option");
      option.value = String(camera.view_index);
      option.textContent = `${camera.view_id || `V${camera.view_index}`} · ${camera.view_index * 45}°`;
      controls.view.appendChild(option);
    }
    data.skeleton.joints.forEach((joint, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = joint.id;
      controls.joint.appendChild(option);
    });

    const clips = (data.motion && data.motion.clips) || [];
    if (clips.length) {
      state.clipIndex = 0;
      const clip = clips[0];
      controls.timeline.max = String(Math.max(0, clip.frames.length - 1));
      controls.play.disabled = clip.frames.length < 2;
      $("clipName").textContent = clip.display_name || clip.id;
      $("motionStatus").textContent = "Canonical motion attached";
      $("motionStatusBox").querySelector("span").textContent =
        "The same M/G/W/CAA is being inspected across the selected sealed clip.";
    } else {
      controls.timeline.disabled = true;
      controls.play.disabled = true;
      $("motionStatus").textContent = "Motion pending selection";
    }
  }

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) || "WebGL shader compile failed");
    }
    return shader;
  }

  function createProgram() {
    if (!gl) return null;
    const vertexSource = `#version 300 es
      precision highp float;
      layout(location=0) in vec2 aPosition;
      layout(location=1) in float aWeight;
      layout(location=2) in float aDepth;
      out float vWeight;
      out float vDepth;
      void main(){
        gl_Position=vec4(aPosition,0.0,1.0);
        vWeight=aWeight;
        vDepth=aDepth;
      }`;
    const fragmentSource = `#version 300 es
      precision highp float;
      in float vWeight;
      in float vDepth;
      uniform int uMode;
      uniform float uOpacity;
      out vec4 outColor;
      vec3 heat(float t){
        vec3 c0=vec3(0.035,0.095,0.145);
        vec3 c1=vec3(0.055,0.79,0.88);
        vec3 c2=vec3(0.56,0.34,0.98);
        vec3 c3=vec3(1.0,0.74,0.31);
        if(t<0.36) return mix(c0,c1,t/0.36);
        if(t<0.72) return mix(c1,c2,(t-0.36)/0.36);
        return mix(c2,c3,(t-0.72)/0.28);
      }
      void main(){
        if(uMode==1){
          float w=clamp(vWeight,0.0,1.0);
          float a=uOpacity*(0.10+0.90*smoothstep(0.01,1.0,w));
          outColor=vec4(heat(w),a);
        }else{
          float d=clamp(vDepth,0.0,1.0);
          vec3 c=mix(vec3(0.08,0.28,0.38),vec3(0.22,0.58,0.72),1.0-d);
          outColor=vec4(c,uOpacity*0.17);
        }
      }`;
    const program = gl.createProgram();
    gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || "WebGL link failed");
    }
    return {
      program,
      buffer: gl.createBuffer(),
      mode: gl.getUniformLocation(program, "uMode"),
      opacity: gl.getUniformLocation(program, "uOpacity"),
    };
  }

  let fieldProgram = null;
  if (gl) {
    try {
      fieldProgram = createProgram();
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    } catch (error) {
      console.warn(error);
      fieldCanvas.style.display = "none";
    }
  } else {
    fieldCanvas.style.display = "none";
  }

  function resize() {
    const rect = stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    for (const canvas of [fieldCanvas, overlayCanvas]) {
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }
    render();
  }

  function updateProjected() {
    const camera = data.cameras[state.view];
    const width = overlayCanvas.width;
    const height = overlayCanvas.height;
    state.projected = state.positions.map((point) => projectPoint(point, camera, width, height));
    return camera;
  }

  function drawField() {
    if (!gl || !fieldProgram) return;
    gl.viewport(0, 0, fieldCanvas.width, fieldCanvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const showWeights = controls.weights.checked;
    const showMesh = controls.mesh.checked;
    if (!showWeights && !showMesh) return;

    const weights = selectedWeights();
    const projected = state.projected;
    let minDepth = Infinity;
    let maxDepth = -Infinity;
    for (const point of projected) {
      minDepth = Math.min(minDepth, point.depth);
      maxDepth = Math.max(maxDepth, point.depth);
    }
    const depthSpan = Math.max(1.0e-9, maxDepth - minDepth);
    const packed = new Float32Array(data.mesh.faces.length * 3 * 4);
    let cursor = 0;
    for (const face of data.mesh.faces) {
      for (const vertex of face) {
        const point = projected[vertex];
        packed[cursor++] = point.clipX;
        packed[cursor++] = point.clipY;
        packed[cursor++] = weights[vertex];
        packed[cursor++] = (point.depth - minDepth) / depthSpan;
      }
    }

    gl.useProgram(fieldProgram.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, fieldProgram.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, packed, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 16, 8);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 16, 12);
    gl.uniform1i(fieldProgram.mode, showWeights ? 1 : 0);
    gl.uniform1f(
      fieldProgram.opacity,
      Number(showWeights ? controls.weightOpacity.value : controls.meshOpacity.value)
    );
    gl.drawArrays(gl.TRIANGLES, 0, data.mesh.faces.length * 3);
  }

  function drawTopology() {
    if (!controls.mesh.checked) return;
    const opacity = Number(controls.meshOpacity.value);
    overlay.save();
    overlay.globalAlpha = Math.min(0.95, 0.22 + opacity * 0.76);
    overlay.strokeStyle = "rgba(112,226,255,.72)";
    overlay.lineWidth = Math.max(0.75, window.devicePixelRatio || 1);
    overlay.shadowColor = "rgba(56,207,255,.28)";
    overlay.shadowBlur = 3 * (window.devicePixelRatio || 1);
    overlay.beginPath();
    for (const edge of data.mesh.edges) {
      const a = state.projected[edge[0]];
      const b = state.projected[edge[1]];
      overlay.moveTo(a.x, a.y);
      overlay.lineTo(b.x, b.y);
    }
    overlay.stroke();
    overlay.restore();
  }

  function boneShape(a, b, selected, alpha) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy);
    if (length < 1.5) return;
    const ux = dx / length;
    const uy = dy / length;
    const nx = -uy;
    const ny = ux;
    const scale = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const baseWidth = Math.min(11 * scale, Math.max(4 * scale, length * 0.10));
    const midWidth = Math.min(15 * scale, Math.max(6 * scale, length * 0.15));
    const neck = Math.min(length * 0.72, length - 5 * scale);
    const ax = a.x + ux * 2 * scale;
    const ay = a.y + uy * 2 * scale;
    const mx = a.x + ux * neck;
    const my = a.y + uy * neck;
    const tx = b.x - ux * 2 * scale;
    const ty = b.y - uy * 2 * scale;

    const gradient = overlay.createLinearGradient(a.x, a.y, b.x, b.y);
    if (selected) {
      gradient.addColorStop(0, `rgba(255,191,93,${0.93 * alpha})`);
      gradient.addColorStop(1, `rgba(255,103,168,${0.96 * alpha})`);
    } else {
      gradient.addColorStop(0, `rgba(120,235,255,${0.82 * alpha})`);
      gradient.addColorStop(1, `rgba(118,150,255,${0.72 * alpha})`);
    }
    overlay.fillStyle = gradient;
    overlay.strokeStyle = selected
      ? `rgba(255,238,197,${alpha})`
      : `rgba(213,250,255,${0.88 * alpha})`;
    overlay.lineWidth = 1.1 * scale;
    overlay.shadowColor = selected ? "rgba(255,116,175,.72)" : "rgba(87,221,255,.60)";
    overlay.shadowBlur = selected ? 13 * scale : 8 * scale;
    overlay.beginPath();
    overlay.moveTo(ax + nx * baseWidth, ay + ny * baseWidth);
    overlay.lineTo(mx + nx * midWidth, my + ny * midWidth);
    overlay.lineTo(tx, ty);
    overlay.lineTo(mx - nx * midWidth, my - ny * midWidth);
    overlay.lineTo(ax - nx * baseWidth, ay - ny * baseWidth);
    overlay.closePath();
    overlay.fill();
    overlay.stroke();

    overlay.shadowBlur = 0;
    overlay.strokeStyle = selected
      ? `rgba(72,22,54,${0.62 * alpha})`
      : `rgba(8,46,64,${0.58 * alpha})`;
    overlay.lineWidth = 1.4 * scale;
    overlay.beginPath();
    overlay.moveTo(a.x + ux * 5 * scale, a.y + uy * 5 * scale);
    overlay.lineTo(b.x - ux * 7 * scale, b.y - uy * 7 * scale);
    overlay.stroke();
  }

  function jointDot(point, selected, alpha) {
    const scale = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    overlay.shadowColor = selected ? "rgba(255,174,91,.9)" : "rgba(91,226,255,.75)";
    overlay.shadowBlur = (selected ? 13 : 7) * scale;
    overlay.fillStyle = selected
      ? `rgba(255,223,150,${alpha})`
      : `rgba(221,251,255,${0.95 * alpha})`;
    overlay.strokeStyle = selected
      ? `rgba(255,116,168,${alpha})`
      : `rgba(80,192,224,${alpha})`;
    overlay.lineWidth = 2 * scale;
    overlay.beginPath();
    overlay.arc(point.x, point.y, (selected ? 5.2 : 3.8) * scale, 0, Math.PI * 2);
    overlay.fill();
    overlay.stroke();
  }

  function drawRig(camera) {
    if (!controls.rig.checked) return;
    const alpha = Number(controls.rigOpacity.value);
    const width = overlayCanvas.width;
    const height = overlayCanvas.height;
    const projectedJoints = state.joints.map((point) => projectPoint(point, camera, width, height));
    overlay.save();
    overlay.globalAlpha = 1;
    data.skeleton.joints.forEach((joint, index) => {
      if (joint.parent_index === null || joint.parent_index === undefined) return;
      boneShape(
        projectedJoints[joint.parent_index],
        projectedJoints[index],
        index === state.joint,
        alpha
      );
    });
    projectedJoints.forEach((point, index) => jointDot(point, index === state.joint, alpha));
    overlay.restore();
  }

  function drawFallbackWeights() {
    if (gl || !controls.weights.checked) return;
    const values = selectedWeights();
    overlay.save();
    overlay.globalAlpha = Number(controls.weightOpacity.value);
    for (let i = 0; i < state.projected.length; i += 1) {
      const w = values[i];
      if (w < 0.04) continue;
      const point = state.projected[i];
      const radius = 1.5 + 4 * w;
      overlay.fillStyle = w > 0.7 ? "#ffbd58" : w > 0.35 ? "#8b6cff" : "#17bdd6";
      overlay.beginPath();
      overlay.arc(point.x, point.y, radius, 0, Math.PI * 2);
      overlay.fill();
    }
    overlay.restore();
  }

  function updateArtwork() {
    const path = frameArtwork(currentFrame(), state.view);
    artwork.style.opacity = controls.artwork.checked ? controls.artworkOpacity.value : "0";
    if (!path) {
      artwork.removeAttribute("src");
      return;
    }
    if (artwork.dataset.path === path) return;
    artwork.dataset.path = path;
    if (!imageCache.has(path)) {
      const image = new Image();
      image.src = path;
      imageCache.set(path, image);
    }
    artwork.src = path;
  }

  function updateBoneMetrics() {
    const joint = data.skeleton.joints[state.joint];
    const parent = joint.parent_index === null || joint.parent_index === undefined
      ? "ROOT"
      : data.skeleton.joints[joint.parent_index].id;
    const values = selectedWeights();
    let count = 0;
    let peak = 0;
    for (const value of values) {
      if (value > 0) count += 1;
      peak = Math.max(peak, value);
    }
    $("boneName").textContent = joint.id;
    $("boneParent").textContent = parent;
    $("weightedVertexCount").textContent = count.toLocaleString();
    $("peakWeight").textContent = peak.toFixed(3);
    $("selectedBoneLabel").textContent = joint.id;
    $("selectedBoneLabel").classList.toggle(
      "visible",
      controls.rig.checked || controls.weights.checked
    );
  }

  function updateLabels() {
    const clip = currentClip();
    const frame = currentFrame();
    $("viewLabel").textContent = `V${state.view}`;
    if (!clip) {
      $("frameLabel").textContent = "REST";
      $("timeLabel").textContent = "0.000 s";
    } else {
      $("frameLabel").textContent = `${state.frame + 1} / ${clip.frames.length}`;
      const time = frame && frame.time_seconds !== undefined
        ? Number(frame.time_seconds)
        : state.frame / Number(clip.fps || 24);
      $("timeLabel").textContent = `${time.toFixed(3)} s`;
    }
    if (controls.weights.checked) $("modeLabel").textContent = "SKIN WEIGHT FIELD";
    else if (controls.mesh.checked && controls.rig.checked) $("modeLabel").textContent = "STRUCTURE X-RAY";
    else if (controls.mesh.checked) $("modeLabel").textContent = "CANONICAL TOPOLOGY";
    else if (controls.rig.checked) $("modeLabel").textContent = "QUALIFIED RIG";
    else $("modeLabel").textContent = "FINAL CHARACTER";
  }

  function render() {
    if (!overlayCanvas.width || !overlayCanvas.height) return;
    const frame = currentFrame();
    state.positions = framePositions(frame);
    state.joints = frameJointPositions(frame);
    const camera = updateProjected();
    overlay.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    drawField();
    drawFallbackWeights();
    drawTopology();
    drawRig(camera);
    updateArtwork();
    updateBoneMetrics();
    updateLabels();
  }

  function setPreset(name) {
    document.querySelectorAll(".preset").forEach((button) => {
      button.classList.toggle("active", button.dataset.preset === name);
    });
    if (name === "final") {
      controls.artwork.checked = true;
      controls.rig.checked = false;
      controls.mesh.checked = false;
      controls.weights.checked = false;
      controls.artworkOpacity.value = "1";
    } else if (name === "rig") {
      controls.artwork.checked = true;
      controls.rig.checked = true;
      controls.mesh.checked = false;
      controls.weights.checked = false;
      controls.artworkOpacity.value = "0.42";
    } else if (name === "mesh") {
      controls.artwork.checked = true;
      controls.rig.checked = false;
      controls.mesh.checked = true;
      controls.weights.checked = false;
      controls.artworkOpacity.value = "0.22";
    } else if (name === "weights") {
      controls.artwork.checked = true;
      controls.rig.checked = false;
      controls.mesh.checked = true;
      controls.weights.checked = true;
      controls.artworkOpacity.value = "0.16";
    } else if (name === "xray") {
      controls.artwork.checked = true;
      controls.rig.checked = true;
      controls.mesh.checked = true;
      controls.weights.checked = false;
      controls.artworkOpacity.value = "0.20";
    }
    render();
  }

  function bindEvents() {
    controls.view.addEventListener("change", () => {
      state.view = Number(controls.view.value);
      render();
    });
    controls.joint.addEventListener("change", () => {
      state.joint = Number(controls.joint.value);
      state.selectedWeights = null;
      render();
    });
    [
      controls.artwork,
      controls.rig,
      controls.mesh,
      controls.weights,
      controls.artworkOpacity,
      controls.rigOpacity,
      controls.meshOpacity,
      controls.weightOpacity,
    ].forEach((control) => control.addEventListener("input", render));

    document.querySelectorAll(".preset").forEach((button) => {
      button.addEventListener("click", () => setPreset(button.dataset.preset));
    });

    controls.timeline.addEventListener("input", () => {
      state.frame = Number(controls.timeline.value);
      render();
    });
    controls.play.addEventListener("click", () => {
      if (!currentClip()) return;
      state.playing = !state.playing;
      controls.play.textContent = state.playing ? "❚❚" : "▶";
      state.lastTick = performance.now();
    });
  }

  function tick(now) {
    const clip = currentClip();
    if (state.playing && clip && clip.frames.length > 1) {
      const fps = Number(clip.fps || 24);
      const interval = 1000 / fps;
      if (now - state.lastTick >= interval) {
        const steps = Math.max(1, Math.floor((now - state.lastTick) / interval));
        state.lastTick += steps * interval;
        let next = state.frame + steps;
        if (next >= clip.frames.length) {
          if (clip.loop !== false) next %= clip.frames.length;
          else {
            next = clip.frames.length - 1;
            state.playing = false;
            controls.play.textContent = "▶";
          }
        }
        state.frame = next;
        controls.timeline.value = String(state.frame);
        render();
      }
    }
    requestAnimationFrame(tick);
  }

  prepareControls();
  bindEvents();
  state.view = Number(controls.view.value || 0);
  state.joint = Number(controls.joint.value || 0);
  const observer = new ResizeObserver(resize);
  observer.observe(stage);
  resize();
  render();
  requestAnimationFrame(tick);
})();
