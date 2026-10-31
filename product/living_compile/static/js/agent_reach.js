(() => {
  const canvas = document.querySelector('#agentReach');
  const shell = document.querySelector('.awaken-shell');
  if (!canvas || !shell) return;

  const ctx = canvas.getContext('2d');
  const glyphs = ['01', '11', 'λ', 'Σ', 'Δ', 'φ', 'bind', 'wake', 'reach', 'resolve', 'alive'];
  let streams = [];
  let motes = [];
  let last = performance.now();

  function resetStream(stream, seed = Math.random()) {
    stream.angle = Math.random() * Math.PI * 2;
    stream.z = seed * 0.92 + 0.02;
    stream.speed = 0.000065 + Math.random() * 0.00011;
    stream.drift = (Math.random() - 0.5) * 0.00012;
    stream.phase = Math.random() * Math.PI * 2;
    stream.glyph = glyphs[(Math.random() * glyphs.length) | 0];
    stream.tail = 0.22 + Math.random() * 0.3;
    stream.hueBias = Math.random();
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(innerWidth * dpr);
    canvas.height = Math.floor(innerHeight * dpr);
    canvas.style.width = `${innerWidth}px`;
    canvas.style.height = `${innerHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const count = Math.max(34, Math.floor(innerWidth / 26));
    streams = Array.from({length: count}, (_, i) => {
      const stream = {};
      resetStream(stream, i / count);
      return stream;
    });

    motes = Array.from({length: Math.max(36, Math.floor(innerWidth / 34))}, () => ({
      angle: Math.random() * Math.PI * 2,
      radius: 12 + Math.random() * Math.min(innerWidth, innerHeight) * 0.42,
      speed: 0.00008 + Math.random() * 0.00016,
      size: 0.45 + Math.random() * 1.1,
      phase: Math.random() * Math.PI * 2,
    }));
  }

  function rgba(stream, alpha) {
    if (stream.hueBias > 0.72) return `rgba(165,255,218,${alpha})`;
    if (stream.hueBias > 0.28) return `rgba(109,236,170,${alpha})`;
    return `rgba(117,226,205,${alpha})`;
  }

  function pointFor(stream, z, now) {
    const w = innerWidth;
    const h = innerHeight;
    const vanishingX = w * 0.5;
    const vanishingY = h * 0.455;
    const active = shell.classList.contains('compiling');
    const spreadX = w * (active ? 0.66 : 0.54);
    const spreadY = h * (active ? 0.59 : 0.47);
    const warp = Math.pow(Math.max(0.001, z), active ? 1.72 : 1.92);
    const wobble = Math.sin(now * 0.0014 + stream.phase + z * 4.0) * (3 + z * 12);
    const angle = stream.angle + stream.drift * now;
    return [
      vanishingX + Math.cos(angle) * spreadX * warp + Math.cos(angle + Math.PI / 2) * wobble,
      vanishingY + Math.sin(angle) * spreadY * warp * 0.84 + Math.sin(angle + Math.PI / 2) * wobble * 0.42,
    ];
  }

  function draw(now) {
    const dt = Math.min(48, now - last);
    const active = shell.classList.contains('compiling');
    const resultVisible = document.querySelector('#result')?.classList.contains('show');
    const intensity = resultVisible ? 0.48 : active ? 1 : 0.31;

    ctx.clearRect(0, 0, innerWidth, innerHeight);
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (const stream of streams) {
      const speedBoost = active ? 2.15 : 0.5;
      stream.z += stream.speed * dt * speedBoost;
      if (stream.z > 1.04) resetStream(stream, 0.015 + Math.random() * 0.04);

      const z0 = Math.max(0.001, stream.z - stream.tail);
      const p0 = pointFor(stream, z0, now);
      const p1 = pointFor(stream, stream.z, now);
      const alpha = Math.min(0.34, (0.025 + stream.z * stream.z * 0.28) * intensity);

      const gradient = ctx.createLinearGradient(p0[0], p0[1], p1[0], p1[1]);
      gradient.addColorStop(0, 'rgba(76,160,118,0)');
      gradient.addColorStop(0.62, rgba(stream, alpha * 0.28));
      gradient.addColorStop(1, rgba(stream, alpha));
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 0.35 + stream.z * 1.35;
      ctx.beginPath();
      ctx.moveTo(p0[0], p0[1]);
      ctx.lineTo(p1[0], p1[1]);
      ctx.stroke();

      if (stream.z > 0.28) {
        const size = 6 + Math.pow(stream.z, 1.55) * (active ? 15 : 9);
        ctx.font = `${size}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = rgba(stream, alpha * 0.92);
        ctx.shadowColor = rgba(stream, Math.min(0.48, alpha * 1.5));
        ctx.shadowBlur = active ? 9 : 3;
        ctx.fillText(stream.glyph, p1[0], p1[1]);
      }
    }

    const cx = innerWidth * 0.5;
    const cy = innerHeight * 0.455;
    for (const mote of motes) {
      mote.phase += mote.speed * dt * (active ? 2.2 : 0.65);
      const breathing = 0.74 + Math.sin(now * 0.001 + mote.phase) * 0.12;
      const r = mote.radius * breathing;
      const x = cx + Math.cos(mote.angle + mote.phase * 0.16) * r;
      const y = cy + Math.sin(mote.angle + mote.phase * 0.16) * r * 0.62;
      const a = (active ? 0.11 : 0.027) * intensity;
      ctx.fillStyle = `rgba(121,239,177,${a})`;
      ctx.fillRect(x, y, mote.size, mote.size);
    }

    if (active) {
      const pulse = 0.5 + Math.sin(now * 0.0045) * 0.5;
      const halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.min(innerWidth, innerHeight) * 0.24);
      halo.addColorStop(0, `rgba(113,238,171,${0.035 + pulse * 0.02})`);
      halo.addColorStop(0.38, `rgba(91,210,159,${0.015 + pulse * 0.008})`);
      halo.addColorStop(1, 'rgba(70,180,137,0)');
      ctx.fillStyle = halo;
      ctx.fillRect(0, 0, innerWidth, innerHeight);
    }

    ctx.restore();
    last = now;
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  resize();
  requestAnimationFrame(draw);
})();
