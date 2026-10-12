# Knight Investor Interactive Demo V1

## Purpose

This is a presentation-only, read-only investor demonstrator. It does not create a second runtime, geometry authority, skin authority, motion authority or appearance authority. Its job is to reveal the real internal structure of the canonical Knight while keeping the presentation polished enough for an investor package.

## Canonical baseline used

The branch starts from `main@82e31b566cdb811e5b1c41fd4f84a6eaeeee7ce3`.

The latest approved main regression path is:

```text
tests/fixtures/single_m_caa_knight_v1/baseline.json
        |
        +-- run_poses.npz        (recorded canonical M poses)
        +-- source_rgba.zip      (source-wins RGB)
        +-- compiled_caa.npz     (sealed CAA fallback)
        |
        v
tools/demo/render_knight_m_caa_v1.py
        |
        v
compiler/realsas_compiler_core/single_m_caa_v1.py
```

That main regression renders 21 recorded M poses across all eight directions, 168 fresh CPU frames total. It does not run inference, retargeting or skinning during replay. The investor viewer therefore does **not** infer G/W from those frames. Instead, the exporter binds the exact `QualifiedSkeletonIR` and `QualifiedMeshSkinIR` from the sealed Knight mechanics run and fail-closes unless its `QualifiedMeshIR` is byte/numerically the same rest M/topology used by the render baseline.

The fixed investor core is therefore:

```text
M   = exact canonical QualifiedMeshIR used by the approved single-M render
G   = exact QualifiedSkeletonIR bound to that Knight mechanics witness
W   = exact QualifiedMeshSkinIR on that same M
CAA = source-wins RGB + sealed compiled CAA on M face/barycentric addresses
```

Motion is deliberately outside this first commit. Exact clips will be selected by the user and attached later without changing M/G/W/CAA.

## Visual language

The viewer is not a debug screenshot. Presentation styling is intentionally richer while remaining data-faithful:

- **Rig:** tapered editor-style bone glyphs with joint sockets and selection glow. Endpoints are the real projected canonical joint positions for the current frame.
- **Topology:** the actual canonical M triangle edges, projected from the current posed vertex positions. Connectivity is never replaced for presentation.
- **Weights:** the exact canonical per-vertex selected-bone weights, linearly interpolated over each posed M triangle by WebGL. Color is presentation only; scalar values are not altered.
- **Artwork / CAA:** canonical source-wins + CAA output underneath the structural overlays.
- **Timeline:** when clips are attached, artwork, posed M, joints and weight field share one frame index/time source.

The demo provides investor-friendly presets (`Final`, `Rig`, `Mesh`, `Weights`, `X-Ray`) plus independent opacity controls.

## Offline package

Build the core from an exact sealed Knight run:

```bash
python tools/investor_demo/export_knight_investor_demo_v1.py \
  --run-root /path/to/exact/knight/run \
  --output /tmp/RealSaS_Knight_Investor_Demo
```

Then open:

```text
/tmp/RealSaS_Knight_Investor_Demo/START_DEMO.html
```

No server, CDN, npm install or network access is required. The exporter writes a self-contained `data/demo-data.js`, rest CAA views, authority manifest and sealed single-M material bundle.

## Motion plug-in contract

`data.motion.clips` is intentionally empty now. The viewer already accepts clip frames shaped as:

```text
clip
  id
  display_name
  fps
  loop
  frames[]
    time_seconds
    posed_xyz              # canonical M vertex order
    joint_world_positions  # canonical G joint order or id map
    artwork_by_view         # V0..V7 CAA frame paths
```

When the exact investor clips are selected, the clip baker must prove that every `posed_xyz` frame uses the same M vertex order/count and that every joint frame uses the same G identity. CAA frames must come from the canonical single-M source-wins + CAA renderer (or its exact qualified successor), not from a second presentation mesh.

## Non-claims

This package is a presentation consumer. It does not close grip, semantic draw order, unresolved depth ties, native pixel parity, product qualification or any currently open scientific gate. Those remain properties of the product/runtime line, not of the investor viewer.
