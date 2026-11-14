# RSS.vNext Stage 3: PyTorch3D compiled rasterization

**Production choice:** CUDA Stage 3 uses the upstream Meta PyTorch3D triangle
rasterizer via `models/rss_vnext/pytorch3d_raster_backend_v1.py`. The
first-party `BustedTorchRasterV1` is a bounded CPU reference only.
There is no silent reference fallback on CUDA: a missing or ABI-incompatible
PyTorch3D extension fails with `RSS_VNEXT_PYTORCH3D_COMPILED_RASTER_REQUIRED`
or `RSS_VNEXT_PYTORCH3D_CUDA_RASTER_EXECUTION_FAILED`.

This is an integration of an external renderer, **not** a new raster algorithm.

## Upstream, license and deployment

- Source: https://github.com/facebookresearch/pytorch3d
- License: https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE
  (**BSD-style**, copyright/notice/disclaimer retention required for binary
  distributions; no Meta endorsement without permission).
- API: `pytorch3d.renderer.mesh.rasterize_meshes.rasterize_meshes` and
  `pytorch3d.structures.Meshes`.
- Upstream main inspected on 2026-11-14 at
  `3eac7c62fed39a09768cd08119c5e1bf4daef559` (source reference, **not
  yet an installation-qualified GPU artifact**).
- Installation must match the deployed CUDA toolkit, Python and PyTorch ABI.
  PyTorch3D's public installation matrix is not proof of compatibility with
  RSS CI's newer PyTorch version. Build a pinned image on the actual A100
  worker and record its wheel SHA256, `torch.__version__`,
  `torch.version.cuda`, `pytorch3d` source SHA and compiler toolchain.
  Example *build candidate*, not a qualified install:
  `python -m pip install --no-build-isolation
  "git+https://github.com/facebookresearch/pytorch3d.git@3eac7c62fed39a09768cd08119c5e1bf4daef559"`.
  GPU image approval is blocked until the true CUDA smoke/gradient benchmark
  and source-render comparison pass.

## Exact Stage-3 adapter contract

1. Camera input uses GL NDC: right-positive X, up-positive Y, and source
   depth in [-1,+1]. PyTorch3D is **left-positive** X, up-positive Y,
   **positive** near-camera Z and uses a longer-axis NDC extension for
   rectangular images. The adapter flips X, rescales both XY axes and maps
   Z monotonically to [0.1,0.9] (preserving depth order).
2. `rasterize_meshes` is called with one front-most face per pixel,
   `blur_radius=0`, `perspective_correct=False`,
   `cull_backfaces=False`. Every fragment uses the same indexed
   `FlexiCubes` face list. There is no average triangle depth.
3. The adapter converts PyTorch3D's top-left raster to the existing
   bottom-left `(w0,w1,edge_alpha,face_id+1)` contract; Stage 3 flips
   the final output **once** into its source teacher image convention.
4. Surface points/normals are barycentrically interpolated from the exact
   vertex-addressed witness, then queried against the **same IRIS H(x)** for
   Graphics and Semantic role outputs. Metric depth is recomputed using
   the original source camera forward vector, never a normalized/z-buffer
   proxy. Blank pixels retain face ID zero and zero alpha.
5. The signed face-edge distance supplies a differentiable interior
   antialias gradient without changing first-hit face ownership. This is a
   one-sided coverage approximation; exact pixel/edge equivalence to the
   historical reference is **not yet established** on the real compiled
   CUDA path. Do not promote shader parity on CPU-only regression.
6. Backend identity is included in Stage-3 raster receipts. CPU reference,
   PyTorch3D compiled and injected research stubs are distinct IDs.

## Qualification before model training and publication

- Fast CPU CI: deterministic coordinate conversion, non-square aspect ratio,
  face-order first-hit, hard-coverage/background, interpolation, provenance,
  missing-extension fail-closed, autograd through mocked PyTorch3D interface.
  These tests **do not qualify the external binary**.
- Native CPU extension smoke is conditional; GPU smoke explicitly requires
  CUDA and fails if PyTorch3D cannot import/execute. Neither is faked by
  stand-in shapes.
- On the actual A100 build run:
  `PYTHONPATH=. python tools/research/rss_vnext_pytorch3d_cuda_qualification_v1.py
  --report /tmp/rss_vnext_pytorch3d_cuda_receipt.json`
  This probe must not run on a CPU stand-in. It records real CUDA first-hit,
  backprop, GPU device, memory and latency, but does not itself qualify the
  whole product. Also back-propagate through H(x) graphics/semantic heads,
  compare visible interior alpha/depth/normal and 32x32/128x128 first-hit
  pixels to CPU reference under the same camera, including tall/wide views.
- Benchmark full representative layered-mesh Stage 3 (faces, pixels, roles,
  six views) with `torch.cuda.synchronize()`, measured max memory,
  peak/steady median latency and gradient time. Register the binary/wheel hash
  in the training environment before large-corpus work.
- Do not interpret a green synthetic CPU test as proof of full-source 2D
  visibility, animation readiness, unseen accuracy or GPU throughput.
