# `.rss` / `.realsas` Runtime Contract v2

## Ownership boundary

The canonical compiler owns mesh topology, rig interpretation, weights, binding, deformation, secondary motion, contact, render order and motion qualification. The deployment runtime only loads and samples the selected baked product. Host adapters may convert coordinates explicitly, but must not reinterpret compiler truth.

## Container

A deployment package is a ZIP archive or unpacked directory containing:

- `manifest.json` — inspectable package contract and integrity metadata;
- `runtime/realsas_runtime.rsr` — little-endian binary product stream;
- `textures/*.png` — source-view RGBA textures.

`.rss` is the preferred deployment extension. `.realsas` is accepted as an interchange alias; authoring bundles remain richer than deploy packages.

## Binary v2

- Magic: `RSRT\0\2\0\0`
- Version: `uint32 = 2`
- Numeric storage: little-endian IEEE-754 `float32` and `uint32`
- Terminal integrity: CRC32 over every preceding binary byte
- Texture integrity: CRC32 of each compressed PNG payload stored beside its view record
- Coordinate system: UTF-8 identifier embedded in the stream
- Unit scale: `units_per_pixel` embedded in the stream
- Capabilities: explicit feature bitset embedded in the stream

The remainder contains deterministic view/mesh topology, baked clip frames, and per-frame draw-order indices. UVs are normalized `[0,1]`.

## Engine-neutral render contract

`manifest.json` declares the host rendering semantics explicitly: sRGB RGBA8 textures with straight alpha, source-over blending, linear minification/magnification, clamp-to-edge sampling, no depth test, and compiler-owned per-frame draw order. These are data-contract requirements, not engine-specific material assets. A host adapter may implement them with its native renderer but must not silently reinterpret them. ABI v3 exposes the optional deterministic reference post-process profile and native baked-clip mixing. Clip mixing blends two already-baked canonical vertex frames; draw order switches discretely at the declared midpoint. Neither feature alters canonical mesh, weights or compiler-authored motion.

## Stable C ABI

`rs_runtime_abi_version()` reports the host ABI version. Schema evolution and ABI evolution are independent. Host integrations should check both the ABI and feature flags, then use `rs_runtime_coordinate_system()` and `rs_runtime_units_per_pixel()` before converting to engine space.

The current runtime reads v1 packages for compatibility and emits v2 packages only. v1 lacks embedded texture CRC and explicit coordinate/feature fields; the runtime returns conservative defaults for those packages.
