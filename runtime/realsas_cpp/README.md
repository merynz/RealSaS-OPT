# RealSaS C++ Runtime

Engine-neutral C++17/C ABI playback for compact `.rss` or `.realsas` deployment archives.

The offline compiler and canonical reference runtime bake the selected canonical deformation result together with each clip's qualification bit. The native runtime does **not** rebuild rigging, weights, deformation, draw order, or compiler hypotheses. It executes the packed product contract exactly and never promotes unqualified content.

## Runtime guarantees

- opens ZIP archives or unpacked package directories;
- validates the `.rsr` CRC32 stream before parsing;
- validates each compressed texture payload against the CRC32 embedded in `.rsr`;
- exposes an explicit coordinate/unit contract instead of relying on engine assumptions;
- exposes views, RGBA textures, meshes, UVs, triangles, clip qualification and source hashes;
- interpolates baked mesh frames deterministically;
- preserves per-frame draw order;
- requires neither Python nor the RealSaS compiler in a shipped application;
- provides a stable C ABI for Unity, Unreal, Godot, custom engines and WASM hosts;
- samples deterministic baked clip crossfades and midpoint-owned draw order through ABI v3;
- includes a software reference renderer for conformance checks and an optional deterministic cinematic post-process profile. Production engines normally submit the same mesh data to their renderer and may reproduce or replace only this presentation layer.

The v2 coordinate contract is `screen_2d.top_left.x_right.y_down.uv_top_left`: one runtime unit equals one source pixel, position and UV origins are top-left, +X points right, +Y points down.

```bash
cmake -S runtime/realsas_cpp -B build/realsas_runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/realsas_runtime -j
./build/realsas_runtime/realsas_runtime_demo character.rss \
  --clip idle --view V0 --time 0.53 --out frame.png
```

The runtime qualification bit is preserved per clip. The library can inspect unqualified clips, but never upgrades them to production-qualified animation.

Install and consume it as a normal CMake package:

```bash
cmake --install build/realsas_runtime --prefix build/realsas_runtime_install
```

```cmake
find_package(RealSaSRuntime 0.2 CONFIG REQUIRED)
add_executable(game main.cpp)
target_link_libraries(game PRIVATE RealSaS::Runtime)
```

The installed package exports `RealSaS::Runtime`, public C/C++ headers, a versioned CMake config, and the native library.

See [`FORMAT.md`](FORMAT.md) for the packed contract and C ABI ownership boundary.
