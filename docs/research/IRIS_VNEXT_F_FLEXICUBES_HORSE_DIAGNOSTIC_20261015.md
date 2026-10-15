# IRIS-vNext F exact FlexiCubes Horse integration diagnostic

**Date:** 2026-10-15  
**Status:** diagnostic integration evidence only — not a teacher-field PASS, not a matched F-arm PASS, no architecture freeze

---

## 1. Purpose

This diagnostic answers a deliberately narrow question:

> Can the exact bound NVIDIA FlexiCubes implementation execute on a real FIT8 morphology under RealSaS raw-source geometry semantics, and what topology does it emit across a coarse resolution sweep when driven by a non-authoritative diagnostic scalar field?

It does **not** answer whether FlexiCubes is the final F geometry backend. The scalar field used here is not the preregistered FIT8 teacher field required by the matched physical court.

---

## 2. Exact source and upstream identity

Horse exact source bytes:

```text
Horse.gltf.bin SHA256
3deb61550dff1d2786d04b6e8559d63ad3907d6ab606ba28ce0af074ed96341b
```

Raw source geometry is read with the same semantics as `tools/corpus/gltf_bind_scene_v1.py`:

```text
raw render vertices = 4400
source triangles     = 2182
position-equivalent geometric vertices = 1093
geometric components = 1
Euler characteristic = 2
boundary edges        = 0
non-manifold edges    = 0
```

The 1093-vertex equivalence result is stable across decimal weld precision from 1e-12 through 1e-5 in this diagnostic and matches the prior FIT8 structural census.

Exact upstream reference:

```text
repository = nv-tlabs/FlexiCubes
commit     = 4cc7d6c3d0cee83c011ce36721b81adff0dd7db6

flexicubes.py SHA256 = b2f7e89cb7b78cfa63cc838936222d519cc8daa864786232223d9ebbebdf032f
tables.py     SHA256 = de9535d097dc7df266a8e8098efbe6f46d923287536d5f7f7ee924171267c442
```

These are the exact bytes preserved by the successful GitHub Actions reference smoke artifact.

---

## 3. Diagnostic scalar provenance

The source closed manifold is isotropically normalized. A surface voxelization is filled, padded, and converted to a signed Euclidean distance-transform lattice:

```text
negative = inside
positive = outside
```

This field is explicitly:

```text
DIAGNOSTIC_VOXEL_FILL_SIGNED_EDT_NOT_TEACHER_TRUTH
```

It is useful for backend integration and gross topology behavior. It is **not** admissible as `QUALIFIED_FIT8` extraction truth and must not be used for matched F-A/F-B/F-R/F-C quality selection.

---

## 4. Exact FlexiCubes sweep

| Target long-axis cells | Effective res | Candidate V | Candidate F | Components | Euler | Boundary | Non-manifold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16×40×36 | 1,844 | 3,684 | 1 | 2 | 0 | 0 |
| 48 | 20×56×48 | 3,814 | 7,624 | 1 | 2 | 0 | 0 |
| 64 | 24×72×62 | 6,378 | 12,752 | 1 | 2 | 0 | 0 |
| 96 | 32×104×90 | 13,850 | 27,696 | 1 | 2 | 0 | 0 |
| 128 | 40×136×116 | 24,046 | 48,092 | 1 | 0 | 0 | 0 |

The CPU extraction call itself remained sub-second in this local diagnostic; that timing is environment-specific and is not a production performance benchmark.

---

## 5. N=128 autopsy

The N=128 topology drift must **not** be attributed directly to FlexiCubes.

The same voxel occupancy used to build the signed EDT was independently surfaced with a marching-cubes diagnostic and produced:

```text
components = 1
Euler      = -4
boundary   = 0
nonmanifold = 0
```

FlexiCubes on the same diagnostic scalar produced:

```text
components = 1
Euler      = 0
boundary   = 0
nonmanifold = 0
```

Therefore the diagnostic scalar/voxel teacher is itself not topologically faithful to the source at this resolution, and the two extraction families resolve the resulting ambiguous/discretized topology differently.

The correct conclusion is:

```text
N128 DIAGNOSTIC FIELD IS NOT A SOURCE-TOPOLOGY AUTHORITY
```

not:

```text
FLEXICUBES FAILS AT N128
```

and not:

```text
FLEXICUBES PASSES HORSE GENERICALLY
```

---

## 6. What this actually proves

The diagnostic supports the following narrow statements:

1. The exact bound upstream FlexiCubes implementation is executable in the RealSaS reference adapter path.
2. Real Horse morphology can be carried through that implementation and produces a clean single closed manifold for the 32–96 diagnostic sweep.
3. Extraction topology is sensitive to the scalar field authority; higher resolution alone does not make a bad/discretized teacher automatically trustworthy.
4. The matched physical court must bind every F arm to the same qualified teacher/extraction truth and exact dense-lattice authority.
5. The newly added exact dense-grid ordering authority is necessary because FlexiCubes ambiguity resolution uses the regular volume adjacency implied by cube row ordering.

---

## 7. Next authoritative step

Do **not** extend this diagnostic by inventing a better ad-hoc SDF.

The next meaningful experiment is:

```text
qualified FIT8 source
    -> preregistered exact/qualified field + extraction truth
    -> exact dense grid authority
    -> exact upstream FlexiCubes F_A reference arm
    -> matched physical measurements
```

Only then compare F-A / F-B / F-R / F-C on:

- exact topology;
- symmetric surface error;
- normal error;
- eight-view depth and silhouette;
- false bridges and missing sheets;
- self-intersection;
- thin-structure retention;
- residual recall;
- payload and memory.

The working M remains the downstream regression target; this diagnostic concerns only the generic upstream reconstruction/extraction path.
