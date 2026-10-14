# FIT8 rendered / mechanical census — Phase 1 diagnostic

**Date:** 2026-10-14  
**Status:** diagnostic evidence only — NOT import admission, NOT teacher authority, NOT scientific PASS  
**Branch:** `agent/unified-iris-stdgenpp-foundation`

This phase follows the exact-byte FIT8 source-structure census. Its purpose is to test whether the structural findings are actually visible or mechanically consequential under the RealSaS canonical camera/motion setting before additional IRIS-vNext leaves are frozen.

The first diagnostic deliberately separates two courts:

```text
STATIC VISIBILITY / RAY COURT
    bind/rest source geometry
    canonical eight-view orthographic camera
    component/material visibility and overlap structure

DYNAMIC MECHANICAL COURT
    animation-evaluated skinning
    deformation consequence statistics
    pose-conditioned overlap structure
```

They are not silently merged. In particular, raw glTF bind geometry is not assumed to equal the default scene pose.

---

## 1. Camera / framing contract used

The diagnostic reproduces the repository's existing controlled-eight-view basis and framing policy:

- view order: `S, SE, E, NE, N, NW, W, SW`;
- yaw: `0,45,90,135,180,225,270,315` degrees;
- global `+Z` screen up;
- level orthographic orbit;
- one common subject center and one common orthographic half-extent;
- frame scale `1.12`;
- no per-view recenter;
- no per-view scale.

The production authority remains `tools/observation/blender_controlled_8view_v1.py`. Phase 1 used an independent CPU raster diagnostic and therefore still requires parity against the production renderer before its pixel measurements can become teacher authority.

---

## 2. Static 256x256 visibility census

The component identity below is source-polygon connectivity for the FBX subjects and position-equivalence diagnostic connectivity for the glTF subjects. The glTF component partition is therefore still diagnostic until geometric-equivalence qualification closes.

| Subject | Diagnostic components | Visible components / view | Visible material regions / view | Max distinct component stack | Max ordered component-pair relations / view |
|---|---:|---:|---:|---:|---:|
| Snake | 3 | 2-3 | 2-7 | 3 | 4 |
| Bat | 12 | 5-12 | 1 | 5 | 30 |
| T-Rex | 1 | 1 | 3-5 | 1 | 0 |
| Horse | 1 | 1 | 3-8 | 1 | 0 |
| Spider | 15 | 8-15 | 1-2 | 6 | 51 |
| GreenBlob | 2 | 1-2 | 1 | 2 | 2 |
| Shark | 1 | 1 | 2 | 1 | 0 |
| George | 61 | 30-59 | 1 | **17** | **395** |

### Immediate interpretation

The result strengthens an already-known contract rejection:

```text
material region != draw-order / presentation primitive
```

George has one material region yet tens of simultaneously visible geometric components, a maximum 17-component ray stack in the static diagnostic, and hundreds of distinct ordered component relations in a single canonical view.

Spider independently shows that this is not only a mech-specific failure: two material regions coexist with up to fifteen visible components and six-component depth stacks.

This does **not** yet choose `owner slot` versus `fragment` as the final CAO draw-order primitive. Dynamic owner/overlap evidence is still required for that decision.

---

## 3. Are Bat / George open surfaces visually significant?

The open-surface representation audit left one explicit freeze question:

> Are the Bat/George open components visibly/mechanically significant, or mostly exporter debris?

Phase 1 performed a resolution sweep with the same canonical framing.

### Bat

| Raster resolution | Closed visible fraction | Open visible fraction | Non-manifold visible fraction |
|---:|---:|---:|---:|
| 128 | 0.727727 | 0.155051 | 0.117222 |
| 256 | 0.727006 | 0.155692 | 0.117302 |
| 512 | 0.727001 | 0.155928 | 0.117070 |

At 256, approximately **27.3%** of first-hit visible foreground belongs to diagnostic components that are not closed (`open + non-manifold`).

### George

| Raster resolution | Closed visible fraction | Open visible fraction |
|---:|---:|---:|
| 128 | 0.880865 | 0.119135 |
| 256 | 0.880080 | 0.119920 |
| 512 | 0.881234 | 0.118766 |

Approximately **12%** of first-hit visible foreground belongs to diagnostic open components.

### Resolution stability

The fractions change only at roughly the `1e-3` absolute scale across 128 -> 256 -> 512. Therefore the observation is not explained by one raster resolution or isolated subpixel fragments.

### Current architecture consequence

The hypothesis that open support can be ignored as negligible exporter debris is now strongly disfavored.

However the glTF component/open classification is still based on diagnostic geometric equivalence. Before changing this leaf to final `LOCKED_BY_FIT8_MEASUREMENT`, production import/geometric-equivalence qualification must confirm that the same visually significant source regions remain open/two-sided under teacher authority.

The priority of the following abstraction is therefore increased:

```text
u(x) = surface proximity / unsigned distance
f(x) = signed or locally-oriented level-set evidence where valid
q(x) = sign/orientation validity
s(x) = surface existence / support
```

The result argues for treating `s(x)` as a first-class learned/evidentiary quantity instead of assuming all surface support can be recovered from global negative-SDF occupancy.

---

## 4. Dynamic glTF mechanical diagnostic

Phase 1 also evaluated the four glTF FIT8 subjects through their native animation clips. Each clip was sampled at five uniformly spaced times. Skinning uses glTF joint transforms + inverse bind matrices. These statistics are **diagnostic**, not qualified mechanical teacher values; the mechanical-reference carrier and importer parity court are not yet sealed.

The most useful first statistic is the worst **p99** observed across sampled poses, not a single raw maximum. Raw maxima are retained only as diagnostics because isolated tiny or degenerate regions can dominate them.

| Subject | Pose samples | max-over-pose edge stretch p99 | max-over-pose face condition p99 | sampled face-area ratio range (raw extrema) |
|---|---:|---:|---:|---:|
| Bat | 20 | 1.181 | 1.618 | 0.495 - 2.595 |
| Horse | 65 | 1.654 | 4.482 | 0.036 - 3.913 |
| GreenBlob | 45 | 1.184 | 1.645 | 0.580 - 1.704 |
| George | 100 | 1.001 | 1.202 | 0.402 - 9.034 |

### Interpretation

1. Dynamic supervision should not be only a mean loss. Horse shows a materially heavier p99 tail than Bat/GreenBlob.
2. Raw extrema and distributional tail must remain separate. George contains large localized raw extrema while the p99 field remains much calmer; a single global maximum would overstate the deformation regime of most of the surface.
3. T's mechanical training/qualification should therefore preserve at least distributional/tail statistics plus localized failure identity, not only global averages.

No threshold is frozen from these numbers yet.

---

## 5. Targeted dynamic overlap diagnostic

A smaller 128x128 dynamic overlap probe was run on Bat and George to test whether overlap complexity disappears when the source is animated.

### Bat

Selected clips: `Bite_Front`, `Flying`, `HitRecieve`; start/mid/end samples.

- maximum distinct component stack: **5**;
- maximum ordered component-pair relations per view: **29-30** across the sampled poses.

The overlap graph is therefore persistent under motion in this slice.

### George

Selected clips: `Dance`, `Pickup`, `Run`, `SwordSlash`; start/mid/end samples.

- maximum distinct component stack: **8-9** in sampled animated poses;
- maximum ordered component-pair relations per view: approximately **237-297**.

The precise graph changes with pose, but remains high-cardinality. This is evidence against a fixed global material-layer order.

The static and dynamic stack maxima are not expected to match exactly: static uses the bind/rest court while dynamic uses animation-evaluated skinning.

---

## 6. A source-truth mistake caught during the prototype

An early diagnostic classified T-Rex as non-manifold after fan-triangulating source n-gons. The authored polygon edge topology itself is closed/manifold; the false classification was created by analyzing derived triangulation chords as if they were authored surface boundaries.

The diagnostic was corrected so that:

```text
surface class / boundary authority
    <- authored polygon boundary topology

raster triangles
    <- provenance-bound derived triangulation
```

After the correction, Snake, T-Rex, Spider and Shark remain closed under the FBX source-polygon diagnostic, consistent with the exact-byte source census.

This is direct evidence for the existing schema rule that raw topology and derived triangulation must remain different truth artifacts.

---

## 7. What Phase 1 resolves and what it does not

### Strongly reinforced / ready for qualification

- open/two-sided surface support is not a theoretical corner case;
- material ID cannot be presentation/draw-order authority;
- all-hit / pairwise overlap truth is necessary for generic subjects;
- mechanical consequence needs tail/localized statistics, not only averages;
- source polygon boundary truth must not be replaced by triangulation diagnostics.

### Still open

- final qualified owner/material region cardinality;
- owner-slot versus fragment-level CAO draw-order primitive;
- component merge/split rules;
- support-grid resolution / sparse active-cell pitch;
- exact local sign/orientation policy for important open components;
- semantic-equivalent extraction behavior at open boundaries;
- full dynamic census for FBX subjects;
- T decoder family and triangle budget;
- final pose-frame sampling policy.

---

## 8. Required Phase 2

Phase 2 must turn the diagnostic into repository authority rather than merely increasing sample count.

Required work:

```text
A. production static teacher renderer
   - exact source import
   - canonical 8 cameras
   - RGBA/depth/normal
   - owner/material/component ID
   - exact surface address
   - all-hit ray stack
   - amodal/decomposition renders

B. import/raster parity
   - independent CPU diagnostic vs production renderer
   - source face/address agreement
   - foreground/depth/normal agreement

C. qualified open-surface audit
   - bind diagnostic components back to source identity
   - determine which Bat/George open regions are intentional/significant
   - validate local orientation / two-sided material behavior

D. dynamic full-FIT8 court
   - glTF + FBX native clips
   - qualified mechanical-reference carrier
   - pose joint matrices and posed surface
   - stretch/area/condition/Jacobian
   - dynamic all-hit / overlap graph

E. sparse-support pitch sweep
   - exact source-surface support cells
   - 48/64/96/128-class coarse candidates by normalized pitch
   - conservative dilation
   - false-negative holes
   - false-positive bridges/shells
   - thin/open-boundary recall
```

Only after B/C/E should the F representation and support-grid resolution be frozen.

---

## 9. Current architectural reading

Phase 1 pushes the architecture toward the following interpretation without yet freezing its implementation:

```text
shared scene field
    |
    +-- proximity / local level-set evidence
    +-- explicit surface-support evidence
    +-- owner/material relational evidence
    +-- appearance evidence
              |
              v
      sparse/high-resolution F
              |
              v
      animation-carrier T
              |
              v
         Compiler courts
              |
       canonical M + CAO
```

The important change is not "replace SDF with UDF". The evidence instead says that **distance/sign and surface existence are separate questions**. A qualified metric SDF remains valuable where the source really defines a closed orientable component; generic RealSaS must also represent authored surfaces for which that global sign statement is false.

No scientific or product PASS is claimed.
