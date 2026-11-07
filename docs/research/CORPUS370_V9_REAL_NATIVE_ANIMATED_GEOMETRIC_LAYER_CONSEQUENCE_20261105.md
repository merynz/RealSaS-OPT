# Corpus370 V9 Real Source Authored Motion vs Geometry-Only Layer Edit Results

Date: 2026-11-05. Research source: https://github.com/merynz/RealSaS-OPT/actions/runs/42014234494
Status: REAL HOSTED 4/4 SUCCESS (validator + 3 original-source Blender workers). PR #327 DRAFT, not merged. 0 artist-drawable owner labels, 0 training admission.

## Why V9 matters

The previous V7 rest-pose rendered edit consequences separated some 8/16/32/64 anonymous source geometry hypotheses, but it was uncertain whether they were meaningful under native animated deformation. V8 showed exact source face order and sampled native posed vertices are reconstructible for 80 original native frames across these 3 source subjects; it did not render animation.

V9 reimported the exact ORIGINAL animated source asset and original action/slot bindings as in the v4 teacher exporter, verified every selected evaluated original-source pose against the SHA-locked native teacher pose vertices and polygon signatures, built a material/UV-preserving triangle soup from exact source rest faces, then moved those source faces to the genuine authored source pose. Thus no made-up FK/rig movement and no source object or bone names used to define drawable labels.

At each source, 3 different genuine authored-action frames were selected solely by high source geometry displacement, under four original cameras [0,2,4,6] at 192x192, 32 Cycles samples, producing SOURCE_ORIGINAL, SOUP_FULL, SOUP_NOOP and anonymous group remove/isolate edit arms.

## Real execution and provenance

- GitHub Actions https://github.com/merynz/RealSaS-OPT/actions/runs/42014234494 — 4 of 4 jobs SUCCESS, 3 of 3 original-source workers SUCCESS.
- Full per-source private receipts and real NPZ arrays stored at gdrive:corpus370_preparation_v1/hosted_native_animated_partition_private_v9/42014234494-1.
- Independent original-source-vs-soup RGBA court JSON and real NPZ bytes downloaded separately and all 6 SHA-256 hashes independently confirmed.
- The initial hosted run 42013088875 **failed because of a false V9 source replay contract**: v4 intentionally freezes triangle lists in REST and uses exact polygon topology signature for posed frames. V9 incorrectly demanded a posed triangle list. Corrected the renderer to match original v4 teacher semantics; rerun succeeded without changing teacher/source/candidate evidence. Source failures were not suppressed or passed off as visual accuracy.

## Authored animated source parity and edit consequence numbers

Numbers below are *additional pixel instances across three native poses and four original camera views* measured against the finest AVAILABLE nested geometry candidate. This is a proxy edit reference, not an artist-authored group, and makes finer hypotheses trivially optimal against themselves. Do not treat 8/16/32/64 as production-optimized group counts.

| Original subject | Poses x cameras | Render arms | Source vs Soup qualified | Group count | Extra direct edit pixels vs finest | Extra isolate pixels vs finest |
|---|---:|---:|---:|---:|---:|---:|
| CAT | 3 x 4 | 528 | 12 / 12 | 8 | 65,910 | 192,675 |
| CAT | 3 x 4 | — | — | 16 | 51,669 | 147,869 |
| CAT | 3 x 4 | — | — | 32 | 7,038 | 54,298 |
| CAT | 3 x 4 | — | — | 64 | 0 (reference) | 0 (reference) |
| TRICERATOPS | 3 x 4 | 240 | 12 / 12 | 8 | 721 | 3,120 |
| TRICERATOPS | 3 x 4 | — | — | 16 | 0 | 0 |
| TRICERATOPS | 3 x 4 | — | — | 32 | 0 (reference) | 0 (reference) |
| COWFISH | 3 x 4 | 336 | 12 / 12 | 8 | 4,626 | 9,631 |
| COWFISH | 3 x 4 | — | — | 16 | 10 | 32 |
| COWFISH | 3 x 4 | — | — | 32 | 0 (reference) | 0 (reference) |

Total: 9 native authored pose instances, 36 original-source pose/camera cases, **1,104 genuinely rendered arms**, 36/36 bounded original-source-to-posed-soup parity cases and 36/36 zero soup/no-op differences. All source-camera frames had nonempty original alpha occupancy.

Full original vs posed soup **max absolute sample error**: CAT 0.0026014; TRICERATOPS 0.0264810; COWFISH 0.0. All source-vs-soup **p99 pixel signal** cases qualified at <=0.025, and all noop maximum-signal cases were zero. Therefore do NOT claim all RGBA pixel values are bit-identical. Further native-resolution CAA is NOT authorized.

Per-source independent court SHA-256:
- CAT d53baa0edeed2b570cef5f84394d91adfca2a5c0efbb721e959b5780e2e40e4b.
- TRICERATOPS 1289010d4999fe1bb21384963189c0c067940003d096c3ca35f077921554274c.
- COWFISH 683b4d63e0b164db3fa9fe644c0e0353ce2b920efdfe29a9e3dd43d3735ed113.

Per-source original RGBA NPZ SHA-256:
- CAT 024cbd674f97d921a666f73a188e9e2b739f5a8f358c1b3593745f7a11566bab.
- TRICERATOPS d882dccebe9f1d5f06052ab94a047b47a90fd873b8a8a2a5dba6db4e9fc4adc8.
- COWFISH 6f8178bd5cdfe84c800982c3983efa4c15d7da5256c6b61361ec45d3313e8bcc.

Complete machine-checkable aggregate: canonical/CORPUS370_NATIVE_ANIMATED_SOURCE_RASTER_REAL_RESULT_V9.json.

## What this DOES and DOES NOT prove

Proves (boundedly):
- Real original-source authored animation can be matched in source-world geometry and original appearance by an explicitly material/UV-preserving, index-preserving posed triangle soup.
- Different geometry-only nested group hypotheses have materially different full-image removal/isolation consequences under sampled original authored animations.
- Fixed 8-way decomposition can be too coarse for controlled render-edit probes; no fabricated semantic part names or mechanical bone labels were needed for this diagnostic.

Does NOT prove:
- 64 groups for CAT or 32 for the others is optimal for an artist-editable puppet. Fine candidate is a self-reference, not independently verified drawable truth.
- Group members share a mechanical owner, source attachment, independent controller, original author edit intentions, draw-order identity or native resolution alpha CAA.
- Unseen morphology generality across all 352 observation PASS / 18 FAIL Corpus370 members, or model training authority.

## Relation to StdGEN and StdGEN++ (reviewed from original papers)

StdGEN++: https://arxiv.org/html/2601.07660 ; original StdGEN CVPR 2025 https://arxiv.org/html/2411.05738.
StdGEN++ uses Dual-Branch S-LRM and true semantic-aware 3D surface extraction with inside-to-outside training. Three main geometric semantic categories: body/clothing/hair, plus texture-layer face supervision. StdGEN Appendix H explicitly cites a VRoid-Hub garment-label convention; StdGEN++ excludes 8 of 109 source samples from layered metrics due to ambiguous/defective semantic labels. Semantic priors solve a defined ontology, but cannot be assumed to identify arbitrary puppet editability on new morphologies. Conversely, geometry-only layer inference is underdetermined from passive renders/motion alone. Our differentiated hypothesis is not the absence of *all priors*, but the absence of *fixed semantic class ontology*, with objective source edit consequences, explicit abstention and later controlled causal supervision.

## Next independent gates

1. Generate split/merge equivalence classes and minimal edit addressability constraints instead of selecting whichever geometry-only partition is finest. Include positive and negative independent artist/edit controls or explicitly abstain when two structures remain observationally equivalent.
2. Source-authorized independent control perturbation: shams, root-only motion, local rig controls, reversibility, occluded attachment and adjacent surfaces. Separate mechanical causal ownership from drawable identity.
3. Native-resolution 8-view original and edited source CAA: alpha bleed, seaming, depth/visibility, draw-order, hidden surface reveal, asymmetric occlusions and edit-rest fidelity; no hardcoded humanoid names.
4. Only qualified typed evidence can enter IRIS/TESSA loss masking or materialization after original SHA lineage, no source owner IDs as student labels.

This evidence is a bounded source-geometry/edit consequence measurement, not training admission or a commercially ready puppet.
