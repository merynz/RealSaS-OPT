# RealSaS — Exact Dehydrated Continuation Context — 2026-09-13

> Purpose: exact continuation packet for a fresh chat. If the operator says **"dehydrate yourself"**, treat this file as the first human-written continuation snapshot, then reconcile against `CURRENT_STATE.md` and the cited canonical/result artifacts. Do not resurrect superseded FIT1 product authority.

## 0. Active branch and claim boundary

- active repair branch: `repair/mage-full-subject-reclosure-20260912`
- branch head at this snapshot: `c69ab78eac42ec89d2bed650fd9aa607d6f44a00`
- same-Mage witness: FIT2
- `PRODUCT_PASS`: **NOT CLAIMED**
- unseen/FIT8/LOFO: still blocked until same-Mage FIT2 reclosure
- no merge to `main` without explicit operator authorization

## 1. Corrected upstream authority

### H1

- source run: `20260912T074348Z`
- H1 checkpoint SHA-256: `76fc68a8c6f2bed80ae8a678649006c875bc96b586065da8914e7f61528c8ce5`
- product-clipped zero-surface SHA-256: `56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925`
- observation-bound product surface, not hidden-teacher/global-watertight authority

### GSA8192

- nodes: `8171`
- edges/relations: `23656`
- observed/completed: `7391 / 780`
- geometry lineage: `65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da`
- tensorization hash: `fe351362e195164805cebb0f63b74d1861ef123458b20ac56607016e00a6c67e`

## 2. Geppetto FIT2 — terminal stage PASS verified

The previous `CURRENT_STATE.md` still said Geppetto was running. That is stale.

Exact Drive result:

`REALSAS_MAGE_FULL_SUBJECT_RECLOSURE_20260912/MAGE_FIT2_PIPELINE_REFIT/GEPPETTO_REFIT/MAIN/GEPPETTO_FIT2_CORRECTED_SUBSTRATE_RESULT.json`

Verified facts from the result/evidence bundle:

- result schema: `RealSaS.MageFIT2.GeppettoCorrectedSubstrateRefit.v1`
- status: `FIT2_GEPPETTO_TERMINAL_PASS`
- closure step: `12160`
- terminal streak: `48 / 48`
- required terminal checks: `48`
- final check: PASS
- evaluation seeds: `[11, 23, 47, 89]`
- every final seed: generated `22`, PASS, root accuracy `1.0`, parent accuracy `1.0`, unsupported joints `0`, illegal parents `0`, deform roots `1`
- final free eval teacher feedback: `false`
- historical Geppetto checkpoint loaded: `false`
- thresholds changed: `false`
- surface: `8171` nodes / `23656` edges
- GSA lineage: `65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da`
- tensorization: `fe351362e195164805cebb0f63b74d1861ef123458b20ac56607016e00a6c67e`
- result repo/source head: `135e6def2ba7112b8471586115d11bf1b936677d`

Persisted evidence manifest status: `PASS`.

Important artifact hashes:

- checkpoint: `2e1f35d196af6eea957bb6377bc0b0e1b045c20dc652d798b74b107198ae80e9`
- result JSON: `91a7d4cb45f0402a0980a59a548c0555137fddd288f508500b8fdee6ab99afb5`
- run log: `14377a0e4302e24fc000f18989c0311f717e3c46db3e90d70bcfbd66255694e6`
- 8-view skeleton contact sheet: `865b215d88bc13e730adfcc7377755c4e3a25aec4d6f89af849e37ef9f453eaa`

`ARACHNE_FIT2_INPUT_HANDOFF.json` exists and says:

- status: `AUTHORIZED_BY_GEPPETTO_FIT2_TERMINAL_PASS`
- `arachne_refit_authorized=true`
- exact Geppetto checkpoint hash matches the evidence manifest
- exact GSA lineage and tensorization hashes match corrected FIT2 authority

Claim boundary: this is a **Geppetto stage terminal PASS**, not `PRODUCT_PASS`. The result itself intentionally has `product_pass_claimed=false` and `promotion_authorized=false`.

## 3. UI skeleton overflow root cause — isolated

The bad Living Compile/UI screenshot did **not** imply Geppetto generated a bad skeleton.

Root cause isolated in the old UI adapter path:

- old `scene.py::_joint_scene_rows()` did not use the qualified directional joint pivot;
- it substituted mean raster position of support-surface nodes, or a nearest-surface fallback;
- on the real Mage graph this produced tens to >100 px of 2D anchor error in cardinal views;
- therefore the same qualified skeleton could look correct in the Geppetto contact sheet and displaced/outside the artwork in UI.

The correct path is qualified directional joint binding. Static rig overlay must be withheld if directional binding is unavailable. Animated rig overlay must likewise be withheld unless qualified per-frame control/joint state exists.

Do not blame mesh or Geppetto for this specific UI overlay displacement.

## 4. Mesh science — exact progression

### 4.1 Uniform legal-Steiner family

Global uniform legal-Steiner `n={2,3,4}` was falsified as final product meshing family.

Reason:

- coverage approached historical CDT in places;
- inherited skinny parent triangles remained skinny under uniform barycentric subdivision;
- V2/V6 boundary residuals plateaued.

Legal support route survived.

### 4.2 Legal-Steiner ceiling

Pixelwise supported ceiling proved geometric reachability around `~97–99%` recall with precision `1.0` depending on view. This was a diagnostic ceiling, not emitted-mesh/product PASS.

### 4.3 Global adaptive boundary + quality Delaunay

Preregistered global treatments `B4_G16 -> B2_G12 -> B1_G8` were falsified as a **global final solver**.

B1 produced high-quality admitted triangles but only about `~70%` recall because global Delaunay followed by strict locality/alpha rejection discarded too much topology.

Interpretation retained:

> adaptive triangulation is useful as a **local repair/recovery operator**, not as the sole whole-character reconstruction solver.

### 4.4 Current experiment: baseline-preserving adaptive patch CDT

Prereg/source:

- `canonical/FIT2_BASELINE_PRESERVING_ADAPTIVE_PATCH_CDT_PREREG_20260913.md`
- runner: `experiments/mage_full_subject_reclosure_v1/run_fit2_baseline_preserving_adaptive_patch_cdt_v1.py`
- current source/preflight branch head: `c69ab78eac42ec89d2bed650fd9aa607d6f44a00`
- self-hosted preflight run: `34752599677` — PASS

Frozen order:

1. `P1_B2_G10`
2. `P2_B1_G8`
3. `P2_B1_G6`

Core architecture:

> **sealed historical CDT = coverage authority**  
> **adaptive support-derived CDT = local quality/recovery operator**

This is one final mesh, not two overlaid meshes.

Baseline is **NOT assumed to be all correct**.

Current code scans every baseline face with the frozen raster shape-quality policy. Faces that fail minimum angle / maximum aspect are `initial_bad` and become local quality-repair cavities. Healthy faces remain unchanged. Residual recovery only adds legal supported geometry where baseline coverage is missing.

Important nuance:

- current experiment treats a baseline face that **passes the frozen quality gate** as `KEEP`;
- therefore it proves/targets **policy sufficiency**, not global mesh optimality;
- if deformation evidence later shows that merely-passing baseline faces should be improved further, that is a **fresh quality-uplift experiment/prereg**, not a post-hoc mutation of this run.

### 4.5 P1 result already observed while run continues

`P1_B2_G10` greatly improved coverage without alpha spill:

| View | baseline recall | final recall | precision | largest hole | P1 status |
|---|---:|---:|---:|---:|---|
| V0 | 92.589% | 97.230% | 100.000% | 0.997% | quality FAIL |
| V1 | 94.061% | 98.948% | 100.000% | 0.322% | PASS |
| V2 | 90.557% | 97.679% | 100.000% | 1.373% | quality FAIL |
| V3 | 94.130% | 98.339% | 100.000% | 0.645% | quality FAIL |
| V4 | 92.850% | 97.343% | 100.000% | 1.001% | quality FAIL |
| V5 | 93.799% | 98.863% | 100.000% | 0.395% | PASS |
| V6 | 90.758% | 97.502% | 100.000% | 1.374% | PASS |
| V7 | 94.366% | 98.230% | 100.000% | 0.625% | quality FAIL |

Residual parent recovery was almost complete (`188/190`, `228/232`, `306/307`, etc.).

Remaining P1 failures were only inherited shape-quality outliers. P1 quality cavity repair accepted none (`Qpatch=0/...`), so the same pathological baseline faces remained. Total bad faces reported after P1: `18` across V0/V2/V3/V4/V7.

This means P1 already strongly validates the asymmetric hybrid idea for coverage, but does **not** close the frozen quality policy.

At snapshot time `P2_B1_G8` / `P2_B1_G6` are still pending/running. Do not infer their outcome.

## 5. Mesh decision rule after P2/P3

Do not change the running thresholds or append a treatment.

After the run finishes:

1. inspect exact result JSON + contact sheet, not stdout alone;
2. require monotonic `final coverage >= baseline coverage` in every view;
3. if a preregistered treatment closes full frozen policy in all 8 views, record that diagnostic closure exactly;
4. if only the same small set of baseline quality outliers remain, inspect rollback reasons for `Qpatch` and decide a **fresh local quality-repair experiment** if needed;
5. do not silently decide that the baseline is universally high-quality merely because it is the coverage authority;
6. do not globally remesh healthy baseline regions just to chase prettier triangles without deformation evidence.

If a higher product-quality bar than the current frozen angle/aspect policy is desired, first measure the full baseline/final distributions plus actual deformation strain/flip behavior. Then preregister a quality-uplift objective. Do not optimize planar triangle aesthetics in isolation from deformation mechanics.

## 6. Immediate execution order

The old order saying Geppetto is running is superseded.

Current order:

1. H1 corrected product surface — DONE
2. GSA8192 — DONE
3. Geppetto FIT2 terminal PASS — VERIFIED
4. Geppetto 8-view evidence + hash manifest — VERIFIED
5. baseline-preserving mesh experiment P1/P2/P3 — ACTIVE, final decision pending
6. Arachne FIT2 — **AUTHORIZED BY GEPPETTO HANDOFF; may proceed independently of waiting for the mesh experiment to finish**, provided it consumes the exact corrected S/G handoff
7. QualifiedSkinIR / exact W proof
8. final strict directional mesh decision + exact mesh-skin transfer
9. component/mechanical/deformation proof
10. professional motion closure
11. exact runtime/export identity reclosure
12. only then `PRODUCT_PASS`

## 7. New-chat recovery instruction

When starting a fresh chat, read in this order:

1. `canonical/DEHYDRATED_CONTEXT_20260913.md`
2. `CURRENT_STATE.md`
3. `canonical/REHYDRATION_PACKET.md`
4. current mesh prereg + exact mesh result if the P1/P2/P3 run has finished
5. Geppetto FIT2 result/evidence manifest/handoff as needed

Do not rely on stale statements that Geppetto is still running. Do not relabel the mesh partial P1 result as final. Do not start unseen/generalization work before same-Mage FIT2 closure.
