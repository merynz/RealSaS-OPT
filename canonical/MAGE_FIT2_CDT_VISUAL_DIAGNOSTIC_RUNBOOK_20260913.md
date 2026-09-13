# RealSaS — Mage FIT2 Exact CDT Visual Diagnostic Runbook — 2026-09-13

**Status:** `ACTIVE_DIAGNOSTIC_RUNBOOK__PRE_GW__NOT_PRODUCT_PASS`  
**Branch:** `repair/mage-full-subject-reclosure-20260912`  
**Purpose:** render the actual sealed observation-domain CDT triangle mesh for V0..V7 and locate remaining uncovered source-alpha regions before fresh G/W product closure.

## Decision

The self-hosted GitHub runner remains healthy. The recent CDT diagnostic failures were caused by workflow/environment and local-artifact discovery assumptions, not by the runner service itself. A probe confirmed that the only locally discoverable `ZERO_SURFACE_PRODUCT_CLIPPED.npz` was an old artifact with SHA-256 `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b`, while the corrected H1 authority is SHA-256 `56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925`.

Therefore local-machine recursive search is forbidden for this diagnostic. The temporary local-search workflows were retired. The canonical operational route is a Drive-first Run-All notebook using the exact paths and SHA bindings already used by the Mage FIT2 Geppetto notebook family.

## Exact authority

- corrected H1 run: `20260912T074348Z`
- corrected zero surface SHA-256: `56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925`
- exact GSA8192 replay: `8171` surface nodes / `23656` full-safe relations
- existing sealed CDT report: `experiments/mage_full_subject_reclosure_v1/MWB2_CDT_EXACT_RECLOSURE_REPORT_20260912.json`
- renderer: `experiments/mage_full_subject_reclosure_v1/render_fit2_cdt_8view_diagnostic_v1.py`

The renderer must reproduce the sealed per-view candidate lineage, mesh lineage, face count, recall and precision before any image is accepted.

## Notebook contract

Notebook filename:

`REALSaS_MAGE_FIT2_EXACT_CDT_8VIEW_RUN_ALL_DRIVE_V1.ipynb`

Required behavior:

1. mount `/content/drive/MyDrive`;
2. use exact corrected-H1 Drive paths; no Downloads/Desktop search and no fallback;
3. verify zero-surface, V0..V7 camera and V0..V7 RGBA SHA-256 values before replay;
4. pin repository source semantics to an explicit repair-branch commit containing the exact renderer;
5. rebuild the sealed GSA8192 substrate and exact directional CDT;
6. fail closed on any candidate-lineage, mesh-lineage, face-count, recall or authority drift;
7. print concise scientific stdout per view: vertices, edges, faces, recall, precision, IoU, largest uncovered connected component and frozen product-floor flags;
8. write and display real `source | CDT wire/fill | uncovered-red` evidence for V0..V7;
9. save outputs and a manifest under Drive;
10. make no product-mesh PASS claim. This diagnostic answers only where the historical exact CDT still leaves uncovered artist pixels.

## Claim boundary

The historical exact CDT behavioral result remains approximately `90.56%..94.37%` source-alpha recall with `100%` precision. Several views therefore remain below the frozen FIT2 product recall floor of `0.94`. Visual evidence is required to distinguish thin silhouette/boundary loss from product-killing local holes in mechanically meaningful pieces.

Fresh Geppetto/Arachne mechanics are not required to render this diagnostic, but fresh G/W remain required for current FIT2 product mesh, mesh-skin, component, deformation and runtime closure.

## Runner policy after this incident

The self-hosted runner remains valid for deterministic tests and repository-local workflows. It must not be used as an implicit artifact-discovery service. Future experiment jobs that consume external scientific artifacts must use an explicit artifact root/manifest and exact expected hashes, or a Drive-first notebook with the same bindings. Environment executables (`python3`/venv) must also be explicit.
