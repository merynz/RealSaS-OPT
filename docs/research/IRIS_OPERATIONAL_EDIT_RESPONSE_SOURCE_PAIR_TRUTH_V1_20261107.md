# IRIS operational edit-response relation target V1 — original physical source candidate

**2026-11-07 · stacked research on PR #371.** No main, existing IRIS/TESSA architecture, Corpus370 training or product deployment changed.

## The practical problem identified by audit

The current IRIS V3 pair truth `models/iris/vnext/graphics_truth_v3.py` rejects `face_a == face_b`. Yet independently verified Cat original source edit experiments (#369 and #371) establish that one original triangle can contain physically supported UV subaddresses with **different interventions' effects**, across two native poses. Face-pair labels cannot represent this. Training `cut_probability` or `same_drawable_affinity` by treating one whole triangle as a uniform edit unit would invent data.

In addition, current V3 adapter abstains for all static drawable/attachment relations. The first needed relation is simpler and measured: **two source surface loci have the same or different observed edit-response signature under a declared intervention suite**.

## Source fact, not artist label

For an independently source-courted UV surface locus `x` and interventions `q_0, q_1, q_2`, the measured Boolean response signature is `s(x)=[changed(x,q_0),changed(x,q_1),changed(x,q_2)]`.

The existing immutable original source court has explicitly physically verified **both same-Cycles original samples**, original pose ordinals **0 and 14**, true original source face and UV edit address, and both:
- intended edit effect >0.025 premultiplied RGBA,
- other two independently applied edit arms <=0.025 for the same physical sample pixel.

It qualifies the same surface barycentric/UV material address across the two poses without pretending that Cycles stochastic rays at both frames are identical.

We now construct:
`observed_edit_signature_agreement(x,y) := 1[s(x)==s(y)]`

**Only pairs on the same original triangle, same camera view, with distinct source barycentric addresses, each independently source-certified across the two original poses, enter this first dataset.** Within each (face,view,edit) group select the first and last certified pose0 pixels for a positive pair; for every two independently observed edit operations on one (face,view) select one pair of certified pixels for a negative pair. Every pair is traceable back to exact physical source evidence. No heuristic distance or invented owner segmentation is used.

Positive means *same observed signature under these three operations*, not same drawable or identical future response to arbitrary controls. Negative means *different observed signature under these operations*, not different drawable. Both can be implemented by a single drawable with internal addressable masks or other equivalent runtimes.

This is **not** a universal source truth claim or full proof of a viable puppet. It is the first locally measured operational relation family that can be fed to an eventual IRIS surface-locus relation consumer without relying on anthropomorphic semantics.

## Source lineage and independent checks

- Parent real source report: `PRIVATE_CAT_POSE_FACE_UV_ANCHOR_PHYSICAL_V1.json` in hosted source run `42307569041-1`, exact SHA `974c1a2a4cdbd51f2d799cc29a8d3b4ef3bbb36ba6442a3cc79c8f541d44ca6b`.
- Parent physically independent source court: `PRIVATE_INDEPENDENT_CAT_POSE_FACE_UV_ANCHOR_COURT_V1.json`, SHA `3ab8cbedd37d31e3fed6ddb554fab6fe5c80294076b7aaf94897b713e4bdad8c`.
- The source producer emits every selected pair with original face/barycentric addresses (TEACHER PRIVATE), source camera, both pose pixel conditions, intervention-op indices (TEACHER ONLY), a binary measured signature relation and physical validity. **No teacher operation ID/face ID is a student input feature**.
- The exact original source `teacher.npz` (SHA `c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603`) is reopened. Each surface locus's original barycentric coordinate is evaluated on the same original triangle at **both pose0 and pose14** to produce typed **source-world 3D query points**. The independent derived-target court reconstructs every 3D point by its own per-vertex weighted sum; this is a real source-space 3D query binding, **not yet IRIS-normalized**.
- Separate derived-target court does not import the producer; it independently reselects every expected pair from the same already physically courted, exact SHA-locked source and validates pair values, addresses, counts and forbidden promotions. It does **not** rerender or independently repeat the original pixel-level physical court; that was done by #371.
- Actual source-run private outputs remain on Drive, with exact post-upload byte readback. Public GitHub contains only scripts, tests, mathematical scope and aggregate receipts.

## Training and downstream integration gates still OPEN

This truth has the following meanings:
- `source_evidence_valid = true` for these two-pose/three-operation source outcomes.
- `iris_training_authorized = false` because the V3 surface-locus query consumer, canonical normalization and held-out data lineage are not yet qualified.
- `same_drawable_owner = UNKNOWN`, `attachment_relation = UNKNOWN`, `proposed_compiler_cut = UNDECIDED`.
- No generic Alpha, reveal, attachment/release, secondary motion, or whole-puppet parity follows.

This prototype does **not** overwrite the current seven V3 Relations outputs. An actual model-facing surface-locus head must explicitly learn measurable edit signatures/response relations with a held-out intervention/pose court. The Compiler can then use this typed evidence when choosing independent subaddress controls, before applying its own partition and complexity policy.

## Immediate next sequence

1. Audit/qualify this first real source target result and independent byte readback. **If FAIL, correct against observed cause, not by adjusting source class counts.**
2. Bind barycentric source addresses to the exact IRIS query-coordinate normalization and add explicit real-data joint consumer receipt; never pass source op IDs to student inputs.
3. Extend the **same measurement semantics** to original Triceratops palette interventions and another morphology — not merely project Cat's UV operations onto them.
4. Make original source interventions and their independent held-out results the consequence evaluator for Compiler-selected candidate representations.
5. Only then add new IRIS operational head/loss and Corpus370 family admission. Preserve unknown heads as masked, not zero.

Files:
- `tools/corpus/derive_operational_edit_response_pair_truth_v1.py`
- `tools/corpus/court_operational_edit_response_pair_truth_v1.py`
- `tests/corpus/test_operational_edit_response_pair_truth_v1.py`
- `.github/workflows/original-cat-operational-edit-pair-truth-v1.yml`
