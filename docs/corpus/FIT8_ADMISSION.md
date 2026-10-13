# Fit8 diversity and source admission

Date: 2026-10-13. Selection is source discovery, not a research execution receipt.

[FIT8_DIVERSITY_SELECTION_V1.json](../../canonical/FIT8_DIVERSITY_SELECTION_V1.json)
pins eight exact source files from the sealed October 4 corpus. Knight remains
the separate fixed regression witness. All eight retrieved files matched the
sealed file ledger. Import, pose, appearance, lineage and render admission remain
pending; no optimizer has been started by this selection.

| Subject | Coverage | Main stress |
|---|---|---|
| Snake | Serpentine | Continuous bends, self overlap, ground contact |
| Bat | Flying | Wing folds, thin surfaces, wing/body occlusion |
| T-Rex (`Trex.fbx`) | Dinosaur | Long tail, short forelimbs, jaw and nonhuman balance |
| Horse | Quadruped | Four contacts, crossed gait, neck/tail chains |
| Spider | Arthropod | Many limb chains, crossings and ground clearance |
| GreenBlob | Amorphous | Few controls, squash/stretch and contact |
| Shark | Aquatic | Axial swimming, fins and no ground assumption |
| George (textured) | Mechanical | Articulated rigid parts, local frames and materials |

These labels are evaluation metadata. Production modules must not branch on
them, source names, joint indices or pack IDs. A cosmetic sibling is not another
coverage class. T-Rex is the actual dinosaur pack source, not the humanoid-rigged
`Ultimate Monsters/Big/Dino.gltf`. Snake, Spider, T-Rex and Shark are FBX sources
outside the previous GLTF-only truth audit; their raw hashes do not prove import
or mechanical truth fidelity.

## Admission starts at the first render

The following gates qualify each subject separately, in dependency order. A
failure keeps that subject and its diagnostics in the roster; it does not trigger
an automatic easier substitute. Replacement needs a versioned selection reason,
retained rejected identity and refrozen court before fitting. Model scores must
not determine replacement.

| Existing boundary | Required evidence before continuing |
|---|---|
| Stage01/02: source bytes and provenance | Exact source, material/image dependencies and license bytes; source revision; verified ledger membership. Working derivatives stay outside immutable raw sources |
| Stage03/04: import and full subject | All meshes, armatures, rigid equipment and attachment nodes accounted for; parent/local/world matrices, authored axes, units and visibility recorded. Missing images, lost objects or ambiguous rig/animation bindings block admission |
| Stage05: pose and camera | Explicit source bind/rest pose and native first-frame pose, distinct when authored that way; one recorded subject-to-canonical transform for geometry, G, W, appearance and motion. Inverse binds and evaluated rest reconstruction agree. No humanoid A-pose assumption |
| Stage06/07: first eight observations | Source-native and imported source renders compared with the same camera, pose, material policy and frame settings; eight level orthographic views with one center/scale. No per-view pose adjustment, recentering, silhouette repair or attachment repositioning |
| Stage08: normalization | Whole subject including equipment fits the declared domain; axes, handedness, ground/root origin and inverse transform are explicit. Dynamic validation uses a separately sealed fixed camera envelope, never per-frame recentering |
| Training admission | Per-subject approved pose/scene/render evidence, usable geometry/rig/skin/appearance/native-motion truth, source dependency closure and leakage component assignment. Geometry-only or partially imported truth is not silently promoted |

Source bind/rest, frame0 and animation start are separate addresses. Frame0 may
be an authored pose rather than rest identity. Compare against the appropriate
source reference; never force identity to hide a rest-frame discrepancy.

Attachment validation must independently check parent and target correspondence,
bind/rest transforms, local grip/anchor, offset, rotation, scale and any inherited
root/object motion. Record the source relation even when the equipment is a
separate rigid mesh or node rather than skinned vertices. Proximity to a hand,
zero centroid residual or a plausible picture alone does not establish a correct
grip. No selection label is an attachment consumer instruction.

Store per-view source/import/rest/frame0 evidence, visible ownership and contact
residuals, clipping/coverage loss and material/alpha differences. Check terminal
poses and loop boundaries for each native motion where applicable. Do not invent
looping semantics or require ground contact from flying/aquatic actors. Thresholds,
reference hashes, sample times and acceptance interpretation must be sealed before
measuring a court; this planning document is not such a quantitative seal.

After source admission, compare the **first compiled M/G/W + appearance render**
to that admitted reference at matching cameras and poses, before broader motion
testing. Source/render admission cannot close compiled contact, attachment or
semantic draw-order failures. Preserve every failed image and attribute it to
source, import, pose/scene, observation, model, Compiler or Runtime ownership.
An average score must not conceal a failed subject or view.

## Fit and regression rules

Use the same architecture and one shared parameter set per model across the
eight subjects, with the same Compiler contracts. No per-subject checkpoint,
anatomy shortcut, renderer patch or hidden teacher input can support the shared
Fit8 claim. Source rig/mesh/weights are teacher/evaluation truth, not image-only
model inputs. Native motion can be explicit court input, with its authority
recorded. Declared shared architecture changes are allowed, with impact and
matched baseline comparisons; they invalidate the real consumers.

Report every subject and module, worst-case failures and timing alongside any
aggregate. A fix must rerun affected admitted subjects and the matched Knight
regression. Independent qualified artifacts retain exact identities under the
Go released-DAG resolver; missing artifacts do not become implicit reuse.

The old September [Fit8 prereg](../../canonical/FIT8_SOURCE_TEXTURED_AUTHORITY_PREREG_V1.json)
is a separate historical selection and must keep its bytes. The new roster does
not imply an optimizer schedule, checkpoint seal, new live Attempt or product
promotion. Execute scientific work only through the existing
[Go entry/run/exit contract](../platform/EXECUTION_LANES.md) on deployed main.

## Lineage and unseen

Split LOFO by leakage-connected skeleton, topology and material lineage, across
pack boundaries. All eight lineage component IDs are still pending. Different
coverage labels do not establish lineage independence.

KayKit is excluded from this roster. However, the older Fit8 prereg already names
Skeleton Mage and Skeleton Warrior; a creator-level unseen claim requires an
exact past-use/derivative audit. Reserve only demonstrably unconsumed identities
and their connected lineages, seal the holdout before new fitting, and distinguish
creator holdout from a fully unseen creator. Fit8 success is fit evidence; LOFO
and unseen are later separate courts.

## First implementation work before admission

Source inspection at the recorded main baseline exposes concrete apparatus gaps:

| Existing implementation | Observed capability and required work |
|---|---|
| `tools/observation/blender_controlled_8view_v1.py:reset_and_import` | FBX-only import; frame0 plus armature REST. Add an explicit GLTF/FBX source contract and isolated source bind/rest evaluation covering animated object/root transforms and NLA/action state before comparing renders |
| Materializer `armature_rows` / `mesh_rows` | Bones, parents, groups and bounds are recorded; complete node local/world matrices and inverse-bind reconstruction evidence are not emitted. Seal these alongside rigid attachment relations |
| Materializer `force_unlit` | Transfers base color to emission; source alpha has no explicit preserved alpha connection. Qualify alpha/material policy against the source reference before calling its silhouettes authoritative |
| `adapters/preflight_v2.py:materialize_observation_render_stage` | Materializes already supplied exact raster bytes. It is not a source renderer. Bind new admission evidence through the actual Stage03–08 consumers; an external report alone must not mark those missing checks passed |

These are source-inspection limits, not observed failures of any selected
character. Extend the common apparatus, prove pose/scene/material fidelity on
real sources and seal an admission court before optimizer work. Do not change a
source pose, conceal missing material, or convert format silently just to obtain
a picture. The source identity verifier below cannot close these gaps.

## Common source/rest engineering producer

`tools/observation/blender_controlled_8view_v2.py` imports FBX with animation
disabled and GLTF/GLB through an explicit animation-free derivative outside raw
source directories. Nodes, default morph state, skins, materials and binary
buffers remain unchanged in that derivative. The producer records imported
node/bone frames, effective parent/world transforms and a common invertible
normalization, then renders a fixed eight-view orbit with explicit clip planes.
The existing v1 Knight producer is unchanged.

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python tools/observation/blender_controlled_8view_v2.py -- \
  --source /absolute/path/exact-roster-source.gltf \
  --expected-source-sha256 <exact-roster-sha256> \
  --subject-id <roster-subject-key> \
  --out-dir /absolute/path/fresh-candidate-directory
```

Defaults are 1024 px, shared frame scale 1.12, minimum border margin 8 px and
Cycles CPU with seed 0 / 16 samples / no denoising / Standard view transform.
Each receipt binds the source, producer/helper bytes, Blender binary, camera,
scene snapshot and pixels. Output must be fresh; a failed/partial directory is
retained for diagnosis and cannot carry a stale success receipt into a retry.

Unlit conversion retains active upstream color/alpha and downstream shader
graphs. An explicit imported legacy `OPAQUE` mode keeps its dormant BSDF alpha
inactive across the CPU backend. The first common run exposed this distinction
on the FBX sources: transporting the inactive alpha value as transparency made
three subjects lose views and Spider nearly disappear. A common mode transport
fix corrected that engineering failure; no source names or anatomy labels enter
the implementation. Native FBX opacity semantics still require independent
source/material parity before admission. Unknown/new material conventions must
be qualified with their exact importer and renderer; an engineering picture is
not authority to infer a source opacity repair.

The [October 13 engineering receipt](../../canonical/FIT8_SOURCE_REST_ENGINEERING_SMOKE_20261013.json)
records all eight exact sources producing 64 nonempty 256 px candidate views,
minimum margins of 12–13 px and no raw mutation in Blender 4.0.2. Synthetic
Blender tests separately exercise nested root/rigid attachment transforms,
active linked alpha with a downstream shader and dormant opaque alpha. Run:

```bash
python -m unittest tests.repository.test_source_scene_contract_v2 -v
blender --background --factory-startup --python-exit-code 1 \
  --python tests/fixtures/source_scene_v2/blender_smoke.py
```

This is source engineering, not a new Go Attempt or source/truth admission.
Snapshot consistency checks bindings inside the imported scene; effective local
matrices are reconstructed from imported world matrices. They do not prove raw
node correspondence, authored axes/units, inverse-bind reconstruction or correct
grip. Independent rest/frame0/native motion/material references and typed
Stage03–08 consumers remain pending. The engineering receipt must not substitute
for those gates or mark the first compiled puppet render accepted.

## Four-model notebook preparation

[FIT8_TRAINING_READINESS_V1.json](../../canonical/FIT8_TRAINING_READINESS_V1.json)
is the delivery checklist, not a second Attempt/workflow ledger. Its initial
status is blocked. Public fit order remains IRIS → TESSA → AXIS → MIRA, with one
shared parameter/optimizer state per model over all eight admitted subjects.
Existing single-subject runners and solved-tree checkpoints are not a common
Fit8 entrypoint. The record names the concrete source gaps for each model.

The first deliverable is the actual IRIS A100 fit notebook, after admission,
released Go cohort bindings and frozen fit/evaluation policy. It must embed
frozen hash-bound source, run all in a clean runtime, mount Drive, use `/content`
for active work/cache, print short true step/loss/per-subject/time/GPU logs and
persist compact results. Checkpoints bind code/inputs/policy and model,
optimizer, AMP, RNG and progress; atomic save and interruption/resume must be
preflighted. No live GitHub token/checkout is required. A selection preview or
CPU source-preparation notebook cannot be presented as this A100 fit notebook.
Notebook path, hash, observed download URL and preflight evidence stay null until
the exact delivered file and its required evidence exist.

## Source identity reproduction

```bash
python tools/corpus/verify_fit8_sources_v1.py \
  --audit-bundle /absolute/path/RealSaS_Truth_Corpus_AUDIT_UPLOAD.zip \
  --source-root /absolute/path/materialized-corpus \
  --output /absolute/path/source-identity-receipt.json
```

The source root preserves the roster's `raw/...` relative paths. The audit ZIP
contains the selected GLTF files; selected FBX bytes can be supplied from the
immutable corpus or the recorded official locators. The verifier checks corpus
seal, manifest, ledger and all eight exact bytes. It performs no import/render,
qualification, Registry write, model fit or product promotion.
