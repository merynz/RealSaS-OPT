# Canonical CAA mechanics/geometry audit continuation

## Scope and authorization

The user's explicit continuation target is the existing
`audit/canonical-caa-mechanics-geometry-lock-20261002` experiment. Use the
`realsas-wsl-1660ti` self-hosted runner. Do not promote this experiment to `main`
or claim product/unseen closure from an audit result.

Goal: generic topology, geometry, deformation and actual-motion qualification on
Knight as FIT1, followed by render of the **same** qualified artifact chain.
Only failures excluded by those measurements may then be attributed to appearance
or presentation. Sampled-frame PASS does not prove continuous-time motion or all
possible future clips/subjects.

## Verified starting evidence

| Evidence | Run | Actual scope/result |
|---|---:|---|
| V9 static optimizer | 37057338450 | 14,399 vertices, 28,810 faces; zero angle/aspect violations; manifold topology PASS |
| Absolute source fidelity | 37057950583 | FAIL, all eight views; not a source-fidelity seal |
| Source fidelity A/B | 37058482343 | Measurement completed; baseline and V9 both fail the frozen Stage13 policy |
| Original dynamic phase1 | 37059197366 | Aborted before dynamics: `MESH_CANDIDATE_SURFACE_LINEAGE_MISMATCH` |

Exact V9 candidate lineage:
`3ab6b0ae02e362079fed11ceb627acf9be795004e790ee897c98a1eadc14edab`.
V9 artifact ID: `11249525461`.
Refined inverse run: `37023349486`, artifact `11234220602`.
Refined inverse SHA-256:
`8aebe4e230dea633ad0f031c8c0d4086872498350561495339ebeda9fd343231`.

Source-fidelity limits remain unchanged: recall/precision/component recall >=
0.999; coherent hole <= 0.00025; interior uncovered <= 0.0005; silhouette-edge
p95 <= 0.5 px. V9 observed extrema: recall 0.9935819769, precision 0.9848946538,
hole 0.0023609399, interior uncovered 0.0023001072, edge p95 3.6055512755 px.
The source Stage13 artifact itself records `every_view_passed=false` and was
admitted as `PASS_DEMO_ONLY`. A successful A/B workflow is not absolute geometry
fidelity PASS, nor does it isolate the single V8-to-V9 move: its baseline is the
original Stage18 candidate.

## Changes in this continuation

1. `b35b4d72e5f5f3c7c575e53270b79f2486ddee23` fixes the dynamic audit's
   authority inputs. Rebuild the exact refined surface from the sealed inverse;
   use V9's partition and original matching carrier recipe; retain all lineage
   validators. The V9 geometry bytes are not modified.
2. Generic `refined_surface_skin_proposal_v1.py` proposes parent-constant skin
   transport using exact dense cluster membership. Both source and target
   centroids/order must replay; cross-parent merging, missing rows, stale rig,
   invalid simplex and ordering errors fail closed. It emits a **proposal**, not
   a mechanical proof. The numeric simplex is requalified separately. It does
   not infer Arachne's prediction at the new positions or decide whether a fresh
   inference/refit is necessary. Source parent weight values are preserved.
3. Dynamic replay run `37064886261` consumes those inputs, stores the refined
   surface/skin/receipt, G3 report and per-clip checkpoints, and measures rest and
   worst-frame full intersections. Visible-subset pairs are filtered from the
   already computed full census without rerunning the same SAT predicate.
4. `e2cf296971446ad8568e84316616be888a22e26d` adds the intrinsic per-frame
   court and owner-localization consumer. Actual-motion edge extension and exact
   degeneracy alone miss severe flattening: the synthetic counterexample has
   max edge ratio 1, area ratio 0.001 and condition 1000. The new court uses the
   existing frozen area/condition policy, independent of screen orientation.
5. Follow-up run `37065836498` consumes the first run's exact artifacts and checks
   all 51 sampled frames, verifies motion-frame hashes, localizes failing face
   IDs/supports/weight discontinuities, and exports sampled positions for reuse.
   It does **not** yet prove intersections on all 51 frames.

Local checks: 14 transport/skin accounting tests and 3 intrinsic frame tests pass.
These are synthetic implementation checks, not Knight qualification evidence.

## Next actions and closure requirements

- Read the two continuation runs' actual results; do not infer success from queue
  state or workflow names. Preserve evidence if either fails.
- If a transported-weight failure appears, distinguish parent-constant transfer
  error from actual Arachne evidence and topology error through controlled
  comparisons. Do not retrain or change thresholds based only on the failure.
- If the phase1 and all-frame conditioning courts pass, perform the complete
  self-intersection census on **every** sampled frame. Worst edge stretch does not
  identify the worst collision frame. Classify rest intersections and any growth
  of pre-existing overlap; subtracting rest pair IDs alone cannot prove absence
  of worsening penetration.
- Source fidelity remains independently open. Its owner must be measured across
  the dense source, compaction and repaired candidate; appearance cannot silently
  compensate for missing geometric support.
- Product integration: Stage14/15 own generic surface construction/qualification;
  Stage18/19 own mesh/static admission; Stage32/35/36 own skin and dynamic
  qualification/transfer. Dynamic deformation is not a Stage14-only seal.
- Before render, replay affected bindings/proofs and show that the runtime uses
  the tested candidate/rig/skin/motion identity. Audit-only objects cannot silently
  become product authority. Record any remaining demo-only scope explicitly.

## 2026-10-02 21:30 UTC measured failure and follow-up

Run `37064886261` completed with scientific `FAIL_PHASE1_MECHANICS`, not an apparatus exception.
Artifact `11252813278` binds the exact V9/refined-surface/transported-skin chain.
G3 fails area ratio (both bounds) and condition; maximum condition 2136.782204031984.
Across sampled motion, maximum edge ratio is 1307.7031818661376, maximum edge counts
above 10/4 are 812/1249, and posed degenerate face count is zero. The selected worst
frame has 9653 intersection pairs versus 3389 at rest; 6555 pairs are new by pair-ID
subtraction. These are not exhaustive all-frame intersection results, and preexisting
penetrations are not accepted as a valid rest surface.

Run `37065836498` completed and localized the failures (artifact `11252996685`):
96 failed micro probes and 51/51 failed sampled motion frames. The highest-stretch
short edges have near-disjoint endpoint skin weights (L1 approximately 2). This is
an observed association, not proof that a particular learning or topology change is
the sole cause. Parent-constant transport has failed qualification; no dynamic seal
or render closure is granted. Original Stage18/skin under the same rig/tracks/policy
is the next differential measurement. No threshold is relaxed and no seam is cut.

A reporting defect was also found: `_joint_pose_v2` returns the derived **rest joint
frame set hash**, not a per-motion-frame hash. Earlier `motion_frame_hash` fields
are mislabeled and constant across all 51 frames. The follow-up now verifies exact
replay against the archived sampled geometry bytes, records the correct rest-frame
field, and adds a hash over clip/time, evaluated skin matrices and posed XYZ. The
old phase1 report is preserved as historical evidence, not silently rewritten.

## Appearance/presentation gate repair

Source-owned Stage42 projection/native rendering still has 2D positions and fixed
UVs without canonical depth/order authority. Stage45 previously treated native /
reference byte parity plus orientation and stretch as sufficient, even though both
renderers can reproduce the same unqualified painter order.

The reference rasterizer now counts alpha-positive contributors at each sampled
pixel before compositing. Stage45 rejects any multi-contributor pixel until a
qualified depth/order contract exists; diagnostics explicitly scope this to sampled
pixels (not a subpixel or continuous-time geometric proof). Shared triangle edges
use the existing top-left rule, and transparent contributions do not count. The
IR validator and Stage46 closure also require this occlusion evidence, so old PASS
receipts cannot bypass the new gate. Generic tests cover overlapping colors,
face-order reversal, transparent samples, shared edges, Stage45 parity with/without
collisions, and rejection of old PASS receipts.

This closes an unsound admission path; it does **not** implement missing depth
transport or coherent visual-triangle deformation bindings. Those remain open,
alongside the measured mechanical and absolute source-fidelity failures.

## 2026-10-02 21:48 UTC differential result and fresh inference request

Run `37068452456` (commit `afaf4c0b9fbe2e3e7b6563195fc925bab16c1639`,
artifact `11254170095`) completed the original Stage18 versus V9 comparison.
All 51 replayed V9 poses are byte-identical to the prior archived poses. Both
meshes fail conditioning, but the catastrophic magnitude is a regression of the
V9 geometry/topology plus transported-skin combination:

| Clip | Original max edge ratio | V9 max edge ratio | Original bad faces/frame | V9 bad faces/frame |
|---|---:|---:|---:|---:|
| Idle | 2.07193 | 364.65035 | 2 | 156–173 |
| Run | 3.55756 | 1043.29591 | 1–16 | 633–762 |
| Slash | 3.61554 | 1307.70318 | 1–16 | 610–799 |

Original has 28,331 faces; V9 has 28,810. This comparison holds rig, numeric
source weight field, motion source, retarget rule and policy fixed, while surface
refinement, mesh geometry/topology and weight transport differ. It does not
identify a single one of those changed factors as the sole causal owner.

Local analysis of the exact archived 51 poses finds 1,257 unique edges stretching
above 4x (841 above 10x). Of the >4x edges, 1,076 are identity-bound edges already
present in the refined surface relations; 79 have nonidentity support bindings.
None of the >4x edges' identity-bound endpoints moved relative to their refined
surface node. The worst edge (candidate indices 4884,10219) is inherited directly
from refined surface connectivity, rest length 0.0006387936215529746, and bridges
base clusters 8403 and 8058. Thus the final single-vertex V9 adjustment did not
create this worst edge. For edges whose endpoint weight L1 difference is <1e-10,
maximum stretch over the 51 poses is 1.0000000032895184. These are diagnostic
associations, not grounds to mutate weights or erase faces.

User explicitly requested fresh rig **and** skin inference on the new surface.
Model asset inspection run `37069671533`, artifact `11254465379`, verified:

- ML runtime `/home/monster/realsas-ml312/bin/python`, Python 3.12,
  Torch 2.7.0+cu118, GTX 1660 Ti available.
- Fit authority resides in parent `SUBJECT2_KNIGHT_DEMO_V2_20260924`; solved child
  does not duplicate Stage26/27/30/31 files.
- Geppetto sealed checkpoint is available (46,011,564 bytes).
- Arachne sealed Stage31 checkpoint is available (558,860,805 bytes), but is the
  original V6 closure model, not the corrected UNIFORM_ALL_PROJECTED arm used by
  the current skin. Fresh inference must fetch the corrected arm checkpoint and
  verify its result's canonical weight digest against the committed rebound
  report (41ab25238d7901653896daba1b5921a340824561893b581d2fe3aed9811bb7e8).

Commit `dac0f302b16e5eb5a34fff7a25b98f648afcc122` starts actual inference run
`37070427256`: exact 15,490-node refined surface -> freshly inferred qualified
Geppetto rig -> corrected V6 checkpoint inference -> frozen G3 and 51-frame court.
This is teacher-free inference, no training, no old-row transport. CPU FP32 and
four threads are explicit apparatus choices; original A100 BF16 parity is not
claimed. Geppetto uses predeclared seed 11; new rig retargeting uses the existing
rule and records its resulting mapping. The V6 readout class was copied unchanged
from sealed source SHA 39ff4f7a2051f0d9c1251a5999f472b9926e83deacbedcedc8e92eea424dce01
into an inference-only module; runtime verifies AST identity. Its chunking and
row/joint permutation test passed on the self-hosted runner.

At this checkpoint Geppetto inference and compiler structural qualification have
completed successfully; the corrected Arachne checkpoint is being retrieved. No
fresh skin result or dynamic PASS has been observed. Even a sampled-conditioning PASS still requires intersection,
source fidelity and appearance/presentation closure before any product claim.

## 2026-10-02 22:20 UTC fresh result and qualification scope correction

Run `37070427256` completed with artifact `11254498087`. Both inference lanes
finished, but **neither may be described as meeting the complete product standard**.
Rig structural qualification found an optimal 28-joint arborescence with no
blockers. This does not establish semantic joint placement or motion correctness.
Skin structural qualification covers all 15,490 rows, but its receipt explicitly
sets `product_skin_evidence_complete=false`, `row_confidence_available=false` and
`supervision_coverage=null`. Its downstream mechanical proof is mandatory and fails.

- New rig lineage: `a6b0ad63f688920eb138e9e4305508f63da3943e348b32b43b1a0e393484ae8c`.
- New skin lineage: `a76d0e6893d18fde2cf200985088689498fca3c078cbfc4d2247352c4ca19603`.
- Corrected checkpoint: `63e589b679461daf7cf859eb0c10e4b5f9c285b055d18f114d0ba348f2829892`.
- G3 FAIL, maximum condition 2179.9684077385773; 51/51 motion frames FAIL.
- Maximum edge ratio 355.8847115545404. Despite the lower extreme, unique edges
  above 4x increased from 1,257 to 1,995, and above 10x from 841 to 1,228. This is
  not evidence of general improvement. Intersection proof was not executed.

The fresh script normalizes FP32 softmax rows in float64 before submitting the
proposal. Recorded correction is max-row 1.6038227060315497e-7 and aggregate
1.833692925243228e-4. The latter exceeds the qualifier's 1e-4 aggregate budget,
but that budget currently measures only corrections to the *submitted*, already
normalized proposal (receipt aggregate ~4.494e-13). The historical sealed V6 source
also canonicalizes rows before qualification, so this comparison alone does not
establish a historical contract violation. It does establish that the qualifier
receipt is NOT an end-to-end raw-model-output correction certificate. Preserve
both measurements; do not silently label the complete raw-output path PASS.

Next diagnostic controls use the same corrected checkpoint on the original
surface: (1) existing rig plus fresh skin, including archived-weight replay delta;
(2) fresh rig plus fresh skin. These isolate inference-apparatus/precision effects
from refinement effects. They mint no product authority and do not weaken gates.

## 2026-10-02 22:31 UTC original-surface inference controls completed

Run `37072500679`, commit `874f89cadb3afc503de0ab24c6e5520533dc0c15`,
artifact `11255580761` completed both controls and exited 2 for measured scientific
failure, not an apparatus exception. Original surface has 12,090 nodes.

| Original-surface arm | G3 max condition | Failed motion frames | Maximum motion edge ratio |
|---|---:|---:|---:|
| Existing rig + archived skin (earlier baseline) | Not rerun in this job | 51/51 | 3.61553549 |
| Existing rig + fresh CPU FP32 skin | 32.28742663888593 | 51/51 | 18.14963526817917 |
| Fresh CPU rig + fresh CPU FP32 skin | 7458.209765794128 | 51/51 | 184.64133276429234 |

With original surface and qualified rig held fixed, fresh skin differs from the
archived weight field: row-L1 mean 0.004808228385607706, p95 0.02798091127904944,
maximum 0.6133617368875417; 7 of 12,090 dominant-joint assignments change.
The exact same corrected checkpoint is verified, but numerical replay parity is
therefore NOT established. The historical source runs backbone and readout under
CUDA BF16 autocast; this replay used CPU FP32. Precision, backend and inference
port differences are not separately isolated by this control.

Fresh rig on the original surface has lineage
`4d1dfb2c86faefcfd3e68000215cdc3eb591102e6805a97506b1b91a9aeaa0fe`.
Changing rig also changes the skin conditioning, predicted weights and derived
retarget mapping; this comparison is not a rig-only causal intervention. It does
show that new-surface topology is not necessary for the new inference path to
produce severe dynamic failure. Do not conclude that topology alone owns the
regression, or that retraining alone is required.

Both controls still report only structural inference qualification, with no
product authority. Full fresh rig/skin PASS is rejected. Next diagnostic priority
is to recover and verify the original inference numerical contract on the original
surface/rig before using this new inference path as a causal control for refinement.
Do not weaken thresholds, smooth weights, or alter authority artifacts to pass.

## 2026-10-03 continuation: constrained A100 budget

User can afford approximately two further A100 hours in Colab and requests notice
if refitting is necessary. Do not launch or prescribe a refit without resolving
the inference replay discrepancy first. CPU precision control holds original
surface, rig, checkpoint and decoder chunk fixed while crossing backbone/readout
FP32 and BF16 autocast. Archived skin is read only after predictions for evaluation.
Raw weights and diagnostic normalized weights are preserved; this experiment
performs no qualification and claims no CUDA BF16 equivalence or product PASS.
