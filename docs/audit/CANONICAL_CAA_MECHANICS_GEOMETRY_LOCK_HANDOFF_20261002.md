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
