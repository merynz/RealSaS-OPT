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
