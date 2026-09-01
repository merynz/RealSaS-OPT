# RealSaS — Investor Demo Task Board V1

**Objective:** single-specimen learned E2E architecture closure within the experimental demo branch.

Status vocabulary: `[x] done`, `[~] active`, `[ ] ready`, `[-] blocked`, `[!] human/input decision`.

## Lane 0 — governance / prereg

- [x] create experimental branch from canonical main
- [x] freeze claim and specimen firewalls in `README.md`
- [x] freeze numerical/structural acceptance contract before specimen and optimizer
- [x] record architecture base audit/readiness gate
- [ ] initialize run ledger / decision log / risk register

## Lane I — IRIS demo learned producer

- [~] audit current IRIS V2 external contract + reusable historical scaffolds
- [ ] implement generic 8-view image encoder and current-contract forward-depth/support/uncertainty heads
- [ ] implement fixed-camera observation packing -> `ObservationEvidenceIR`
- [ ] implement generic IRIS teacher target adapter for fit/eval only
- [ ] implement V1 IRIS acceptance metrics
- [ ] implement checkpoint save/load + provenance
- [ ] synthetic no-specimen schema/gradient smoke

Design constraint: no learned direct common-frame P authority; `P = O + dF` remains analytic.

## Lane G — Geppetto demo model

- [x] audit current Compiler `SkeletonProposalIR -> qualify_skeleton` seam
- [x] audit R6 clean-room functional obligations
- [ ] implement deterministic `GeppettoConditioningAdapter(RiggingSurfaceIR)`
- [ ] implement generic surface encoder
- [ ] implement generic bounded joint-query/existence/position/root heads
- [ ] implement generic directed parent-edge scoring head
- [ ] serialize output to existing `SkeletonProposalIR`
- [ ] implement teacher matching/evaluation metrics
- [ ] synthetic proposal -> existing Compiler qualification PASS
- [ ] checkpoint/provenance support

## Lane A — Arachne demo model

- [x] audit current Compiler `SkinProposalIR -> qualify_skin` seam
- [ ] implement generic training-only dense teacher target projection to admitted surface
- [ ] implement deterministic `ArachneConditioningAdapter(RiggingSurfaceIR, QualifiedSkeletonIR)`
- [ ] implement generic geometry+skeleton-conditioned influence predictor
- [ ] emit all admitted-surface rows as existing `SkinProposalIR`
- [ ] implement static skin metrics
- [ ] existing Compiler skin qualification synthetic PASS
- [ ] checkpoint/provenance support

For the demo, a direct influence-field predictor is permitted; a production `SkinFieldCodec` claim is not made by this branch.

## Lane M — editable deformation carrier

- [x] audit MWB-0/MWB-1 types and validation
- [ ] freeze branch-local generic view-local triangulation policy
- [ ] implement support-bound mesh candidate with no hidden geometry
- [ ] qualify/validate mesh through existing Compiler APIs where applicable
- [ ] implement generic qualified skin transfer to mesh
- [ ] assemble `CanonicalPuppetGraph.v2`

## Lane P — deformation / animation proof harness

- [x] audit current runtime boundary; native source is historical/external
- [ ] implement generic LBS deformation consumer over qualified mesh/skeleton/skin
- [ ] implement frozen generic pose probe set
- [ ] implement teacher deformation evaluator for fit/eval only
- [ ] implement generic preset demo clip(s) independent of specimen identity
- [ ] export frame sequence + video/GIF-ready artifacts
- [ ] bind a branch-local proof report to exact product/checkpoint/input hashes

## Lane F — final inference firewall

- [ ] implement fresh-process inference CLI
- [ ] permit only images + fixed cameras + checkpoints + generic config
- [ ] forbid all teacher/source-rig/source-mesh paths in inference schema
- [ ] destroy/isolate training objects before final run
- [ ] add static specimen-specific-code scanner
- [ ] record teacher reachability assertion in final run manifest

## Gate R — architecture readiness before specimen

- [-] complete every item in `DEMO_ARCHITECTURE_READINESS_V1.md`
- [-] seal `DEMO_ARCHITECTURE_READY_V1 = PASS`
- [-] only after PASS: open specimen selection

## Lane S — specimen selection (BLOCKED UNTIL READY)

- [-] freeze generic specimen envelope before seeing candidates
- [-] enumerate candidate data only after READY
- [-] select specimen without code modification
- [-] record selection rationale and source/license/provenance

## Lane T — explicitly authorized single-specimen fit

- [-] fit IRIS to frozen acceptance contract
- [-] fit Geppetto to frozen acceptance contract
- [-] fit Arachne to frozen acceptance contract
- [-] seal checkpoints and metrics

Optimizer steps in this lane are branch-local user-authorized experimental demo steps only. They do not change main authority.

## Lane E — teacher-firewalled E2E inference

- [-] launch fresh final-inference process
- [-] image-only IRIS PASS
- [-] surface compile PASS
- [-] Geppetto + Compiler skeleton PASS
- [-] Arachne + Compiler skin PASS
- [-] mesh/mesh-skin PASS
- [-] product assembly PASS
- [-] deformation/probe PASS
- [-] generic animation render PASS

## Lane V — investor artifacts

- [-] input 8-view panel
- [-] IRIS evidence/depth panel
- [-] Geppetto proposal + qualified skeleton panel
- [-] Arachne weights/deformation panel
- [-] editable puppet/deformation capture
- [-] animation capture
- [-] 60–150 second primary video
- [-] 20–30 second short cut
- [-] concise claim/firewall caption
- [-] delivery bundle + run manifest

## Current critical path

`generic IRIS -> generic Geppetto -> generic Arachne -> mesh/binding -> deformation/playback -> inference firewall -> ARCHITECTURE_READY -> specimen -> three fits -> fresh E2E -> investor capture`
