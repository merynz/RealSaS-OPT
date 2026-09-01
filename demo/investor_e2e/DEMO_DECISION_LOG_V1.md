# RealSaS — Investor Demo Decision Log V1

Append-only decisions. A later change must add a new entry; earlier entries are not silently rewritten.

## D-001 — Separate experimental branch

Decision: investor demo work lives on `demo/investor-single-specimen-e2e`, based from main commit `bba22f25313b32aa4f8c2fbe418e3f2590198f38`.

Reason: main remains scientific/product continuation authority. Demo memorization fitting must not silently alter main optimizer/training authorization.

## D-002 — Architecture proof, not generalization proof

Decision: single-specimen memorization is explicitly allowed after architecture READY. Generalization claims are explicitly forbidden.

## D-003 — Specimen selected after model architecture

Decision: no real specimen may be selected, named or used before `DEMO_ARCHITECTURE_READY_V1 = PASS`.

Reason: prevents the implementation from being designed around the answer.

## D-004 — Learned parameters may memorize; source code may not

Allowed: optimizer-produced parameter values that memorize the selected specimen.

Forbidden: specimen ID branches, copied target coordinates, copied topology, copied weight tables, hardcoded output arrays, specimen-specific thresholds or cached predictions used as inference.

## D-005 — Source-freeze anti-hack firewall

At architecture READY, record the exact Git commit/tree containing all demo model/adapters/evaluator/firewall code. After specimen selection:

- changing learned parameters/checkpoints is allowed;
- changing generic data/config bindings is allowed only where the frozen schema explicitly permits it;
- changing model/adaptor/evaluator/inference source code invalidates READY and requires a new pre-specimen architecture audit before the demo can be claimed.

This source-freeze is the primary enforceable anti-specimen-hack control.

## D-006 — Preserve current Compiler authority

Geppetto emits `SkeletonProposalIR`; Compiler owns the final tree/root/canonical IDs.

Arachne emits `SkinProposalIR`; Compiler owns reference legality/simplex/bounded repair/final qualified skin.

The demo will not duplicate these authorities inside learned modules.

## D-007 — IRIS keeps analytic geometry boundary

IRIS predicts camera-forward depth/support/uncertainty. Common-frame `P` remains analytic under the fixed camera contract. Older direct-P/N controlled IRIS code may inform implementation patterns but is not the demo output authority.

## D-008 — Arachne demo may bypass a production codec claim

For the investor architecture proof, Arachne may be a direct generic geometry+skeleton-conditioned influence-field predictor. It must still be learned, generic, emit `SkinProposalIR`, and pass Compiler/deformation gates.

This does not claim the production SkinFieldCodec problem is solved or superseded on main.

## D-009 — Native runtime is not falsely claimed

Current main records native C++17 source as historical external byte authority rather than current executable source. The demo may first use a branch-local generic deformation/playback harness consuming the qualified product. It may not label that harness as a promoted native runtime.

## D-010 — Final inference firewall

Final inference runs in a fresh process with only canonical images, fixed camera contract, frozen checkpoints and generic code/config. Teacher/source-rig/source-weight/source-mesh paths are absent from the inference schema.
