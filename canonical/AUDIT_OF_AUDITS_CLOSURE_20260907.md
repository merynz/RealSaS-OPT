# RealSaS — Audit of Audits Closure — 2026-09-07

**Intended bootstrap status:** `BOOTSTRAP_AUDIT_CLOSED`  
**Scope:** repository context/provenance reconstruction and continuity governance  
**FIT1 gate anchor:** `f6ce5dbc8719d6b6c592a4e060d8f1b38056b8ee`  
**First executable FIT base:** `de1a44cae1195dd9cbad3b23ef75d58ae80aa9b3`

## Closure meaning

This closes the **bootstrap audit-of-audits for context reconstruction**. It does **not** retroactively certify every historical experiment/report as scientifically valid or current.

The closure is honest because repository knowledge is divided into explicit classes:

1. current/important semantics explicitly indexed in the continuity spine;
2. every FIT1-descendant change recoverable through an exhaustive commit-level lineage;
3. high-signal artifacts that already existed at the FIT1 gate retained as explicit historical-provenance residuals;
4. any artifact fitting none of those classes is `UNEXPLAINED` and is a fail-closed continuity regression after closure.

Binding disposition: `canonical/AOA_ARTIFACT_DISPOSITION_V1.json`.

## What was added to make future context reconstruction deterministic

- `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md` — mandatory learned+deterministic subsystem responsibility expansion.
- `canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md` — scientific epoch map from the FIT1 gate to current state.
- `tools/render_fit1_commit_lineage.py` — exhaustive FIT1-descendant commit/provenance renderer across live refs.
- `canonical/FIT1_COMMIT_LINEAGE_V1.md/.json` — generated exact commit discovery view.
- ancestry-aware `tools/audit_context_coverage.py` — explicit post-closure fail-closed coverage classes.
- no-active-experiment-safe `tools/render_rehydration_packet.py` — fixes the stale assumption that a current gate/branch must always be active.
- local self-hosted `live_authority_map.yml` now generates the FIT1 lineage before census/coverage and commits all generated continuity views.
- `AGENTS.md` now forces subsystem ownership + FIT1 lineage into first-read context.

## Mandatory ownership conclusion

RealSaS scientific reasoning must not treat model names as isolated systems.

### IRIS

`observations/cameras -> learned signed/support evidence -> deterministic GSA/RiggingSurfaceIR assembly/provenance -> learned consumers`

IRIS does not own canonical skeleton IDs, final legal tree, canonical weights, or hidden character completion.

### Geppetto

`lossless RiggingSurfaceIR -> learned SkeletonProposalIR/control/root/parent/mechanical-salience evidence -> Compiler exact graph qualification -> QualifiedSkeletonIR/canonical IDs`

> **Geppetto is proposal, not canonical rig authority.**

Compiler may enforce legal graph constraints and fail close. Compiler may not silently repair Geppetto with geometry-only semantic deduplication, cardinality repair, missing deform-node synthesis, or mechanical-salience heuristics and then call the repaired output a Geppetto success.

### Arachne

`qualified surface+skeleton -> learned semantic skin/deformation proposal -> deterministic Compiler skin/mesh/reference/simplex qualification -> editable qualified deformation state`

Bounded numerical projection is allowed only under an explicitly assigned semantic owner. Compiler must not become a hidden second semantic skinning model.

Primary ownership sources:

- `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md`
- `canonical/DETERMINISTIC_DOWNSTREAM_LAYER_OWNERSHIP_OVERLAP_AUDIT_20260831.md`

## FIT1-to-current scientific conclusions preserved by AOA

### FIT1 scope

FIT1 is a controlled one-witness mechanism/capacity/product-science qualification gate. The architecture remains generic/generalization-oriented; Mage-specific architectural shortcuts are forbidden. FIT1 is **not** unseen-family generalization proof.

### IRIS

After FIT opened, IRIS underwent generic evidence-access strengthening rather than Mage-specific redesign. On 2026-09-03 the AOA lineage includes a real scientific invalidation: the old IRIS V2 learned input contract admitted privileged alpha/mask/view-slot shortcuts. Affected learned checkpoints/metrics remain quarantined for clean observation-only claims.

The later RGB-only/camera-relation/full-frame firewall repair closes the source seam prospectively; it does not retroactively cleanse the quarantined learned results.

The scene-first signed V3 line later clarified the ownership split: IRIS owns learned signed/observation evidence, while compaction, robust local geometry/provenance and `RiggingSurfaceIR` assembly live deterministically in GSA.

Authorities:

- `canonical/IRIS_LEAK_SCOPE_20260903.md`
- `canonical/IRIS_PRIVILEGED_INPUT_FIREWALL_REPAIR_V1_20260903.md`
- `canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md`

### Geppetto / RigAnything

AOA corrected an implemented-vs-executed continuity error. The authoritative Causal Repair V2 staircase is:

`R2 FAIL -> C1 full-surface cross-attention PASS -> STOP`

Therefore C2 conditional diffusion was **implemented but not authoritatively run in that staircase**. C3/C4 fuller RigAnything mechanisms exist in source but do not have an authoritative full-formulation closure.

Authority: `canonical/GEPPETTO_RIGANYTHING_LINEAGE_V1.md`.

### AR-01

Both AR0 and AR1 completed to 16,384. Neither satisfied the preregistered terminal 48-check stability gate, therefore categorical verdict is:

`AR01_NO_TERMINAL_CLOSURE`

- AR0 showed strong same-witness reachability/capacity and a long exact streak, but terminal stability failed.
- AR1 learned the teacher-forced structural task but failed badly free-running, giving a concrete generated-state exposure/recurrent-error-amplification failure signature.
- AR-01 was a minimal feedback isolation, **not** a full RigAnything C3/C4 verdict.
- no causal winner, Geppetto refreeze, generalization claim or next experiment is authorized by AR-01 alone.

Authority: `canonical/AR01_RESULT_20260907.md`.

### Arachne

Early Arachne/SkinFieldCodec Mage FIT artifacts remain scoped historical evidence. They are not current complete Arachne qualification because the authoritative upstream IRIS/GSA/Geppetto boundaries changed afterward.

Product-science order remains:

`Geppetto genuine closure/refreeze -> Arachne FIT1 on current qualified upstream outputs -> SkinTokens clean-room mechanism audit/adaptation -> deterministic skin/mesh/deformation qualification -> Mage idle/breathing witness`

## Branch disposition

- `main` alone is continuation authority.
- zero active experiment branches is currently valid.
- explicitly registered evidence branches remain evidence-only.
- all other non-main branches are `EVIDENCE_ONLY_UNREGISTERED` by safe default unless explicitly promoted/activated.
- AOA does not delete evidence branches automatically; deletion would be a separate provenance-aware cleanup transaction.

This prevents branch sprawl from becoming scientific authority sprawl even when many historical refs remain physically present.

## Closure non-claims

AOA closure does not mean:

- every old report was reread line-by-line and scientifically reaffirmed;
- every historical numerical metric is promoted into current state;
- every source-coded mechanism was executed;
- every branch is canonical or should be merged;
- FIT1 is complete;
- generalization is proven;
- Geppetto is refrozen;
- Arachne training is authorized;
- an active experiment exists.

For an exact historical claim, the future agent must still inspect the exact source/prereg/result/commit. AOA guarantees that the route to that evidence is deterministic rather than dependent on chat memory.

## Post-closure regression rule

After `canonical/BOOTSTRAP_COVERAGE_STATE_V1.json` becomes `BOOTSTRAP_AUDIT_CLOSED`:

- `tools/render_fit1_commit_lineage.py` must remain able to enumerate the FIT1-descendant context lineage;
- `tools/audit_context_coverage.py` must report **zero `UNEXPLAINED` high-signal artifacts**;
- `tools/render_rehydration_packet.py` must validate with zero or one active experiment as actually declared by current authority;
- local self-hosted continuity CI must fail if those invariants drift.

The goal of this closure is not maximal documentation. It is that a future chat can recover **what the system is, who owns each responsibility, what was actually tested, what was only implemented, what was invalidated, what remains open, and where the exact evidence lives** without the user rebuilding the context manually.
