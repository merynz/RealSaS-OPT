# RealSaS — Arachne Information-Preservation Audit V1 — Phase 3 Semantic Calibration

**Date:** 2026-09-10  
**Status:** `OPEN_RESEARCH_AUDIT__EVIDENCE_ONLY__NO_ARCHITECTURE_OR_TRAINING_CHANGE_AUTHORIZED`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`  
**Parent records:** Phase 1 primary audit + Phase 2 evidence census.  
**Continuation authority remains:** `main/CURRENT_STATE.md`.  
**Current A0 token-capacity experiment remains untouched.**

This phase asks a stricter question than “does a tensor exist?”:

> **Does the tensor have a scientifically established semantic meaning that justifies preserving/exposing it to a downstream learned consumer?**

The answer materially changes two Phase-1/2 provisional concerns. Some apparent information loss is real; some apparent “evidence” is not actually calibrated evidence and therefore should not be propagated by habit.

---

## 1. IRIS V3 `log_uncertainty`: public name, but not calibrated uncertainty in the promoted Mage lineage

### 1.1 Source fact

Current `models/iris/v3/scene_first_signed_v3.py` exposes two named field outputs:

- `sdf`;
- `log_uncertainty`.

The zero-surface decoder subsequently keeps only `sdf`, so Phase 2 correctly localized a source-level drop at the signed-grid/zero-surface interface.

### 1.2 Training-lineage forensic

The actual promoted V3/Signed-Mesh checkpoint is:

- Drive: `SIGNED_CHECKPOINT.pt`, file ID `1015IJL4sB8zPGYOO-QGdDZcTcoLRad6v`;
- SHA-256: `766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2`;
- checkpoint schema: `RealSaS.Cleanroom.H1.SignedMesh.Checkpoint.v2`;
- step: `512`;
- warm start: H1 V1 run `20260904T214354Z`.

The V2 notebook explicitly rebuilds the V1 scene-first network, replaces only the old unsigned-distance head with a signed scalar head, and warm-starts **everything except `field.dist.*`**. Therefore `field.unc.*` is inherited unchanged at initialization.

The V2 training loop then does:

`ss,_ = MODEL.query(...)`, `sb,_ = ...`, `sh,_ = ...`

and optimizes only:

`SmoothL1(signed field, signed target) + .25*background_margin + .50*hull_margin`.

No uncertainty term appears in the V2 objective.

Checkpoint-state forensic independently confirms this:

- serialized model state has 103 parameter tensors;
- optimizer param group contains 103 parameter IDs;
- optimizer state exists only for IDs `0..100`;
- IDs `101` and `102` have no Adam state;
- positional tensor-shape matching for IDs `0..100` is exact;
- the last two model parameters are `field.unc.weight` and `field.unc.bias`;
- `field.unc.weight` and `field.unc.bias` are **bitwise identical** between the V1 warm-start checkpoint and the promoted V2 signed checkpoint.

Therefore the V2 signed-field phase supplied **zero optimizer updates** to the uncertainty head.

### 1.3 What V1 trained the head to do

The preceding V1 scene-first notebook does use the auxiliary `unc` output, but not as an uncertainty likelihood or calibrated error predictor. Its loss is:

`reg = .001 * (us.square().mean() + ub.square().mean())`

and total:

`loss = local_UDF_loss + .5*background_loss + reg`.

Thus the only direct training pressure on `unc` in V1 is a small **zero-centering L2 regularizer**. There is no target such as residual magnitude, variance NLL, calibration bin, coverage objective, or probabilistic scoring rule. The V1 history/result records no uncertainty calibration metric.

### 1.4 Revised verdict F-01

The prior label `UNKNOWN_CALIBRATION` can now be narrowed:

`IRIS_V3_log_uncertainty = LEGACY_AUXILIARY_SCALAR__NOT_CALIBRATED_AS_UNCERTAINTY_IN_PROMOTED_MAGE_LINEAGE`.

Consequences:

- the zero-surface boundary does indeed discard the scalar;
- **but there is currently no scientific reason to classify that discard as loss of useful learned uncertainty evidence**;
- wiring it into GSA/Arachne now would propagate a semantically overnamed channel rather than preserve a proven information source;
- canonical prose that describes the current V3 head as established “uncertainty evidence” is stronger than the actual training evidence and should be reconciled at audit closure.

Current severity is therefore changed from “possible downstream evidence loss” to:

- `P1 GOVERNANCE / CONTRACT-SEMANITCS DRIFT`;
- `P2 OR LOWER for downstream feature loss` unless a future properly supervised/calibrated uncertainty head is introduced.

No repair is authorized by this finding. The correct short-term action is **do not consume this scalar as uncertainty**.

---

## 2. Geppetto evidence census: distinguish calibrated/owned evidence from tempting but weak side channels

Promoted `GeppettoReferenceStrengthRawOutputV1` exposes many tensors before proposal/Compiler boundaries:

- positions/coarse positions;
- stop/existence/root/salience logits;
- support-presence + dense support logits;
- internal-parent + all-pair-parent logits;
- position log-sigma;
- surface attention;
- diffusion loss/training flags.

Not all of these deserve downstream preservation.

### 2.1 Dense support logits -> top-8 support anchors

Training contract:

- `support_target_k = 8`;
- for each teacher mechanical locus, target support indices are the **8 nearest compact surface nodes** in normalized geometric space;
- support-index loss maximizes probability on that nearest-8 set;
- proposal generation then keeps the top `support_topk` IDs if support presence is admitted.

Therefore full `support_logits[joint, surface]` are learned **mechanical-locus proximity/support evidence**, not skin ownership probabilities.

The proposal boundary compresses this dense score field to top-8 IDs. That is a real information reduction, but it must not be automatically “repaired” by exposing all raw logits to Arachne. Their calibration as a transferable soft field has never been established. Correct status:

`DENSE_SUPPORT_LOGITS -> COMPRESSED_TO_SPARSE_TOP8__POTENTIALLY_USEFUL_BUT_UNCALIBRATED_FOR_A1`.

If tested later, use as soft positive mechanical evidence only. Never a hard skin mask.

### 2.2 Mechanical salience is not a useful calibrated magnitude on current FIT1

The training loss explicitly states that every projected target control survived the mechanical-core rule, so **every target salience label is positive**. The source comment explicitly says this does not establish calibrated salience magnitude or generalization.

Exact final Mage proposal values confirm saturation:

- min salience probability ≈ `0.9995263`;
- median ≈ `0.9999938`;
- max ≈ `0.9999996`;
- std ≈ `9.78e-05`.

Thus the earlier Phase-1 idea that `mechanical_salience_probability` is an obviously valuable Arachne side channel is downgraded. On current evidence it is nearly constant and non-discriminative.

Verdict:

`MECHANICAL_SALIENCE = PROPOSAL_EVIDENCE_EXISTS__FIT1_MAGNITUDE_NOT_CALIBRATED__DO_NOT_PRIORITIZE_FOR_A1`.

### 2.3 Position sigma has a stronger semantic basis, but still needs calibration

Unlike salience, `position_log_sigma` participates in an explicit heteroscedastic position-NLL term during Geppetto training. Final proposal metadata retains `position_sigma_normalized` until Compiler qualification.

Exact final Mage joint-mean sigma range:

- min ≈ `0.0022492`;
- median ≈ `0.0042210`;
- max ≈ `0.0171970`.

This is non-constant and has a training objective tied to locus error. It is therefore a more plausible downstream confidence signal than salience. But FIT1 does not prove cross-character calibration. Status:

`POSITION_SIGMA = SEMANTICALLY_TRAINED__CALIBRATION/GENERALIZATION_UNKNOWN__P1_CANDIDATE_FOR_QUALIFIED_EVIDENCE_SIDECAR_AFTER_AUDIT`.

### 2.4 Joint confidence is narrow on Mage

Proposal `confidence = existence_probability * exp(-mean_position_sigma)`.

Exact final Mage distribution:

- min ≈ `0.9826607`;
- median ≈ `0.9957801`;
- max ≈ `0.9977526`;
- std ≈ `0.0029024`.

It is legal proposal evidence and Compiler consumes it for graph optimization, but current FIT1 shows little dynamic range. Downstream utility is not established independently of sigma/existence.

### 2.5 Root and parent evidence are authority-sensitive

Root score is sharply resolved on Mage: max ≈ `0.9999355` for `P:GRS:0000`, while the median non-root score is near zero. Compiler is the canonical root/tree owner and must remain so.

For the 21 selected parent edges in the exact final qualified Mage graph:

- each Compiler-selected parent is also the local highest proposal score for that child on this witness;
- however proposal-score margins against the best alternative are often small;
- minimum selected-vs-best-alternative margin ≈ `2.384e-7`;
- median ≈ `0.00291425`;
- `6/21` margins are `<1e-4`;
- `8/21` are `<1e-3`;
- `12/21` are `<1e-2`.

This does **not** invalidate the exact Compiler solution. It does show why raw proposal alternatives and canonical authority must not be conflated.

If Arachne ever benefits from “structural certainty”, the preferred owner-clean representation is not raw all-pair logits. It should be a **Compiler-qualified structural confidence/margin sidecar** derived from the accepted solve, leaving the selected tree immutable.

### 2.6 Model-private tensors remain private by default

`control_states`, internal recurrence parent logits, raw surface attention, and diffusion training loss are not public semantic evidence merely because they exist. They stay `MODEL_PRIVATE_LATENT` / training-only unless separately promoted by a scientific contract.

---

## 3. Old A1 compression is more severe than the nominal 20D/8D widths suggest

Phase 1 already proved structural compression. Exact current Mage cache statistics now show several nominal channels collapse or duplicate.

### 3.1 Joint feature support scalar is exactly constant

All 22 qualified Mage joints have 8 sparse support anchors. Old A1 computes:

`support_count = min(len(support_surface_ids),16)/16`.

Therefore every joint receives exactly:

`support_count = 8/16 = 0.5`.

The full spatial mapping of 176 joint→surface support references / 155 unique anchor nodes is reduced to a scalar that is **identical for every joint**.

This is stronger than “compressed”: on the current Mage witness the old joint support feature has **zero discriminative information**.

Other exact joint-feature facts in the current cache:

- columns 0..2 normalized joint position: variable;
- root bit: variable;
- parent-present bit: variable;
- support-count column: one unique value `0.5`;
- depth: 8 unique values;
- trailing constant: one unique value `1.0`.

### 3.2 Surface summary contains redundancy/constants

For the current 950-node Mage cache:

- `normal_valid` is exactly `1.0` for every node;
- `mean_relation_score` is exactly `1.0` for every node;
- `raster_count` is **bitwise identical** to `support_count` for every node under the current scene-first contract;
- all exact per-view raster XY values are absent from the 20D summary;
- exact GSA edge identity/edge metadata are absent; only normalized degree + mean score remain.

Thus the nominal 20D width overstates effective information content on this witness.

### 3.3 Pair geometry remains genuinely useful structure

`pair_geometry` is `[950,22,10]` and `pair_mask` is fully active for all 20,900 point-joint pairs on Mage. This explicit point/control/parent-segment relation is a valid part of the old A1 lineage and should remain a candidate inductive bias in the V7-native design.

### 3.4 Revised P0 statement

The P0 finding is now stronger and more precise:

> The old A1 does not merely use a compact representation. It discards exact legal graph/raster/support mappings and, on Mage, several retained summary channels are constant or duplicate. The future V7-native A1 consumer boundary must be re-established from the full legal `S + Qualified G` substrate rather than by extending the old 20D/8D cache schema.

This still does **not** prove which rich channels improve predictive performance. Feature admission remains an ablation question.

---

## 4. Evidence-preservation rule refined

The audit now distinguishes three cases that looked similar at Phase 1:

### Case A — proven useful legal evidence is compressed

Example: exact GSA edge/raster mapping and joint→surface support-anchor identity are available upstream but compressed by old A1.

Action: preserve at the consumer boundary; ablate neural use.

### Case B — a named “evidence” head exists but lacks the claimed semantics

Example: current IRIS V3 `log_uncertainty` lineage.

Action: do **not** propagate merely to avoid “information loss”; first fix/reclassify the semantic contract.

### Case C — evidence was consumed by the canonical owner

Example: Geppetto root/parent alternatives consumed by Compiler.

Action: downstream gets canonical state. If uncertainty about the accepted state is product-useful, expose an owner-qualified confidence sidecar rather than the raw alternatives as competing authority.

Compact rule:

> **Preserve semantics, not tensors. Preserve legal evidence, not arbitrary activations. Qualification may consume decision evidence, but must not silently erase downstream-relevant confidence if that confidence has a calibrated contract.**

---

## 5. Arachne design implications — still no implementation authorization

This phase changes the planned A1 inventory in useful ways:

- `IRIS log_uncertainty`: **exclude from baseline input** under current lineage; not calibrated uncertainty.
- exact per-view raster/support and GSA graph: retain as legal rich S boundary candidates.
- joint→surface support IDs: retain exact sparse mapping, but as sparse mechanical evidence, never hard/exhaustive ownership.
- Geppetto salience magnitude: low priority / do not assume useful.
- Geppetto position sigma: calibration audit candidate; possible qualified evidence sidecar.
- raw parent/root alternatives: do not expose as competing authority.
- Compiler-qualified structural confidence/margin: possible future sidecar if an Arachne ablation justifies it.
- old 10D point↔segment geometry: preserve as candidate inductive bias.

No current A0 gate, A0 token-count experiment, A1 authorization, or product claim changes.

---

## 6. Next audit actions

1. Complete an explicit field-by-field `S + Qualified G -> old A1 -> proposed V7-native boundary` matrix with legal/semantic/constant/duplicate status.
2. Specify a **no-training** candidate `QualifiedSkeletonEvidence` sidecar contract, but do not implement it; include only fields whose owner/semantic calibration is defensible.
3. Recover Mage staff/shield rigid-attachment classification separately from the Arachne A0 path.
4. After current A0 token arms finish, run regional row-error diagnostics and latent-interface audit on frozen artifacts.
5. At audit closure, reconcile canonical text that currently overstates V3 uncertainty semantics.

**No implementation, optimizer run, or scientific promotion is authorized by this Phase-3 record.**