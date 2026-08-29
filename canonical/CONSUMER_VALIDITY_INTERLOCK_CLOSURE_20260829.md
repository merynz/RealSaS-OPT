# RealSaS — Consumer Validity Interlock Closure — 2026-08-29

**Status:** `PASS__SIX_OF_SIX_CONSUMER_INTERLOCK_CLOSED__SACRIFICIAL_PROFILE_ONLY`

## Question closed

Can the current RealSaS observable 2.5D substrate be consumed end to end by a minimal skeleton/skin route through the exact current Compiler, and is the final deformation response genuinely coupled to skeleton geometry rather than being silently hidden by the skin consumer?

For the frozen sacrificial profile `ClosedRiggingVolume V0 / InteriorRiggingSubstrate V0 -> G0 -> Compiler -> A0 -> Compiler -> deformation/proof`, the answer is **yes**.

This is a consumer-validity result, not a product-quality Geppetto/Arachne claim.

## Original six-step interlock

The preregistered `CONSUMER_VALIDITY_INTERLOCK_V1.md` required:

1. deterministic `ClosedRiggingVolume V0` / `InteriorRiggingSubstrate V0`;
2. minimal G0 consuming the actual substrate;
3. Compiler skeleton qualification;
4. minimal A0 consuming the qualified skeleton and surface evidence;
5. Compiler skin qualification + deformation/proof replay on clean substrate;
6. explicit coupling probes preventing A0 from silently masking a systematically bad G0 skeleton.

All six are now evidenced on the branch `consumer-interlock-v0-20260829`.

## Steps 1–5 — clean exact-Compiler route

Frozen real witness:

- asset: `asset_551ea351b43a1787d0f55536`
- surface nodes: `512`
- G0 joints / edges: `5 / 4`
- qualified joints: `5`
- qualified skin rows: `512 / 512`
- Compiler graph solver: `chu_liu_edmonds_maximum_spanning_arborescence`
- optimizer status: `optimal_arborescence_super_root`
- optimizer optimality proven: `true`
- proposal IDs disjoint from Compiler canonical IDs: `true`
- all product joint IDs minted as `J:*`: `true`
- skin max simplex residual before bounded repair: `1.341104507446289e-07`
- clean deformation finite / nontrivial / bounded: `true / true / true`
- clean proof PASS: `true`
- proof and runtime bound to exact product state: `true`

Clean result artifact already recorded as:

`experiments/consumer_interlock_20260829/CONSUMER_INTERLOCK_EXACT_COMPILER_CLEAN_RESULT_V1.json`

The exact historical Compiler vendor raw SHA remains:

`3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850`

On coupling-code head `bb7daa3bcb66571642900dd246aab8550fcb235d`, clean workflow run `33241894722` completed successfully.

## Step 6 — explicit G0/A0 coupling probe

The coupling test was preregistered before outcome in:

`experiments/consumer_interlock_20260829/CONSUMER_COUPLING_PROBE_PREREG_V1.md`

Frozen corruption:

`p_bad = root + 0.25 * (p_clean - root)` for every non-root G0 joint.

Only joint positions changed. Proposal IDs, graph edges, scores, confidence, surface supports, and hard-required topology were preserved.

Measured corruption magnitude:

- joint-position RMS: `0.31006343780462375`
- normalized by surface bbox diagonal: `0.21196194321606632`

Thus the corruption was materially larger than the preregistered minimum `0.05`.

Both corrupted routes still passed structural qualification:

- corrupted skeleton: `5 / 5` qualified joints;
- corrupted recomputed A0: `512 / 512` qualified rows;
- corrupted frozen-clean-weight diagnostic route: `512 / 512` qualified rows.

### Primary coupling result

After allowing A0 to fully recompute on the bad skeleton, final posed surface deviation from the clean route was:

- normalized posed-coordinate RMSE: `0.015168392024714485`
- normalized P95 point delta: `0.02837774072761807`
- mean point delta: `0.019208901494996912`
- max point delta: `0.04725191887231861`

Both preregistered detectability floors passed:

- `POST_COMP_RMSE_NORM >= 0.005`: PASS;
- `POST_COMP_P95_NORM >= 0.010`: PASS.

A0 did materially react to the corrupted skeleton:

- mean per-row skin L1 change: `0.2687926005329492`
- P95 per-row skin L1 change: `0.5763698374343423`.

The frozen-clean-weight route had normalized RMSE `0.010852395661949734`; the diagnostic compensation ratio was therefore `1.3976998717340667`.

Interpretation: on this severe corruption A0 did **not** erase the bad-skeleton consequence. Recomputing A0 weights actually increased, rather than hid, the final response deviation. The ratio is diagnostic only and was not a selection threshold.

The coupling proof was bound to exact corrupted product state:

`7412bc4d6fa9bdd7d272396542d63b65067034cab3026986af56b95647c19f92`

Coupling workflow run:

- run ID: `33241894766`
- code/prereg head: `bb7daa3bcb66571642900dd246aab8550fcb235d`
- conclusion: `success`
- workflow artifact digest: `sha256:2d5f099dbba06a8518aaec3a9082926f3ea4d35cf0209bc0e1067fc79ba00455`
- committed result JSON SHA-256: `39a3f42e650e60e6bcb14f0f368c0964d5c8bfc30a069a3468ce389c1f9f845f`

Result:

`experiments/consumer_interlock_20260829/CONSUMER_INTERLOCK_COUPLING_PROBE_RESULT_V1.json`

## Scientific consequence

The cheap consumer-validity question is closed for the sacrificial G0/A0 profile:

- the observable substrate reaches real downstream consumers;
- the exact Compiler accepts and canonicalizes their outputs;
- skin and deformation execute on the resulting product state;
- final behavior is measurably sensitive to systematic skeleton-geometry corruption even after A0 adapts.

Therefore the next scientific question is no longer whether the current substrate has any viable downstream consumer route. The next controlled question is representation accessibility / geometry precision under a fixed learner, beginning with the preregistered DINOv2 frozen-representation ladder after governance synchronization.

## Scope boundary

This closure does **not** claim:

- G0 is product-quality Geppetto;
- A0 is product-quality Arachne;
- the current IRIS learner is product-ready;
- the current depth tolerance object automatically transfers to future G1/A1 consumers;
- training is authorized merely by this file.

Product-level tolerance remains consumer-profile-specific and must be replayed when Geppetto/Arachne materially change.

## Continuation

Before opening DINOv2 candidate training:

1. promote this closure through the canonical branch process;
2. synchronize `CURRENT_STATE.md` with the already-closed FIT_PROXY32 audit and this six-step interlock;
3. retain one human-readable roadmap stating: depth tolerance closed -> old learner residual audit closed -> consumer interlock closed -> controlled DINOv2 S/B/L/g representation ladder next;
4. preregister the controlled ladder with matched family/sample exposure and identical post-extractor trainable capacity;
5. only then execute training/evaluation.
