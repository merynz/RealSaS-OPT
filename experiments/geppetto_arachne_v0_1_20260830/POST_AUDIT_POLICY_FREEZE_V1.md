# Geppetto/Arachne V0.1 — Post-Audit Policy Freeze V1

**Date:** 2026-08-30  
**Status:** `POLICY_FROZEN__C0_APPARATUS_NEXT__OPTIMIZER_NOT_AUTHORIZED`

This document consumes the complete preregistered FIT skeleton-projection audit and freezes the four policy decisions required before Geppetto/Arachne C0 training apparatus may be built.

## Audit authority

- audit status: `PASS_MEASUREMENT_COMPLETE__POLICY_FREEZE_NEXT`
- complete FIT Geppetto scope: `2914 / 2914` hard-contract PASS
- Arachne-capable subset: `2540`
- source commit: `2c2a8c1e361a1b80a34d18c42f1ffacf151f18e1`
- selection SHA-256: `af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9`
- audited asset-set SHA-256: `908609d8044a96d812bd0e2f0ce928c470c1fca86e47c26ba03be0a2fb372e99`

No model output was used to make these decisions.

## 1. Geppetto C0 control capacity / overflow

Observed deform-control distribution:

- min / p50 / p95 / p99 / max = `3 / 39 / 80 / 129.87 / 328`
- assets above 64 controls = `789 / 2914` (`27.08%`)
- assets above 128 controls = `35 / 2914` (`1.20%`)
- assets above 160 controls = `17 / 2914` (`0.583%`)
- all 17 `>160` assets are from `objaverse_animated_originals`

**Freeze:** `GEPPETTO_C0_MAX_CONTROLS = 160`.

Rationale: 64 would truncate a material fraction of valid FIT truth. Moving from 160 toward the observed maximum would force the quadratic all-pairs parent-evidence graph to be sized by a tiny extreme tail. The 160-control C0 profile covers `2897 / 2914` (`99.417%`) of the audited FIT population while retaining bounded quadratic cost.

**Overflow contract:** `ABSTAIN_NO_TRUNCATION`.

A projected skeleton with more than 160 deform controls is not silently clipped, subsampled, merged, or reinterpreted. It is structurally outside the clean C0 consumer profile and must be reported as overflow. This is a profile limitation, not a claim that the underlying asset is invalid or permanently unsupported.

## 2. Skeleton projection policy

Observed helper-chain evidence:

- assets with any helper skip: `11 / 2914`
- total projected deform-parent relations with one skipped helper: `14`
- maximum skipped-helper chain length: `1`
- exact zero-length deform bones: `0`
- near-zero deform bones at the frozen diagnostic threshold: `0`

**Freeze:** retain `NEAREST_DEFORM_ANCESTOR` exactly as implemented by `SkeletonTeacherProjectionV1`.

The audit does not support inventing a more complex helper-collapse rule. Source tails remain teacher/audit provenance and are forbidden production Arachne features.

Multi-root is a real corpus regime rather than an error:

- multi-root assets: `164 / 2914` (`5.63%`)
- projected roots p50 / p95 / max = `1 / 2 / 19`

**Freeze:** preserve the projected forest and supervise root evidence as a set/multi-hot teacher signal. Do not invent a synthetic teacher super-root. Compiler remains the authority for final product root/tree qualification.

If an exact zero-length deform bone appears under the frozen authority in a future corpus revision, fail closed and reopen the projection audit rather than silently repairing it.

## 3. Arachne helper/non-deform skin policy

Observed non-deform skin evidence:

- corpus non-deform mass fraction: `4.748351768e-08`
- assets with non-deform mass: `1 / 2540`
- affected asset: `asset_25d49db6bfdb3df4c32454bf`
- affected-asset non-deform mass fraction: `0.0005328706061275463`
- nearest-deform-ancestor counterfactual untransportable fraction: `0`

**Freeze:** `EXCLUDE_FROM_C0__NO_TRANSPORT__NO_RENORMALIZATION`.

The clean Arachne C0 teacher target uses only exact projected deform columns. No helper-column mass is moved to an ancestor, deleted and renormalized, or otherwise semantically rewritten. Any asset with positive non-deform skin mass is outside Arachne C0 until a separately preregistered transport semantics is justified.

This avoids adding a new target convention to solve a one-asset, corpus-negligible exception.

## 4. Clean C0 admission

Under the frozen policy:

- Geppetto C0 eligible: `2897 / 2914`
- Geppetto structural-overflow abstentions: `17`
- Arachne C0 eligible: `2527 / 2540`
- Arachne exclusions due to `>160` controls: `12`
- Arachne exclusions due to positive non-deform skin mass: `1`

Multi-root, one-hop helper-parent projection, and ordinary helper bones are **not** exclusion criteria.

## Authority / non-authority

Frozen here:

1. C0 control capacity and overflow behavior;
2. nearest-deform-ancestor teacher projection;
3. multi-root teacher preservation;
4. no Arachne helper-mass transport;
5. clean C0 structural admission rules.

Not frozen here:

- final product maximum joint count;
- Geppetto optimizer/hyperparameters/checkpoint;
- SkinFieldCodec capacity / FSQ levels;
- Arachne learned architecture/checkpoint;
- product skin sparsification/max-influence policy;
- IRIS admissible region under the future learned consumer profile.

## Next executable gate

Build, without optimizer steps:

1. deterministic Geppetto/Arachne C0 admission manifests from the audited FIT authority;
2. Geppetto C0 training/evaluation apparatus bound to `max_controls=160` and explicit overflow reporting;
3. SkinFieldCodec preregistration and reconstruction ceiling experiment on exact admitted Arachne C0 skin fields;
4. zero-step apparatus preflight.

Only after those pass may Geppetto C0 optimizer steps be authorized.
