# Geppetto/Arachne V0.1 — Skeleton Projection Corpus Audit Prereg V1

**Status:** `FROZEN_BEFORE_FIT_AUDIT_RESULT`  
**Scope:** read-only data-plane audit; no model training and no consumer prediction.

## Question

Before selecting Geppetto G0.1 capacity and before defining Arachne helper-column transport, what does the **FIT** master corpus actually contain after the already-coded `SkeletonTeacherProjectionV1` projection?

The master schema uses `FIT` rather than `TRAIN`; therefore “TRAIN skeleton-projection corpus audit” means the complete canonical `FIT ∧ capabilities.geppetto=true` population. Arachne-specific metrics are computed only on the nested `capabilities.arachne=true` subset.

## Frozen input authority

For each selected asset the only rig/skin teacher authority is:

`<root>/master/assets/<canonical_asset_id>/primary_geometry.npz`

Required exact array names:

- `bone_heads [B,3]`
- `bone_tails [B,3]`
- `parents [B]`
- `deform_mask [B]`
- `skin [P,B]` only when the selected asset is Arachne-capable.

No alias search, transpose, bone reindex, helper deletion, skin renormalization, mass transport, or source-specific reinterpretation is allowed in the audit. Any required-array or axis mismatch is a hard data-contract failure and remains visible in the result.

## Frozen skeleton measurements

`SkeletonTeacherProjectionV1` remains unchanged during this audit:

- one model target per deform bone;
- control location = source bone head;
- helper/non-deform bones are not targets;
- parent skips to nearest deform ancestor;
- multiple projected roots are preserved;
- cycles fail closed.

The audit records, per asset and in aggregate:

1. source bone count;
2. deform-control count;
3. skipped-helper count;
4. projected root count and multi-root incidence;
5. helper-chain hops skipped between each deform control and its nearest deform ancestor;
6. exact zero-length deform bones;
7. near-zero deform bones using the fixed **diagnostic-only** threshold  
   `1e-8 * max(skeleton_bbox_diag, 1)`.

The exact-zero statistic is the primary zero-length fact. The near-zero statistic cannot by itself trigger a projection-policy change.

## Frozen skin measurements

For Arachne-capable FIT assets, `skin` must have shape `[point,bone]` with `bone == len(deform_mask)`. Silent transpose is forbidden.

The audit records:

- total skin mass;
- mass in deform columns;
- mass in non-deform/helper columns;
- per-row non-deform mass fraction;
- non-deform columns carrying nonzero mass;
- for measurement only, how much non-deform mass has a nearest deform ancestor and how much has none.

Tiny negative numerical residue in `[-1e-8,0)` is reported and clipped only for nonnegative mass accounting. Any weight `< -1e-8`, non-finite value, or bone-axis mismatch is a hard failure.

**No helper-column mass is transported by this audit.** In particular, the nearest-deform-ancestor calculation is a counterfactual diagnostic, not Arachne truth.

## Leakage/firewall

The runner must not:

- read Geppetto/Arachne predictions;
- open DEV, VALIDATION, SEALED_QUAL, EXTERNAL_HOLDOUT, or historical sealed/test rows;
- mutate corpus evidence;
- rerender;
- train or tune any model;
- choose a checkpoint;
- alter `SkeletonTeacherProjectionV1`.

## Outcome semantics

- `FAIL_HARD_DATA_CONTRACT`: at least one selected asset cannot be audited under the frozen contract.
- `SMOKE_ONLY__NOT_FREEZEABLE`: a deliberately truncated `--max-assets` run.
- `PASS_MEASUREMENT_COMPLETE__POLICY_FREEZE_NEXT`: the complete selected FIT scope was measured with zero hard failures.

A PASS here is **not** a model-quality PASS and authorizes no training by itself.

## Next gate

After the full FIT result is sealed, and before seeing model outputs, freeze:

1. G0.1 `max_controls` / overflow-abstain policy;
2. whether the current nearest-deform-ancestor projection remains valid in the observed helper/multi-root/zero-length regime;
3. Arachne non-deform skin-column transport policy;
4. any exclusion/abstention rule needed for structurally unsupported FIT tails.

Only after that policy freeze do we proceed to Geppetto training apparatus and `SkinFieldCodec`.
