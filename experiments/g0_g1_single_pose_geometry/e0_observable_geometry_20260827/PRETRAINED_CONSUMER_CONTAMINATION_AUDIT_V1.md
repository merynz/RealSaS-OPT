# RealSaS — Pretrained Downstream Consumer Contamination Audit V1

**Date:** 2026-08-27  
**Status:** `POLICY_FROZEN__EXACT_EXTERNAL_MEMBERSHIP_PARTIALLY_UNAVAILABLE`

## Purpose

External pretrained auto-riggers are allowed only as downstream engineering/OOD diagnostics until training-set provenance is resolved asset-by-asset. They do not replace RealSaS family-disjoint scratch qualification.

## RigAnything

Public evidence states training used RigNet plus 9,686 curated high-quality rigged Objaverse shapes. The audited public inference release does not provide an authoritative list of those 9,686 Objaverse IDs.

Policy:

- RealSaS Objaverse-origin asset + RigAnything checkpoint => `PRETRAIN_CONTAMINATION_UNKNOWN` unless exact exclusion/inclusion can be proven.
- Unknown assets may be used for distribution-shift debugging.
- Unknown assets may not support a clean pretrained generalization claim.
- RigAnything checkpoint/code remains research-only under the Adobe Research License in the current RealSaS product registry.

## SkinTokens / TokenRig

Recommended checkpoint provenance is public: ArticulationXL 2.0 (70%), VRoid Hub (20%), ModelsResource (10%) plus GRPO refinement. Exact processed splits are not yet released as authoritative membership in the audited checkpoint release.

Policy:

- RealSaS ArticulationXL2 / VRoid-RigXL / ModelsResource-linked lineage => not pristine by default for TokenRig pretrained evaluation.
- Exact object-level exclusion may later promote an asset to `PRETRAIN_CLEAN_PROVEN` if authoritative external split membership becomes available.
- Other sources (for example independently sourced CC0 KayKit/Quaternius) are not automatically clean: source-identity or derivative overlap must still be audited before a clean pretrained claim.

## Effect on EXTERNAL_HOLDOUT

The RealSaS `EXTERNAL_HOLDOUT` seal is **not globally revoked**. It remains valid for RealSaS-native models that did not train on it.

When evaluating an external pretrained consumer, cleanliness becomes consumer-specific:

```text
RealSaS holdout clean
    !=
external-pretrain clean
```

Reports must therefore carry both statuses separately.
