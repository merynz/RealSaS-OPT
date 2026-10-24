# IRIS Frozen Semantic Provider V3 — PartCLIPSeg

Date: 2026-10-12
Status: research-only, non-authoritative

## Motivation

Knight V2 established that label-independent SAM2 localization is useful while generic CLIPSeg naming remains the dominant semantic error source.  V3 therefore keeps SAM2 local masks and replaces only the naming field with the official zero-shot PartCLIPSeg checkpoint.

## Invariants

- canonical IRIS geometry source/checkpoint unchanged;
- no geometry inference, refit, reskin or retarget in the semantic provider;
- no second visual mesh or geometry authority;
- external semantics remain evidence, not truth;
- runtime grip/order/secondary-motion behavior remains unchanged;
- Compiler qualification remains authoritative.

## V3 path

```text
source RGBA
  -> SAM2 segment-everything (local label-free masks)
  -> isolated official PartCLIPSeg dense part fields
  -> one semantic label OR abstain per local mask
  -> provider locality/collapse gates
  -> existing FrozenSemanticMaskEvidenceV1
  -> existing canonical-M fusion / relation diagnostics
```

PartCLIPSeg is isolated because its official repository carries a legacy Transformers fork.  The worker runs at pinned upstream commit `8abceefff960920019096982688e3e8209ed39de`, loads the official Pascal-Part-116 zero-shot checkpoint, records the checkpoint SHA-256 and checkpoint-to-model load coverage, and writes only dense probability fields to the current RealSaS environment.

## Naming qualification

`models/iris/semantic_part_naming_v1.py` requires:

- dominant class probability;
- top-vs-second class margin;
- local-region evidence stronger than the label's whole-subject baseline;
- part probability stronger than PartCLIPSeg background evidence.

Ambiguous regions abstain.  Labels supported by fewer than two source views are removed before provider evidence serialization by default.

## Experiment decision

V3 is an ablation of naming quality, not a product promotion.  Compare V2 and V3 on the same Knight source/M baseline:

1. false labels such as V2 `horn`;
2. hand / forearm / head naming;
3. cape / cloth naming;
4. sword / shield naming;
5. semantic owner fragmentation after M lift;
6. downstream grip candidate multiplicity;
7. cape/forearm source-overlap and pose-order diagnostics.

A better V3 provider should reduce false semantic identities and owner fragmentation without changing M or recorded motion.  Runtime consumers remain out of scope until semantic identity qualification is sufficiently clean.
