# RigAnything Code-Level Reference Audit V1 — Rigging Authority Erratum

**Date:** 2026-09-06  
**Applies to:** `canonical/RIGANYTHING_CODE_LEVEL_REFERENCE_AUDIT_V1.md`  
**Status:** `AUTHORITATIVE_ERRATUM`

The earlier audit correctly assigned canonical root/tree ownership to the RealSaS Compiler, but wording such as “selects the admitted root/parents” could be read as if the current canonical graph optimizer itself also performed node admission, duplicate fusion or bounded deform-node completion.

That interpretation is incorrect.

## Correct authority split

The visible `optimize_canonical_graph_v18_98` solver operates on the node set already present in `CanonicalGraphOptimizationRequest` (minus explicit forbidden-node constraints). In the current arborescence route it selects the global root and parent edges and proves a legal connected tree. It does not, by itself, infer which Geppetto proposal joints should exist.

As of the rigging-line closure patch the product route is explicitly:

```text
SkeletonProposalIR
  -> admit_skeleton_proposal_v1
  -> AdmittedSkeletonProposalIR
  -> CanonicalGraphOptimizationRequest
  -> optimize_canonical_graph_v18_98
  -> canonical joint IDs
  -> QualifiedSkeletonIRV2
```

Current product admission policy is deliberately conservative:

- all contract-valid native proposal joints are preserved;
- unsupported product joints fail closed;
- geometry-only duplicate fusion is forbidden;
- synthesized deform-node completion budget is zero.

Therefore native Geppetto STOP/cardinality remains a real product obligation. A Geppetto proposal with the wrong joint count is not automatically repaired by the canonical tree optimizer.

This erratum supersedes any sentence in the original audit that could imply otherwise. The original clean-room RigAnything functional decomposition and license firewall remain unchanged.
