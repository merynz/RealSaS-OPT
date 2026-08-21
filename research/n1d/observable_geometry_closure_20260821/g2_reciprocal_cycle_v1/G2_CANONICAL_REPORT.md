# G2 Reciprocal Cycle — Canonical Decision

**Decision: PASS**

## Frozen question

On a separately frozen confirmation16 surface, does adding observation-only same-view B-to-A reciprocal descriptor cycle rank to frozen G1 descriptor+geometry proposal evidence improve proposal identity selection while preserving exact G1 geometry, exact V5 R/support, matching cardinality and solver semantics?

## Preregistered gates

- Overall hit delta: **+5.093 pp** vs required **+3.0 pp** — PASS.
- Mean regret ratio G2/G1: **0.863468** vs max **0.90** — PASS.
- G1-vs-cycle disagreement hit delta: **+8.474 pp** vs required **+8.0 pp** — PASS.
- Family mean-regret non-worse: **16/16** vs required **12/16** — PASS.

## Primary result

- Overall oracle-best hit: G1 **33.47%** (1255/3750) → G2 **38.56%** (1446/3750); **+191 hits**.
- Mean proposal regret: **1.5357px → 1.3261px**, a **13.65% reduction**.
- Median proposal regret: **1.1270px → 0.7523px**, a **33.25% reduction**.
- Disagreement stratum n=2372: hit **30.99% → 39.46%**; mean regret **1.6405px → 1.3270px**.

## Family guard

All 16/16 families reduced or preserved mean proposal regret. Smallest relative reduction was family 15284 (~5.48%); largest was family 16414 (~26.90%).

## Interpretation

The preregistered G2 formulation passes. Same-view reciprocal descriptor cycle is therefore promoted as an additional deterministic, observation-native frontend primitive on this confirmation16 surface.
This does **not** prove learned generalization of a new network component, does not authorize a weight sweep or same-surface retune, and does not by itself establish the complete 2.5D mechanical surface.

## Provenance

- Full-byte preflight SHA256: `61fb0290efadd783bcdc4d82513fc25807f1f8b0c5c9944abbf3631c6bf928e0`
- Pretruth freeze SHA256: `fac07740e4290bed18a93efabae3f6c6b2f302e4159676ea4f3d89eda5ad98ec`
- Pretruth persistence authority SHA256: `cb49eaa9214ac4538c6b005ee9e263249d13d1bf2ec69b3bfd26bd6810d951a7`
- Truth-open authority SHA256: `e8868f2499bf2eebfbea852b3658dbbcef9517e8fd663e33011ed12c942caa61`
- Truth evaluation SHA256: `81b3ad2f97d7695129fe3d0e6e2eb0e1424a205a2dbd7ce4d204f1b76d55e2a0`
- Decision SHA256: `6274a1b9e62846489e422eb444c56719826d93e00adf28d59817e41eeecd011a`
