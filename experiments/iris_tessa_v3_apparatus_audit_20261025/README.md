# RealSaS — IRIS/TESSA V3 Research Apparatus Audit

**Date:** 2026-10-25  
**Status:** `ACTIVE__NO_TRAINING_AUTHORIZED`  
**Parent canonical main:** `d97d0c51240905c07cc788a6be80b69f95aebb3e`  
**Scope:** IRIS VNext Evidence Model V3 + TESSA Structuralizer V3 + their current research/product bridges.

## Question

Before building causal Truth139 supervision or running FIT8 learning experiments, is the current IRIS/TESSA V3 apparatus structurally capable of carrying the intended hypothesis without a hidden representation, spatial-resolution, fusion-placement, relation-expressivity, objective, sampling, or sequence-memory bottleneck predetermining the outcome?

The hypothesis under audit is:

> geometry, mechanics and graphics evidence queried from one shared continuous scene can support role-free causal evidence; that evidence can improve TESSA structuralization under matched capacity; downstream Compiler consumers can then resolve legal canonical structures without a fixed semantic taxonomy.

This audit does **not** ask whether the hypothesis is true. It asks whether the apparatus gives the hypothesis a fair, falsifiable test.

## Inputs

Current source authority is the exact parent main SHA above. Initial audited modules include:

- `models/iris/vnext/inference_input_v1.py`
- `models/iris/vnext/contracts_v1.py`
- `models/iris/vnext/scene_core_v1.py`
- `models/iris/vnext/model_v3.py`
- `models/iris/vnext/evidence_heads_v3.py`
- `models/iris/vnext/heads_v3.py`
- `models/iris/vnext/presentation_evidence_v1.py`
- `models/iris/vnext/structure_v4.py`
- `models/tessa/v2/model_v2.py`
- `models/tessa/v2/objective_v2.py`
- `models/tessa/v3/config_v3.py`
- `models/tessa/v3/model_v3.py`
- current product inference bridges for IRIS/TESSA.

Historical DINO apparatus audits are evidence about prior failure modes, not authority for the V3 verdict. In particular, prior work already demonstrated that a supposedly fixed decoder/access path can itself determine a representation experiment outcome.

## Forbidden information / firewall

This phase is architecture/apparatus audit only.

Forbidden before a probe is preregistered:

- no FIT8 outcome inspection to tune the audited architecture;
- no Truth139 causal-label outcomes to tune thresholds;
- no product-success or Knight visual quality used as a substitute for accessibility proof;
- no semantic part names, material names, bone names, node names or source component IDs admitted as student inputs;
- no legacy model checkpoint relabeling;
- no model training or ModelRelease promotion authorized by this audit.

## Audit matrix

### A. Observation / representation access

**A1 — RGBA identifiability**  
Current admitted scene input is built from 1024 RGBA by white-compositing to RGB before frozen DINO-S extraction. Determine whether distinct authored RGBA observations can collapse to identical scene inputs while differing in alpha/source-support truth relevant to graphics.

**A2 — foundation spatial access**  
Record exact DINO patch grid, native source resolution, multiscale/earlier-layer access, and any learned native-detail path available to V3.

**A3 — shared-scene spatial ceiling**  
Record scene-query grid and triplane output resolution. Probe whether controlled structures below a preregistered scale remain distinguishable at H(x).

**A4 — camera authority**  
Confirm exact 13-D orthographic camera authority reaches scene construction and relation geometry without re-derivation from style pixels.

### B. Fusion / attention placement

**B1 — IRIS multi-view fusion placement**  
Characterize where view tokens first interact. Current candidate: camera/XY projection per token, flatten all views, then shared scene queries attend to the combined memory. Determine whether the apparatus has any source-token cross-view reconciliation before scene-query compression.

**B2 — TESSA stream fusion placement**  
Characterize whether scene, surface/mechanics and presentation evidence interact before or only after independent Perceiver-style stream compression.

If a later learned probe is required, compare preregistered matched-capacity variants only: current late fusion, a bounded mid-fusion variant, and a bounded early typed-fusion variant. No variant may receive more total conditioning capacity merely by construction.

### C. Relation expressivity

**C1 — static/causal pair geometry access**  
Determine whether role-free static/causal pair heads receive explicit relative geometry or must reconstruct all spatial relation from H alone.

**C2 — view-conditioned relation geometry**  
Verify exact deterministic pair geometry/camera features available to front/occlusion/boundary relations.

**C3 — complete-ray visibility**  
Verify whether complete-stack visibility is merely implemented in source or actually instantiated, trained, and wired into current IRIS/product paths.

### D. Truth/objective geometry

The causal Truth Foundry must retain privileged source-space response before reducing it to any student objective. Candidate truth families are not assumed to share one loss geometry.

Audit objective compatibility for:

- calibrated front probability;
- qualified occlusion/compositing response;
- appearance-support response;
- rigid-transform response;
- deformation trajectory/coherence;
- directed attachment/dependency;
- separation evidence;
- complete-stack visibility.

BCE is not presumed correct for all families. Continuous response, relative-ranking/contrastive, directional, trajectory and calibration objectives remain admissible candidates until a truth contract proves otherwise.

### E. Sampling / support coverage

**E1 — surface witness budget**  
Audit the 1024-witness default against thin/low-area/high-curvature structure coverage.

**E2 — extraction resolution**  
Audit training extraction resolution 24 separately from product inference resolution policy. Do not infer that product resolution repairs a training accessibility bottleneck.

**E3 — relation/ray sampling**  
No future causal relation experiment may use sampling that silently removes rare attachment, thin-support, grazing-angle or occlusion cases. Coverage must be measured by stratum.

### F. Optimization interference

For any multi-objective learned probe, report per-family gradient norms and pairwise gradient cosine statistics on the shared scene trunk. A shared representation failure must be distinguished from destructive optimization interference.

No gradient surgery or adaptive reweighting may be introduced from outcome inspection. Such a change is a new versioned experiment.

### G. TESSA sequence / memory apparatus

**G1 — conditioning compression**  
Record exact token budget and per-stream compression/refinement before decoder access.

**G2 — topology sequence length distribution**  
Before interpreting the 4096-token causal self-attention window, measure actual teacher topology token-length distributions on the intended training/evaluation corpus.

**G3 — position semantics**  
Audit modulo position-embedding behavior beyond the local attention window and distinguish architectural legality from empirical adequacy.

**G4 — objective / free-running court separation**  
Teacher-forced token loss does not prove free-running structural validity. Preserve independent decoded topology courts.

## Initial source facts already visible before probes

These are observations, not final verdicts:

1. Current IRIS V3 consumes frozen DINO-S patch tokens and exact 13-D cameras through a shared scene core.
2. The current RGBA frontend white-composites RGBA to RGB before DINO extraction; alpha is therefore not an independent foundation-token channel.
3. The admitted DINO-S token grid is 37x37 per view.
4. IRIS scene construction uses learned triplane queries and a Transformer decoder over the flattened multi-view token memory.
5. The V3 static relation path is H-pair based; deterministic point/camera geometry is added to the view-conditioned relation path.
6. RayStackVisibilityHeadV3 exists as a separate complete-ray model but is not instantiated by `IRISVNextEvidenceModelV3`.
7. TESSA V3 uses three independently compressed typed memory streams and concatenates them before the autoregressive decoder attends to the combined memory.
8. TESSA V3 reference capacity is 1024 d_model, 24 decoder layers and a matched 1024-token conditioning budget split 384/384/256.
9. TESSA causal self-attention uses a 4096-token local window while the configuration permits much larger topology outputs; adequacy is not inferred from configuration alone.

## Procedure

Phase 0 — static source contract audit (CPU only):

1. Extract machine-readable facts from exact source modules.
2. Fail on silent architecture drift from this preregistration.
3. Produce `static_contract_report.json`.

Phase 1 — deterministic/synthetic accessibility probes (CPU unless measured otherwise):

1. RGBA collision / alpha-identifiability probe.
2. spatial-frequency / thin-structure access probe;
3. pair-geometry expressivity probe;
4. TESSA stream-addressability/fusion probe using synthetic typed evidence;
5. sequence/window legality and corpus-length census where teacher sequences already exist.

Phase 2 — bounded learned apparatus probes (GPU only if Phase 0/1 shows they are necessary):

Any GPU experiment must be separately preregistered before execution. If the local GTX 1660 Ti is insufficient, execution moves to an explicit A100 notebook only after the user is informed. No result-dependent architecture sweep is allowed.

Phase 3 — readiness decision:

Only after the apparatus is considered fair may the causal Truth Foundry be built/trained against V3 and later FIT8 experiments begin.

## Outputs

Required audit outputs:

- `static_contract_report.json`
- `OPEN_QUESTIONS.md`
- one report per deterministic/synthetic probe;
- later, if needed, separately preregistered GPU experiment records;
- final `V3_APPARATUS_READINESS_DECISION.md`.

## Readiness rule

There is intentionally no single scalar PASS threshold for the whole apparatus yet. A readiness decision requires:

- no known observation collapse that makes a required target unidentifiable without an explicit experimental arm or repair;
- no unmeasured spatial bottleneck below the target structural scale;
- fusion placement either justified by deterministic evidence or covered by a matched-capacity ablation;
- every proposed causal target paired with a truth representation and an objective whose geometry matches that target;
- sampling coverage measured for rare structural strata;
- shared-trunk optimization interference measured for the multi-objective regime;
- TESSA sequence/memory limits measured against real target distributions;
- product/Compiler consumer incompleteness kept separate from representation-learning verdicts.

If any item is unresolved, status remains `NOT_READY_FOR_CAUSAL_TRUTH_TRAINING` rather than forcing a PASS/FAIL of the scientific hypothesis.

## Authority

A PASS of this audit authorizes only construction/execution of the next preregistered causal truth/model experiment. It does not authorize:

- product capability activation;
- ModelRelease promotion;
- FIT8 genericity claims;
- a claim that the shared-scene hypothesis is true;
- a claim that V3 is the final architecture.
