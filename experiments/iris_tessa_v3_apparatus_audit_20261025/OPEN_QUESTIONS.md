# IRIS/TESSA V3 Apparatus Audit — Open Questions

**Status:** active; no training authorized.

This file separates proven apparatus facts from unresolved research questions. A source-level limitation is not promoted into a scientific conclusion without the appropriate probe.

## Closed / proof-level findings

### A1 — RGBA/source-support identifiability

**Status:** `PROVEN_BOUNDARY`.

Current Shape scene construction white-composites authored RGBA to RGB before frozen DINO-S. Distinct authored RGBA observations with different alpha can therefore map to identical RGB before the learned shared scene. The executable witness is in `rgba_identifiability_probe_v1.py`.

Consequence: alpha/source-support distinctions are not identifiable from current H alone for those observation equivalence classes.

This does **not** decide whether alpha should be learned by the shared scene or supplied as deterministic presentation authority. That scientific scope decision remains open.

### E1 — current surface-witness sampler has no component guarantee

**Status:** `PROVEN_BOUNDARY`.

`deterministic_surface_witness_v1` samples evenly in extracted vertex-array index, not by component, area, boundary cause, thinness, curvature, attachment/contact stratum or visibility stratum. `surface_witness_coverage_probe_v1.py` proves that a non-empty contiguous vertex subset can receive zero witnesses.

This is a guarantee failure, not a claim that current FIT8 subjects actually lose components.

### D1 — current V3 occlusion truth has no negative supervision

**Status:** `PROVEN_CURRENT_TRUTH_LIMIT`.

Outside tests, the current `GraphicsPairTruthRowV3` producer is `relation_truth_v3_adapter_v1.py`. Its valid occlusion support is positive-only and it rejects values other than 1.0, while `training_v3.py` uses validity-masked BCE for relation targets.

Consequence: current occlusion BCE cannot teach calibrated non-occluding negatives. Truth Foundry must supply qualified negative support and N/A/abstention semantics before occlusion learning is scientifically meaningful.

### G1 — TESSA reference local-window semantics measured

**Status:** `SEMANTICS_MEASURED__ADEQUACY_OPEN`.

Post-window self-attention is chunk-end anchored. With window=4096 and query_chunk=256, a full post-window chunk gives direct one-layer history from 3841 tokens at the first query through 4096 at the last. Incremental generation intentionally reproduces this reference behavior, so this is not classified as an implementation bug.

Adequacy remains open until real teacher topology sequence/dependency-span distributions are measured.

## High-priority open questions

### A2 — foundation feature access

Current V3 uses frozen DINO-S final normalized patch tokens only (`x_norm_patchtokens`) after whole-canvas 1024 -> 518 bicubic/antialiased RGB preprocessing. Grid is 37x37, patch size 14. No earlier-layer or multiscale DINO features enter the current scene contract.

Questions:

1. Are causal graphics/mechanics distinctions below this feature scale still recoverable through the shared scene apparatus?
2. Which failures are due to foundation access versus the later triplane/field route?
3. Does an RGBA/native residual path become necessary only for alpha/support, or also for thin local geometry?

Next probe: controlled spatial-frequency/thin-structure accessibility, with treatment variables changed one at a time.

### A3 — shared-scene spatial capacity

Reference V3 uses 32x32 learned queries per plane, then one stride-2 transpose convolution to a 64x64 triplane. H is sampled from this triplane and passed through the local field network.

This is not automatically a 64x64 hard semantic-resolution limit because channels and the nonlinear local field can encode subcell variation. Therefore no failure is claimed from the number alone.

Next probe must distinguish:

- information already lost before scene construction;
- information present in DINO tokens but inaccessible after scene-query compression;
- information present in triplanes but not expressible by the local H/field decoder.

### B1 — IRIS cross-view fusion placement

Current per-view tokens receive image projection + camera projection + XY projection, then all views are flattened. Learned scene queries attend to the combined memory. There is no source-token cross-view self-attention before scene-query compression.

Open question: is query-mediated fusion sufficient for ambiguous correspondence, symmetry, hidden-surface and interleaving cases?

Preferred first test: a tiny controlled correspondence task that isolates placement without changing representation width. Only if that is inconclusive should a reference-scale learned ablation be considered.

### B2 — TESSA cross-stream fusion placement

Current raw sources are approximately:

- scene: 3 x 64 x 64 = 12,288 tokens -> 384 memory tokens;
- surface/mechanics: 1,024 witnesses -> 384 memory tokens;
- presentation: variable token count -> 256 memory tokens.

Each stream is independently projected, Perceiver-style cross-attended into learned queries, and self-refined before the streams are concatenated. Scene/surface/presentation memory tokens do not directly cross-attend to one another before this compression; topology decoder state integrates the concatenated memories later.

Open question: does exact presentation-endpoint <-> surface-locus addressability survive independent compression strongly enough for topology structuralization?

Preferred sequence:

1. deterministic/addressability inspection;
2. tiny matched-capacity synthetic lookup/structural task on CPU;
3. only if needed, preregister reference-capacity late-vs-mid/early fusion A/B. Reference TESSA may require A100; user must be informed before execution.

### C1 — static causal relation geometry

Current static relation head receives H_a and H_b only. View-conditioned relations additionally receive explicit point-pair geometry and camera features.

H can in principle encode absolute geometry, so lack of explicit pair geometry is not an identifiability proof by itself.

Open question: for causal static targets such as rigid-transform support, deformation coherence and directed dependency, is forcing geometry to be reconstructed implicitly from H an avoidable optimization/sample-complexity bottleneck?

Next probe: matched synthetic relation task with and without explicit relative geometry, holding pair-network capacity fixed.

### C2 — complete-ray visibility wiring

`RayStackVisibilityHeadV3` exists, but current `IRISVNextEvidenceModelV3` does not instantiate it. Current product Shape also does not assemble non-null presentation evidence.

Need to decide before training whether global visibility is:

- learned complete-stack evidence;
- deterministic renderer/compiler authority;
- or a hybrid with explicitly separated authorities.

Do not train a head merely because a class exists.

### D2 — causal truth representation and objective geometry

Truth Foundry must preserve privileged response before producing model-specific targets.

Minimum candidate canonical response families:

- appearance response;
- rigid-transform response;
- deformation trajectory/kernel;
- directed intervention dependency;
- separation witness;
- front/depth relation;
- occlusion/compositing contribution;
- complete-stack visibility contribution.

Do not collapse them into one binary identity target.

For each family, decide separately whether the student objective should be calibrated probability/BCE, continuous regression, directional loss, ranking/contrastive loss, trajectory/kernel matching or another geometry. Any thresholded quotient should be derived only after coverage/calibration courts justify it.

### E2 — extraction resolution and product path

Reference config uses extraction resolution 24 and 1,024 witnesses. Current Shape worker uses the same `extraction_training_resolution` value during product inference; no separate coarse-to-fine inference policy is active in that path today.

Open question: what structural scale is representable/recoverable at resolution 24 for diverse morphology?

Need measurement on controlled synthetic geometry before changing the value.

### E3 — relation/ray sampling

Causal relation and visibility supervision will be highly imbalanced if pairs/rays are sampled naively.

Preregister coverage strata before Truth Foundry optimizer step 0, including at least:

- positive/negative/N/A;
- adjacency/non-adjacency;
- thin/low-area supports;
- boundaries by cause;
- attachment/contact;
- overlap/non-overlap;
- grazing-angle;
- alpha/compositing complexity;
- morphology/family.

### F1 — shared-trunk optimization interference

Once multiple truth families exist, measure gradient norms and pairwise cosine statistics on the shared scene trunk. Do not interpret poor learning as a representation failure if objectives systematically cancel one another.

Any adaptive weighting/gradient surgery must be a separately preregistered treatment, not an outcome-dependent repair.

### G2 — real topology sequence distribution

Teacher serialization preserves one global asset sequence and uses truncated training windows without arbitrary chart cuts. Need corpus-level statistics before changing TESSA attention:

- token length distribution;
- face/component/restart counts;
- dependency spans for component/chart closure and long-range repeated structure;
- fraction of scored tokens whose useful dependency exceeds 3841/4096 direct one-layer history;
- distribution relative to the 4096 position period.

This census is CPU/data work if the teacher mesh corpus is locally available; no A100 is required.

## GPU boundary

No current probe in this file requires A100.

Likely GPU-requiring experiments, if earlier probes justify them:

1. reference-scale IRIS representation/access ablation;
2. reference TESSA V3 late-vs-mid/early fusion matched-capacity A/B;
3. multi-objective shared-H gradient/interference measurement at realistic batch/scene scale;
4. later causal-truth IRIS training and TESSA presentation A/B.

Before any such external notebook/A100 execution, create a separate preregistration with exact treatment, control, data, seed, optimizer, budget and stopping rules and inform the user.
