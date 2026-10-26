# IRIS/TESSA V3 Apparatus Audit — Decision Ledger

**Status:** active; no Truth139/FIT8 training authorized.

This ledger separates four states:

1. `PROVEN_REPAIR_REQUIRED` — source/mathematics makes the current contract invalid or insufficient; no learned A/B is needed to justify repair.
2. `MECHANISM_ESTABLISHED` / `TREATMENT_SELECTED_TINY` — a preregistered bounded court isolated the mechanism; real-data/reference-scale confirmation may still be required.
3. `OPEN_RESEARCH_QUESTION` — a matched court is still needed.
4. `DEFER_UNTIL_TRUTH` — the question cannot be answered honestly before causal truth batches exist.

No result in this file grants product authority or genericity claims.

---

## Closed / treatment-selected findings

### A1 — alpha/source-support information loss

**Status:** `PROVEN_REPAIR_REQUIRED`.

Current inference accepts authored RGBA, then white-composites it to RGB before frozen DINOv2-S/14. Distinct authored alpha/RGBA states can therefore become exactly the same scene observation before Z is constructed. The executable witness is `rgba_identifiability_probe_v1.py`.

This is stronger than an appearance-only limitation: a white opaque foreground locus and transparent white background can collapse to the same white RGB observation, so silhouette/geometry support can also be unidentifiable from the current RGB-only shared scene.

Required consequence before architecture freeze: alpha/support information must be preserved either in the shared perception input or through an equally authoritative deterministic route wherever geometry/graphics depend on it. The exact alpha feature contract remains an open treatment question; discarding alpha is no longer eligible.

### A2 — DINO feature access

**Status:** `TREATMENT_SELECTED_TINY`.

Exact frozen DINOv2-S/14 feature-access court found materially more local information at earlier transformer depth than in current final-only `x_norm_patchtokens`:

- L2 local recovery = 1.0 on 2/4/8/12 native-pixel strata;
- current final contract ≈0.832–0.895;
- L2 mean gain ≈+0.131.

The matched dual-level court then held token count and trainable image-adapter parameters exactly constant (197,120) and established:

- `CURRENT_FINAL_512`: mean local 0.864626, min 0.816327;
- `DUAL_L2_FINAL_MATCHED_512`: mean/min local 1.0/1.0;
- dual final-feature cosine mean/min 0.997598/0.996222.

Result records:

- `DINOV2_FEATURE_ACCESS_RESULT_V1.md`
- `DINOV2_DUAL_LEVEL_VIEW_TOKEN_RESULT_V1.md`

Selected direction: preserve the frozen DINOv2-S/14 foundation and expose aligned early-local + final-semantic patch information at the same 37x37 token lattice. Immediate DINOv3 adoption is not authorized by current evidence.

The exact production fusion form remains open; a by-construction final residual plus projected early-detail variant is still eligible.

### A3 — post-foundation 32->64 subcell routing

**Status:** `MECHANISM_ESTABLISHED`.

The preregistered post-foundation subcell-routing court established that a current-like learned query -> stride-2 carrier route can preserve distinct subcell states when those states are already present in source tokens. Therefore the 32-query -> 64-carrier mechanism is not currently the first suspect for the observed local-detail risk.

This does not prove unlimited spatial capacity; extraction and full triplane/Q capacity remain separate questions.

### B1 — source self-attention placement

**Status:** `MECHANISM_ESTABLISHED`.

`IRIS_CROSS_VIEW_FUSION_RESULT_V1.md` established five-seed perfect correspondence for both:

- current query-mediated multi-view fusion; and
- source-self-attention-first treatment.

Random-label firewall stayed at chance. Early source self-attention supplied no measurable gain on the controlled correspondence task.

Selected direction: retain query-mediated fusion as the low-compute baseline. Do not add O(source-token²) source self-attention without a harder court that specifically falsifies current placement.

Camera/backprojection feature geometry remains open and is not covered by this court.

### B2 — TESSA endpoint addressability

**Status:** `TREATMENT_SELECTED_TINY`.

Independent compression did not establish exact relation-endpoint access. Generic joint compression improved a broader lookup court but remained seed-sensitive. The stricter endpoint-binding court then isolated the mechanism:

- current late/unbound mean = 0.503223;
- generic joint/unbound mean = 0.507373;
- endpoint-bound late = 1.0 in all five seeds;
- random-label firewall = 0.506396;
- deterministic binder trainable parameters = 0.

Result: `TESSA_ENDPOINT_BINDING_RESULT_V1.md`.

Selected direction for endpoint relations:

1. preserve typed endpoint refs;
2. convert endpoint coordinates between declared domains explicitly;
3. query `Q(Z,x)` at each endpoint;
4. bind admissible endpoint-local H/mechanics/evidence before presentation compression;
5. let TESSA structuralize the already-addressable relation rather than rediscover the join.

Generic early/joint fusion is not currently justified as the endpoint solution.

### D1 — current occlusion pair truth

**Status:** `PROVEN_REPAIR_REQUIRED`.

Outside tests, the current V3 pair-truth producer supplies valid occlusion support only for positive same-ray cases. `training_v3.py` then uses validity-masked BCE.

Consequence: calibrated non-occluding negatives are not learnable from the current producer. Truth Foundry must provide qualified positive / negative / N-A / abstention support before a pair occlusion objective is meaningful.

### D2 — current ray-visibility truth ontology

**Status:** `PROVEN_REPAIR_REQUIRED`.

`RayStackVisibilityTruthV1.from_complete_depth_order()` encodes the first hit as 1 and all later hits as 0. This is an opaque first-hit ontology and is not sufficient for alpha compositing.

Truth Foundry must preserve complete-stack quantities such as incoming transmittance and per-hit compositing contribution before any model-specific target is derived. A canonical useful form is based on

`w_k = alpha_k * product_{j<k}(1-alpha_j)`.

Whether runtime visibility is learned evidence, deterministic renderer/compiler authority, or hybrid remains open; the current binary truth is not eligible as the final causal graphics truth.

### D3 — mechanical tangent identifiability

**Status:** `PROVEN_REPAIR_REQUIRED`.

Current `mechanical_demand_loss_v3` uses one mechanical validity mask for stretch, tangent, bend and rigidity. Principal tangent direction is mathematically undefined as stretch becomes isotropic (`major ~= minor`).

Required repair: tangent supervision needs its own validity/anisotropy gate or an equivalent continuous weighting rule. Arbitrary teacher orientation in isotropic regions must not become a target.

### E1 — surface-witness sampler

**Status:** `PROVEN_REPAIR_REQUIRED`.

`deterministic_surface_witness_v1` samples evenly in extracted vertex-array index. It is not geometry-order invariant and has no component/area/thin-support/curvature/boundary guarantee.

Historical Horse R48 forensic check (8,008 vertices / 15,608 faces) had 159 extracted connected components; the current 1,024 index sampler missed 42 components, representing about 1.07% of total extracted area. Many are likely extraction pepper/noise, so a naive one-witness-per-component rule is also rejected.

Selected next court: current index sampler versus deterministic geometry-native sampling, with vertex-permutation invariance, geometric coverage radius and qualified thin/low-area support coverage as primary metrics. Do not spend quota on arbitrary extraction fragments merely because they are separate components.

### G1 — TESSA local-window semantics

**Status:** `SEMANTICS_MEASURED__ADEQUACY_OPEN`.

With `local_attention_window=4096` and `query_chunk_size=256`, current chunk-end-anchored attention supplies approximately 3,841 direct historical tokens to the first token of a full post-window chunk and 4,096 to the last. Incremental inference intentionally reproduces the same semantics.

This is not classified as a learned failure. Exact per-query 4096 semantics can be repaired deterministically if desired; usefulness of >4096 dependencies remains a corpus question.

### H1 — TESSA/IRIS coordinate-domain coherence

**Status:** `PROVEN_REPAIR_REQUIRED`.

Current contracts mix coordinate systems inside one TESSA conditioning memory:

- IRIS field/query geometry uses approximately `[-1,1]`;
- extracted surface evidence is converted to TESSA `[-0.5,0.5]`;
- presentation endpoint evidence enforces `[-0.5,0.5]`;
- TESSA output tokenization uses `[-0.5,0.5]`;
- Bridge scene-position features currently use `linspace(-1,1,N)`.

Therefore the same physical locus can be represented at a 2x numerical scale difference between scene and surface/presentation streams.

There is also a cell-center convention mismatch: H samples the 64 carrier with `align_corners=False`, whose natural IRIS centers are `((i+0.5)*2/64 - 1)`, whereas Bridge scene positions use endpoint linspace.

Required deterministic repair before endpoint binding/reference experiments:

- typed coordinate domains;
- fine scene position in TESSA units: `0.5 * ((i+0.5)*2/N - 1)`;
- endpoint query conversion `x_iris = 2*x_tessa` before `Q(Z,x)`;
- coarse scene-query position convention aligned with the 2x carrier/cell-center geometry.

Do not ask TESSA to learn a unit conversion or half-cell offset from data.

---

## High-priority open questions

### A4 — alpha feature contract

**Status:** `OPEN_RESEARCH_QUESTION`.

Alpha must no longer be discarded, but the representation should not inflate token count unnecessarily. DINO input resolution is exactly `518 = 37 * 14`, so each existing DINO patch token aligns with a 14x14 alpha patch (196 observed alpha samples).

Candidate matched treatment: frozen dual-level DINO RGB token plus a patch-aligned alpha residual projection, preserving the 37x37 token lattice and splitting the existing 512-D scene-token budget among final semantics, early detail and alpha evidence.

Do not reduce alpha to patch mean without a court: distinct subpatch silhouettes can share the same mean.

### A5 — camera/backprojection feature geometry

**Status:** `OPEN_RESEARCH_QUESTION`.

Current ViewBank learns separate camera and token-XY embeddings and adds them to the image token. Query-mediated cross-view fusion is established, but the court did not test whether the model should have to rediscover orthographic backprojection.

For a pixel `(u,v)` and camera basis, the ray/basepoint contains multiplicative geometry such as `origin + u*h*right + v*h*up`.

Next court: matched current factorized camera+XY treatment versus deterministic orthographic ray/basepoint features, without changing source-token count or scene-query capacity.

### A6 — shared scene / Q oracle capacity

**Status:** `OPEN_RESEARCH_QUESTION`.

`LocalEvidenceFieldV1` receives triplane samples but does not receive query xyz explicitly. The current design may be sufficient because location is carried by the sampled scene field, but the representation substrate itself has not yet been isolated from the image frontend.

Next court should bypass the image encoder with oracle analytic scene fields and compare current triplane+Q against justified alternatives such as Q+explicit coordinates and/or multiresolution carrier under matched capacity. This is the cleanest way to test the mathematical substrate independently of perception.

### C1 — causal relation head algebra / geometry

**Status:** `OPEN_UNTIL_TRUTH_SCHEMA_STABILIZES`.

Current static head receives only H_a/H_b, while view head also receives deterministic pair/camera geometry. Current heads also mix different algebraic families in common sigmoid vectors.

Future truth families should determine architecture by algebra:

- symmetric support/coherence;
- directed attachment/dependency;
- complementary/tie-aware front order;
- symmetric projected boundary relation;
- ray-global compositing/visibility.

Do not add absolute geometry indiscriminately or keep one generic output head merely for convenience. Once causal quantities are fixed, run matched geometry/algebra courts.

### C2 — presentation schema V2

**Status:** `OPEN_UNTIL_TRUTH_SCHEMA_STABILIZES`.

`RoleFreePresentationEvidenceV1` currently represents many optional families as scalar [0,1] values. Causal deformation response, directed dependency and compositing truth may require richer continuous/vector or family-specific parameterizations.

Truth Foundry must keep privileged raw response richer than the model-facing presentation view. Do not force canonical causal truth into the current scalar-probability schema.

### E2 — extraction topology observability / resolution

**Status:** `OPEN_RESEARCH_QUESTION`.

Current product Shape uses extraction resolution 24. The pinned grid spans IRIS `[-1,1]`, giving spacing `2/24 = 0.083333` in IRIS units (`0.041667` in TESSA units). Field signs are evaluated on this base grid before extraction deformation parameters, so topology with no mixed-sign cube cannot be recovered downstream by FlexiCubes beta/alpha/gamma.

This makes resolution a surface-evidence observability parameter, not merely output mesh density.

Next court: analytic shapes/gaps/thin supports over 24/32/48/64, followed by a real normalized morphology-scale census. Use silhouette/local-thickness/gap statistics, not authored triangle edge length, as the real target scale.

### E3 — relation/ray sampling

**Status:** `OPEN_UNTIL_TRUTH_BUILDER_CONTRACT`.

Preregister coverage strata before Truth Foundry optimizer step 0, including positive/negative/N-A, adjacency, thin/low-area support, causal boundary family, attachment/contact, overlap, grazing angle, alpha/compositing complexity and morphology/family.

### F1 — shared-trunk optimization interference

**Status:** `DEFER_UNTIL_TRUTH`.

Once real truth families exist, measure shared-scene gradient norms and pairwise cosine statistics. Do not diagnose representation failure if objectives systematically cancel one another.

Any adaptive weighting, PCGrad-style treatment or other gradient surgery must be separately preregistered after a conflict is measured.

### G2 — real topology dependency span

**Status:** `OPEN_CPU_CENSUS`.

Historical FIT8 teacher-forced sequences are below the current 131,072 product decode cap (largest known George sequence ≈53,516 valid tokens), so raw decode budget is not presently the first blocker.

Remaining question: what useful dependency spans actually exceed the current 3,841–4,096 direct one-layer history? Measure component/chart closure and long-range repeated-structure dependencies before increasing the attention window.

### R1 — V3 training apparatus / geometry teacher

**Status:** `OPEN_RESEARCH_CONTRACT`.

V3 defines heads/objectives/checkpoint/inference surfaces but there is no current non-test end-to-end V3 optimizer orchestration that trains the full hypothesis. This is useful for the audit: the architecture can still be changed before outcome-driven patching.

Before real training, explicitly choose geometry teacher/objective. Do not silently inherit legacy silhouette-derived geometry if authored Truth Foundry geometry can provide a more appropriate full-scene supervision court.

---

## GPU boundary

No currently authorized court in this ledger requires A100.

Potential GPU-requiring stages only after CPU/proof closure:

1. reference-scale IRIS dual-level+alpha/camera treatment validation;
2. oracle/shared-scene capacity treatment if tiny courts remain ambiguous;
3. reference TESSA with bound causal presentation evidence;
4. realistic multi-objective gradient/interference measurement;
5. later Truth139/FIT8 learning.

Before external notebook/A100 execution, create a separate preregistration with treatment/control/data/seeds/optimizer/budget/stopping rules and inform the user.
