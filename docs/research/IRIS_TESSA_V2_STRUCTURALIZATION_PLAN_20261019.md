# IRIS → TESSA V2 Structuralization Plan — 2026-10-19

Status: **SOURCE IMPLEMENTED / DEDICATED GATE GREEN / NO TRAINING AUTHORIZATION**

Branch: `research/iris-vnext-hybrid-tessa-v2-20261019`

Base source: `f78b6fb647bd6e514b7fbe5e8b5026e06c9d5c5b`

## 1. Decision

TESSA V2 is not a second surface-reconstruction model and it is not merely a small IRIS head.

The preferred split is:

```text
8-view source observations
        ↓
IRIS shared continuous scene carrier H / triplanes
        ↓
high-fidelity continuous field + sparse/high-resolution surface witness
        ↓
OPTIONAL TESSA V2 structuralization / learned retopology layer
        ↓
learned mesh candidate set
        ↓
Compiler qualification / support binding / canonical M
```

IRIS owns learned continuous scene evidence. TESSA owns learned explicit structural mesh proposals. Compiler remains the only product authority that may mint canonical M.

TESSA is deliberately falsifiable: the exact same IRIS state must support both a direct-surface→Compiler path and a TESSA→Compiler path. TESSA becomes mandatory only if the preregistered A/B court proves downstream value.

This is intentionally the intersection of different research lessons rather than a forced concatenation of papers:

- StdGEN++ / InstantMesh lineage: preserve geometric/raster supervision through representation transitions; use high-resolution active/sparse surface computation; treat thin-structure continuity as an explicit optimization problem.
- MeshAnything V2 lineage: dense/extracted surfaces are not automatically good artist/production meshes; autoregressive shape-conditioned mesh generation can learn explicit connectivity and lower-face structural meshes; AMT improves sequence efficiency by exploiting adjacency.
- MeshRipple-class evidence: truncated/local autoregression can break long-range topology; topological traversal order and global structural memory matter for large meshes.
- FACE-class evidence: raw coordinate-token streams are not the only viable granularity; sequence compression is a first-class scalability problem.

The branch does **not** claim any external system as RealSaS authority. They are architecture evidence only.

## 2. What stays from TESSA V1

Keep unless a court falsifies it:

- 1024 coordinate bins in canonical `[-0.5, 0.5]` space;
- explicit COMPONENT boundaries;
- adjacency-aware mesh serialization rather than independent-face soup;
- proposal-only authority;
- Compiler support binding and qualification after decoding;
- max policy of 32768 vertices / 32768 faces as an admission ceiling, not a demonstrated capability claim;
- fail-closed quantized-identity collision policy;
- no teacher topology, rig, skin or privileged product state in inference input.

## 3. TESSA V2 architecture

### 3.1 Role and admissible inputs

TESSA moves from an IRIS `structure head` abstraction to a separately typed **structuralization layer**.

It consumes only model-produced / admissible IRIS evidence:

- shared scene planes;
- high-resolution or best-available SurfaceWitness XYZ;
- witness normals;
- raw local H features;
- explicitly trained mechanical evidence;
- later, only explicitly admitted structural-continuity evidence.

Historical field/mechanical uncertainty scalars are **not** V2 conditioning because V8 supplied no calibration objective for them. They remain frozen/hidden unless a future typed calibrated objective is introduced.

TESSA must not consume source/teacher topology as inference conditioning.

### 3.2 Reference-class capacity

Do not start from a reduced diagnostic model.

The integrated V8 topology path was only:

- width 512;
- 8 heads;
- 8 decoder layers;
- 256 memory tokens.

Standalone historical TESSA was larger at 768 / 12 heads / 12 layers / 384 memory tokens. MeshAnything V2 uses an OPT-350M-class decoder. TESSA V2 therefore starts at:

- model width: **1024**;
- attention heads: **16**;
- decoder depth: **24**;
- feed-forward width: **4096**;
- coordinate bins: **1024**;
- typed scene-memory tokens: **512**;
- typed surface/mechanics-memory tokens: **512**;
- total conditioning memory: **1024 tokens**;
- local exact causal context: **4096 tokens**;
- causal global structural-history summaries for older completed blocks;
- BF16 + activation checkpointing / memory-efficient attention expected for training.

These are research defaults, not a mobile runtime promise.

### 3.3 Remove the replacement bottleneck

The V8 structure path concatenated scene and witness source tokens and resampled them into only 256 learned topology-memory tokens. V2 keeps scene and surface/mechanics as separately typed streams before fusion:

```text
rich IRIS scene evidence ──→ 512 scene memory ───────┐
                                                     ├─→ TESSA decoder
surface + H + mechanics ───→ 512 surface memory ────┘
```

Ablations may later reduce memory, but the first serious model must not prove capability through an arbitrarily narrow bridge.

### 3.4 Large-mesh topology context

The V1 trailing-window attention was not sufficient evidence for a 32768-face capability claim. V2 therefore keeps exact recent topology context and adds causal summaries of older completed history blocks. No future token contributes to a summary visible to an earlier query.

The current global-history implementation is a source-level architectural mechanism, not yet a scientific proof of 32k generation quality or throughput.

### 3.5 Free-running generation

V2 contains a strict streaming grammar state machine and a correctness/reference autoregressive generator:

```text
model logits
    ↓
strict grammar mask
    ↓
greedy/sampled next token
    ↓
streaming grammar state
    ↓
EOS / bounded stop
    ↓
decode
    ↓
independent topology court
```

The reference generator intentionally re-evaluates the prefix and makes **no high-throughput claim**. A cached incremental decoder must prove token parity before replacing it.

## 4. Loss and evidence redesign

### 4.1 Topology

Token cross entropy remains necessary but is not sufficient.

```text
L_tessa =
    λ_token      * CE(next token)
  + λ_coord      * metric coordinate loss
  + λ_special    * structural-token loss
```

Coordinate tokens are ordered geometry. Plain CE treats a one-bin miss and a five-hundred-bin miss as equally wrong. V2 therefore adds a metric loss on expected coordinate position.

Free-running decoded meshes are evaluated outside the gradient path by an independent court:

- grammar validity / EOS completion;
- component count;
- boundary edges / loops / open chains;
- non-manifold edges;
- degenerate faces;
- finite/domain validity;
- later, source-surface fidelity, support binding, G3 and G5 results.

A falling teacher-forced CE is never a topology PASS.

### 4.2 Geometry / extraction

The 24^3 differentiable V8 training court must not be confused with final spatial capability. The repository already contains an older sparse T512→T1024 zero-surface decoder. The post-V8 redesign should recover/adapt a sparse high-resolution training/evaluation path rather than raising one dense cube blindly.

Transferable StdGEN++ supervision:

- silhouette/mask;
- depth;
- normal;
- surface distance;
- FlexiCubes deviation;
- continuity/hole regularization only under source-defensible selectors.

### 4.3 Relation/order

The relation redesign follows a strict rule:

> If geometry is deterministically available, do not force a latent head to rediscover it.

Current camera features already contain normalized camera origin, right, screen-up, forward and normalized half extent. V3 relation prediction therefore receives, in addition to full-width H(A)/H(B):

- XYZ_A / XYZ_B;
- ΔXYZ and distance;
- camera-space depth_A / depth_B / Δdepth;
- projected UV_A / UV_B / ΔUV;
- same-ray residual;
- camera embedding.

Front/order remains pairwise. Visibility is removed from the pairwise abstraction and belongs to a complete ray-stack/view head or deterministic raster authority because global visibility depends on all hits on the ray.

### 4.4 No orphan authority outputs

Every inference output must be one of:

- directly supervised;
- calibrated against an explicit target;
- deterministic / derived;
- typed diagnostic-only and forbidden from product authority.

V3 consequences already implemented:

- no unsupervised relation confidence in the new pair API;
- no pairwise visibility output;
- no appearance confidence or unsupervised appearance descriptor;
- no public/trainable uncalibrated mechanical uncertainty;
- historical scene uncertainty row is frozen and hidden from V3 evidence;
- contact persistence has a validity-typed slot but **must abstain unless source authority exists**.

### 4.5 Contact-persistence correction

The historical teacher type can represent `contact_persistence`, but the active Fit8 `_mechanical_bank` calls the teacher **without `contact_face_mask`**. Therefore current prepared truth has `contact_persistence_valid=False`; it is not a usable training target.

Policy:

```text
no authored/source-defensible contact mask
        ↓
contact_persistence_valid = false
        ↓
zero gradient / ABSTAIN
```

Do not reinterpret the stored zero as a negative label. A future contact materializer must define and audit source authority before this family can train.

## 5. StdGEN++ vs learned mesh structuralization

StdGEN++ demonstrates high-fidelity explicit surfaces extracted from learned fields, including high-resolution thin/hollow structures and decomposition-aware mesh training. That does not automatically prove the extracted face layout is the best production/animation canonical M.

MeshAnything / MeshRipple-class ideas add value only if RealSaS needs a structurally different mesh from that witness: lower face count, learned connectivity, more useful topology and controlled production structure.

Therefore TESSA V2 is justified only by an A/B court:

```text
A: high-resolution IRIS/FlexiCubes witness → Compiler
B: same IRIS evidence/witness → TESSA V2 → Compiler
```

The implemented A/B receipt requires all of the following before TESSA can become mandatory:

1. static Compiler qualification preserved;
2. G5/multiview qualification preserved;
3. G3/dynamic qualification preserved;
4. support binding preserved;
5. source-surface p95/max fidelity within preregistered tolerance;
6. mask/depth/normal fidelity within tolerance;
7. no unacceptable component/non-manifold/degenerate/open-chain regression;
8. meaningful face-count reduction.

If direct surface passes and TESSA adds no meaningful structural economy, the policy explicitly prefers the direct path. TESSA does not survive merely because it exists.

## 6. Current source verification

Dedicated source gate on PR #122 is green at the current implemented A/B slice:

- source compile: PASS;
- reference capacity/authority boundary: PASS;
- **26 V2/V3 architecture + free-running + A/B tests: PASS**;
- **23 historical TESSA/Relation regression tests: PASS**.

This is source/contract evidence only. It is **not** a scientific TESSA pass.

## 7. Scientific admission

No optimizer step is authorized by this document.

Before a TESSA V2 fit:

1. exact V2 teacher topology corpus and free-running court materialized;
2. direct high-resolution IRIS surface baseline materialized through the same Compiler courts;
3. sparse/high-resolution surface path selected and source-fidelity supervision defined;
4. V8/old-TESSA baseline reproducible;
5. capacity/resource preflight on the intended accelerator;
6. A/B thresholds preregistered before evaluating the new checkpoint;
7. decoded-mesh evaluator remains independent from training loss;
8. contact persistence remains abstained unless a separate authority materialization qualifies it.

## 8. Implementation state / next work

Implemented:

- reference-capacity TESSA V2 config;
- dual typed scene/surface memory;
- causal global structural-history attention;
- metric-aware topology objective;
- strict free-running grammar and generation reference;
- independent decoded topology court;
- geometry-explicit Relation V3;
- separate ray-stack visibility head;
- objective-complete appearance/mechanical V3 APIs;
- IRIS/TESSA module separation;
- direct-surface vs TESSA downstream A/B promotion court;
- dedicated source gate + historical regression protection.

Next, before training:

1. restore/adapt the sparse T512→T1024 IRIS surface path to the vNext carrier;
2. add source-defensible mask/depth/normal high-resolution surface supervision;
3. materialize direct-surface→Compiler baseline receipts;
4. build exact V2 topology training corpus/readiness receipt;
5. leave contact persistence abstained unless an authored-contact authority is independently qualified;
6. run capacity/resource preflight and preregister the first V2 training court.
