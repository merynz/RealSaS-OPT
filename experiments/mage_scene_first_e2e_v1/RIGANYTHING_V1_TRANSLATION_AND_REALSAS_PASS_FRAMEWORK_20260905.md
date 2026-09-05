# RealSaS — RigAnything v1 Translation, Architecture Delta, and Real PASS Framework

Status: `DIRECTION_FROZEN__CHALLENGERS_CODED__SCIENTIFIC_RUNS_NOT_YET_PROMOTED`

Date: 2026-09-05

Primary external source: RigAnything v1, arXiv `2502.09615v1`  
https://arxiv.org/html/2502.09615v1

RealSaS branch inspected before this report:  
`e2e/mage-scene-first-v1-20260905 @ c1c42c1e9f062b80d929005232539d02d495cc95`

This document intentionally does **not** revive GFDR or any previously failed architecture. Historical ideas are not product authority. The goal is to compare the current scene-first IRIS→GSA→RiggingSurfaceIR→Geppetto→Compiler stack against mechanisms that RigAnything demonstrates successfully, then translate only the mechanisms that remain justified.

---

## 1. Executive decision

The current B1/B1a/B1s exact-41-on-31 series is useful as an **optimization / identifiability diagnostic**, but it must not remain the RealSaS product PASS definition.

The product question is not:

> Did Geppetto reconstruct the authored teacher rig exactly?

The product question is:

> Did **Geppetto + deterministic qualification + Compiler** construct a clean, legal, editable rig whose mechanical/deformation behavior is sufficient for the character?

Therefore RealSaS now separates three levels:

1. **FIT-OPT** — teacher-relative training diagnostic. Near-exact or exact fitting may be useful here to prove that an optimization/supervision path is learnable. No product authority.
2. **FIT-PRODUCT** — same-family product gate on the compiled/qualified rig. This is the real Mage PASS.
3. **GEN-PRODUCT** — the same product metrics on family-disjoint/unseen data. FIT and GEN must not use different definitions of product quality.

The currently running phase-switch notebook may finish. Its result remains FIT-OPT diagnostic evidence only. No further optimizer micro-surgery is authorized merely to force authored-teacher exact multiplicity after that readout.

---

## 2. What RigAnything v1 actually does

### 2.1 Input representation

RigAnything samples 1024 mesh-surface points and corresponding normals. Each shape token starts as:

`[x, y, z, nx, ny, nz]`

and is lifted with an MLP. The released config uses 1024 points and a maximum of 64 joints.

The RealSaS current scene-first substrate is not poorer in raw evidence. `RiggingSurfaceIR` already carries:

- 3D surface position `P`;
- robust derived normal;
- 8-view support/visibility;
- exact-camera raster bindings;
- predicted-surface local topology relations;
- provenance / lineage.

Current Geppetto V2 conditions on 24D surface features containing normalized XYZ, local covariance spectrum, radius, normal, support bits/fraction, and raster statistics.

**Conclusion:** the largest RigAnything gap is not that RealSaS lacks geometry evidence. The more important gaps are in how Geppetto uses that evidence during autoregressive skeleton generation.

### 2.2 Full shape-token access during every autoregressive step

RigAnything concatenates shape tokens and skeleton tokens and processes them with a hybrid attention mask:

- shape tokens self-attend globally;
- each skeleton token can attend to **all shape tokens**;
- skeleton tokens causally attend to earlier skeleton tokens.

Its conditional next-joint distribution is explicitly:

`P(j_k | T_<k, H)`

where `H` is the full sequence of shape tokens, not a single pooled vector.

Current Geppetto V2 instead encodes the surface to token memory **and a mean-pooled descriptor**, then its autoregressive recurrence is driven primarily by pooled shape state + previous latent control states. The full surface memory is retained, but is not freshly cross-attended by the locus-generating autoregressive state at every step; it is used again for support logits.

This is a material architectural difference.

### 2.3 Probabilistic continuous joint positions via diffusion

RigAnything models the continuously valued next joint with conditional diffusion. The paper's ablation is unusually strong:

- full RigAnything skeleton IoU: `0.765`;
- replacing joint diffusion with deterministic L2: `0.308`.

The authors explicitly attribute the failure to joint positions collapsing toward averaged/mid-axis locations under sibling/tree ambiguity, while diffusion captures multiple plausible position modes.

The released code uses a cosine diffusion process, epsilon prediction, an AdaLN-conditioned denoising MLP, and iterative sampling.

Current Geppetto V2 already acknowledges locus ambiguity with a 3-mode continuous head. B1a proved this has enough representational capacity for the Mage exact-multiplicity diagnostic under oracle-stable identity. However, that does **not** establish that a fixed three-mode head is the best mechanism for unseen structural ambiguity. RigAnything provides direct evidence that conditional joint diffusion deserves a controlled challenger.

### 2.4 Generated/teacher joint geometry is fused back into skeleton state

RigAnything does not generate a joint and then forget its actual geometry. After obtaining `j_k`, it fuses the sampled joint position into context for connectivity; skeleton tokens themselves are built from joint position, parent position, and sequence-position embeddings. During training, ground-truth next-joint geometry is used for connectivity conditioning; during inference, sampled geometry is used.

Current Geppetto deliberately removed hard-MAP locus feedback after a previous multimodal crossover failure. That repair was justified for the old hard-MAP feedback mechanism. It does not prove that **all** geometry feedback is undesirable.

The correct challenger is therefore not “restore the old hard-MAP recurrence.” It is a new, explicitly controlled mechanism in which probabilistically sampled/teacher-forced geometry is tokenized and fed back, with crossover telemetry and no promotion by assumption.

### 2.5 Sequence ambiguity is treated as ambiguity, not a single sacred order

RigAnything serializes skeletons in BFS order, but explicitly randomizes the order of joints at the same BFS depth during training. Its appendix also states that one shape can admit multiple valid skeleton topologies.

This matters for RealSaS because exact authored row identity is not a product invariant. Structural serialization remains useful as a deterministic FIT-OPT diagnostic, but the general training path should not force an arbitrary sibling ordering to become semantic identity.

### 2.6 RigAnything's evaluation is not exact-rig equality

For skeleton prediction RigAnything reports RigNet-family metrics:

- IoU;
- Precision;
- Recall;
- CD-J2J;
- CD-J2B;
- CD-B2B.

Its reported full-model result on RigNet is approximately:

- IoU `0.765`;
- Precision `0.786`;
- Recall `0.765`;
- CD-J2J `0.033`;
- CD-J2B `0.034`;
- CD-B2B `0.019`.

A successful RigAnything prediction is therefore **not** required to have exact ground-truth joint count, exact joint identity, exact topology, or exact positions.

Connectivity is also evaluated conditionally with ground-truth joints, and skinning is evaluated conditionally with the ground-truth skeleton. Their methodology deliberately separates component questions rather than treating one end-to-end exact reconstruction score as the only scientific signal.

### 2.7 Their teacher data is curated before learning

RigAnything does not treat every raw authored control as sacred ground truth. It filters invalid/poorly aligned rigs, excludes rigs with more than 64 joints, and notes that many over-64 cases are facial/hair rigs. Low-quality rigging/skinning examples are removed.

This reinforces a RealSaS rule:

> Exact FIT is meaningful only when the target representation itself is product-relevant.

If raw artist helpers, redundant coincident controls, facial/hair-detail controls, or technical assembly controls are not part of the desired first product rig, exact reconstruction of them is not a product objective.

---

## 3. Current RealSaS vs RigAnything v1 — mechanism matrix

| Mechanism | RigAnything v1 | Current RealSaS | Decision |
|---|---|---|---|
| surface evidence | 1024 XYZ+normal points | ~1024 scene-first nodes with XYZ+normal+view/raster/topology | keep RealSaS richer evidence |
| global shape processing | shape-token self-attention | surface local+global encoder | keep |
| per-joint full shape access | skeleton token attends all shape tokens | autoregressive locus state mainly sees pooled surface descriptor | **add challenger** |
| continuous locus ambiguity | conditional joint diffusion | fixed 3-mode locus head | **add diffusion challenger** |
| generated geometry feedback | sampled/GT joint + parent geometry fused into skeleton state | hard-MAP locus deliberately not fed back | **add new safe challenger; do not restore old mechanism** |
| sibling-order ambiguity | same-depth BFS order randomized | deterministic structural serialization in current diagnostic | keep serialization for FIT-OPT; add randomized equivalent-order training challenger |
| parent prediction | conditioned on generated/GT current joint | all-pairs parent head after latent sequence | compare after geometry-feedback challenger |
| teacher exactness | similarity metrics, not exact equality | B1-family exact multiplicity diagnostic | demote exactness to FIT-OPT only |
| canonical graph authority | learned rig is final skeleton | Compiler owns canonical IDs/root/legal tree | **keep RealSaS deterministic authority** |
| functional product proof | mostly GT similarity metrics | Compiler/runtime can support deformation/motion proof | **RealSaS should go beyond RigAnything** |

---

## 4. RealSaS PASS must score the total deterministic + learned system

The primary product metric target is not raw `SkeletonProposalIR`.

It is the qualified/compiled product:

`RiggingSurfaceIR → Geppetto proposals → deterministic qualification → Compiler canonical skeleton → skin/mesh/deformation/runtime proof`

Raw Geppetto scores remain diagnostics used to localize failures.

### 4.1 Reference skeleton scores — literature compatibility layer

Implement RigNet/RigAnything-family reference telemetry on the **compiled skeleton**:

- `SkeletonIoU(τ)`;
- `Precision(τ)`;
- `Recall(τ)`;
- `CD-J2J`;
- `CD-J2B`;
- `CD-B2B`.

These scores answer “how similar is the compiled result to the curated reference rig?” They do **not** alone decide RealSaS product PASS.

A first executable implementation is added in `compiler/realsas_compiler_core/rig_quality_v1.py`. Before publishing direct numerical comparisons against RigAnything/RigNet, the evaluator must be parity-checked against the exact RigNet metric implementation/tolerance convention.

### 4.2 Necessary Mechanical Coverage — identity/count-independent core metric

Let the curated reference rig induce infinitesimal deformation modes `J_T` and the compiled RealSaS rig induce modes `J_C`. Define reference-to-candidate residual:

`E_T→C = ||J_T - P_span(J_C) J_T||_F / ||J_T||_F`

and coverage:

`Coverage_T→C = 1 - E_T→C`.

This directly answers whether the compiled rig can reproduce the mechanically relevant deformation subspace, independent of joint identity or joint count.

A 28-control compiled rig can therefore beat a 41-control authored rig if it reproduces the necessary mechanical space with less redundancy.

Executable subspace telemetry is added in `rig_quality_v1.py`.

### 4.3 Finite deformation / motion equivalence

Infinitesimal coverage is not enough. For finite motion probes, fit candidate controls to each reference deformation and report:

- deformation RMS;
- deformation p95;
- max error;
- per-view silhouette IoU;
- per-view reprojection p95;
- foldover/inversion/tearing violations;
- contact violations;
- unsupported motion / solver abstention.

These measurements belong on the compiled rig and should feed the existing motion-proof chain.

### 4.4 Cleanliness / editability

Product quality must penalize unnecessary controls rather than rewarding raw teacher count replication.

For each candidate control, estimate the fraction of its deformation mode not explained by the span of the other controls. This yields per-control unique-contribution telemetry. A redundant-control threshold is **not** hard-coded yet; it must be calibrated.

Hard invariants:

- illegal parent = 0;
- cycle = 0;
- unsupported canonical control = 0;
- stale lineage = 0;
- skin simplex / qualification = PASS;
- required fail-closed behavior = PASS.

---

## 5. Threshold calibration — no invented product numbers

RealSaS must not invent PASS thresholds from intuition.

Before freezing `FIT-PRODUCT V1`, construct controlled variants of the Mage reference rig.

### GOOD variants — must remain product-equivalent

Examples:

- remove a zero-effect helper;
- merge demonstrably deformation-equivalent coincident controls;
- collapse a redundant collinear subdivision;
- replace authored structure with a cleaner graph that preserves deformation span and finite motion probes.

### BAD variants — must fail

Examples:

- delete a necessary articulation control;
- move a critical pivot by controlled fractions of object extent;
- attach a branch to the wrong parent;
- collapse a necessary limb/tail/wing branch;
- corrupt skin weights;
- add unsupported or mechanically harmful controls.

Freeze thresholds only where the measured `worst GOOD` and `best BAD` populations separate. If they do not separate, the metric is not yet an adequate product gate.

This calibration must happen before unseen/generalization evaluation. The same frozen gate is then used for FIT-PRODUCT and GEN-PRODUCT.

---

## 6. Deterministic boundary work still matters

The prior GSA/RiggingSurface audit found two real boundary mismatches:

1. the B1 compact fixture drops production raster bindings, zeroing four Geppetto conditioning channels;
2. GSA emits predicted-surface topology, but current Geppetto locality reconstructs Euclidean KNN and does not consume those relations.

Those facts remain valid, but the old A0-A3 experiment was specified using the exact-teacher B1 gate. Its result would still be diagnostic rather than product authority.

The more important deterministic defect is the current pure spatial voxel compactor: nearby but mesh-disconnected surface patches can be averaged into one compact node. That can destroy shape evidence before any model sees it.

A topology-preserving same-voxel connected-component challenger is now added as:

`compiler/realsas_compiler_core/substrate/topology_preserving_compactor_v1.py`

It is additive and not product-wired. It must be evaluated with surface fidelity / coverage / visibility / topology invariants and then, if non-inferior, with the new Geppetto challenger. It carries no teacher rig/skin information.

---

## 7. New RigAnything-mechanism challenger coded now

Added research-only module:

`models/geppetto/challengers/riganything_mechanisms_v1.py`

It does **not** mutate canonical Geppetto V2. It implements the following translation:

1. reuse the current richer RealSaS surface encoder;
2. every autoregressive control step freshly cross-attends to the full surface-token memory;
3. next-joint 3D position is trained with a conditional cosine diffusion epsilon-prediction objective;
4. inference samples the joint with deterministic DDIM-style reverse stepping from random initial noise;
5. joint geometry and parent geometry are embedded and fused into the token supplied to subsequent steps;
6. teacher-forced geometry is supported during training, matching RigAnything's conditional connectivity idea;
7. an equivalent same-depth BFS-order randomizer is provided for training ambiguity experiments;
8. Compiler remains canonical graph authority; this module cannot mint product identities.

The diffusion implementation is a RealSaS adaptation, not copied source code. It intentionally uses the mechanism demonstrated by RigAnything while preserving our contracts and compute scale.

---

## 8. Scientific execution order from here

### Running now

Allow the already-started gate-relative epsilon phase-switch notebook to finish. Seal its readout as FIT-OPT diagnostic only.

### Work that does not need to wait

Already prepared/coded in parallel:

- RigAnything v1 architecture delta report;
- full-surface autoregressive cross-attention challenger;
- conditional 3D joint-diffusion challenger;
- safe geometry/parent feedback challenger;
- equivalent BFS-depth-order randomization helper;
- executable reference + functional rig-quality metrics;
- topology-preserving GSA compactor challenger.

### Next scientific experiments

Do **not** launch another optimizer-only authored-teacher exactness campaign.

Run a mechanism ladder on the same Mage scene-first evidence, with each arm frozen before results:

- `C0 CURRENT`: canonical Geppetto V2;
- `C1 FULL_SURFACE_XATTN`: only fresh full-surface cross-attention;
- `C2 XATTN_PLUS_DIFFUSION`: add conditional joint diffusion;
- `C3 XATTN_DIFFUSION_GEOMETRY_FEEDBACK`: add sampled/teacher-forced joint+parent geometry feedback;
- `C4 C3_PLUS_EQUIVALENT_ORDER_AUG`: randomize same-depth equivalent teacher ordering during training.

The purpose of this ladder is **mechanism identification**, not to require exact teacher reconstruction as the final product criterion.

For component FIT-OPT, exact/near-exact teacher diagnostics can remain on the dashboard to detect training collapse. Promotion, however, is based on the compiled system's new product scores.

### In parallel / immediately after substrate validation

Compare current spatial compactor vs topology-preserving challenger on:

- compact node budget;
- dense-surface reconstruction dispersion p50/p95/max;
- disconnected-patch fusion count (must be zero for challenger by construction);
- topology preservation;
- exact-camera visibility support stability;
- normal stability;
- downstream product metrics once the new Geppetto challenger is available.

### Product closure

The next meaningful Mage milestone is:

`IRIS → GSA → Geppetto challenger/winner → deterministic Compiler → compiled skeleton → Arachne/skin → deformation & motion proof → FIT-PRODUCT PASS`

Only after FIT-PRODUCT is frozen and passed do we move the same gate to family-disjoint/unseen GEN-PRODUCT.

---

## 9. What is explicitly *not* being done

- No GFDR revival.
- No raw authored teacher rig becomes product authority.
- No exact joint count/multiplicity is promoted to product PASS.
- No old hard-MAP feedback mechanism is restored.
- No diffusion promotion merely because RigAnything used it; it enters as a controlled challenger.
- No GSA topology or raster ablation result can alone promote product quality.
- No new product threshold is invented before GOOD/BAD calibration.
- No generalization claim from Mage.

---

## 10. Bottom line

RigAnything v1 demonstrates that the problem we are fighting is not unsolved in the abstract. Its strongest directly relevant evidence is:

- per-joint access to the full shape-token set;
- probabilistic continuous joint generation with diffusion;
- explicit use of generated joint geometry in evolving skeleton context;
- training that respects sequence/topology ambiguity;
- evaluation by rig similarity/quality rather than exact authored-rig identity.

RealSaS should adopt/test those mechanisms where they fill actual gaps, while retaining the parts that are stronger in our architecture:

- scene-first multiview evidence;
- richer RiggingSurfaceIR;
- deterministic canonical graph authority;
- fail-closed compilation;
- functional deformation/motion proof.

The new direction is therefore not “copy RigAnything.” It is:

> **Use RigAnything's demonstrated probabilistic autoregressive skeleton mechanisms inside the richer RealSaS evidence/compiler system, and judge success on the compiled rig's functional quality.**
