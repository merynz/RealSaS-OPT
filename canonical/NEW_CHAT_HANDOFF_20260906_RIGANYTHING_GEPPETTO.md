# RealSaS — New Chat Handoff — 2026-09-06

Status: `ACTIVE_HANDOFF__CONTINUE_DIRECTLY_FROM_HERE`

Canonical branch at handoff creation:

`e2e/mage-scene-first-v1-20260905 @ ba634955777479ee05a5b199742710b1736123b5`

Latest canonical commit before this handoff:

`ba634955777479ee05a5b199742710b1736123b5` — **Add RigAnything mechanism challengers and functional rig quality framework**

Primary current architecture/direction report:

`experiments/mage_scene_first_e2e_v1/RIGANYTHING_V1_TRANSLATION_AND_REALSAS_PASS_FRAMEWORK_20260905.md`

Primary external reference:

RigAnything v1 — arXiv `2502.09615v1`  
https://arxiv.org/html/2502.09615v1

---

# 0. How the next chat should use this handoff

If the user opens a new chat and says **“repoda direkt kaldığımız yerden devam et”**, do not re-ask for the project state and do not restart the architecture discussion from scratch.

Read this file first, then re-fetch the branch HEAD, inspect the referenced current files if needed, and continue from the **active scientific state** below.

The user is explicitly trying to avoid continuity loss, scope drift, and late “actually we had not tested that” surprises. Distinguish measured facts from hypotheses and keep the architecture audit-first discipline.

Do **not** revive historical GFDR or any failed historical architecture merely because an old idea resembles a current mechanism. Historical failed systems remain historical. The user explicitly rejected “GFDR geri gelsin”; only current mechanisms justified by present evidence should be carried forward.

---

# 1. Product / system intent

RealSaS converts a static 8-view 2D character sheet into an editable 2D puppet with skeleton, skinning/deformation, canonical graph, and runtime motion.

Current conceptual split:

- **IRIS** — scene/perception geometry evidence.
- **GSA / RiggingSurfaceIR** — deterministic geometric substrate and surface evidence.
- **Geppetto** — learned mechanical **proposal** generator, not canonical graph authority.
- **Arachne** — downstream skin/deformation proposal stage.
- **Compiler** — deterministic canonical authority: qualification, graph selection, repair/completion boundaries, canonical IDs, runtime product construction.
- **Runtime / proof chain** — deformation and motion validation.

The intended product is not “reconstruct the authored teacher rig exactly.” The intended product is a clean, editable, functionally sufficient rig, potentially simpler than the teacher and allowed to discard redundant/helper controls.

This distinction is now a hard rule.

---

# 2. Current top-level scientific doctrine

## 2.1 Exact teacher fitting is diagnostic, not product authority

Three levels are now explicitly separated:

### `FIT-OPT`

Teacher-relative optimization / supervision diagnostic.

Purpose:

> Can the model actually fit/memorize the supervision contract under a controlled same-family setting?

Near-exact or exact teacher fitting can be useful here. It is allowed to be stricter than the product goal because it is a training-capacity / identifiability diagnostic.

**No product promotion authority.**

### `FIT-PRODUCT`

The real Mage same-family product gate.

Authority is the **compiled/qualified system output**, not raw Geppetto proposals:

`IRIS → GSA/RiggingSurfaceIR → Geppetto → deterministic qualification → Compiler → compiled rig`

A raw Geppetto output may look unlike the teacher and still be a product PASS if deterministic downstream construction yields a clean mechanically sufficient rig.

### `GEN-PRODUCT`

The same product metrics, frozen, on family-disjoint/unseen data.

FIT and GEN must not use different definitions of “good rig.”

---

# 3. IRIS current status

IRIS scene-first signed geometry is already the promoted 1FIT core.

Canonical core: `SceneFirstSignedGeometryV3`.

Mage measured depth quality from the promoted run:

- overall depth p95: `0.04648` <= `.05` PASS
- MAE: `.02045`
- median: `.00696`
- worst-view p95: `.07193` <= `.08` PASS
- per-view p95 roughly:
  - S `.0301`
  - SE `.0391`
  - E `.0401`
  - NE `.0719`
  - N `.0632`
  - NW `.0674`
  - W `.0399`
  - SW `.0352`
- Y-span ~`96.3%`
- no depth collapse
- support monotonic hard-tail evidence was good

Important wording:

- **IRIS generic architecture / 1FIT contract is closed/promoted.**
- **IRIS unseen/generalization is not yet proven.**

Do not reopen IRIS merely because Geppetto is still under work unless new evidence points upstream.

---

# 4. Current GSA / RiggingSurfaceIR state

Mage scene-first GSA witness:

- `950` compact surface nodes
- `2813` compact topology edges
- robust normals
- exact 8-view support masks
- exact-camera raster bindings exist in production reconstruction
- teacher truth is not used by the GSA producer

Source run: `20260904T220929Z`

Current production scene-first builder:

`compiler/realsas_compiler_core/substrate/scene_first_signed.py`

Current `RiggingSurfaceIR` carries:

- surface `P`
- support views
- raster bindings
- derived normal
- local relations
- lineage/provenance

A central read-only validator exists:

`compiler/realsas_compiler_core/substrate/validation.py`

Current boundary audit report:

`experiments/mage_scene_first_e2e_v1/GSA_RIGGINGSURFACE_BOUNDARY_AUDIT_V1.md`

Key measured findings from that audit:

1. The B1/B1a/B1s Mage fixture is genuinely a quantized snapshot of the **current scene-first GSA**, not an old D2 path.
2. The compact experimental fixture dropped production raster bindings, so the last four Geppetto raster conditioning channels became zero even though production GSA provides nontrivial raster evidence.
3. GSA local topology is preserved, but current Geppetto V2 does not consume `RiggingSurfaceIR.local_relations`; it rebuilds Euclidean KNN internally.
4. On Mage, Euclidean KNN covers almost all direct GSA edges, but adds thousands of non-direct topology neighbors. This is a semantic boundary difference, not yet proven causal.
5. RiggingSurfaceIR itself was weakly enforced as a type boundary; a validator was added, but it is not yet wired into product construction as a mandatory interlock.

Frozen A0-A3 boundary design exists in:

`experiments/mage_scene_first_e2e_v1/GSA_GEPPETTO_CONDITIONING_ABLATION_DESIGN_FREEZE_V1.md`

Frozen arms:

- A0: stripped raster + current Euclidean KNN
- A1: production raster + current Euclidean KNN
- A2: stripped raster + `GSA_GRAPH_16`
- A3: production raster + `GSA_GRAPH_16`

However, after the RigAnything comparison, these boundary ablations are no longer the only next direction; see Sections 9–12.

---

# 5. Current Geppetto V2 architecture facts

Canonical current model:

`models/geppetto/v2/geppetto_candidate_v2.py`

Key facts:

- latent autoregressive set proposal
- model dim `192`
- local/global surface encoder
- autoregressive control generation
- 3-mode continuous position hypothesis head
- parent all-pairs head
- support logits against surface memory
- stop/existence/root outputs
- teacher IDs are not product IDs

Intentional design choice:

- previous hard-MAP locus is **not** fed back into later recurrence, because earlier experiments showed multimodal MAP crossover could corrupt later states.

Important current limitation revealed by RigAnything comparison:

- the surface encoder retains token memory, but locus-generating autoregressive recurrence is driven primarily by pooled/global state + previous latent states;
- it does **not** freshly cross-attend the full surface-token memory at every generated joint step in the RigAnything way.

Current conditioning:

`models/geppetto/v2/geppetto_conditioning_v2.py`

Current Geppetto V2 surface feature width: `24`.

Current loss:

`models/geppetto/v2/geppetto_loss_v2.py`

Current loss still has teacher-relative geometry/topology/support targets for training diagnostics. This is acceptable for FIT-OPT but must not be confused with product authority.

Teacher projection is explicitly training/eval-only:

`models/geppetto/v2/training_targets_v2.py`

---

# 6. Geppetto diagnostic history that must be preserved

## 6.1 Original B1 multiplicity

Forced `41` authored teacher controls on `31` exact loci.

Dynamic assignment / exact multiplicity run failed through 8192.

Result:

- best p95 ~`.03744`
- exact multiplicity not closed
- assignment/group flips extremely high

This established that the original training identity mechanism was problematic.

## 6.2 B1a fixed-assignment oracle

Step-0 canonical Hungarian correspondence frozen once.

Result: **PASS**.

- first single PASS `7872`
- stable PASS `8000`
- final `8192`
- p95 `.0039796`
- max `.0050179`
- outside `0`
- exact multiplicity true

Interpretation:

Stable persistent training identity is sufficient to remove the obstruction in the B1 regime. Decoder capacity exists under stable identity.

This was an oracle diagnostic only, not a product solution.

## 6.3 B1s structural serialization

Generic deterministic structural ordering, no teacher row identity oracle.

Historical constant AdamW `eps=1e-8` run:

- improved substantially over original B1
- best around step `10752`, p95 `.021379`
- only slots `34,35,36` remained outside exact capture at best
- then catastrophic collapse by step `10816`
- correspondence itself remained fixed

This pointed away from assignment churn and toward optimizer/recurrent trajectory instability.

## 6.4 Corrected optimizer fork

From the exact 10752 historical state:

- baseline `eps=1e-8` collapses
- grad clip alone does not solve
- LR/10 avoids immediate collapse
- `eps=1e-4` avoids collapse and improves

Interpretation:

Small Adam denominator / effective step-size instability is causally implicated in the late collapse.

## 6.5 Rescue confirmatory from 10752

Continue the 10752 B1s state with `eps=1e-4`.

Result: authoritative **RESCUE PASS**.

- first stable original-gate PASS: `12544`
- max stable streak: `39`
- best step: `15790`
- best outside: `0`
- best occupancy: `0`
- best p95 ~`.00528624`
- final `16384`
- final outside `0`
- final occupancy `0`
- final p95 `.00788724`

This proved `eps=1e-4` is sufficient as a **late rescue/stabilizer** from the near-converged state.

## 6.6 Full step-0 constant `eps=1e-4`

Started from initialization with `eps=1e-4` from step 1.

Result: **FAIL** through 16384.

- best p95 `.0424168`
- best outside `15`
- final p95 `.0405985`
- final outside `16`
- final occupancy `16`

Interpretation:

`eps=1e-4` is not a good constant-from-start optimizer policy. It over-damps/coarsens early learning even though it stabilizes late learning.

## 6.7 Current running phase-switch notebook — ACTIVE AT HANDOFF

Notebook:

`RealSaS_MAGE_GEPPETTO_B1S_PHASE_EPS_SWITCH_A100_RUN_ALL_V1_NO_TOKEN_PREFLIGHTED_FINAL.ipynb`

Policy:

- start AdamW `eps=1e-8`
- check every 64 steps
- when structural-slot p95 <= `1.5 × capture_radius` for **3 consecutive checks**, switch once and irreversibly to `eps=1e-4`
- no model/loss/GSA/serializer/LR/WD change
- primary 8192, max 16384

Historical Mage replay should trigger this generic condition at step `10240`; the notebook treats that step only as a deterministic parity assertion, **not** as the policy itself.

### Latest visible live progress at handoff

The user reported:

- at step `15616`
- `PASS streak = 41`

This means the exact FIT-OPT gate had already remained PASS for roughly 2600 steps and was essentially scientifically answered before final packaging.

**Do not treat this as final until the Drive summary/manifest/ZIP are read after completion.**

When the user next says the notebook is finished, first search Google Drive for:

`RealSaS_MAGE_GEPPETTO_B1S_PHASE_EPS_SWITCH_A100_V1_NO_TOKEN`

Then inspect:

- `LATEST.txt`
- contract folder
- `B1S_PHASE_SWITCH_SUMMARY.json`
- `FINAL_MANIFEST.json`
- `RESULT_PACKAGE.json`

Verify:

- actual switch step
- first single PASS
- first stable PASS
- max/final streak if logged
- final outside / occupancy / p95
- ZIP SHA
- that phase policy started at eps=1e-8 and switched only once

Expected scientific interpretation if final remains stable PASS:

> `FIT-OPT` optimizer path closed: coarse learning with eps=1e-8 + gate-relative late stabilization with eps=1e-4 is sufficient for this exact-teacher diagnostic.

Then **stop opening further optimizer micro-surgery experiments solely to force authored-teacher exact multiplicity**.

---

# 7. Why exact authored-teacher reconstruction is not the product target

This became explicit again during the RigAnything comparison.

RigAnything does train against ground-truth rig supervision, but its product/evaluation philosophy is not “exactly reconstruct every authored teacher joint identity and multiplicity.”

Important principles from RigAnything v1:

- skeleton generation is probabilistic because multiple valid skeleton topologies/orderings can exist for a shape;
- sibling ordering is ambiguous;
- dataset rigs are curated/filtered rather than treated as sacred authored truth;
- models with excessive joints / invalid structure / bad alignment are filtered;
- the method is praised for producing a **reasonable number of joints** aligned to the shape;
- evaluation uses similarity metrics (IoU, precision, recall, Chamfer-family distances), not exact graph identity equality.

Therefore RealSaS must preserve this distinction:

- teacher rig = supervision/reference/evaluator
- Compiler product = canonical authority

A teacher with 41 controls and a RealSaS compiled rig with 27 controls can still be a product PASS if the 27-control rig preserves necessary mechanical/deformation behavior and is cleaner.

---

# 8. RigAnything v1 mechanisms that matter for us

The current direction report already records the full comparison:

`experiments/mage_scene_first_e2e_v1/RIGANYTHING_V1_TRANSLATION_AND_REALSAS_PASS_FRAMEWORK_20260905.md`

The highest-value differences are:

## 8.1 Full surface-token access at each autoregressive joint step

RigAnything uses a hybrid attention mask:

- shape tokens globally self-attend
- each skeleton token attends to **all shape tokens**
- skeleton tokens causal-attend to previous skeleton tokens

So the next-joint distribution is conditioned on full shape memory:

`P(j_k | T_<k, H_surface)`

Current Geppetto V2 is more bottlenecked by pooled/global recurrence.

This is a likely mechanism behind RigAnything’s clean spatial joint placement.

## 8.2 Conditional joint diffusion

RigAnything models continuous joint coordinates with conditional diffusion.

This is not a minor implementation detail. Their ablation is strong:

- deterministic L2 joint regression: skeleton IoU around `0.308`
- full diffusion model: skeleton IoU around `0.765`

The paper explicitly describes deterministic regression as collapsing joints toward an average/mid-axis solution under ambiguity.

Important nuance:

- this is **ambiguity-induced averaging/modal collapse**;
- it is not automatically the same phenomenon as our 10752→10816 Adam optimizer catastrophe.

But the mechanism is highly relevant to our long-standing sibling/order/multimodal locus difficulty.

## 8.3 Generated geometry feedback

RigAnything re-embeds generated joint geometry into the autoregressive state and uses joint/parent geometry when predicting connectivity / later context.

Current Geppetto deliberately avoids hard-MAP locus feedback because of earlier crossover instability.

Therefore any geometry feedback challenger must be controlled and explicit; do not simply restore the old mechanism.

## 8.4 Equivalent sibling-order handling

RigAnything treats sibling ordering as non-semantic ambiguity rather than one sacred serialization.

RealSaS should test same-depth / equivalent sibling-order augmentation rather than overfitting arbitrary authored row identity.

---

# 9. New RigAnything mechanism challenger already coded

New file committed at `ba634955...`:

`models/geppetto/challengers/riganything_mechanisms_v1.py`

This is **research-only** and does not mutate canonical Geppetto V2.

It contains:

1. full surface-memory cross-attention per generated control step;
2. conditional 3D joint diffusion;
3. current joint geometry + parent geometry feedback into subsequent recurrence;
4. separate challenger config/output types.

It retains RealSaS’s richer 24D scene-first conditioning and Compiler authority.

Do not silently promote this challenger wholesale. The next scientific work should isolate mechanisms.

Desired additive ladder:

- `C0` = current Geppetto
- `C1` = C0 + full-surface cross-attention
- `C2` = C1 + conditional joint diffusion
- `C3` = C2 + safe joint/parent geometry feedback
- `C4` = C3 + equivalent sibling-order augmentation

Mechanisms should be measured separately so we know what fixed what.

The user explicitly believes joint diffusion looks promising and is frustrated by spending a long time rediscovering a problem that RigAnything already solved. The current architecture direction should respect that evidence rather than continuing endless bespoke optimizer variants.

---

# 10. New functional rig quality / PASS framework already coded

New file:

`compiler/realsas_compiler_core/rig_quality_v1.py`

Current utilities include:

- skeleton reference telemetry (IoU/precision/recall-like matching and Chamfer-family scores)
- mechanical deformation-space coverage
- finite deformation RMS/p95/max
- per-control unique contribution / redundancy telemetry

Important caveat:

The current RigAnything/RigNet-like reference metrics must be parity-checked against the exact external evaluator before publishing apples-to-apples numbers. They are reference diagnostics, not current product authority.

## 10.1 Mechanical coverage concept

For teacher/reference deformation modes `J_T` and compiled candidate modes `J_C`:

`E_T→C = ||J_T - P_span(J_C) J_T||_F / ||J_T||_F`

Coverage is approximately:

`1 - E_T→C`

This metric is independent of joint identity/count.

This is a core product idea: a simpler compiled rig can PASS if it spans the mechanically necessary deformation space.

## 10.2 Finite deformation proof

Local Jacobian/span coverage is not enough. The product gate must also use finite motion/deformation probes:

- deformation RMS
- p95
- max error
- silhouette/reprojection consistency when available
- foldover/inversion/tearing checks
- contact/motion violations

These should be evaluated after deterministic downstream construction, not on raw proposal positions alone.

## 10.3 Cleanliness/editability

Product quality should penalize unnecessary controls, but not by raw equality to teacher joint count.

Telemetry should include:

- redundant controls
- zero-effect deform controls
- unsupported controls
- illegal parent/cycle/root structure
- skin simplex/correction quality
- canonical graph legality

Thresholds are not yet frozen.

---

# 11. How real product thresholds should be calibrated

Do **not** invent arbitrary PASS thresholds now.

Use controlled perturbation calibration:

## GOOD variants that should remain product PASS

Examples:

- delete truly redundant coincident helper
- merge deformation-equivalent controls
- remove zero-effect helper
- collapse unnecessary collinear subdivision

## BAD variants that should FAIL

Examples:

- delete critical articulation joint
- shift pivot by controlled fractions of body extent
- wrong parent connection
- collapse an important branch
- corrupt skinning weights
- add pathological redundant controls

Then freeze thresholds from the separation between:

- worst GOOD
- best BAD

This is the correct route to a real `FIT-PRODUCT` gate.

---

# 12. New topology-preserving compactor challenger already coded

New file:

`compiler/realsas_compiler_core/substrate/topology_preserving_compactor_v1.py`

Purpose:

Current adaptive voxel compaction can spatially merge disconnected same-voxel surface patches. The challenger only fuses dense vertices if they are:

1. in the same voxel, **and**
2. connected through mesh edges within that voxel.

It uses no teacher rig/skin information.

This is additive research code, not promoted production behavior yet.

---

# 13. Recommended direction after the running notebook finishes

The user explicitly asked whether we should “sit and wait.” Answer: **no**. The repo already moved forward while the notebook was running.

Recommended order now:

## A. Close the current phase-switch FIT-OPT diagnostic

When final result arrives:

- retrieve Drive artifacts
- record exact result
- if stable PASS, mark optimizer diagnostic path closed
- no more optimizer micro-surgery for authored-teacher exact multiplicity unless a future experiment specifically needs it

## B. Keep the boundary scientifically clean

Before major challenger training:

- prefer production-like scene-first `RiggingSurfaceIR`
- restore actual raster evidence rather than carrying the intentionally stripped B1 fixture as the natural model input
- verify `RiggingSurfaceIR` invariants
- measure current compactor vs topology-preserving challenger

The old A0-A3 boundary experiment remains useful if needed for attribution, but do not let it become another months-long detour. Its role is to separate “input evidence lost at boundary” from “generation architecture problem.”

## C. Run the RigAnything mechanism ladder

Priority:

1. `C1`: full surface-token cross-attention
2. `C2`: conditional joint diffusion
3. `C3`: safe geometry feedback
4. `C4`: sibling-order augmentation

Use the same scientific substrate and training conditions per comparison.

The most informative immediate comparison is likely `C1 → C2`, because RigAnything provides strong causal evidence that diffusion prevents deterministic joint-position collapse under ambiguity.

## D. Move evaluation authority toward compiled output

Do not wait until Arachne/runtime are perfect to define product metrics.

Implement/calibrate progressively:

- compiled skeleton mechanical coverage
- compiled finite deformation probes
- canonical graph legality
- redundancy/cleanliness

Then extend the same framework through Arachne/skin and runtime motion.

---

# 14. Compiler authority must remain central

Current skeleton qualifier:

`compiler/realsas_compiler_core/rig.py`

Current behavior:

- proposal IDs are not canonical
- surface lineage must match
- node/edge proposals are converted to graph candidates
- canonical graph optimizer chooses root/parents
- canonical IDs are assigned only after optimization

Current product type explicitly separates deform roots from future technical assembly root semantics.

This matters for evaluation:

**Real PASS is not a Geppetto-only score.**

If Geppetto emits noisy-but-sufficient proposals and the deterministic stack constructs a clean valid rig, the product can PASS.

If Geppetto matches teacher geometry beautifully but Compiler/Arachne/runtime yields poor deformation, the product must FAIL.

---

# 15. Arachne / downstream note

Arachne has not yet been re-entered as the current active stage because Geppetto architecture is still being closed.

Once Geppetto mechanism direction is chosen, the E2E sequence remains:

`IRIS → GSA → Geppetto → Arachne → Compiler → Runtime dynamic proof`

Do not declare full Mage E2E product pass before skin/deformation/runtime are actually exercised.

---

# 16. FIT8 / generalization context

FIT8 family selection is already frozen for future post-Mage work:

1. Mage
2. Goblin Male
3. Cow
4. Pug
5. Witch
6. Elf
7. Rogue
8. Cleric

The seven non-Mage witnesses are intentionally supposed to come from raw Drive banks via fresh generic native-1024×8 preparation, not by quietly reusing the old prepared master corpus.

FIT8 scientific training/evaluation remains blocked until the Mage 1FIT architecture is actually closed.

However, prep work can proceed separately.

Generalization has not been proven for IRIS or Geppetto merely because Mage fits.

---

# 17. Current repo hygiene / operational rules

- Single canonical repo.
- Active branch: `e2e/mage-scene-first-v1-20260905`.
- Re-fetch HEAD before every new write.
- Avoid unnecessary pushes because push-triggered workflows run automatically.
- A previous accidental branch `dummy-unused` exists pointing to an old head; do not use it.
- Older audit-temp branch also exists; do not use it.
- Do not create parallel patch streams.
- Prefer additive challengers until evidence justifies canonical replacement.
- Record canonical reports after meaningful scientific results.
- No dummy assets for product evidence.

At handoff creation, `model-mainline-source-gate` for commit `ba634955...` had passed. A push-triggered `mage-scene-first-geppetto-fit-v1` workflow also launched automatically; do not mistake that workflow side effect for a new scientific challenger experiment.

---

# 18. Current user preference / collaboration style relevant to continuation

The user wants:

- concise but technically substantial answers
- architecture audit before major changes
- measured fact vs hypothesis clearly separated
- no scope drift
- no late revelation that a supposed gate never actually tested what was claimed
- no simplifying final architecture merely to make a demo pass
- literature mechanisms should be used when they solve an already-known problem rather than reinventing the entire field
- deterministic layers are a first-class part of RealSaS and must be included in product scoring

When saying “PASS,” always name the level:

- `FIT-OPT PASS`
- `FIT-PRODUCT PASS`
- `GEN-PRODUCT PASS`

Never say a narrow teacher-exact diagnostic is “RealSaS solved.”

---

# 19. Immediate next-turn checklist

If the user says **“bitti”** or similar after opening the next chat:

1. Re-fetch branch HEAD.
2. Search Drive for `RealSaS_MAGE_GEPPETTO_B1S_PHASE_EPS_SWITCH_A100_V1_NO_TOKEN`.
3. Read LATEST → summary → final manifest → result package.
4. Verify the phase switch and final stable gate metrics.
5. Write a canonical result report to the branch.
6. Mark the exact-teacher optimizer series as `FIT-OPT` diagnostic closure if stable PASS.
7. Continue directly to the current RigAnything mechanism / production-boundary plan; do not reopen old GFDR or another optimizer tweak by default.

If the user says **“repoda direkt kaldığımız yerden devam et”** without mentioning the notebook result:

- read this handoff
- inspect current HEAD
- continue from the scientific plan above
- if the phase-switch result is still missing, ask only whether it finished **after** checking whether Drive already contains the completed artifacts.

---

# 20. Short current-state summary

The project is no longer stuck on “can Geppetto represent 41 teacher controls?” B1a already showed it can under stable identity. B1s + optimizer diagnostics identified a late Adam effective-step instability, and the current phase-switch run appears to have achieved a very long exact FIT-OPT streak.

The more important architectural direction is now:

- keep the successful scene-first IRIS/GSA substrate;
- clean deterministic GSA→Geppetto evidence boundaries;
- test RigAnything-proven generation mechanisms, especially full surface-token access and conditional joint diffusion;
- keep Compiler as canonical authority;
- define actual product PASS on the compiled rig’s mechanical/deformation behavior rather than authored teacher identity/count.

That is the state from which the next chat should continue.
