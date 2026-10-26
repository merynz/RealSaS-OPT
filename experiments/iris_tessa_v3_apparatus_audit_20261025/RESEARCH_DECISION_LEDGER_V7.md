# IRIS/TESSA Research Decision Ledger V7

**Date:** 2026-10-26  
**Status:** `ACTIVE_CANONICAL_RESEARCH_MEMORY__PRE_V3_1_CLOSURE`  
**Supersedes for working decisions:** V6. Older ledgers remain provenance.  
**Authority:** architecture-search only.

## Research operations

- Synthetic/analytic/bounded micro-learning courts use sealed GitHub workflows.
- Workflow success is execution evidence only; scientific verdict comes from the preregistered report.
- New jobs use `timeout-minutes: 360` as a hung/runaway guard. Runtime is not a scientific gate.
- Preserve `question -> exact treatment/control -> metric -> result -> interpretation -> carry-forward choice -> unresolved dependency` after every court.
- Real dataset/checkpoint/training/product authority returns to Platform `Experiment -> Attempt -> Artifact/CAS` after architecture freeze / Truth Builder.

## Closed carry-forward decisions

### O1 — observation foundation and alpha/support contract

Selected pre-Truth contract:

`frozen DINOv2-S/14 -> aligned L2 + FINAL evidence + patch-aligned RAW_ALPHA64 residual`

Evidence:

- final-only DINO loses local detail; aligned L2+FINAL preserves it;
- alpha cannot be discarded before scene construction;
- scalar/mean-alpha is insufficient;
- alpha court run `40315204922` selected `DUAL_RAW_ALPHA64`;
- raw-alpha64 achieved mean/min alpha recovery `1.0/1.0`, mean final cosine `0.99738`, local recovery `1.0`, within the current adapter budget;
- no-alpha and mean-alpha controls remained at chance (`~0.499`).

DINOv3 adoption remains unauthorized.

### O2 — cross-view and camera contract

Retain query-mediated cross-view fusion. Retain current factorized camera + UV representation. Held-out yaw/scale camera V2 reached `1.0/1.0`; deterministic backprojection supplied zero gain.

### O3 — deterministic coordinate coherence

Typed coordinate domains, explicit IRIS/TESSA conversion and exact carrier cell centers are mandatory. Learned unit/half-cell correction remains ineligible.

### O4 — shared-scene degree

Pre-Truth default is **shared trunk + protected private GM/GP subspaces (S2)**. Full undifferentiated shared-Z defaults showed >15% synthetic negative transfer. Authored causal Truth Stage B can overturn this provisional choice.

### O5 — endpoint evidence binding

Typed endpoint refs and endpoint-local evidence must be bound before TESSA presentation compression: query `Q(...,x_a)` and `Q(...,x_b)` after deterministic coordinate conversion, then structuralize already-addressable evidence.

### O6 — extraction and witnesses

Carry policy form:

`OBSERVABILITY_GATED_BOUNDED_RESOLUTION_ESCALATION -> deterministic XYZ_FPS`

Witness budget cannot recover supports absent from extraction. Production resolution/escalation thresholds remain deferred to normalized real morphology census.

### O7 — TESSA local history

Retain local attention window `4096` pre-Truth; no evidence to increase. Do not reduce until authored/real learned dependency span exists.

### O8 — query operator class

Oracle-Z V4 run `40318916429` established that the earlier surface calibration failure was primarily a query-capacity/coordinate-basis confound.

Selected query class:

- width `192`;
- three hidden SiLU layers;
- Fourier4 xyz basis `[xyz, sin/cos(2^k*pi*xyz), k=0..3]`;
- V4 calibrated tiny-court LR `3e-3`.

Selected Q192/Fourier4 direct-XYZ control:

- mean nRMSE `0.09863`;
- worst `0.12229`;
- mean local `0.12284`;
- random-label firewall mean `1.28831`.

This freezes the query **class**, not per-scene weights and not a product optimizer.

### O9 — final pre-Truth continuous carrier

V6 selected `DENSE3D` as the final pre-Truth carrier dependency for joint courts.

Frozen dependency file:

`ORACLE_Z_FINAL_CARRIER_SELECTION_V6.json`

Evidence carried by that file:

- all V6 evidence gates true;
- mean Dense3D relative gain `0.05421` over the matched alternative;
- paired optimizer-seed gains `0.03489 / 0.05944 / 0.06830`;
- Dense3D lower on all optimizer seeds;
- no group worse by >10%;
- support-count not worse per seed.

This is a bounded pre-Truth architecture selection, not a claim that Dense3D is universally optimal or deployable as per-scene fitted state.

### O10 — relation gradient ownership

`CONTINUOUS_JOINT_SYSTEM_V1` isolated a joint failure: endpoint relation learning remained strong while GM/GP geometry and negative-transfer gates failed.

Sealed Gradient Ownership V1 then varied only relation-gradient ownership and relation-private low-rank capacity.

Exact evidence:

- probe/workflow head SHA `439ab65bd1f104fa3d615627f1024dbdc426c9c2`;
- run `40335120772`;
- job `121614497333`;
- artifact `iris-gradient-ownership-v1-report`, ID `12656684424`;
- artifact ZIP SHA256 `9bce9d56de7f24e112769a6b05d6dbfc381fc23e0abf9363fa7bdeb521a58a00`;
- status `GRADIENT_OWNERSHIP_CLOSED_TINY_V1`;
- interpretation `OPTIMIZATION_COUPLING`;
- selected treatment `STOPGRAD_RELATION`.

Key matched-arm results:

- `CURRENT_FULL_GRAD`: GM/GP nRMSE `0.15958/0.16393`, relation acc `0.98634`, GM/GP transfer `+36.29%/+22.05%` -> invalid;
- `PRIVATE_REL_ADAPTER_FULL_GRAD`: GM/GP `0.15963/0.16209`, relation `0.98618`, transfer `+36.34%/+20.68%` -> invalid;
- `STOPGRAD_RELATION`: GM/GP `0.12461/0.12774`, relation `0.98558`, transfer `+6.43%/-4.89%` -> valid;
- `PRIVATE_REL_ADAPTER_STOPGRAD`: same geometry and valid relation, but +0.603% parameters -> valid but loses preregistered tie-break to no-adapter stop-grad.

Controls:

- random-relation firewall `0.503743` -> PASS;
- rolled-carrier local error increase `9.248654` relative -> PASS;
- adapter budget <=5% -> PASS.

Carry forward exactly:

`relation head <- stopgrad(H_GP(a)), stopgrad(H_GP(b))`

Relation BCE trains the relation head but is not allowed to update endpoint GP state, GP adapter, shared Q or carrier. GM/GP continuous losses continue to train the geometry path normally.

A LoRA/private relation adapter remains allowed as a later matched treatment if a separate capacity limitation appears, but it is **not selected** by current evidence.

### O11 — Truth Builder structural requirements

- positive-only occlusion truth is ineligible;
- first-hit binary visibility is not alpha-compositing truth;
- ordered-ray transmittance/contribution quantities are required;
- mechanics tangent truth needs anisotropy/validity handling;
- relation families must declare symmetry/direction/complementarity/ray-global algebra before final truth-dependent heads.

## Active court

### Continuous Joint System V2 — selected gradient ownership treatment

Authorized by Gradient Ownership V1. V2 must rerun the full V1 joint apparatus and controls with the **only scientific architecture change** being `STOPGRAD_RELATION` on relation evidence.

Frozen:

- final V6 carrier = `DENSE3D`;
- query = `Q192_D3_FOURIER4`;
- S2 GM/GP private adapters, bottleneck 64;
- scene/truth bank, endpoint target, optimizer, LR, steps, pool, evaluation and all V1 gates;
- no LoRA/private relation adapter unless V2 itself fails in a way that requires a separately preregistered capacity court.

Required V2 controls remain:

- GM and GP single-family oracles;
- S2 bound selected arm;
- full-shared bound placement control under the same stop-grad ownership rule;
- unbound endpoint control under the same stop-grad ownership rule;
- random-relation firewall under stop-grad;
- rolled-carrier ownership control.

V3.1 freeze remains unauthorized until V2 and the remaining observation/fusion prerequisites close.

## Remaining pre-V3.1 closure

1. execute and close `Continuous Joint System V2` with selected `STOPGRAD_RELATION` treatment;
2. verify/close `Observation Endpoint Interaction` result authority;
3. verify/close `DINOv2 V3.1 Fusion Form` result authority;
4. assemble `IRIS_TESSA_V3_1_ARCHITECTURE_FREEZE.md` only if all pre-freeze dependencies are scientifically closed;
5. freeze explicit post-freeze Truth Builder dependencies.

After architecture freeze, authored causal Truth Builder work and real model training return to Platform authority, followed by FIT8, then LOFO/unseen.
