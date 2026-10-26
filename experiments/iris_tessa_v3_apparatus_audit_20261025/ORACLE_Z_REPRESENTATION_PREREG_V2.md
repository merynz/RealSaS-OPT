# Oracle-Z Representation Court V2 — preregistration

**Status:** `SEALED_BEFORE_EXECUTION`  
**Execution authority:** Platform `Experiment -> Attempt -> Artifact Registry/CAS` only.  
**Source authority:** reviewed `main`; this review branch does not itself authorize a scientific execution.  
**Supersedes:** `ORACLE_Z_REPRESENTATION_PREREG_V1.md` only as the next apparatus design; V1 result remains immutable and classified `APPARATUS_INCONCLUSIVE__ESCALATE`.

## Research question

When perception is bypassed and scene state is optimized directly, which part of the current `Z -> Q(Z,x)` design limits continuous surface evidence: carrier representation, query-network sharing, optimization, or target spatial frequency?

V1 could not answer this because carrier, shared-Q capacity and optimization were entangled; even the dense volumetric positive control failed.

## Hypothesis under test

For RealSaS surface-structured evidence, a triplane carrier is not an obvious representation blocker if:

1. a per-scene upper-bound decoder can fit held-out continuous surface truth at the relevant frequency/thickness scale;
2. the same carrier remains decodable by a shared Q across scenes with a bounded generalization gap; and
3. a calibrated dense-3D control demonstrably solves an easier genuinely volumetric target, proving the court can expose a representation mismatch.

This court does **not** assume that one homogeneous shared scene is globally optimal. It tests only carrier/query capacity after perception is removed.

## Fixed arms

Carrier scalar budget is matched as closely as the V1 geometry permits and is reported exactly.

1. `TRIPLANE`
   - three 2-D planes;
   - bilinear `XY/XZ/YZ` sampling;
   - `align_corners=False`;
   - no explicit xyz in the base query.

2. `TRIPLANE_XYZ`
   - same carrier as `TRIPLANE`;
   - explicit xyz appended to Q;
   - diagnostic for coordinate reconstruction burden, not an automatic production choice.

3. `MULTIRES_TRIPLANE`
   - two triplane scales under approximately the same carrier-scalar budget;
   - tests whether thin/local evidence benefits from multiscale state rather than a larger total budget.

4. `DENSE3D`
   - trilinear 3-D carrier under the same carrier-scalar budget;
   - adversarial control, not a proposed default production representation.

No additional representation family may be added after seeing V2 outcomes. Hash grids, sparse voxels, tensor-factorized 3-D and other challengers require a separately preregistered V3 court.

## Court decomposition

### Court A — optimizer calibration

Optimization is calibrated on an **independent calibration family** not used for architecture selection.

- carrier: `DENSE3D`;
- decoder: per-scene Q;
- target: low-frequency smooth 3-way volumetric field;
- candidate learning rates fixed before execution: `1e-2`, `3e-3`, `1e-3`;
- fixed step budget per candidate: `1200`;
- selection: lowest held-out normalized RMSE, with ties resolved toward the smaller learning rate;
- selected optimizer settings are then frozen for all representation arms.

Calibration validity requires mean held-out nRMSE `<= 0.08` and worst-seed nRMSE `<= 0.12` on the easy volumetric target. If this fails, the entire V2 representation verdict is `APPARATUS_INCONCLUSIVE`.

### Court B — per-scene carrier/Q upper bound

Each scene receives its own carrier **and its own Q**. This intentionally gives the representation a generous upper bound and removes cross-scene shared-Q pressure.

Truth families:

- `SURFACE_STRUCTURED`: anisotropic authored components with geometry + three component-local evidence channels;
- `THIN_SURFACE`: the same family with controlled minimum thickness;
- `VOLUMETRIC_3WAY`: smooth genuine xyz interaction.

Difficulty is fixed in one shared truth domain for all arms. Define the reference spacing from the current/base triplane carrier as `h_ref = 2/24` in normalized IRIS coordinates. Every arm sees the **same** target at each ladder point:

- surface/thin ladder: minimum radius approximately `4h_ref`, `2h_ref`, `1h_ref`, `0.5h_ref`;
- volumetric ladder: maximum integer spatial frequency `1`, `2`, `4`, `6` cycles over the normalized domain.

No arm-specific rescaling of truth is allowed.

Held-out continuous query points are never optimized directly.

### Court C — shared-Q burden

For every carrier family, repeat the same scenes/difficulty ladder with a single Q shared across scenes while retaining per-scene carriers.

Report:

- mean/worst nRMSE;
- boundary nRMSE for surface families;
- shared-Q gap relative to the corresponding per-scene upper bound;
- seed variance.

A carrier is not blamed for a failure that appears only after replacing per-scene Q with shared Q; that is classified as a `SHARED_Q_CAPACITY_OR_ALIGNMENT` issue.

### Court D — negative-control firewall

Train on dynamically resampled coordinate-independent random targets under the same optimizer. Held-out performance must remain near chance/high error. A suspiciously low error invalidates the court.

## Seeds and data isolation

- comparison seeds: `731`, `947`, `1217`, `1601`, `1999`;
- calibration scenes/seeds are disjoint from comparison scenes/seeds;
- exact scene definitions are generated deterministically from sealed numeric seeds;
- no FIT8, Knight, Truth139 or authored semantic IDs are used.

## Interpretation rules

### Apparatus validity

V2 is interpretable only if:

1. Court A calibration passes;
2. the random-target firewall does not show leakage;
3. all compared arms use the frozen optimizer selected in Court A;
4. held-out continuous points remain disjoint from optimization samples.

### Surface-capacity support

At each difficulty, `TRIPLANE` receives `SURFACE_CAPACITY_SUPPORTED_TINY_V2` only if the **per-scene upper bound** has:

- mean nRMSE `<= 0.08` on `SURFACE_STRUCTURED`;
- mean nRMSE `<= 0.12` on `THIN_SURFACE`;
- worst-seed nRMSE `<= 0.18` on both;
- boundary mean nRMSE `<= 0.16`.

Failure at one difficulty does not retroactively fail easier scales; the output is a measured capacity curve.

### Explicit xyz

`TRIPLANE_XYZ` is preferred only if, at a difficulty where both courts are valid, it improves worst-seed shared-Q surface error by `>=20%` relative while not degrading another surface family by more than `0.01` absolute nRMSE.

### Multires

`MULTIRES_TRIPLANE` is preferred only if it improves worst-seed `THIN_SURFACE` error by `>=20%` at one or more hard ladder points while `SURFACE_STRUCTURED` remains within `+0.01` absolute nRMSE and carrier budget is not larger than the admitted tolerance recorded by the report.

### Dense control

A dense-grid win on `VOLUMETRIC_3WAY` is expected and proves court sensitivity. Dense 3-D is not selected for production from that result alone.

A triplane representation concern becomes credible only when:

- calibration is valid;
- `DENSE3D` solves the same target/difficulty under the same frozen optimizer;
- `TRIPLANE` fails already in the per-scene upper-bound court;
- the failure repeats across seeds.

Otherwise the classification is not `TRIPLANE_FAIL`.

## Platform execution contract

The scientific execution must be represented by:

- one preregistered Platform `Experiment` for this V2 question;
- one Platform `Attempt` per exact execution/retry identity;
- exact reviewed `main` `EngineRelease` / research plan;
- immutable JSON report registered in Artifact Registry/CAS;
- the Attempt linked to the Experiment;
- no ProductRevision and no ModelRelease promotion.

GitHub Actions may run syntax/unit/engineering checks but may not be cited as the canonical V2 scientific execution.

## Non-claims

This court does not establish:

- learnability from images;
- optimal shared-scene factorization;
- causal Truth correctness;
- interaction with DINO/alpha/camera evidence;
- TESSA structuralization quality;
- Compiler success;
- FIT8/LOFO/unseen genericity;
- production carrier resolution.

A later shared-scene-factorization court and joint tiny-system court remain mandatory even if V2 passes.
