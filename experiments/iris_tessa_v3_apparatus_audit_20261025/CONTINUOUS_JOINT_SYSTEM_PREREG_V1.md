# Continuous Joint System Court V1 — preregistration

**Status:** SEALED BEFORE JOINT OUTCOME  
**Authority:** final bounded pre-V3.1 interaction/falsification court only.

## Dependency amendment (pre-outcome)

V5 produced a valid but threshold-sensitive **provisional** carrier decision, so a sealed V6 optimization-robustness court was added before any joint-system execution. The joint court therefore consumes the final V6 pre-Truth carrier selection rather than the provisional V5 choice. No joint-system scientific constants or gates below changed.

## Trigger / frozen dependency

This court MUST NOT choose its own carrier. It consumes exactly:

`ORACLE_Z_FINAL_CARRIER_SELECTION_V6.json`

produced after the sealed V6 carrier robustness court. If V6 does not produce a valid final pre-Truth carrier selection, this court returns blocked and V3.1 cannot freeze.

The query class is already frozen by V4: `Q192_D3_FOURIER4`.

## Purpose

Test whether the selected continuous carrier/query can jointly support geometry/mechanics-like and graphics/presentation-like continuous evidence while using the selected S2 placement:

`shared carrier Z -> shared query Q(Z,x)=H0 -> private H_GM / H_GP adapters`

and whether endpoint binding remains necessary after this continuous representation is introduced.

This is the final synthetic architecture falsification before V3.1 freeze. Observation/DINO/alpha compatibility is handled by the separate observation-endpoint interaction court.

## Scene family

Reuse the V3 analytic surface scenes for five seeds and factors `4,2,1` times `h_ref`, both `SURFACE_STRUCTURED` and `THIN_SURFACE`.

For every scene/query point define:

- GM response: the exact V3 4-D surface response;
- GP response: the same geometry support weights with an independently seeded 4-D appearance/presentation coefficient bank, so geometry is shared but family-private response information differs;
- directed endpoint relation: for two sampled points, whether GP channel 1 at endpoint A is greater than GP channel 1 at endpoint B.

## Architecture

Selected carrier parameters are scene-specific latent state for this oracle architecture court. They are not deployable per-scene fitted parameters.

A single shared Q uses:

- selected carrier sample;
- Fourier4 xyz basis;
- width 192;
- three SiLU layers;

and emits shared local `H0`.

Selected S2 arm applies separate residual private adapters:

- `H_GM = A_GM(H0)`;
- `H_GP = A_GP(H0)`.

GM/GP heads read their respective private state. The bound relation head receives endpoint-local `H_GP(a), H_GP(b)` before relation compression.

## Arms

1. `GM_SINGLE_ORACLE` — selected carrier/Q + GM adapter/head only.
2. `GP_SINGLE_ORACLE` — selected carrier/Q + GP adapter/head only.
3. `S2_JOINT_BOUND` — selected full assembly.
4. `FULL_SHARED_JOINT_BOUND` — no GM/GP private adapters; same carrier/Q and local heads.
5. `S2_JOINT_UNBOUND` — local heads unchanged; relation receives global compressed GP state instead of endpoint-local state.
6. `S2_JOINT_BOUND_RANDOM_RELATION` — relation-label firewall.

All joint arms use the same selected carrier type and Q class. Report parameter counts; single-family arms are oracle references, not budget competitors.

## Optimization

- LR `3e-3`;
- 6000 steps per arm;
- fixed batch/pool/eval construction;
- five scene seeds;
- no runtime-based truncation;
- GitHub job uses the 360-minute runaway guard.

## Metrics / gates

Use normalized RMSE for GM and GP, and relation accuracy.

`S2_JOINT_BOUND` must satisfy:

- GM overall mean nRMSE <=0.15;
- GP overall mean nRMSE <=0.15;
- relation mean accuracy >=0.95;
- relation target positive fraction in `[0.35,0.65]`.

Negative transfer relative to matching single-family oracle must be <=15% for both GM and GP error.

Ownership / controls:

- unbound relation accuracy <=0.60 while GM/GP remain within 15% of selected bound local errors;
- random-relation firewall mean accuracy in `[0.48,0.52]`;
- rolling scene carrier state across scenes must increase selected-arm mean local error by >=25%;
- full-shared arm is not allowed to beat S2 by >=10% on both GM and GP simultaneously. Sharing-degree authority remains Stage A; this gate only detects a catastrophic placement contradiction.

## Verdict

If all gates pass: `CONTINUOUS_JOINT_SYSTEM_CLOSED_TINY_V1`, authorizing V3.1 architecture freeze.

If they fail, V3.1 remains open and the failure owner must be isolated rather than weakening the gate post hoc.

## Non-claims

No real-model training, Truth Builder label adequacy, product authority, FIT8 genericity, or final relation-loss choice is proven here.
