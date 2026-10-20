# RealSaS-OPT — Current State

**Date:** 2026-10-20  
**Production authority:** `main` until reviewed merge  
**Active research branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Source-gate vehicle:** draft PR #130 — `IRIS: H-GEO-P3 scale-transfer stage localization`  
**Active research lane:** `SURFACE_TESSA`  
**Active experiment:** `H-GEO-P3-SCALE-XFER`  
**Runtime source:** pending final green freeze  
**Parked lane:** `RELATION_V10` — not cancelled  
**Scientific status:** **NO GLOBAL IRIS FIT8 PASS / NO TESSA PASS / SURF-A0 FALSIFIED / SURF-C0 MEASURED / SURF-A1 FAIL / H-REP-P0 FAIL WITH LOCAL SIGNAL / H-GEO-P1 FAIL WITH MATERIAL RESCUE / H-GEO-P2 FAIL WITH 3-OF-4 ABSOLUTE GEOMETRY GATES PASS + SHARED-H RETENTION FAIL / H-GEO-P3 FAIL WITH 3-OF-4 ABSOLUTE GEOMETRY GATES PASS + EXACT SHARED-V8 PRESERVATION / EFREQ SEALED UNRESOLVED / SCALE-XFER ZERO-OPTIMIZER STAGE LOCALIZATION ACTIVE**  
**Optimizer policy:** **zero optimizer steps are authorized in H-GEO-P3-SCALE-XFER.** No P4, SURF-A2 or TESSA optimizer step is authorized.

Machine-readable continuation authority: `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`.

## Product invariant

```text
8-view observations
        ↓
IRIS shared visual / semantic / mechanical evidence
        +
qualified explicit metric geometry evidence
        ↓
qualified high-fidelity direct surface
        ↓
OPTIONAL TESSA structuralization / learned retopology
        ↓
Compiler qualification
        ↓
ONE canonical animation-ready M
```

There is no independent deformable visual mesh. TESSA may structuralize correct geometry; it may not repair unqualified IRIS geometry.

## Immutable V8 authority

- source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- step `12000`;
- checkpoint SHA256 `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding SHA256 `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`.

## Closed geometry evidence

- `SURF-A0`: historical raw-field transition falsified.
- `SURF-C0`: native V8 extraction measured diagnostic debt.
- `SURF-A1`: extraction rescue improved quality but failed locked high-fidelity gates.
- `H-REP-P0`: sealed V8 H contains strong local geometry signal but fails absolute metric fidelity.
- `H-GEO-P1`: exact-triangle field-only retraining materially improved H but remained below absolute gates.
- `H-GEO-P2`: scene+field retraining passed distance/normal/near and all vs-P1 improvement gates, but endpoint and shared-H retention failed.
- `H-GEO-P3`: dedicated geometry carrier preserved V8 exactly and passed distance/normal/near, but endpoint p95 remained failed and worsened versus P2.

### H-GEO-P3-EFREQ — sealed `UNRESOLVED`

Authority: `docs/research/IRIS_H_GEO_P3_EFREQ_RESULT_20261020.json`.

- execution source `e3b46f3bc98cad0c48f37854de64de996192f667`;
- audit SHA256 `ab4ca656442de0addb9468c6dcd7bf3e1e23b5a874fb4e5b675f2079720f2d04`;
- query CSV SHA256 `41bf4f6bf1d90fa639e01fc799c032fb276ef51dd833fed1891ce4ce7c87a16d`;
- optimizer/probe optimizer steps `0 / 0`;
- locked signals all false, therefore classification `UNRESOLVED`;
- measured tails nevertheless had smaller projected footprint, more non-adjacent surface competition, and P3 H_geo was smoother than P2 H at fine perturbation scales.

The result authorizes only expanded zero-optimizer stage localization.

## Active experiment — H-GEO-P3-SCALE-XFER

Question:

> At which stage do endpoint-relevant fine spatial distinctions first become inadequate or ambiguous: frozen 37×37 view tokens, P3 scene/carrier transfer, or the single closest-surface target itself?

Authority:

- `canonical/H_GEO_P3_SCALE_XFER_INPUT_BINDING_V1.json`;
- `canonical/H_GEO_P3_SCALE_XFER_PREREGISTRATION_V1.json`;
- `docs/research/IRIS_H_GEO_P3_SCALE_XFER_DECISION_LEDGER_20261020.md`;
- `tools/iris_vnext/h_geo_p3_scale_xfer_v1.py`;
- `tools/iris_vnext/audit_h_geo_p3_scale_xfer_v1.py`.

The exact EFREQ query CSV is reused byte-for-byte. Tail/control membership cannot be changed.

Measurements:

1. frozen DINO token contrast for nearest-vs-competing exact surface points across all eight views;
2. exact non-adjacent multi-surface margins / normal disagreement;
3. deterministic spatial transfer curves at `1/1024 ... 1/32` IRIS through:
   - P2 H baseline;
   - P3 sampled 64² planes;
   - bilinear 128² planes;
   - learned-refined 128² planes;
   - geometry-field H;
   - continuous-position residual;
   - final H_geo.

Locked classification:

- `TARGET_MULTISURFACE_LIMITED`;
- `VIEW_INPUT_RESOLUTION_LIMITED`;
- `CARRIER_TRANSFER_LIMITED`;
- `MIXED`;
- `UNRESOLVED`.

No classification automatically authorizes training.

## SURF-A2 / TESSA

`SURF-A2-HIRES` remains blocked on SCALE-XFER and the subsequent preregistered geometry decision.

`TESSA-V2-B1` remains blocked until a genuinely qualified high-resolution direct A-arm exists.

## Notebook worker law

```text
open notebook
→ Run all
→ approve Google Drive exactly once
→ exact-hash autonomous execution
→ separated visible phases
→ zero optimizer/probe steps
→ scale-transfer audit + query table
```

No full-Drive discovery, second OAuth, hidden fallback, checkpoint write, source drift or threshold relaxation is allowed.

## Immediate continuation

```text
seal EFREQ result                                    [DONE]
        ↓
SCALE-XFER binding + preregistration                 [DONE]
        ↓
zero-optimizer stage-localization source + CI        [ACTIVE — PR #130]
        ↓
freeze exact final green SCALE-XFER SHA              [NEXT]
        ↓
fast-forward active research branch
        ↓
A100 Run-All SCALE-XFER notebook
        ↓
inspect stage-localized classification
        ↓
preregister exactly the minimal P4 family indicated by evidence
```
