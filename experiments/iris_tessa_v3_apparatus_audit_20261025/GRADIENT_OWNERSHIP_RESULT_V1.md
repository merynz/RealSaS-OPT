# Gradient Ownership / Private Relation Adapter Court V1 — result

**Date:** 2026-10-26  
**Status:** `GRADIENT_OWNERSHIP_CLOSED_TINY_V1`  
**Interpretation:** `OPTIMIZATION_COUPLING`  
**Selected treatment:** `STOPGRAD_RELATION`

## Exact execution authority

- Branch: `research/iris-tessa-v3-apparatus-audit-v1`
- Exact probe/workflow head SHA: `439ab65bd1f104fa3d615627f1024dbdc426c9c2`
- Workflow run: `40335120772`
- Job: `121614497333` (`gradient-ownership-scientific`)
- Runner: GitHub-hosted Ubuntu
- Scientific artifact: `iris-gradient-ownership-v1-report`
- Artifact ID: `12656684424`
- Artifact ZIP SHA256: `9bce9d56de7f24e112769a6b05d6dbfc381fc23e0abf9363fa7bdeb521a58a00`
- Scientific schema: `RealSaS.GradientOwnershipCourt.v1`

Workflow success is execution evidence only. The scientific verdict below is the emitted preregistered report interpreted under `GRADIENT_OWNERSHIP_PREREG_V1.md`.

## Frozen dependency

The court consumed `ORACLE_Z_FINAL_CARRIER_SELECTION_V6.json`:

- selected carrier: `DENSE3D`;
- schema: `RealSaS.OracleZFinalCarrierSelection.v6`;
- status: `FINAL_PRETRUTH_CARRIER_SELECTION_V6`;
- all V6 evidence gates true;
- Continuous Joint dependency authority true.

No carrier, truth family, endpoint semantics, optimizer, loss weighting, step budget, batch construction, GM/GP adapter, or query-class change was admitted.

## Scientific arms

| Arm | GM nRMSE | GP nRMSE | Relation acc. | GM neg. transfer | GP neg. transfer | Joint-valid |
|---|---:|---:|---:|---:|---:|---|
| `CURRENT_FULL_GRAD` | 0.159582 | 0.163927 | 0.986344 | +36.29% | +22.05% | NO |
| `STOPGRAD_RELATION` | 0.124614 | 0.127742 | 0.985579 | +6.43% | -4.89% | YES |
| `PRIVATE_REL_ADAPTER_FULL_GRAD` | 0.159634 | 0.162092 | 0.986182 | +36.34% | +20.68% | NO |
| `PRIVATE_REL_ADAPTER_STOPGRAD` | 0.124614 | 0.127742 | 0.986377 | +6.43% | -4.89% | YES |

Single-family references:

- `GM_SINGLE_ORACLE`: GM overall nRMSE `0.117088`, surface-local `0.148925`;
- `GP_SINGLE_ORACLE`: GP overall nRMSE `0.134311`, surface-local `0.166168`.

The rank-16 private relation adapter adds only `0.6031%` trainable parameters, so adapter arms were budget-admissible. It nevertheless failed to rescue the full-gradient arm. The stop-gradient arm without a private relation adapter tied the stop-gradient+adapter arm on geometry and used fewer parameters, therefore the preregistered selection rule chooses `STOPGRAD_RELATION`.

## Selected-arm controls

- ordinary mean surface-local nRMSE: `0.161258`;
- rolled-carrier mean surface-local nRMSE: `1.652679`;
- rolling carrier relative error increase: `9.248654` (= ~`924.9%`, far above the +25% gate);
- random-relation firewall accuracy: `0.503743` (inside `[0.48,0.52]`);
- relation target positive fraction: `0.5`;
- selected adapter-overhead gate: PASS (selected treatment has no adapter).

## Scientific interpretation

The failed V1 joint behavior is causally consistent with **relation-loss optimization coupling into the shared geometry path**. Relation learning itself is not the problem: full-gradient arms reach ~0.986 relation accuracy while violating GM/GP geometry and negative-transfer gates. Adding private low-rank relation capacity without changing gradient ownership does not repair those failures. Blocking relation BCE gradients at endpoint GP evidence restores both continuous tasks while preserving relation accuracy.

Carry forward exactly:

`relation head <- stopgrad(H_GP(a)), stopgrad(H_GP(b))`

The relation head remains trainable. GM/GP continuous losses continue to train the GP adapter, shared Q and scene carrier normally. A private LoRA/low-rank relation adapter is **not selected pre-Truth**, but remains an admissible later treatment if a subsequent matched court demonstrates a separate relation-capacity limitation.

## Authority

This result authorizes one next step only:

- full `Continuous Joint System V2` rerun using `STOPGRAD_RELATION` plus the original V1 controls and unchanged scientific gates.

It does **not** authorize:

- V3.1 architecture freeze;
- Truth Builder training;
- FIT8 training;
- product authority changes.
