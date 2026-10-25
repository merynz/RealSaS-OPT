# RealSaS — IRIS Cross-View Attention Placement Court Prereg V1

**Date:** 2026-10-25  
**Status:** `SEALED_BEFORE_OUTPUTS`  
**Execution class:** tiny synthetic CPU mechanism court.

## Question

Can the current IRIS placement — independently projected view tokens followed by canonical scene queries cross-attending the flattened all-view memory — perform cross-view correspondence robustly, or is source-token self-attention before scene queries materially beneficial?

## Task

- canonical scene: fixed 4x4 lattice = 16 loci;
- views: four fixed orthographic in-plane rotations: 0, 90, 180, 270 degrees;
- every canonical locus receives four independent fair bits, one per view;
- a source token contains `[projected_x, projected_y, cos(theta), sin(theta), signed_payload]`;
- all 64 source tokens are globally randomly permuted every batch;
- 16 canonical scene queries contain only canonical `(x,y)`;
- target at every canonical query is the ordered four-bit vector for views `[0,90,180,270]`.

Payload bits are resampled independently every example, so position/view correspondence is required and no semantic or neighboring-token shortcut exists.

## Arms

### A — CURRENT_QUERY_MEDIATED

- independent source projection `5 -> d_model=64`;
- learned canonical query + coordinate MLP;
- **three TransformerDecoder layers**;
- no source-source self-attention;
- shared `Linear(64,4)` output head per canonical query.

This is the current IRIS fusion placement analogue.

### B — SOURCE_SELF_ATTENTION_FIRST

- identical source projection;
- **one TransformerEncoder layer** over source memory before scene queries;
- then **two TransformerDecoder layers**;
- same query construction and output head.

This trades one decoder layer for one source self-attention layer rather than simply adding depth. Exact trainable parameter counts are reported.

### C — RAW_CORRESPONDENCE_POSITIVE_CONTROL

Use the known synthetic rotations to gather the four correct payloads for each canonical locus and pass them through a small identity-capable linear readout. This validates task generation/readout.

### D — RANDOM_LABEL_FIREWALL

Use A inputs unchanged but independent fair random four-bit targets absent from inputs.

## Seeds / budget

Seeds exactly `731, 947, 1217, 1597, 2017`.

- AdamW lr `1e-3`, betas `(0.9,0.999)`, weight decay 0;
- batch 128;
- exactly 1200 steps;
- BCE-with-logits;
- evaluation 4096 fresh examples/seed/arm;
- CPU only; no tuning/early stopping/schedule.

## Metrics

Per seed/arm:

- bit accuracy over 16x4 outputs;
- BCE.

Aggregate:

- mean/min/std bit accuracy;
- paired B-A gain;
- trainable parameter counts.

## Gates

Firewall D: every seed in `[0.48,0.52]`; otherwise `CROSS_VIEW_FUSION_FIREWALL_INVALID`.

Positive control C: every seed >= `0.995`; otherwise `CROSS_VIEW_TASK_INVALID`.

### Current placement adequate

`CURRENT_QUERY_MEDIATED_CORRESPONDENCE_ESTABLISHED_TINY_V1` when A:

- every seed >= `0.95`;
- mean >= `0.98`.

If A meets this, B may still be reported but no source-self-attention treatment is authorized solely by this task.

### Early source fusion supported

Only if A misses its adequacy gate, call `SOURCE_SELF_ATTENTION_SUPPORTED_TINY_V1` when:

- B every seed >= `0.95`, mean >= `0.98`;
- B beats A in >=4/5 seeds;
- mean paired B-A gain >= `0.10`;
- B trainable parameter count <= A.

If B improves but fails these conditions: `CROSS_VIEW_FUSION_SIGNAL_INCONCLUSIVE_TINY_V1`.

Otherwise: `CROSS_VIEW_CORRESPONDENCE_NOT_ESTABLISHED_TINY_V1`.

## Scope

This court tests attention placement/correspondence mechanism only. It does not test real camera geometry, DINO features, occlusion, causal graphics truth or reference-scale IRIS.

No outcome authorizes Truth139/FIT8 training, product change, ModelRelease promotion or architecture replacement without a subsequent real-data/reference court.
