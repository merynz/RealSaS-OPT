# Representation Sufficiency Battery V1 — Test-C Candidate-Breadth Counterfactual Prereg

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_AFTER_11032_FAILURE_LOCALIZATION_BEFORE_BREADTH_RESULTS`

## Question

Current Test C is RED on family 11032 because the current top-4 descriptor candidate sets often fail to include a target-near B projection. Determine whether this is a semantic/representation information failure or a deterministic candidate-search/truncation failure.

No model weights, descriptor embeddings, raster inputs, geometry, truth target, or learned scoring are changed. Truth is evaluator-only after candidate sets are generated.

## Frozen arms

Baseline is the current parent route:

```text
coarse foreground stride 4
coarse descriptor top 8
±4 px / step 2 refinement
retain final top 4 per visible view
all pairwise rank-3 H_i
```

Two monotone counterfactual arms are fixed before execution:

### Arm R — retention-only

```text
coarse top 8 unchanged
same exact refined pixel pool
retain final top 16 per visible view
```

This asks whether the relevant evidence is already in the refined pool but top-4 truncation discards it.

### Arm S — broader descriptor search

```text
coarse top 32
same ±4 px / step 2 refinement
retain final top 16 per visible view
```

This asks whether the frozen descriptor contains the relevant match but the current coarse-search beam is too narrow.

No threshold is fitted from truth. Candidate ranking remains frozen descriptor cosine. The feasible H set is generated exactly as before from all visible-view pairs.

## Evaluation

Use the already-preregistered Test-C evaluator mapping addendum (`5dc6989b2026a2db7194130242243b0ab6902b09`). Report strict 1x and primary 2x local-surface-scale containment for the same reliably mapped carriers.

Interpretation on 11032:

- if Arm R or S moves primary 2x containment to >=0.90, current Test-C RED is localized to candidate truncation/search capacity rather than absence of target information from the frozen descriptor representation;
- if both remain <0.90, the frozen descriptor/candidate representation remains a hard-tail insufficiency witness and a new observable/evidence mechanism is required;
- regardless of recovery, the existing top4 current production H remains RED and cannot be called sufficient without changing the candidate representation contract.

The counterfactual is truth-open development only and does not change frozen Stage-B qualification.
