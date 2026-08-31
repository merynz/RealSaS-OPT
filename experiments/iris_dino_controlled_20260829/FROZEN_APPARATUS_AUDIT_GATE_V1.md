# RealSaS — Frozen Apparatus Audit Gate V1

**Date:** 2026-08-31  
**Status:** `REQUIRED_BEFORE_NEXT_DINO_TRAINING`  
**Training authorized:** NO

This gate exists because S1 exposed a recurrent program risk: the component held fixed for causal isolation may itself be the active bottleneck.

No new decoder, loss, interface or backbone-scale experiment may open until every item below is inspected from actual source/config/runtime artifacts and given an explicit PASS / KNOWN_LIMITATION / BLOCK decision.

## A. Representation extraction and cache

- exact DINO model ID, source revision and weight SHA;
- exact layer/token used;
- native input -> 518 preprocessing;
- channel normalization and dtype;
- 37x37 token geometry;
- zero-pad `Q_d` implementation;
- per-token FP32 L2 normalization;
- cached-vs-online parity;
- cache membership and ordering;
- no candidate-specific preprocessing.

## B. Sampling and supervision

- exact 4096-locus selection algorithm;
- visible-support definition and mask authority;
- without-replacement versus coverage-repeat behavior;
- deterministic RNG/seed derivation;
- distribution of sampled loci by silhouette distance, thin structure, grazing geometry and broad interior;
- whether any family/view can dominate effective supervision;
- truth construction `d_truth = dot(P-O,F)`;
- no mechanics-derived sampling or weighting.

The audit may diagnose poor coverage. It must not retroactively tune S1.

## C. Shared learner / spatial access

- token stem dimensions and parameter count;
- within-view spatial reasoning;
- cross-view 37->16 pooling, row fusion and 16->37 interpolation;
- context-fuse path;
- native-1024 stencil coordinates/radii and encoder capacity;
- raster-query sampling method;
- final decoder/head and whether neighboring output queries are coupled;
- receptive field available to a native pixel;
- whether the native route can bypass DINO;
- whether DINO information can be lost before the output head.

## D. Loss and optimizer

- exact SmoothL1 implementation and `beta`;
- reduction domain and mask;
- error distribution relative to beta and DTB tolerance;
- mean versus tail sensitivity;
- optimizer type/betas/weight decay;
- MAIN and TAIL LR values;
- fresh-moment reset at step 2049;
- AMP/GradScaler behavior;
- gradient accumulation semantics;
- effective families/styles/views/loci per logical update;
- resume behavior and RNG/sample-stream continuity.

## E. Evaluator and residual carrier

- exact actual-residual mask;
- eligibility rules producing FIT/TRAIN eligible-cell counts;
- zero-reference construction;
- robust local-plane normal operator and fixed parameters;
- D2-support construction;
- Geppetto/Arachne proxy checks and thresholds;
- hard-route failure semantics;
- style aggregation;
- scalar diagnostics may not override direct replay;
- no hidden consumer-specific information enters training.

## F. Corpus substrate

This audit is separate from the active Geppetto/Arachne clean-character audit, but the next IRIS experiment must explicitly state which corpus membership it uses.

Before any clean-C0 causal comparison:
- membership must be frozen before outcomes;
- contamination categories must be defined;
- no threshold may be relaxed because too many assets fail;
- random visual sanity checks must accompany automatic audit results.

## G. Required audit output

The closure must include:
- source file/path + SHA for every audited component;
- exact configuration values;
- known limitations;
- a causal-confound table;
- explicit allowed/forbidden claims;
- one final decision: `PASS_FOR_NEXT_EXPERIMENT`, `BLOCK_AND_FIX`, or `PASS_WITH_NAMED_LIMITATIONS`.

Until then, next DINO training is forbidden.
