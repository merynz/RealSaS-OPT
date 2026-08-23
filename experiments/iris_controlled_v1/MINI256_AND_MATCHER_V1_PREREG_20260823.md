# IRIS Controlled V1 — Evidence Matcher V1 + Mini256 prereg

Date: 2026-08-23. Pre-run status: optimizer for Mini256 = 0. Sealed CAL/DEV/EXTERNAL remain closed.

## Purpose
Measure the **complete currently-supported IRIS evidence system**, not raw Z retrieval, before any 1024/full-corpus scale-up.

## Matcher V1
No D3. No hidden track/owner identity authority. Matcher consumes only predicted P/N/U/Z, observed grid coordinates and known controlled view structure.

Pipeline:
1. controlled row corridor candidate domain;
2. Z_coarse top-8 high-recall retention;
3. FIT-calibrated rank fusion of Z_coarse, Z_fine and uncertainty-normalized P distance inside retained candidates;
4. reciprocal top-4 consistency;
5. 3+-view image-coordinate cycle support;
6. consistency-first final rerank;
7. FIT-calibrated >=99% singleton-precision margin;
8. confident singleton, otherwise adaptive top-4/top-8 set-valued ambiguity.

GT physical track IDs are evaluator-only and forbidden to matcher decisions.

## Mini256
- 224 FIT / 32 TUNE;
- deterministic hash selection;
- exact 64-asset ceiling witness is retained as a subset;
- 12 epochs default;
- corrected v1.1 trainer;
- TUNE selection always uses the same full objective;
- both cel-clean and ink-cel are evaluated by Matcher V1;
- no sealed panel.

## Decision
GREEN requires >=50% TUNE P and N error reduction from random init, Matcher V1 final top4 >=.90, top8 >=.97, adaptive-set coverage >=.98, and calibrated singleton precision >=.97 at >=.10 coverage.

AMBER means learning exists but complete matcher misses GREEN without meeting RED. Do not scale; ablate matcher components.

RED if P or N reduction <.30, matcher top8 <.90, or adaptive-set coverage <.90. Do not scale.

GREEN authorizes a 1024 intermediate run, **not immediate full 3248**.
