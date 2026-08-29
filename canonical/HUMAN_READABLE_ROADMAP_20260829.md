# RealSaS — Human-Readable Continuation Roadmap — 2026-08-29

**Purpose:** one plain-language map of what is actually known, what is closed, and what comes next. Internal experiment codenames remain in sealed files but are not required to follow continuation.

## 1. What geometry accuracy do downstream systems tolerate? — CLOSED

We injected controlled ray-aligned depth errors into real 8-view geometry and replayed the downstream historical D2 consumer chain.

After the robust deterministic local-plane normal derivation, the observed ALL8 transition is approximately:

`0.00250 <= depth RMS critical boundary < 0.00275`

For the matched high-frequency Gaussian-like case, that corresponds roughly to depth absolute-P95 around `0.00490–0.00539`.

This is a measured tolerance chart for the tested consumer profile, not a universal product constant.

## 2. Why did the old learner miss that accuracy? — CLOSED

The FIT_PROXY32 audit established that the historical metric is depth-dominated at the relevant scale and that the old learner failure is **not** explainable as pure unseen-family generalization collapse.

Observed decomposition:

- there is already a global fitting/precision deficit on training families at the large-family rung;
- there is an additional unseen-family gap;
- family/source tails matter strongly;
- increasing family diversity helped transfer while reducing per-family exposure under the fixed 7168-step budget.

Therefore the old 32 -> 128 -> 512 family experiment is not a controlled model-capacity ladder.

Consequence: test pretrained representation accessibility under one common population/exposure budget rather than keep tuning deterministic geometry operators.

## 3. Can the real downstream Compiler route consume our 2.5D substrate? — CLOSED

A minimal sacrificial route passes end to end:

`ClosedRiggingVolume V0 -> InteriorRiggingSubstrate V0 -> G0 -> exact Compiler -> A0 -> exact Compiler -> deformation/proof`

Clean real witness:

- 512 surface nodes;
- 5/5 skeleton joints qualified;
- 512/512 skin rows qualified;
- exact Compiler graph optimizer proves the arborescence;
- Compiler mints canonical `J:*` IDs;
- deformation is finite, nontrivial and bounded;
- proof/runtime bind to the exact product state.

The coupling probe also passes: a severe 75% inward collapse of non-root G0 geometry remains clearly visible in final posed-surface response even after A0 recomputes its weights.

Thus the downstream route genuinely consumes skeleton geometry in this sacrificial profile. This proves route validity, not product-quality Geppetto/Arachne.

## 4. Controlled DINOv2 experiment design — PREREGISTERED

Plain-language question:

> With the learner and downstream route fixed, does a stronger frozen DINOv2 visual representation make the required camera-forward depth easier to learn?

Candidates:

- DINOv2 S — 384-d
- DINOv2 B — 768-d
- DINOv2 L — 1024-d
- DINOv2 g — 1536-d

The important controls are now frozen before candidate outputs:

- exact same historical 512-FIT-family population for all four rungs;
- both fixed styles for every family;
- same deterministic sample/order stream;
- same Mode-G 8-view native-1024 observations and exact cameras;
- same full-canvas 518x518 DINO raster / 37x37 patch grid;
- same native-1024 detail/support path;
- fixed non-trainable zero-pad isometric lift to 1536-d;
- identical post-lift normalization;
- identical trainable fusion/head bytes/config/parameter count;
- depth `d` is the only learned geometry authority;
- no learned normal, camera or risk head in this representation ladder;
- no constant common-risk-band ranking path;
- no candidate-specific stopping/tuning.

Exposure is also frozen:

- update 7168 = historical-budget diagnostic = 112 nominal exposures/family;
- update 32768 = primary comparison = 512 nominal exposures/family.

The 32768-step point is not called asymptotic convergence. It is simply the same substantially less-starved budget for all candidates.

Primary PASS at update 32768 requires both:

- FIT_PROXY32: at least 61/64 direct actual-residual replay PASS, with zero hard route failures;
- TRAIN_DIAG32: at least 61/64 direct actual-residual replay PASS, with zero hard route failures.

Depth RMS/P95 summaries remain diagnostics. They cannot override direct replay.

## 5. What comes next? — ZERO-STEP PREFLIGHT

**Training is still closed.**

Before a single DINO optimizer step, the preflight must prove:

1. the exact historical 512-family membership object is recovered/reconstructed and matches its frozen canonical SHA;
2. all 512 train families and all 32 held-out families exist in native-1024 authority with no substitution;
3. exact official DINO source/model/weight bytes are SHA-sealed;
4. the shared trainable architecture/config is sealed and has exactly equal trainable parameter count for S/B/L/g;
5. one deterministic sample/order manifest is sealed;
6. cached-vs-online frozen token parity passes;
7. the frozen DTB-ND1 evaluator can consume actual candidate residual fields without changing semantics;
8. DEV32 and other closed sets remain unopened.

Current honest blocker: the historical membership hash/selection rule are known, and the old runner identifies the original membership filename, but the exact membership object itself has not yet been recovered from persistent authority. A new 512-set with the same source quota is **not** an acceptable substitute.

## 6. Training status

**DINOv2 S/B/L/g scientific optimizer steps: 0.**

Training becomes eligible only after the zero-step preflight closes cleanly and this preregistered state is promoted to canonical `main`.

## 7. Later branches, only if needed

If all rungs fail because TRAIN_DIAG32 itself is underfit, localize shared learner/optimization adequacy first.

If training fit is adequate but held-out representation access still fails:

1. controlled foundation fine-tuning may be opened as a separately preregistered phase;
2. MapAnything / DA3 or other full geometry systems may be compared as later system-level interventions;
3. final models must still be judged by actual residual fields through frozen downstream replay, not by depth summary metrics alone.

No further cheap deterministic normal/window/hull/persistence tuning should be opened from Phase-1A candidate outcomes.

## One-line state

`depth tolerance measured -> old learner failure decomposed -> real consumer route + coupling validated -> DINO S/B/L/g preregistered -> exact zero-step preflight next -> then, and only then, training`
