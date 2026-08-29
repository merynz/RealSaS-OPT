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

Consequence: test pretrained representation accessibility/capacity under matched exposure rather than keep tuning deterministic geometry operators.

## 3. Can the real downstream Compiler route consume our 2.5D substrate? — CLOSED

A minimal sacrificial route now passes end to end:

`ClosedRiggingVolume V0 -> InteriorRiggingSubstrate V0 -> G0 -> exact Compiler -> A0 -> exact Compiler -> deformation/proof`

Clean real witness:

- 512 surface nodes;
- 5/5 skeleton joints qualified;
- 512/512 skin rows qualified;
- exact Compiler graph optimizer proves the arborescence;
- Compiler mints canonical `J:*` IDs;
- deformation is finite, nontrivial and bounded;
- proof/runtime bind to the exact product state.

The final missing coupling probe also passes: a severe 75% inward collapse of non-root G0 geometry remains clearly visible in final posed-surface response even after A0 is allowed to recompute its weights.

Thus A0 is not silently hiding a systematically bad G0 skeleton in this sacrificial profile.

This proves route validity, not product-quality Geppetto/Arachne.

## 4. What comes next? — CONTROLLED DINOv2 REPRESENTATION LADDER

Scientific question:

> With everything after the frozen visual representation held constant, does stronger pretrained representation make the required depth field accessible at the accuracy we now know downstream needs?

Frozen candidates:

- DINOv2 S — 384-d
- DINOv2 B — 768-d
- DINOv2 L — 1024-d
- DINOv2 g — 1536-d

Required controls before training:

- same training-family population for every rung;
- same samples/order/augmentations;
- same 518x518 foundation raster and 37x37 patch grid;
- same native-1024 detail path;
- fixed isometric lift to 1536-d;
- same post-lift normalization;
- identical trainable fusion/head architecture and identical trainable parameter count;
- same optimizer policy and exposure budget;
- no candidate-specific tuning from scientific outcomes;
- actual predicted residual fields replayed through the frozen downstream consumer/proof chain.

The earlier 32/128/512 family ladder confound must not recur.

## 5. Training status

**DINOv2 S/B/L/g training has not started and is not authorized merely by the consumer-interlock PASS.**

Next executable action is to write and seal the controlled DINO ladder preregistration, including the matched exposure/optimizer-budget rule and exact selection criteria. Only after that seal should candidate training begin.

## 6. Later branches, only if needed

If the frozen DINO ladder does not reach the required region:

1. controlled fine-tuning may be opened as a separately preregistered phase;
2. MapAnything / DA3 or other full geometry systems may be compared separately;
3. the final selected model must be judged by its actual residual field through the exact Compiler/consumer chain, not by depth summary metrics alone.

No further cheap deterministic normal/window/hull/persistence tuning should be opened before the representation test.

## One-line state

`depth tolerance measured -> old learner failure decomposed -> real consumer route + coupling validated -> controlled DINOv2 S/B/L/g preregistration next -> then training`
