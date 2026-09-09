# Arachne Mage A0 FIT1 — V7-C2 Frozen Blend-Boundary Localization Preregister

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Parent result: `335139fc8a29392e5dcb63edde9a47110b786f8d`
Diagnostic source: `7fcc750dad1e79ef9d95d7ff1f5a5454c269fa99`
Upstream SkinTokens reference commit: `273b691d35989d71cd17ff2895fdc735097b92d1`

## Question
The exact upstream SkinTokens top-4 production parity improves C2 but leaves 55 supervised Mage rows where a false joint displaces at least one weak true support joint. Before changing training, determine whether these displacement failures are geometrically concentrated at support boundaries in the same qualitative region targeted by SkinTokens `SamplerMix.sample_on_skin()`.

## Frozen contract
- exact C2 treatment final checkpoint SHA-256 `280d126ecb3177dfd718b651a956a65ca8a96bede1952b0b18f7d9a719bad7d0`
- exact Mage cache SHA-256 `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`
- V7 architecture/config unchanged
- no optimizer, backward, training, warm-start, model mutation, K sweep, threshold sweep, output-map sweep, or teacher-dependent prediction
- prediction support is the already-justified fixed SkinTokens parity mapping: sigmoid scalar fields -> top 4 -> renormalize
- teacher weights are used only for localization/evaluation

## Important limitation
The bound Mage cache contains frozen surface samples/positions but not the original triangle-face adjacency used by upstream SkinTokens `SamplerMix.sample_on_skin()`. Therefore this experiment is a **point-cloud boundary localization proxy**, not an exact replay of the upstream face-mask sampler. It can justify or reject the *direction* of the upstream boundary-aware treatment, but must not claim that a specific upstream face would or would not be selected.

## Operational localization
For each joint independently on the 934 supervised Mage rows:
- support = teacher weight > `1e-8`;
- for every active row, compute Euclidean distance in frozen normalized surface-position space to the nearest inactive row of that joint (`true -> boundary/outside` distance);
- for every inactive row, compute distance to the nearest active row of that joint (`false -> support` distance);
- convert each distance to a joint-normalized empirical percentile against the corresponding active or inactive background distribution;
- convert every positive teacher weight to a joint-normalized percentile within that joint's positive-weight distribution.

For every top-4 displacement row, separately record:
1. each missed true joint: teacher weight percentile and active-to-inactive boundary-distance percentile;
2. each false kept joint: inactive-to-active support-distance percentile.

No one-to-one matching between missed and false joints is required; the sets are characterized independently so pairing heuristics cannot create the effect.

## Primary evidence
Three joint-normalized median percentiles:
- missed-true boundary-distance percentile;
- missed-true positive-weight percentile;
- false-kept near-support distance percentile.

Interpretation is relative to `0.5`, the center of each joint's own empirical background distribution.

Pre-registered descriptive tiers:
- **STRONG boundary alignment:** all three medians <= `0.25`.
- **DIRECTIONAL boundary alignment:** all three medians < `0.50` but strong criterion not met.
- **NO CLEAN boundary alignment:** one or more medians >= `0.50`.

Also report fractions <=0.25 and <=0.50, quartiles, raw normalized distances, support-cardinality slices, and the exact displaced row/pair counts.

## Decision rule
- STRONG or DIRECTIONAL alignment: SkinTokens-style boundary-aware dense sampling becomes the next justified single-variable training treatment. Do not add a custom blend-ratio loss in the same experiment.
- NO CLEAN alignment: do not port boundary-aware sampling merely because upstream uses it; return to localization of the residual blend-ordering error before any training intervention.

## Non-decisions
- no C3 ratio loss authorized by this preregistration;
- no blind 4k continuation;
- no architecture/FSQ/token-count change;
- no A1/generalization authorization;
- top-4 is not silently adopted as the RealSaS product gate by this diagnostic.
