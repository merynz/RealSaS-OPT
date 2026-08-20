# V5 Post-Hoc Pair Diagnostic — Development Only

**Status:** `DIAGNOSTIC_ONLY__NO_SELECTION_AUTHORITY`

This analysis was run **after V5 truth-open and after the V5 source/64 assignments were frozen**. It may guide the next preregistration, but it cannot change or rescue V5.

## Question

Do V5 hard-tail regressions coincide with (a) larger true articulation-induced change in pair relations and/or (b) the frozen V5 objective actively preferring the selected non-oracle proposal over an evaluator-oracle proposal?

## Result

On the 79 rank-3 hard-tail view rows:

- V5-vs-V3A improved rows: n=20, mean pair-articulation index `0.02938`, median `0.02650`.
- unchanged rows: n=47, mean `0.02810`, median `0.02696`.
- worsened rows: n=12, mean `0.03706`, median `0.04201`.

Descriptively, a randomly chosen worsened row has a larger articulation index than a randomly chosen improved row with probability/AUC about `0.675`. A one-sided Mann–Whitney comparison is `p≈0.053`; this was **not preregistered** and is diagnostic only.

For rows where an evaluator-oracle proposal is legal under the frozen peers:

- worsened: 10/10 oracle alternatives have positive total local objective delta (the frozen objective prefers the V5 choice); 6/10 have a positive R-component penalty; 8/10 have a positive G-component penalty;
- improved: 6/19 have positive total local objective delta; 5/19 positive R penalty; 6/19 positive G penalty;
- same: 32/45 positive total local objective delta; 15/45 positive R penalty; 31/45 positive G penalty.

This supports a real **pair-factor reliability / override** problem, but not a pure articulation-only explanation. In particular `14404/carrier32` regresses catastrophically despite a comparatively modest mean articulation index, so a simple articulation threshold would be unjustified.

## Scientific implication

The next experiment must not be described as “F/D is now required.” V5 shows that R still has unused signal, and this diagnostic shows that some full-pair factors are misleading while others correct large errors.

A next R-only preregistration is justified if it changes exactly one thing: **how observable confidence/reliability of pairwise R factors controls their contribution**, without using evaluator truth, F, D, XYZ collapse, or post-truth thresholds. G/proposal evidence and evaluator semantics must remain frozen.

If that controlled R-reliability experiment fails on an untouched validation surface, then adding F or D becomes the justified next information-channel test.

Raw diagnostic SHA-256: `b1faa6a7b0fa6fb6d63eab9ec7b247c026a4f8b15e3569b1eda844150c02bcef`.
