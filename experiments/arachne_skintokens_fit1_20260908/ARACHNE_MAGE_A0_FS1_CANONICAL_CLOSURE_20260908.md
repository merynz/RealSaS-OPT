# RealSaS — Arachne Mage A0 FS1 Closure — 2026-09-08

Status: `FAIL__NO_A0_TERMINAL_CLOSURE`

A0 ran the frozen preregistered 1536-step budget on the sealed FS1 target/cache and terminated with streak `0/3`. The only acceptance family that failed was row-wise reconstruction tail fidelity: raw/qualified row-L1 p95 = `0.11468168/0.11468171` against the frozen `<=0.05` gates. Deformation ratio passed (`0.04871330/0.04871329`), as did finite/nonnegative/simplex, 950-row Compiler qualification, correction budgets, and zero sparsification discard.

The final tail is plateaued rather than budget-limited: row-L1 p95 is ~`0.1143–0.1147` from steps 1376–1536 while cosine LR decays to zero. No threshold, architecture, optimizer, scheduler, target, or training budget was changed after observing the result. A1 remains forbidden.

Post-failure diagnostic (non-authorizing) reconstructed the final W-hat from the sealed step-1536 progress checkpoint. Among 934 supervised rows, 165 exceed L1 0.05 and 59 exceed 0.10; 158/165 bad rows are HIGH-confidence FS1 truth. Bad rows are materially more blended (mean active influences >=1e-3: 2.63 vs 1.70; teacher entropy 0.65 vs 0.29), with additional sharp rigid failures around hand/handslot fields. This supports a representation/objective/capacity diagnostic, not a target-confidence excuse.

A new training attempt requires a separately preregistered post-failure hypothesis.
