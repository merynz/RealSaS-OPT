# Arachne Mage A0 FIT1 — V7-C2 SkinTokens Top-4 Production Parity Result

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Preregister commit: `d021649c0559a283e27f9506029584c3e537329b`
Diagnostic source commit: `cbc940432fdaef4adb2cdbf783f0f54142b6f79e`
C2 treatment model SHA-256: `280d126ecb3177dfd718b651a956a65ca8a96bede1952b0b18f7d9a719bad7d0`
Upstream SkinTokens audited commit: `273b691d35989d71cd17ff2895fdc735097b92d1`

## Contract
Frozen C2 treatment checkpoint. No training, no optimizer/backward, no K sweep, no threshold sweep, no teacher support in prediction mapping, no optional voxel postprocess.

Prediction mapping only:
`sigmoid scalar fields -> keep top 4 predicted joints per row -> renormalize retained weights`.

K=4 is taken from the upstream SkinTokens production export contract. Mage truth maximum support is exactly 4.

## Baseline replay
- row-L1 mean: `0.05629588442160142`
- row-L1 p95: `0.22197734363017302`
- CVaR10: `0.2754472310858055`
- deformation-error ratio: `0.08492328226566315`
- dominant accuracy: `0.9946466809421841`
- teacher dominant top3 inclusion: `1.0`
- inactive predicted mass mean: `0.020908252814482105`

## Exact SkinTokens top-4 parity result
- row-L1 mean: `0.047605848917153165`
- row-L1 p95: `0.17656562418530558`
- CVaR10: `0.23241348651931396`
- deformation-error ratio: `0.0774342343211174`
- dominant accuracy: `0.9946466809421841`
- teacher dominant top3 inclusion: `1.0`
- inactive predicted mass mean: `0.016307435695669493`
- inactive predicted mass p95: `0.08602921705628643`
- p95-tail inactive mass mean: `0.10298022971616964`
- discarded predicted mass mean: `0.005467101267268291`
- discarded predicted mass p95: `0.037398991330854406`
- true-support recall in predicted top4: `0.9690315315315315`
- full true support contained: `879 / 934 = 0.9411134903640257`
- rows where a false joint displaces a true support joint: `55`

Delta top4 minus baseline:
- p95: `-0.04541171944486744`
- deform: `-0.007489047944545746`
- mean: `-0.008690035504448253`
- CVaR10: `-0.04303374456649153`

Top-4 improves every primary error metric but does not close either FIT1 gate.

## Support-cardinality slices under predicted top-4
Truth support size 1 (444 rows):
- true-support recall: `1.0`
- full support contained: `1.0`
- row-L1 p95: `0.005164429363523564`

Truth support size 2 (149 rows):
- true-support recall: `0.9899328859060402`
- full support contained: `0.9798657718120806`
- row-L1 p95: `0.21041882529019287`

Truth support size 3 (330 rows):
- true-support recall: `0.9545454545454546`
- full support contained: `0.8636363636363636`
- row-L1 p95: `0.18943639092836254`

Truth support size 4 (11 rows):
- true-support recall: `0.8409090909090909`
- full support contained: `0.36363636363636365`
- row-L1 p95: `0.1567962987376787`

## C2 teacher-support oracle on the same checkpoint
Diagnostic only; teacher support is not used for product prediction.
- row-L1 mean: `0.019796774312323736`
- row-L1 p95: `0.0801826793484964`
- CVaR10: `0.08930489607657312`
- deformation-error ratio: `0.030969249084591866`

## Causal interpretation
The upstream top-4 production contract is a real but secondary contributor. It removes some diffuse inactive leakage and improves p95/deformation, but it cannot approach the teacher-support oracle because predicted top-4 sometimes contains false joints that displace weak true blend supports.

The remaining learned blocker is now narrower than generic `inactive leakage` or `blend-ratio objective`:

**`WEAK_TRUE_VS_FALSE_SUPPORT_ORDERING_AT_BLEND_BOUNDARIES`**

Evidence:
1. pure 1-joint rows are effectively closed and have perfect top-4 support containment;
2. failures increase with truth support cardinality;
3. 55 supervised rows lose at least one true support joint from predicted top-4;
4. exact teacher-support projection would reduce p95 from ~0.222 to ~0.080 and deformation from ~0.085 to ~0.031, while blind top-4 only reaches ~0.177/~0.077;
5. upstream SkinTokens has a directly relevant training-data mechanism not yet matched by RealSaS: per-bone dense surface sampling over nonzero support plus a geometric near-support neighborhood.

## Decision
- Top-4 parity is informative but not sufficient; do not silently change the RealSaS FIT1 gate or declare closure.
- No blind 4k extension.
- No custom blend-ratio loss authorized from this result.
- No architecture/FSQ/token-count change.
- Before any new training treatment, run a frozen localization diagnostic on the 55 displaced-support rows to verify that their false-positive / missed-true ordering is geometrically concentrated near per-bone support boundaries and weak-weight regions.
- Only if that localization matches the upstream sampler mechanism should the next treatment port SkinTokens-style boundary-aware dense sampling as a single justified variable.
