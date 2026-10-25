# DINOv2 + Patch-Aligned Alpha Residual Court — Preregistration V1

**Status:** sealed before execution.  
**Execution class:** GitHub-hosted CPU.  
**No product / Truth139 / FIT8 authority.**

## Research question

Current IRIS receives authored RGBA but discards alpha by white-compositing before frozen DINOv2-S/14. The identifiability proof shows this can erase both graphics and silhouette support.

The dual-level court separately established that L2+final DINO features can coexist in one 512-D patch token.

This court asks:

> Can a patch-aligned spatial alpha residual be added to that same 512-D scene-token budget while retaining dual-level RGB local detail and final DINO semantics?

It also asks whether a scalar patch-mean alpha summary is sufficient. The answer must be established by a controlled equal-mean alpha task rather than intuition.

## Fixed foundation

Use the exact frozen DINOv2-S/14 authority already admitted by the preceding courts:

- source revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`;
- exact admitted weight SHA256;
- no foundation update;
- existing 37x37 patch lattice.

The RGB local-detail/final-feature corpus is the same controlled corpus used by the dual-level court.

## Alpha construction

Each DINO patch aligns to a conceptual 14x14 = 196-sample alpha patch.

For every training/evaluation row, construct a synthetic 196-D binary alpha patch from **32 independent equal-mass bits**:

- reserve 64 distinct alpha-cell indices as 32 fixed pairs;
- for each bit exactly one member of its pair is alpha=1 and the other is alpha=0;
- all remaining cells are zero;
- therefore every alpha patch has exactly 32 opaque cells and the same patch mean `32/196` regardless of target bits.

Targets are the 32 pair-choice bits.

This construction makes patch mean mathematically uninformative while preserving spatial alpha information.

Alpha targets are generated independently of DINO RGB features and independently across rows/seeds. A no-alpha arm therefore has no legitimate shortcut.

## Arms

All arms emit exactly one 512-D scene image contribution per existing patch token. Token count is unchanged.

1. `DUAL_NO_ALPHA`
   - final DINO -> 384
   - L2 -> 128
   - no alpha input
   - concat = 512
   - this is the current dual-level candidate.

2. `DUAL_MEAN_ALPHA64`
   - final -> 320
   - L2 -> 128
   - scalar alpha mean -> 64
   - concat = 512
   - negative control for spatial alpha information.

3. `DUAL_RAW_ALPHA64`
   - final -> 320
   - L2 -> 128
   - raw 196-D alpha patch -> 64
   - concat = 512.

4. `DUAL_RAW_ALPHA96`
   - final -> 288
   - L2 -> 128
   - raw 196-D alpha patch -> 96
   - concat = 512.

Every candidate adapter must have trainable parameter count <= the existing current final-only adapter (`197,120`). No arm may increase patch-token count.

## Shared readouts / objective

Each arm receives the same three readouts:

- 4-bit local RGB-detail recovery;
- final DINO feature reconstruction, scored by cosine similarity;
- 32-bit alpha-pattern recovery.

Train with equal un-tuned family weights:

`BCE(local) + (1 - cosine(final_reconstruction, final_truth)) + BCE(alpha_bits)`.

No outcome-dependent loss reweighting is permitted in V1.

## Seeds / budget

Use seeds:

`731, 947, 1217, 1597, 2017`

Use the same controlled RGB width strata as the dual-level court (`2,4,8,12` native pixels), the same train/eval corpus sizes, AdamW settings and a fixed step budget declared in the executable probe.

## Validity gates

The court is invalid unless:

- exact DINO authority verification passes;
- final-contract identity check passes;
- all candidate adapter parameter counts are <=197,120;
- alpha target positive fraction is balanced;
- alpha patch mean variance is numerically zero (within tolerance);
- `DUAL_NO_ALPHA` still retains the previously established RGB local/final capability sufficiently to make this a valid shared-capacity test.

## Selection gates

A raw-alpha arm is eligible only if across all seeds/widths:

- mean local RGB-detail accuracy >=0.995 and minimum >=0.98;
- mean final-feature cosine >=0.99 and minimum >=0.98;
- mean alpha-bit accuracy >=0.995 and minimum >=0.98;
- alpha recovery exceeds both `DUAL_NO_ALPHA` and `DUAL_MEAN_ALPHA64` by at least 0.30 absolute accuracy on average;
- parameter count <=197,120.

If both raw64 and raw96 pass, select **raw64** as the smaller alpha subspace. If raw64 fails but raw96 passes, select raw96. If neither passes, no alpha fusion architecture is authorized by V1.

The mean-alpha arm must remain near chance (`<=0.55` mean alpha accuracy) for the equal-mean construction to validate the intended spatial-information test.

## Interpretation limits

A PASS establishes only that spatial alpha can coexist with the dual DINO information inside the bounded patch-token representation. It does not prove real FIT8 superiority, choose an alpha interpolation method for authored 1024 images, or authorize training.

A later real-data court must still validate alpha boundary/support usefulness on authored morphology.
